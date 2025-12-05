#!/usr/bin/env python3
"""
Global phospho-network model with protein-specific inputs,
abundance dynamics, and global/local coupling from PTM SQLite DBs.

PYMOO VERSION: Uses scipy.integrate.odeint (LSODA) for integration
and pymoo.algorithms.soo.nonconvex.de.CMA-ES (Differential Evolution) for optimization.
Parallelized with multiprocessing (32 cores) using StarmapParallelization.
"""

from __future__ import annotations

import argparse
import os
import re
import sqlite3
import pickle
import multiprocessing

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# Pymoo Algorithm imports
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.algorithms.moo.unsga3 import UNSGA3
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.operators.sampling.lhs import LHS

from pymoo.optimize import minimize
from pymoo.core.problem import ElementwiseProblem
from pymoo.parallelization.starmap import StarmapParallelization
from pymoo.termination.default import DefaultSingleObjectiveTermination, DefaultMultiObjectiveTermination

import warnings
from scipy.integrate import odeint
from scipy.integrate import ODEintWarning

warnings.filterwarnings("ignore", category=ODEintWarning)

# Global dims
GLOBAL_K = None
GLOBAL_M = None
GLOBAL_N = None

# ------------------------------------------------------------
#  TIMEPOINTS & DATA
# ------------------------------------------------------------

DEFAULT_TIMEPOINTS = np.array(
    [0.0, 0.5, 0.75, 1.0, 2.0, 4.0, 8.0, 16.0,
     30.0, 60.0, 120.0, 240.0, 480.0, 960.0]
)

EPS = 1e-8


def load_site_data(path, timepoints=DEFAULT_TIMEPOINTS):
    """
    Load time-series from a file.
    """
    df = pd.read_csv(path, sep=None, engine="python")

    value_cols = [c for c in df.columns if c.startswith("v") or c.startswith("x")]
    if len(value_cols) != len(timepoints):
        raise ValueError(f"Expected {len(timepoints)} value columns, found {len(value_cols)}")

    if "Protein" in df.columns:
        prot_col = "Protein"
    elif "GeneID" in df.columns:
        prot_col = "GeneID"
    else:
        raise ValueError("Need either 'Protein' or 'GeneID' column in data.")

    if "Psite" in df.columns:
        has_site = df["Psite"].notna()
    elif "Residue" in df.columns:
        has_site = df["Residue"].notna()
    else:
        raise ValueError("Need either 'Residue' or 'Psite' column in data.")

    df_sites = df[has_site].copy()
    df_prot = df[~has_site].copy()

    proteins_raw = df_sites[prot_col].astype(str).tolist()
    residues_raw = []
    positions = []

    if "Residue" in df_sites.columns:
        for r in df_sites["Residue"].astype(str):
            residues_raw.append(r)
            m = re.match(r"[A-Z]([0-9]+)", r)
            positions.append(int(m.group(1)) if m else np.nan)
    elif "Psite" in df_sites.columns:
        for psite in df_sites["Psite"]:
            psite = str(psite)
            if "_" in psite:
                aa, pos = psite.split("_", 1)
                residues_raw.append(f"{aa}{pos}")
                try:
                    positions.append(int(pos))
                except ValueError:
                    positions.append(np.nan)
            else:
                residues_raw.append(psite)
                m = re.match(r"[A-Z]([0-9]+)", psite)
                positions.append(int(m.group(1)) if m else np.nan)

    positions = np.array(positions, dtype=float)
    sites = [f"{p}_{r}" for p, r in zip(proteins_raw, residues_raw)]
    proteins = sorted(set(proteins_raw))
    prot_index = {p: k for k, p in enumerate(proteins)}
    site_prot_idx = np.array([prot_index[p] for p in proteins_raw], dtype=int)

    Y = df_sites[value_cols].values.astype(float)
    t = np.array(timepoints, dtype=float)

    A_data = None
    A_proteins = None

    if not df_prot.empty:
        A_rows = []
        A_prots = []
        for p, sub in df_prot.groupby(prot_col):
            p_str = str(p)
            if p_str not in prot_index:
                continue
            A_prots.append(p_str)
            A_rows.append(sub[value_cols].values.astype(float).mean(axis=0))

        if A_rows:
            A_data = np.vstack(A_rows)
            A_proteins = np.array(A_prots, dtype=object)

    return sites, proteins, site_prot_idx, positions, t, Y, A_data, A_proteins


def scale_fc_to_unit_interval(Y, use_log=False, high_percentile=90.0):
    """Per site transform FC -> p in [0,1]."""
    N, T = Y.shape
    P = np.zeros_like(Y, dtype=float)
    baselines = np.zeros(N, dtype=float)
    amplitudes = np.zeros(N, dtype=float)
    eps = 1e-6

    for i in range(N):
        y_raw = Y[i].astype(float)
        if use_log:
            y = np.log1p(y_raw)
        else:
            y = y_raw

        b = y[0]
        P_y = np.percentile(y, high_percentile)
        A = P_y - b
        if A < eps:
            A = 1.0

        p = (y - b) / A
        P[i] = p
        baselines[i] = b
        amplitudes[i] = A

    return P, baselines, amplitudes


def row_normalize(C):
    row_sums = C.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0.0] = 1.0
    C_norm = C / row_sums
    return C_norm


# ------------------------------------------------------------
#  DB & MATRIX BUILDERS
# ------------------------------------------------------------

def build_C_matrices_from_db(ptm_intra_path, ptm_inter_path,
                             sites, site_prot_idx, positions,
                             proteins, length_scale=50.0):
    N = len(sites)
    idx = {s: i for i, s in enumerate(sites)}
    Cg = np.zeros((N, N), dtype=float)

    # Intra
    conn_i = sqlite3.connect(ptm_intra_path)
    cur_i = conn_i.cursor()
    for protein, res1, r1, res2, r2 in cur_i.execute(
            "SELECT protein, residue1, score1, residue2, score2 FROM intra_pairs"
    ):
        s1 = f"{protein}_{res1}"
        s2 = f"{protein}_{res2}"
        if s1 in idx and s2 in idx:
            i, j = idx[s1], idx[s2]
            score = 0.8 * (r1 + r2) / 200.0
            if score > Cg[i, j]:
                Cg[i, j] = Cg[j, i] = score
    conn_i.close()

    # Inter
    conn_e = sqlite3.connect(ptm_inter_path)
    cur_e = conn_e.cursor()
    for p1, res1, r1, p2, res2, r2 in cur_e.execute(
            "SELECT protein1, residue1, score1, protein2, residue2, score2 FROM inter_pairs"
    ):
        s1 = f"{p1}_{res1}"
        s2 = f"{p2}_{res2}"
        if s1 in idx and s2 in idx:
            i, j = idx[s1], idx[s2]
            score = 0.8 * (r1 + r2) / 200.0
            if score > Cg[i, j]:
                Cg[i, j] = Cg[j, i] = score
    conn_e.close()

    # Local
    Cl = np.zeros((N, N), dtype=float)
    L = float(length_scale)
    for i in range(N):
        for j in range(N):
            if i == j: continue
            if site_prot_idx[i] != site_prot_idx[j]: continue
            if np.isfinite(positions[i]) and np.isfinite(positions[j]):
                d = abs(positions[i] - positions[j])
                Cl[i, j] = np.exp(-d / L)

    return Cg, Cl


def load_kinase_site_matrix(path, sites):
    df = pd.read_csv(path, sep="\t")
    if "weight" not in df.columns: df["weight"] = 1.0

    site_to_idx = {s: i for i, s in enumerate(sites)}
    kinases = sorted(df["Kinase"].astype(str).unique())
    kin_index = {k: j for j, k in enumerate(kinases)}

    N, M = len(sites), len(kinases)
    K_site_kin = np.zeros((N, M), dtype=float)

    for _, row in df.iterrows():
        s, k, w = str(row["Site"]), str(row["Kinase"]), float(row["weight"])
        if s in site_to_idx and k in kin_index:
            i, j = site_to_idx[s], kin_index[k]
            if w > K_site_kin[i, j]:
                K_site_kin[i, j] = w
    return K_site_kin, kinases


def build_kinase_site_from_kea(ks_psite_table_path, sites):
    df = pd.read_csv(ks_psite_table_path, sep="\t")
    df["substrate_site"] = df["substrate_site"].astype(str).str.upper()
    df["kinase"] = df["kinase"].astype(str).str.upper()

    sites_upper = [s.upper() for s in sites]
    site_to_idx = {s: i for i, s in enumerate(sites_upper)}

    df = df[df["substrate_site"].isin(site_to_idx.keys())].copy()
    if df.empty:
        raise ValueError("No overlap between ks_psite_table and model sites.")

    grouped = df.groupby(["substrate_site", "kinase"]).agg(weight=("pmid", "nunique")).reset_index()
    kinases = sorted(grouped["kinase"].unique())
    kin_index = {k: j for j, k in enumerate(kinases)}

    N, M = len(sites), len(kinases)
    K_site_kin = np.zeros((N, M), dtype=float)

    for _, row in grouped.iterrows():
        s, k, w = row["substrate_site"], row["kinase"], float(row["weight"])
        i, j = site_to_idx[s], kin_index[k]
        if w > K_site_kin[i, j]:
            K_site_kin[i, j] = w

    row_sums = K_site_kin.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0.0] = 1.0
    return K_site_kin / row_sums, kinases


def build_alpha_laplacian_from_unified_graph(pkl_path, kinases, weight_attr="weight_mean"):
    with open(pkl_path, "rb") as f:
        G_full = pickle.load(f)

    kin_to_idx = {k: i for i, k in enumerate(kinases)}
    M = len(kinases)
    A = np.zeros((M, M), dtype=float)

    for u, v, data in G_full.edges(data=True):
        if u in kin_to_idx and v in kin_to_idx and u != v:
            i, j = kin_to_idx[u], kin_to_idx[v]
            w = float(data.get(weight_attr, 1.0))
            if w > A[i, j]:
                A[i, j] = A[j, i] = w

    L = np.diag(A.sum(axis=1)) - A
    return L


# ------------------------------------------------------------
#  NUMPY/SCIPY ODE MODEL
# ------------------------------------------------------------

def decode_theta(theta):
    K = GLOBAL_K
    M = GLOBAL_M
    N = GLOBAL_N
    idx0 = 0

    log_k_act = theta[idx0:idx0 + K];
    idx0 += K
    log_k_deact = theta[idx0:idx0 + K];
    idx0 += K
    log_s_prod = theta[idx0:idx0 + K];
    idx0 += K
    log_d_deg = theta[idx0:idx0 + K];
    idx0 += K
    log_beta_g = theta[idx0];
    idx0 += 1
    log_beta_l = theta[idx0];
    idx0 += 1
    log_alpha = theta[idx0:idx0 + M];
    idx0 += M
    log_kK_act = theta[idx0:idx0 + M];
    idx0 += M
    log_kK_deact = theta[idx0:idx0 + M];
    idx0 += M
    log_k_off = theta[idx0:idx0 + N]

    # Numpy exp and clip
    k_act = np.exp(np.clip(log_k_act, -20, 10))
    k_deact = np.exp(np.clip(log_k_deact, -20, 10))
    s_prod = np.exp(np.clip(log_s_prod, -20, 10))
    d_deg = np.exp(np.clip(log_d_deg, -20, 10))
    beta_g = np.exp(np.clip(log_beta_g, -20, 10))
    beta_l = np.exp(np.clip(log_beta_l, -20, 10))
    alpha = np.exp(np.clip(log_alpha, -20, 10))
    kK_act = np.exp(np.clip(log_kK_act, -20, 10))
    kK_deact = np.exp(np.clip(log_kK_deact, -20, 10))
    k_off = np.exp(np.clip(log_k_off, -20, 10))

    return (k_act, k_deact, s_prod, d_deg, beta_g, beta_l, alpha, kK_act, kK_deact, k_off)


def network_rhs(x, t, theta, Cg, Cl, site_prot_idx, K_site_kin, R, L_alpha, kin_to_prot_idx):
    K = GLOBAL_K
    M = GLOBAL_M
    N = GLOBAL_N

    (k_act, k_deact, s_prod, d_deg,
     beta_g, beta_l, alpha,
     kK_act, kK_deact, k_off) = decode_theta(theta)

    S    = x[0:K]
    A    = x[K:2*K]
    Kdyn = x[2*K:2*K+M]
    p    = x[2*K+M:2*K+M+N]

    # ---------- summaries ----------
    p_sum   = np.bincount(site_prot_idx, weights=p, minlength=K)
    p_count = np.bincount(site_prot_idx, minlength=K)
    p_mean_by_prot = p_sum / np.maximum(p_count, 1)

    p_kin = R @ p  # M-dim

    # site-level context
    coup_g_sites = Cg @ p
    coup_l_sites = Cl @ p
    coup_sites   = np.tanh(beta_g * coup_g_sites + beta_l * coup_l_sites)

    ctx_sum   = np.bincount(site_prot_idx, weights=coup_sites, minlength=K)
    ctx_mean_by_prot = ctx_sum / np.maximum(p_count, 1)

    # ---------- S dynamics ----------
    gamma_S_p   = 1.0
    gamma_S_ctx = 0.5

    effective_drive_S = 1.0 + gamma_S_p * p_mean_by_prot + gamma_S_ctx * ctx_mean_by_prot
    effective_drive_S = np.clip(effective_drive_S, 0.0, 10.0)

    dS = k_act * effective_drive_S * (1.0 - S) - k_deact * S

    # ---------- A dynamics ----------
    gamma_A_S = 0.5
    gamma_A_p = 0.5

    s_eff = s_prod * (1.0 + gamma_A_S * S)
    s_eff = np.clip(s_eff, 0.0, 10.0)

    deg_mod = 1.0 + gamma_A_p * p_mean_by_prot
    deg_mod = np.clip(deg_mod, 0.0, 10.0)

    dA = s_eff - d_deg * deg_mod * A

    # ---------- Kdyn dynamics ----------
    gamma_K_net = 0.5
    gamma_K_S   = 0.3
    gamma_K_A   = 0.3

    u_sub = p_kin
    u_net = -gamma_K_net * (L_alpha @ Kdyn)

    S_for_kin = np.zeros(M)
    A_for_kin = np.zeros(M)
    for j in range(M):
        p_idx = kin_to_prot_idx[j]
        if p_idx >= 0:
            S_for_kin[j] = S[p_idx]
            A_for_kin[j] = A[p_idx]

    u_SA = gamma_K_S * S_for_kin + gamma_K_A * A_for_kin

    u_total = u_sub + u_net + u_SA
    u_total = np.clip(u_total, -10.0, 10.0)

    dK = kK_act * np.tanh(u_total) * (1.0 - Kdyn) - kK_deact * Kdyn

    # ---------- p dynamics ----------
    k_on_eff = K_site_kin @ (alpha * Kdyn)
    S_local  = S[site_prot_idx]
    A_local  = A[site_prot_idx]

    v_on  = (k_on_eff * S_local * A_local + coup_sites) * (1.0 - p)
    v_off = k_off * p
    dp    = v_on - v_off

    return np.concatenate([dS, dA, dK, dp])

def simulate_p_scipy(t_arr, P_data0, theta,
                     Cg, Cl, site_prot_idx,
                     K_site_kin, R,
                     L_alpha, kin_to_prot_idx):
    K = GLOBAL_K
    M = GLOBAL_M
    N = GLOBAL_N
    state_dim = 2 * K + M + N

    x0 = np.zeros((state_dim,))
    x0[K:2 * K] = 1.0
    x0[2 * K:2 * K + M] = 1.0
    x0[2 * K + M:] = P_data0[:, 0]

    xs = odeint(
        network_rhs,
        x0,
        t_arr,
        args=(theta, Cg, Cl, site_prot_idx,
              K_site_kin, R, L_alpha, kin_to_prot_idx)
    )

    A_sim = xs[:, K:2 * K].T
    P_sim = xs[:, 2 * K + M:].T

    return P_sim, A_sim

# ------------------------------------------------------------
#  PYMOO PROBLEM DEFINITION
# ------------------------------------------------------------

class NetworkOptimizationProblem(ElementwiseProblem):
    def __init__(self,
                 t, P_data, Cg, Cl, site_prot_idx, K_site_kin, R,
                 A_scaled, prot_idx_for_A, W_data, W_data_prot,
                 L_alpha, kin_to_prot_idx,
                 lambda_net, reg_lambda,
                 xl, xu,
                 **kwargs):

        n_var = len(xl)
        super().__init__(n_var=n_var, n_obj=3, n_ieq_constr=0, xl=xl, xu=xu, **kwargs)

        self.t = t
        self.P_data = P_data
        self.Cg = Cg
        self.Cl = Cl
        self.site_prot_idx = site_prot_idx
        self.K_site_kin = K_site_kin
        self.R = R
        self.A_scaled = A_scaled
        self.prot_idx_for_A = prot_idx_for_A
        self.W_data = W_data
        self.W_data_prot = W_data_prot
        self.L_alpha = L_alpha
        self.kin_to_prot_idx = kin_to_prot_idx
        self.lambda_net = lambda_net
        self.reg_lambda = reg_lambda

        self.n_p = max(1, self.P_data.size)
        self.n_A = max(1, self.A_scaled.size)
        self.n_var = n_var

    def _evaluate(self, x, out, *args, **kwargs):
        theta = x

        # --- simulate ---
        P_sim, A_sim = simulate_p_scipy(
            self.t, self.P_data, theta,
            self.Cg, self.Cl, self.site_prot_idx,
            self.K_site_kin, self.R,
            self.L_alpha, self.kin_to_prot_idx
        )

        # ---- safety check: explosion / NaNs ----
        if (not np.all(np.isfinite(P_sim)) or
            not np.all(np.isfinite(A_sim)) or
            np.max(np.abs(P_sim)) > 1e6 or
            np.max(np.abs(A_sim)) > 1e6):

            # terrible but finite point; constraints 0 so dominated
            out["F"] = np.array([1e12, 1e12, 1e12])
            # out["G"] = np.zeros(3)
            return

        # ---- objective 1: phosphosite fit ----
        diff_p = (self.P_data - P_sim) * np.sqrt(self.W_data)
        diff_p = np.clip(diff_p, -1e6, 1e6)
        loss_p = np.square(diff_p, dtype=np.float64).sum()
        f1 = loss_p / self.n_p

        # ---- objective 2: protein abundance fit ----
        if self.A_scaled.size > 0:
            A_model = A_sim[self.prot_idx_for_A, :]
            diff_A = (self.A_scaled - A_model) * np.sqrt(self.W_data_prot)
            diff_A = np.clip(diff_A, -1e6, 1e6)
            loss_A = np.square(diff_A, dtype=np.float64).sum()
        else:
            loss_A = 0.0
        f2 = loss_A / self.n_A

        # ---- regularization terms ----
        (k_act, k_deact, s_prod, d_deg,
         beta_g, beta_l, alpha,
         kK_act, kK_deact, k_off) = decode_theta(theta)

        reg = self.reg_lambda * np.dot(theta, theta)
        reg_net = self.lambda_net * (alpha @ (self.L_alpha @ alpha))

        # objective 3: model/network complexity
        f3 = (reg + reg_net) / self.n_var

        out["F"] = np.array([f1, f2, f3])

        # # --------- PARAMETER INEQUALITY CONSTRAINTS (G <= 0) ---------
        #
        # # 1) degradation slower than deactivation: d_deg[k] <= rho * k_deact[k]
        # rho = 0.2
        # c_degrade = np.max(d_deg - rho * k_deact)
        #
        # # 2) alpha L2 norm cap: ||alpha||_2 <= alpha_max
        # alpha_max = 20.0
        # c_alpha_norm = np.linalg.norm(alpha) - alpha_max
        #
        # # 3) kinase time scales ~ signaling time scale
        # k_act_mean = np.mean(k_act)
        # r = 10.0  # one order of magnitude
        # c_kK_upper = np.max(kK_act - r * k_act_mean)
        # c_kK_lower = np.max((k_act_mean / r) - kK_act)
        # c_kK_deact_upper = np.max(kK_deact - r * k_act_mean)
        # c_kK_deact_lower = np.max((k_act_mean / r) - kK_deact)
        # c_kK = max(c_kK_upper, c_kK_lower, c_kK_deact_upper, c_kK_deact_lower)
        #
        # out["G"] = np.array([c_degrade, c_alpha_norm, c_kK])

def bio_score(theta):
    (k_act, k_deact, s_prod, d_deg,
     beta_g, beta_l, alpha,
     kK_act, kK_deact, k_off) = decode_theta(theta)

    t_half_kinase = np.log(2) / kK_deact
    t_half_protein = np.log(2) / d_deg

    median_t_kinase = np.median(t_half_kinase)
    median_t_protein = np.median(t_half_protein)

    T_kinase_prior = 10.0
    T_protein_prior = 600.0

    term1 = (np.log10(median_t_kinase) - np.log10(T_kinase_prior))**2
    term2 = (np.log10(median_t_protein) - np.log10(T_protein_prior))**2

    return term1 + term2

# ------------------------------------------------------------
#  MAIN
# ------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Pymoo Differential Evolution Fit")
    parser.add_argument("--data", required=True)
    parser.add_argument("--ptm-intra", required=True)
    parser.add_argument("--ptm-inter", required=True)
    parser.add_argument("--outdir", default="network_fit_pymoo")
    parser.add_argument("--length-scale", type=float, default=50.0)
    parser.add_argument("--crosstalk-tsv")
    parser.add_argument("--kinase-tsv")
    parser.add_argument("--kea-ks-table")
    parser.add_argument("--unified-graph-pkl")
    parser.add_argument("--lambda-net", type=float, default=0.0)
    parser.add_argument("--pop-size", type=int, default=200, help="Population size multiplier for CMA-ES")
    parser.add_argument("--cores", type=int, default=60, help="Number of cores for multiprocessing")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Load Data
    (sites, proteins, site_prot_idx, positions, t, Y, A_data, A_proteins) = load_site_data(args.data)
    print(f"[*] Loaded {len(sites)} sites, {len(proteins)} proteins.")

    # Crosstalk filtering
    if args.crosstalk_tsv:
        def load_allowed(path):
            df = pd.read_csv(path, sep="\t")
            s = set()
            for c in ["Site1", "Site2"]:
                for p, site in zip(df["Protein"], df[c]):
                    if pd.notna(p) and pd.notna(site): s.add(f"{p}_{site}")
            return s

        allowed = load_allowed(args.crosstalk_tsv)
        mask = np.array([s in allowed for s in sites], dtype=bool)
        sites = [s for s, m in zip(sites, mask) if m]
        positions = positions[mask]
        Y = Y[mask, :]
        site_prot_idx = site_prot_idx[mask]
        prots_used = sorted({s.split("_", 1)[0] for s in sites})
        prot_map = {p: i for i, p in enumerate(prots_used)}
        site_prot_idx = np.array([prot_map[s.split("_", 1)[0]] for s in sites], dtype=int)
        proteins = prots_used
        print(f"[*] Filtered to {len(sites)} sites.")

    # Scaling
    P_scaled, baselines, amplitudes = scale_fc_to_unit_interval(Y)

    if A_data is not None and len(A_data) > 0:
        prot_map = {p: i for i, p in enumerate(proteins)}
        mask_A = [p in prot_map for p in A_proteins]
        A_data = A_data[mask_A]
        A_proteins = A_proteins[mask_A]
        prot_idx_for_A = np.array([prot_map[p] for p in A_proteins], dtype=int)
        A_scaled, A_bases, A_amps = scale_fc_to_unit_interval(A_data)
    else:
        A_scaled = np.zeros((0, P_scaled.shape[1]))
        prot_idx_for_A = np.array([], dtype=int)
        A_bases, A_amps = np.array([]), np.array([])

    # ---- weights based on noise + early emphasis ----

    # time weights
    tau = 10.0
    w_time = np.exp(-t / tau)
    w_time /= w_time.mean()

    # site weights
    logY = np.log1p(np.clip(Y, 1e-3, None))
    diff = np.diff(logY, axis=1)
    sigma_site = np.sqrt((diff ** 2).mean(axis=1) + 1e-8)

    w_site = 1.0 / (sigma_site ** 2 + 1e-4)

    # cap extreme weights (sites that look perfectly flat)
    w_site = np.clip(w_site, 0.1, 20.0)
    w_site /= w_site.mean()

    # protein weights
    if A_data is not None and len(A_data) > 0:
        logA = np.log1p(np.clip(A_data, 1e-3, None))
        diffA = np.diff(logA, axis=1)
        sigma_prot = np.sqrt((diffA ** 2).mean(axis=1) + 1e-8)
        w_prot = 1.0 / (sigma_prot ** 2 + 1e-4)
        w_prot = np.clip(w_prot, 0.1, 20.0)
        w_prot /= w_prot.mean()

    W_data = np.outer(w_site, w_time)

    if A_data is not None and len(A_data) > 0:
        W_data_prot = np.outer(w_prot, w_time)
    else:
        W_data_prot = np.zeros((0, P_scaled.shape[1]))

    # Matrices
    Cg, Cl = build_C_matrices_from_db(args.ptm_intra, args.ptm_inter, sites, site_prot_idx, positions, proteins,
                                      args.length_scale)
    Cg, Cl = row_normalize(Cg), row_normalize(Cl)

    if args.kinase_tsv:
        K_site_kin, kinases = load_kinase_site_matrix(args.kinase_tsv, sites)
    elif args.kea_ks_table:
        K_site_kin, kinases = build_kinase_site_from_kea(args.kea_ks_table, sites)
    else:
        K_site_kin = np.eye(len(sites))
        kinases = [f"K_{i}" for i in range(len(sites))]
    R = K_site_kin.T

    L_alpha = np.zeros((len(kinases), len(kinases)))
    if args.unified_graph_pkl and args.lambda_net > 0:
        L_alpha = build_alpha_laplacian_from_unified_graph(args.unified_graph_pkl, kinases)

    # After you have `proteins` and `kinases`
    prot_map_all = {p: i for i, p in enumerate(proteins)}
    kin_to_prot_idx = np.array(
        [prot_map_all.get(k, -1) for k in kinases],
        dtype=int
    )

    # Globals
    global GLOBAL_K, GLOBAL_M, GLOBAL_N
    GLOBAL_K, GLOBAL_N = len(proteins), len(sites)
    GLOBAL_M = len(kinases)

    # Params and Bounds
    K, M, N = GLOBAL_K, GLOBAL_M, GLOBAL_N
    dim = 4 * K + 2 + 3 * M + N

    # Init params for reference
    # Bounds: approx -11.5 (log 1e-5) to 2.3 (log 10)
    low_b = np.log(1)
    high_b = np.log(10)

    xl = np.full(dim, low_b)
    xu = np.full(dim, high_b)

    # # --- INTELLIGENT BOUNDS SETTING ---
    # # We create specific bounds for specific parameter types rather than one global bound.
    #
    # xl = np.zeros(dim)
    # xu = np.zeros(dim)
    #
    # idx = 0
    #
    # # 1. Protein Kinetics (K)
    # # k_act (Activation): Allow range [1e-5, 100] - Can be fast
    # xl[idx:idx + K] = np.log(1e-5);
    # xu[idx:idx + K] = np.log(100.0);
    # idx += K
    #
    # # k_deact (Deactivation): Allow range [1e-5, 100] - Can be fast
    # xl[idx:idx + K] = np.log(1e-5);
    # xu[idx:idx + K] = np.log(100.0);
    # idx += K
    #
    # # s_prod (Synthesis): Allow range [1e-5, 10]
    # xl[idx:idx + K] = np.log(1e-5);
    # xu[idx:idx + K] = np.log(10.0);
    # idx += K
    #
    # # d_deg (Degradation): *** RESTRICT THIS ***
    # # Force degradation to be slow (e.g., max 0.5).
    # # This forces the model to use PHOSPHORYLATION dynamics, not abundance destruction.
    # xl[idx:idx + K] = np.log(1e-5);
    # xu[idx:idx + K] = np.log(0.5);
    # idx += K
    #
    # # 2. Coupling (2)
    # xl[idx] = np.log(1e-5);
    # xu[idx] = np.log(10.0);
    # idx += 1  # beta_g
    # xl[idx] = np.log(1e-5);
    # xu[idx] = np.log(10.0);
    # idx += 1  # beta_l
    #
    # # 3. Kinase Params (M)
    # # Alpha (Global Strength): Allow high
    # xl[idx:idx + M] = np.log(1e-5);
    # xu[idx:idx + M] = np.log(100.0);
    # idx += M
    # # Kinase Act/Deact
    # xl[idx:idx + M] = np.log(1e-5);
    # xu[idx:idx + M] = np.log(100.0);
    # idx += M
    # xl[idx:idx + M] = np.log(1e-5);
    # xu[idx:idx + M] = np.log(100.0);
    # idx += M
    #
    # # 4. Site Params (N)
    # # k_off (Phosphatase): Allow range [1e-5, 50] - High but not 100
    # xl[idx:idx + N] = np.log(1e-5);
    # xu[idx:idx + N] = np.log(50.0);
    # idx += N

    # ----------------------------------------------------------
    # PYMOO SETUP
    # ----------------------------------------------------------

    # Initialize Pool
    print(f"[*] Initializing multiprocessing pool with {args.cores} cores...")
    pool = multiprocessing.Pool(args.cores)
    runner = StarmapParallelization(pool.starmap)

    # Initialize Problem
    problem = NetworkOptimizationProblem(
        t, P_scaled, Cg, Cl, site_prot_idx, K_site_kin, R,
        A_scaled, prot_idx_for_A, W_data, W_data_prot, L_alpha,
        kin_to_prot_idx,
        args.lambda_net, 1e-4,
        xl, xu,
        elementwise_runner=runner
    )

    # Termination for multi-objective

    termination = DefaultMultiObjectiveTermination(
        xtol=1e-8,
        cvtol=1e-6,
        ftol=0.0025,
        period=10,
        n_max_gen=1000,
        n_max_evals=100000
    )

    print(f"[*] Starting Optimization...")


    # -------------------------------
    # Multi-Objective Algorithms
    # -------------------------------

    # NSGA-II
    # algorithm = NSGA2(
    #     pop_size=args.pop_size,
    #     sampling=LHS(),
    # )

    # NSGA-III
    # ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=12)
    # algorithm = NSGA3(pop_size=args.pop_size, ref_dirs=ref_dirs)

    # U-NSGA-III
    ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=12)
    algorithm = UNSGA3(pop_size=args.pop_size, ref_dirs=ref_dirs)

    res = minimize(
        problem,
        algorithm,
        termination,
        seed=1,
        verbose=True
    )

    # Clean up pool
    pool.close()
    pool.join()

    print(f"[*] Optimization finished. Time: {res.exec_time}")

    F = res.F  # (n_points, 3) Pareto front in objective space
    X = res.X  # decision variables on Pareto front

    # normalize objectives to [0,1]
    F_norm = (F - F.min(axis=0)) / (np.ptp(F, axis=0) + 1e-12)

    bio = np.array([bio_score(theta) for theta in X])
    bio_norm = (bio - bio.min()) / (bio.max() - bio.min() + 1e-12)

    alpha1, alpha2, alpha3, beta = 1.0, 1.0, 0.5, 1.0

    J = (alpha1 * F_norm[:, 0] +
         alpha2 * F_norm[:, 1] +
         alpha3 * F_norm[:, 2] +
         beta * bio_norm)

    best_idx = np.argmin(J)
    theta_opt = X[best_idx]
    F_best = F[best_idx]

    print(f"    Final Loss (F): {F_best}")

    # Final Simulation and Saving
    P_sim, A_sim = simulate_p_scipy(
        t, P_scaled, theta_opt,
        Cg, Cl, site_prot_idx, K_site_kin, R,
        L_alpha, kin_to_prot_idx
    )

    # Save
    (k_act, k_deact, s_prod, d_deg, beta_g, beta_l, alpha, kK_act, kK_deact, k_off) = decode_theta(theta_opt)

    np.savez(
        os.path.join(args.outdir, "fitted_params.npz"),
        theta=theta_opt,
        proteins=np.array(proteins),
        sites=np.array(sites),
        kinases=np.array(kinases),
        k_act=k_act,
        k_deact=k_deact,
        s_prod=s_prod,
        d_deg=d_deg,
        beta_g=beta_g,
        beta_l=beta_l,
        alpha=alpha,
        kK_act=kK_act,
        kK_deact=kK_deact,
        k_off=k_off
    )
    print(f"[*] Saved parameters with labels to {os.path.join(args.outdir, 'fitted_params.npz')}")

    # 1. Site-Level DataFrame
    Y_sim_rescaled = np.zeros_like(Y)
    for i in range(len(sites)):
        Y_sim_rescaled[i] = baselines[i] + amplitudes[i] * P_sim[i]

    df_sites = pd.DataFrame({
        "Protein": [s.split("_")[0] for s in sites],
        "Residue": [s.split("_")[1] for s in sites],
        "Type": "Phosphosite"
    })

    for j in range(len(t)):
        df_sites[f"sim_t{j}"] = Y_sim_rescaled[:, j]
        df_sites[f"data_t{j}"] = Y[:, j]

    # 2. Protein-Level DataFrame
    # Note: A_sim contains normalized abundance. If we have A_scaled, we can potentially rescale A_sim back to FC
    # but since A_sim is internal state, we output it directly. If A_data exists, we use its scaling.

    A_sim_rescaled = np.zeros_like(A_sim)

    # We must rescale A_sim to match the original abundance data scale if available.
    # However, A_sim in the model represents normalized abundance approx [0,1].
    # If we have scaling factors for specific proteins, we use them.
    # A_scaled corresponds to a SUBSET of proteins. We only have baselines/amplitudes for that subset.

    # Initialize rescaled with raw simulation
    A_sim_rescaled = A_sim.copy()

    # If we have observed data, we can try to rescale the simulation to match the observed FC range
    if A_scaled.size > 0:
        for k, p_idx in enumerate(prot_idx_for_A):
            # A_sim[p_idx] is the simulated trace (0-1)
            # Rescale: y = baseline + amplitude * p
            A_sim_rescaled[p_idx] = A_bases[k] + A_amps[k] * A_sim[p_idx]

    df_prots = pd.DataFrame({
        "Protein": proteins,
        "Residue": "",  # Empty
        "Type": "ProteinAbundance"
    })

    for j in range(len(t)):
        df_prots[f"sim_t{j}"] = A_sim_rescaled[:, j]
        df_prots[f"data_t{j}"] = np.nan  # Init with NaN

    # Map observed protein data if it exists
    if A_data is not None and len(A_data) > 0:
        # A_data is (K_subset, T) raw FC
        for k, p_idx in enumerate(prot_idx_for_A):
            for j in range(len(t)):
                df_prots.at[p_idx, f"data_t{j}"] = A_data[k, j]

    # 3. Concatenate
    df_out = pd.concat([df_sites, df_prots], ignore_index=True)

    # Reorder cols
    cols = ["Protein", "Residue", "Type"] + [c for c in df_out.columns if c not in ["Protein", "Residue", "Type"]]
    df_out = df_out[cols]

    df_out.to_csv(os.path.join(args.outdir, "fit_timeseries.tsv"), sep="\t", index=False)
    print(f"[*] Saved fit_timeseries.tsv with sites and proteins.")

    # Plot
    plt.figure(figsize=(10, 10))
    plt.scatter(Y.flatten(), Y_sim_rescaled.flatten(), alpha=0.5, color ='blue', label='Phosphosites')
    plt.scatter(A_data.flatten(), A_sim_rescaled[prot_idx_for_A, :].flatten(), alpha=0.5, color='green', label='Proteins')
    plt.xlabel("Observed")
    plt.ylabel("Simulated")
    plt.plot([Y.min(), Y.max()], [Y.min(), Y.max()], 'r--')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "fit.png"), dpi=300)
    print("[*] Done.")


if __name__ == "__main__":
    main()