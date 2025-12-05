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
import sys
import sqlite3
import pickle
import multiprocessing

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

# Pymoo Algorithm imports
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.algorithms.moo.unsga3 import UNSGA3
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.operators.sampling.lhs import LHS

from pymoo.optimize import minimize
from pymoo.core.problem import ElementwiseProblem
from pymoo.parallelization.starmap import StarmapParallelization
from pymoo.termination.default import DefaultMultiObjectiveTermination

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

    # --- existing log-parameters ---
    log_k_act   = theta[idx0:idx0 + K]; idx0 += K
    log_k_deact = theta[idx0:idx0 + K]; idx0 += K
    log_s_prod  = theta[idx0:idx0 + K]; idx0 += K
    log_d_deg   = theta[idx0:idx0 + K]; idx0 += K

    log_beta_g  = theta[idx0]; idx0 += 1
    log_beta_l  = theta[idx0]; idx0 += 1

    log_alpha   = theta[idx0:idx0 + M]; idx0 += M
    log_kK_act  = theta[idx0:idx0 + M]; idx0 += M
    log_kK_deact= theta[idx0:idx0 + M]; idx0 += M

    log_k_off   = theta[idx0:idx0 + N]; idx0 += N

    # --- NEW: 4 raw γ-parameters (not in log-space) ---
    # order: gamma_S_p, gamma_A_S, gamma_A_p, gamma_K_net
    raw_gamma = theta[idx0:idx0 + 4]
    # idx0 += 4  # not needed afterwards but fine if you add it

    # ---- map logs to positive rates ----
    k_act   = np.exp(np.clip(log_k_act,   -20, 10))
    k_deact = np.exp(np.clip(log_k_deact, -20, 10))
    s_prod  = np.exp(np.clip(log_s_prod,  -20, 10))
    d_deg   = np.exp(np.clip(log_d_deg,   -20, 10))

    beta_g  = np.exp(np.clip(log_beta_g,  -20, 10))
    beta_l  = np.exp(np.clip(log_beta_l,  -20, 10))

    alpha   = np.exp(np.clip(log_alpha,   -20, 10))
    kK_act  = np.exp(np.clip(log_kK_act,  -20, 10))
    kK_deact= np.exp(np.clip(log_kK_deact,-20, 10))

    k_off   = np.exp(np.clip(log_k_off,   -20, 10))

    # ---- map γ's to bounded real values via tanh ----
    # chosen max magnitudes; adjust if needed
    gamma_max = np.array([2.0, 2.0, 2.0, 2.0])  # [S_p, A_S, A_p, K_net]
    gammas = gamma_max * np.tanh(raw_gamma)

    gamma_S_p, gamma_A_S, gamma_A_p, gamma_K_net = gammas

    return (k_act, k_deact, s_prod, d_deg,
            beta_g, beta_l, alpha,
            kK_act, kK_deact, k_off,
            gamma_S_p, gamma_A_S, gamma_A_p, gamma_K_net)

def network_rhs(x, t, theta, Cg, Cl, site_prot_idx, K_site_kin, R, L_alpha, kin_to_prot_idx):
    # clip negative states to zero
    x = np.maximum(x, 0.0)

    K = GLOBAL_K
    M = GLOBAL_M
    N = GLOBAL_N

    (k_act, k_deact, s_prod, d_deg,
     beta_g, beta_l, alpha,
     kK_act, kK_deact, k_off,
     gamma_S_p, gamma_A_S, gamma_A_p, gamma_K_net) = decode_theta(theta)

    # unpack state
    S    = x[0:K]
    A    = x[K:2 * K]
    Kdyn = x[2 * K:2 * K + M]
    p    = x[2 * K + M:2 * K + M + N]

    # ---------- site-level global + local context ----------
    coup_g = beta_g * (Cg @ p)         # (N,)
    coup_l = beta_l * (Cl @ p)         # (N,)
    coup   = np.tanh(coup_g + coup_l)  # (N,)

    # per-protein aggregates
    num_p  = np.bincount(site_prot_idx, weights=p,    minlength=K)
    num_c  = np.bincount(site_prot_idx, weights=coup, minlength=K)
    den    = np.bincount(site_prot_idx, minlength=K)
    den[den == 0] = 1.0

    mean_p_per_prot   = num_p / den           # ⟨p⟩_protein
    mean_ctx_per_prot = num_c / den           # ⟨coup⟩_protein

    # ---------- 1) S dynamics ----------
    # drive from phosphorylation + contextual coupling
    D_S = 1.0 + gamma_S_p * mean_p_per_prot + mean_ctx_per_prot
    D_S = np.clip(D_S, 0.0, None)

    dS = k_act * D_S * (1.0 - S) - k_deact * S

    # ---------- 2) A dynamics ----------
    # synthesis boosted by S
    s_eff = s_prod * (1.0 + gamma_A_S * S)
    s_eff = np.clip(s_eff, 0.0, None)

    # degradation boosted by mean phosphorylation
    D_deg = 1.0 + gamma_A_p * mean_p_per_prot
    D_deg = np.clip(D_deg, 0.0, None)

    dA = s_eff - d_deg * D_deg * A

    # ---------- 3) Kdyn dynamics ----------
    # substrate signal from sites
    u_sub = R @ p            # (M,)

    # network smoothing from kinases
    if L_alpha is not None and L_alpha.size > 0:
        u_net = -(L_alpha @ Kdyn)   # Laplacian term
    else:
        u_net = 0.0

    U = u_sub + gamma_K_net * u_net
    dK = kK_act * np.tanh(U) * (1.0 - Kdyn) - kK_deact * Kdyn

    # ---------- 4) p dynamics ----------
    k_on_eff = K_site_kin @ (alpha * Kdyn)   # (N,)

    S_local = S[site_prot_idx]
    A_local = A[site_prot_idx]

    v_on  = (k_on_eff * S_local * A_local + coup) * (1.0 - p)
    v_off = k_off * p

    dp = v_on - v_off

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
         kK_act, kK_deact, k_off,
         gamma_S_p, gamma_A_S, gamma_A_p, gamma_K_net) = decode_theta(theta)

        reg = self.reg_lambda * np.dot(theta, theta)
        reg_net = self.lambda_net * (alpha @ (self.L_alpha @ alpha))

        # objective 3: model/network complexity
        f3 = (reg + reg_net) / self.n_var

        out["F"] = np.array([f1, f2, f3])

# ------------------------------------------------------------
#  MAIN
# ------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Pymoo Differential Evolution Fit")
    parser.add_argument("--data", required=True)
    parser.add_argument("--ptm-intra", required=True)
    parser.add_argument("--ptm-inter", required=True)
    parser.add_argument("--outdir", default="network_fit_pymoo")
    parser.add_argument("--crosstalk-tsv")
    parser.add_argument("--kinase-tsv")
    parser.add_argument("--kea-ks-table")
    parser.add_argument("--unified-graph-pkl")
    parser.add_argument("--lambda-net", type=float, default=0.0)
    parser.add_argument("--length-scale", type=float, default=50.0)
    parser.add_argument("--reg-lambda", type=float, default=0.01, help="L2 regularization lambda")
    parser.add_argument("--pop-size", type=int, default=100, help="Population size multiplier for CMA-ES")
    parser.add_argument("--cores", type=int, default=60, help="Number of cores for multiprocessing")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    class TeeLogger:
        def __init__(self, logfile):
            self.terminal = sys.stdout
            self.log = open(logfile, "w")

        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)

        def flush(self):
            self.terminal.flush()
            self.log.flush()

    # log file inside output directory
    log_path = os.path.join(args.outdir, "run.log")
    sys.stdout = TeeLogger(log_path)
    sys.stderr = TeeLogger(log_path)

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

    # Matrices that don't depend on hyper-parameters (except length_scale for Cl)
    # For Cg/Cl we will rebuild inside the hyper-parameter loop for each length_scale.
    # Here we only build once as a placeholder; they will be overwritten in the loop.
    Cg, Cl = build_C_matrices_from_db(
        args.ptm_intra, args.ptm_inter,
        sites, site_prot_idx, positions, proteins,
        args.length_scale
    )
    Cg, Cl = row_normalize(Cg), row_normalize(Cl)

    if args.kinase_tsv:
        K_site_kin, kinases = load_kinase_site_matrix(args.kinase_tsv, sites)
    elif args.kea_ks_table:
        K_site_kin, kinases = build_kinase_site_from_kea(args.kea_ks_table, sites)
    else:
        K_site_kin = np.eye(len(sites))
        kinases = [f"K_{i}" for i in range(len(sites))]
    R = K_site_kin.T

    # Build Laplacian independent of lambda_net (we'll scale via lambda_net inside the objective)
    L_alpha = np.zeros((len(kinases), len(kinases)))
    if args.unified_graph_pkl:
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

    # Params and Bounds (common to all hyper-parameter combos)
    K, M, N = GLOBAL_K, GLOBAL_M, GLOBAL_N
    dim = 4 * K + 2 + 3 * M + N + 4

    # Init params for reference
    # Bounds: approx -11.5 (log 1e-5) to 2.3 (log 10dim = 4 * K + 2 + 3 * M +
    # low_b = np.log(1)
    # high_b = np.log(10)

    # xl = np.full(dim, low_b)
    # xu = np.full(dim, high_b)

    # --- INTELLIGENT BOUNDS SETTING ---
    # We create specific bounds for specific parameter types rather than one global bound.

    xl_base = np.zeros(dim)
    xu_base = np.zeros(dim)

    idx = 0

    # 1. Protein Kinetics (K)
    # k_act (Activation): Allow range [1e-5, 100] - Can be fast
    xl_base[idx:idx + K] = np.log(1e-5)
    xu_base[idx:idx + K] = np.log(100.0)
    idx += K

    # k_deact (Deactivation): Allow range [1e-5, 100] - Can be fast
    xl_base[idx:idx + K] = np.log(1e-5)
    xu_base[idx:idx + K] = np.log(100.0)
    idx += K

    # s_prod (Synthesis): Allow range [1e-5, 10]
    xl_base[idx:idx + K] = np.log(1e-5)
    xu_base[idx:idx + K] = np.log(10.0)
    idx += K

    # d_deg (Degradation): *** RESTRICT THIS ***
    # Force degradation to be slow (e.g., max 0.5).
    # This forces the model to use PHOSPHORYLATION dynamics, not abundance destruction.
    xl_base[idx:idx + K] = np.log(1e-5)
    xu_base[idx:idx + K] = np.log(0.5)
    idx += K

    # 2. Coupling (2)
    xl_base[idx] = np.log(1e-5)
    xu_base[idx] = np.log(10.0)
    idx += 1  # beta_g
    xl_base[idx] = np.log(1e-5)
    xu_base[idx] = np.log(10.0)
    idx += 1  # beta_l

    # 3. Kinase Params (M)
    # Alpha (Global Strength): Allow high
    xl_base[idx:idx + M] = np.log(1e-5)
    xu_base[idx:idx + M] = np.log(100.0)
    idx += M
    # Kinase Act/Deact
    xl_base[idx:idx + M] = np.log(1e-5)
    xu_base[idx:idx + M] = np.log(100.0)
    idx += M
    xl_base[idx:idx + M] = np.log(1e-5)
    xu_base[idx:idx + M] = np.log(100.0)
    idx += M

    # 4. Site Params (N)
    # k_off (Phosphatase): Allow range [1e-5, 50] - High but not 100
    xl_base[idx:idx + N] = np.log(1e-5)
    xu_base[idx:idx + N] = np.log(50.0)
    idx += N

    # 5. Gamma coupling parameters (4 scalars, raw space)
    # order must match decode_theta: [gamma_S_p, gamma_A_S, gamma_A_p, gamma_K_net]
    xl_base[idx:idx + 4] = -3.0   # tanh(-3..3) ~ [-0.995, 0.995], then * 2.0 → roughly [-2, 2]
    xu_base[idx:idx + 4] =  3.0
    idx += 4

    # optional sanity check
    assert idx == dim, f"Bounds indexing mismatch: idx={idx}, dim={dim}"

    # ----------------------------------------------------------
    # SERIAL HYPER-PARAMETER GRID
    # ----------------------------------------------------------

    # Define grids around the user-specified values
    reg_lambdas = [args.reg_lambda / 10.0, args.reg_lambda, args.reg_lambda * 10.0]
    length_scales = [max(5.0, args.length_scale / 2.0), args.length_scale, args.length_scale * 2.0]
    lambda_nets = [0.0, max(0.001, args.lambda_net), max(0.01, args.lambda_net * 10.0 if args.lambda_net > 0 else 0.1)]

    results_summary = []

    for reg_lambda in reg_lambdas:
        for length_scale in length_scales:
            for lambda_net in lambda_nets:
                print("\n" + "=" * 80)
                print(f"[*] Hyper-parameter combo: reg_lambda={reg_lambda}, "
                      f"length_scale={length_scale}, lambda_net={lambda_net}")
                print("=" * 80)

                # Rebuild Cg/Cl for this length_scale
                Cg, Cl = build_C_matrices_from_db(
                    args.ptm_intra, args.ptm_inter,
                    sites, site_prot_idx, positions, proteins,
                    length_scale
                )
                Cg, Cl = row_normalize(Cg), row_normalize(Cl)

                # Copy bounds (they are common)
                xl = xl_base.copy()
                xu = xu_base.copy()

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
                    lambda_net, reg_lambda,
                    xl, xu,
                    elementwise_runner=runner
                )

                # Termination for multi-objective
                termination = DefaultMultiObjectiveTermination(
                    xtol=1e-8,
                    cvtol=1e-6,
                    ftol=0.0025,
                    period=10,
                    n_max_gen=100,
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

                F = res.F
                X = res.X

                # ----- SELECTION FOCUSING ON f1, f2 -----
                f1 = F[:, 0]
                f2 = F[:, 1]
                f3 = F[:, 2]

                # Normalize first two objectives
                eps = 1e-12
                f1n = (f1 - f1.min()) / (f1.max() - f1.min() + eps)
                f2n = (f2 - f2.min()) / (f2.max() - f2.min() + eps)

                # Distance to ideal point (0,0) in the (f1,f2) plane
                d12 = np.sqrt(f1n ** 2 + f2n ** 2)

                f3n = (f3 - f3.min()) / (f3.max() - f3.min() + eps)

                w_f3 = 0.1  # small weight: complexity as gentle penalty
                w_bio = 0.1  # small weight: biological plausibility

                J = d12 + f3n

                best_idx = np.argmin(J)
                theta_opt = X[best_idx]
                F_best = F[best_idx]

                print("[*] Best solution (prioritizing phosphosite + protein fit):")
                print("    f1 (P-sites) =", F_best[0])
                print("    f2 (protein) =", F_best[1])
                print("    f3 (network complexity) =", F_best[2])

                # Final Simulation and Saving
                P_sim, A_sim = simulate_p_scipy(
                    t, P_scaled, theta_opt,
                    Cg, Cl, site_prot_idx, K_site_kin, R,
                    L_alpha, kin_to_prot_idx
                )

                # Save per hyper-parameter combo in its own subdirectory
                tag = f"reg{reg_lambda}_L{length_scale}_lnet{lambda_net}".replace('.', 'p')
                outdir_hp = os.path.join(args.outdir, tag)
                os.makedirs(outdir_hp, exist_ok=True)

                # Save parameters
                (k_act, k_deact, s_prod, d_deg,
                 beta_g, beta_l, alpha,
                 kK_act, kK_deact, k_off,
                 gamma_S_p, gamma_A_S, gamma_A_p, gamma_K_net) = decode_theta(theta_opt)

                np.savez(
                    os.path.join(outdir_hp, "fitted_params.npz"),
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
                    k_off=k_off,
                    gamma_S_p=gamma_S_p,
                    gamma_A_S=gamma_A_S,
                    gamma_A_p=gamma_A_p,
                    gamma_K_net=gamma_K_net
                )
                print(f"[*] Saved parameters with labels to {os.path.join(outdir_hp, f'fitted_params.npz')}")

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
                    for k_idx, p_idx in enumerate(prot_idx_for_A):
                        # A_sim[p_idx] is the simulated trace (0-1)
                        # Rescale: y = baseline + amplitude * p
                        A_sim_rescaled[p_idx] = A_bases[k_idx] + A_amps[k_idx] * A_sim[p_idx]

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
                    for k_idx, p_idx in enumerate(prot_idx_for_A):
                        for j in range(len(t)):
                            df_prots.at[p_idx, f"data_t{j}"] = A_data[k_idx, j]

                # 3. Concatenate
                df_out = pd.concat([df_sites, df_prots], ignore_index=True)

                # Reorder cols
                cols = ["Protein", "Residue", "Type"] + [c for c in df_out.columns if c not in ["Protein", "Residue", "Type"]]
                df_out = df_out[cols]

                df_out.to_csv(os.path.join(outdir_hp, f'fit_timeseries.tsv'), sep="\t", index=False)
                print(f"[*] Saved fit_timeseries.tsv with sites and proteins.")

                # Collect summary for this hyper-parameter combo
                results_summary.append({
                    "reg_lambda": reg_lambda,
                    "length_scale": length_scale,
                    "lambda_net": lambda_net,
                    "f1": float(F_best[0]),
                    "f2": float(F_best[1]),
                    "f3": float(F_best[2]),
                    "J": float(J[best_idx])
                })

                # Visualizations

                plt.figure(figsize=(8, 8))
                plt.scatter(Y.flatten(), Y_sim_rescaled.flatten(), alpha=0.5, color='blue', label='Phosphosites')

                if A_data is not None and A_data.size > 0:
                    plt.scatter(
                        A_data.flatten(),
                        A_sim_rescaled[prot_idx_for_A, :].flatten(),
                        alpha=0.5,
                        color='green',
                        label='Proteins'
                    )

                plt.xlabel("Observed")
                plt.ylabel("Simulated")
                plt.plot([Y.min(), Y.max()], [Y.min(), Y.max()], 'r--')
                plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(outdir_hp, "fit.png"), dpi=300)
                plt.close()

                # Plot Pareto Front
                plt.figure(figsize=(8, 6))
                plt.scatter(F[:, 0], F[:, 1], c=J, cmap='viridis', s=50)
                plt.colorbar(label='Combined Score J')
                plt.xlabel("Objective 1: Phosphosite Fit Loss")
                plt.ylabel("Objective 2: Protein Abundance Fit Loss")
                plt.title("Pareto Front")
                plt.tight_layout()
                plt.savefig(os.path.join(outdir_hp, "pareto_front.png"), dpi=300)
                plt.close()

                # Plot parameter distributions
                param_names = [
                    "k_act", "k_deact", "s_prod", "d_deg",
                    "beta_g", "beta_l",
                    "alpha", "kK_act", "kK_deact", "k_off",
                    "gamma_S_p", "gamma_A_S", "gamma_A_p", "gamma_K_net"
                ]

                param_slices = [
                    (0, K), (K, 2 * K), (2 * K, 3 * K), (3 * K, 4 * K),
                    (4 * K, 4 * K + 1), (4 * K + 1, 4 * K + 2),
                    (4 * K + 2, 4 * K + 2 + M), (4 * K + 2 + M, 4 * K + 2 + 2 * M),
                    (4 * K + 2 + 2 * M, 4 * K + 2 + 3 * M),
                    (4 * K + 2 + 3 * M, 4 * K + 2 + 3 * M + N),
                    (4 * K + 2 + 3 * M + N, 4 * K + 2 + 3 * M + N + 1),
                    (4 * K + 2 + 3 * M + N + 1, 4 * K + 2 + 3 * M + N + 2),
                    (4 * K + 2 + 3 * M + N + 2, 4 * K + 2 + 3 * M + N + 3),
                    (4 * K + 2 + 3 * M + N + 3, 4 * K + 2 + 3 * M + N + 4)
                ]

                for name, (start, end) in zip(param_names, param_slices):
                    plt.figure(figsize=(8, 6))
                    param_values = X[:, start:end].flatten()
                    plt.hist(param_values, bins=30, alpha=0.7)
                    plt.xlabel(f"Log Parameter Values: {name}")
                    plt.ylabel("Frequency")
                    plt.title(f"Distribution of Parameter: {name}")
                    plt.tight_layout()
                    plt.savefig(os.path.join(outdir_hp, f"param_distribution_{name}.png"), dpi=300)
                    plt.close()

                # Plot alpha values on kinases for all kianses with label, no mean
                plt.figure(figsize=(10, 6))
                alpha_values = theta_opt[4 * K + 2:4 * K + 2 + M]
                plt.bar(kinases, alpha_values)
                plt.xticks(rotation=90)
                plt.xlabel("Kinases")
                plt.ylabel("Alpha Values (Log Scale)")
                plt.title("Alpha Values for Kinases")
                plt.tight_layout()
                plt.savefig(os.path.join(outdir_hp, "alpha_values_kinases.png"), dpi=300)
                plt.close()

                # Plot heatmap of kinase-kinase Laplacian if available
                if L_alpha.size > 0:
                    plt.figure(figsize=(8, 6))
                    sns.heatmap(L_alpha, xticklabels=kinases, yticklabels=kinases, cmap='viridis')
                    plt.title("Kinase-Kinase Laplacian Heatmap")
                    plt.tight_layout()
                    plt.savefig(os.path.join(outdir_hp, "kinase_laplacian_heatmap.png"), dpi=300)
                    plt.close()

                # Plot parameter correlations across Pareto front, no mean, all points
                plt.figure(figsize=(10, 8))
                param_matrix = X  # shape: (n_points, dim)
                corr = np.corrcoef(param_matrix.T)

                # Build one label per parameter dimension using param_names + param_slices
                labels = []
                for name, (start, end) in zip(param_names, param_slices):
                    for idx in range(start, end):
                        # local index within that block
                        local_idx = idx - start
                        labels.append(f"{name}[{local_idx}]")

                # sanity check: labels must match number of parameters
                assert corr.shape[0] == len(labels), (
                    f"Label/parameter mismatch: corr has {corr.shape[0]} dims, labels has {len(labels)}"
                )

                sns.heatmap(
                    corr,
                    xticklabels=labels,
                    yticklabels=labels,
                    cmap='coolwarm',
                    center=0
                )
                plt.xticks(rotation=90, fontsize=6)
                plt.yticks(fontsize=6)
                plt.title("Parameter Correlation Heatmap Across Pareto Front")
                plt.tight_layout()
                plt.savefig(os.path.join(outdir_hp, "parameter_correlation_heatmap.png"), dpi=300)
                plt.close()

    # ----------------------------------------------------------
    # SUMMARY ACROSS HYPER-PARAMETER GRID
    # ----------------------------------------------------------
    if results_summary:
        df_summary = pd.DataFrame(results_summary)
        summary_path = os.path.join(args.outdir, "hyperparam_summary.tsv")
        df_summary.to_csv(summary_path, sep="\t", index=False)
        print(f"[*] Saved hyper-parameter summary to {summary_path}")

        # Simple scatter: reg_lambda vs length_scale, colored by (f1+f2)
        plt.figure(figsize=(8, 6))
        sc = plt.scatter(
            df_summary["reg_lambda"],
            df_summary["length_scale"],
            c=df_summary["f1"] + df_summary["f2"],
            s=80
        )
        plt.xscale("log")
        plt.xlabel("reg_lambda")
        plt.ylabel("length_scale")
        cbar = plt.colorbar(sc)
        cbar.set_label("f1 + f2 (lower is better)")
        plt.title("Hyper-parameter grid: fit quality vs reg_lambda & length_scale")
        plt.tight_layout()
        plt.savefig(os.path.join(args.outdir, "hyperparam_scatter_reg_length.png"), dpi=300)
        plt.close()

        # Optionally another: lambda_net vs (f1+f2)
        plt.figure(figsize=(8, 6))
        plt.scatter(
            df_summary["lambda_net"],
            df_summary["f1"] + df_summary["f2"],
            s=80
        )
        plt.xlabel("lambda_net")
        plt.ylabel("f1 + f2 (lower is better)")
        plt.title("Hyper-parameter effect of lambda_net on fit quality")
        plt.tight_layout()
        plt.savefig(os.path.join(args.outdir, "hyperparam_lambda_net_vs_fit.png"), dpi=300)
        plt.close()

if __name__ == "__main__":
    main()