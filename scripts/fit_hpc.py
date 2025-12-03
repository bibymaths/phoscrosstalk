#!/usr/bin/env python3
"""
Global phospho-network model fitting script optimized for MULTI-NODE HPC.

Changes from Single-Node:
- Uses mpi4py.futures.MPIPoolExecutor instead of multiprocessing.
- Scales across multiple nodes (e.g., 4 nodes * 64 cores = 256 workers).

Usage (via Slurm):
  srun python -m mpi4py.futures scripts/fit_hpc_mpi.py ...
"""

from __future__ import annotations

import argparse
import os
import re
import sqlite3
import pickle
import sys

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from scipy.integrate import odeint

# Pymoo imports
from pymoo.core.problem import ElementwiseProblem
from pymoo.parallelization.starmap import StarmapParallelization
from pymoo.algorithms.soo.nonconvex.cmaes import CMAES
from pymoo.optimize import minimize
from pymoo.termination.default import DefaultSingleObjectiveTermination

# MPI Import
try:
    from mpi4py.futures import MPIPoolExecutor
except ImportError:
    print("[!] Error: mpi4py is not installed. Please run: conda install mpi4py")
    sys.exit(1)

# Global dims
GLOBAL_K = None
GLOBAL_M = None
GLOBAL_N = None

# ------------------------------------------------------------
#  CONSTANTS & UTILS
# ------------------------------------------------------------

DEFAULT_TIMEPOINTS = np.array(
    [0.0, 0.5, 0.75, 1.0, 2.0, 4.0, 8.0, 16.0,
     30.0, 60.0, 120.0, 240.0, 480.0, 960.0]
)


def load_site_data(path, timepoints=DEFAULT_TIMEPOINTS):
    """Load time-series data from CSV."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")

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


def scale_fc_to_unit_interval(Y, high_percentile=90.0):
    N, T = Y.shape
    P = np.zeros_like(Y, dtype=float)
    baselines = np.zeros(N, dtype=float)
    amplitudes = np.zeros(N, dtype=float)
    eps = 1e-6

    for i in range(N):
        y_raw = Y[i].astype(float)
        b = y_raw[0]
        P_y = np.percentile(y_raw, high_percentile)
        A = P_y - b
        if abs(A) < eps:
            A = 1.0

        p = (y_raw - b) / A
        P[i] = p
        baselines[i] = b
        amplitudes[i] = A

    return P, baselines, amplitudes


def row_normalize(C):
    row_sums = C.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0.0] = 1.0
    C_norm = C / row_sums
    return C_norm


def build_C_matrices_from_db(ptm_intra_path, ptm_inter_path,
                             sites, site_prot_idx, positions,
                             proteins, length_scale=50.0):
    N = len(sites)
    idx = {s: i for i, s in enumerate(sites)}
    Cg = np.zeros((N, N), dtype=float)

    if os.path.exists(ptm_intra_path):
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

    if os.path.exists(ptm_inter_path):
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
        return np.zeros((len(sites), 1)), ["UnknownKinase"]

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
    if not os.path.exists(pkl_path):
        return np.zeros((len(kinases), len(kinases)))

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
#  ODE MODEL
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


def network_rhs(x, t, theta, Cg, Cl, site_prot_idx, K_site_kin, R):
    K = GLOBAL_K
    M = GLOBAL_M
    N = GLOBAL_N

    (k_act, k_deact, s_prod, d_deg, beta_g, beta_l, alpha, kK_act, kK_deact, k_off) = decode_theta(theta)

    S = x[0:K]
    A = x[K:2 * K]
    Kdyn = x[2 * K:2 * K + M]
    p = x[2 * K + M:2 * K + M + N]

    dS = k_act * (1.0 - S) - k_deact * S
    dA = s_prod - d_deg * A

    coup_g = beta_g * (Cg @ p)
    coup_l = beta_l * (Cl @ p)
    coup = np.tanh(coup_g + coup_l)

    u = R @ p
    dK = kK_act * np.tanh(u) * (1.0 - Kdyn) - kK_deact * Kdyn

    k_on_eff = K_site_kin @ (alpha * Kdyn)
    S_local = S[site_prot_idx]
    A_local = A[site_prot_idx]

    v_on = (k_on_eff * S_local * A_local + coup) * (1.0 - p)
    v_off = k_off * p
    dp = v_on - v_off

    return np.concatenate([dS, dA, dK, dp])


def simulate_p_scipy(t_arr, P_data0, theta, Cg, Cl, site_prot_idx, K_site_kin, R):
    K = GLOBAL_K
    M = GLOBAL_M
    N = GLOBAL_N
    state_dim = 2 * K + M + N

    x0 = np.zeros((state_dim,))
    x0[K:2 * K] = 1.0
    x0[2 * K:2 * K + M] = 1.0
    x0[2 * K + M:] = P_data0[:, 0]

    xs = odeint(network_rhs, x0, t_arr, args=(theta, Cg, Cl, site_prot_idx, K_site_kin, R))

    A_sim = xs[:, K:2 * K].T
    P_sim = xs[:, 2 * K + M:].T

    return P_sim, A_sim


# ------------------------------------------------------------
#  PYMOO PROBLEM
# ------------------------------------------------------------

class NetworkOptimizationProblem(ElementwiseProblem):
    def __init__(self,
                 t, P_data, Cg, Cl, site_prot_idx, K_site_kin, R,
                 A_scaled, prot_idx_for_A, W_data, L_alpha,
                 lambda_net, reg_lambda,
                 xl, xu,
                 K_dim, M_dim, N_dim,
                 **kwargs):

        n_var = len(xl)
        super().__init__(n_var=n_var, n_obj=1, n_ieq_constr=0, xl=xl, xu=xu, **kwargs)

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
        self.L_alpha = L_alpha
        self.lambda_net = lambda_net
        self.reg_lambda = reg_lambda
        self.K_dim = K_dim
        self.M_dim = M_dim
        self.N_dim = N_dim

    def _evaluate(self, x, out, *args, **kwargs):

        global GLOBAL_K, GLOBAL_M, GLOBAL_N
        GLOBAL_K = self.K_dim
        GLOBAL_M = self.M_dim
        GLOBAL_N = self.N_dim

        theta = x

        P_sim, A_sim = simulate_p_scipy(
            self.t, self.P_data, theta,
            self.Cg, self.Cl, self.site_prot_idx,
            self.K_site_kin, self.R
        )

        diff_p = (P_sim - self.P_data) * np.sqrt(self.W_data)
        loss_p = np.sum(diff_p ** 2)

        if self.A_scaled.size > 0:
            A_model = A_sim[self.prot_idx_for_A, :]
            diff_A = (A_model - self.A_scaled)
            loss_A = np.sum(diff_A ** 2)
        else:
            loss_A = 0.0

        reg = self.reg_lambda * np.sum(theta ** 2)

        (k_act, k_deact, s_prod, d_deg, beta_g, beta_l, alpha, kK_act, kK_deact, k_off) = decode_theta(theta)
        reg_net = self.lambda_net * (alpha @ (self.L_alpha @ alpha))

        total_loss = loss_p + loss_A + reg + reg_net

        n_data_points = self.P_data.size + self.A_scaled.size
        if n_data_points > 0:
            total_loss /= n_data_points

        out["F"] = total_loss


# ------------------------------------------------------------
#  MAIN
# ------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="HPC MPI Network Fit")
    parser.add_argument("--data", required=True)
    parser.add_argument("--ptm-intra", required=True)
    parser.add_argument("--ptm-inter", required=True)
    parser.add_argument("--outdir", default="network_fit_mpi")
    parser.add_argument("--length-scale", type=float, default=50.0)
    parser.add_argument("--crosstalk-tsv")
    parser.add_argument("--kinase-tsv")
    parser.add_argument("--kea-ks-table")
    parser.add_argument("--unified-graph-pkl")
    parser.add_argument("--lambda-net", type=float, default=0.0)
    parser.add_argument("--pop-size", type=int, default=200, help="Population size for CMA-ES")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print(f"[*] Loading data from {args.data}...")
    (sites, proteins, site_prot_idx, positions, t, Y, A_data, A_proteins) = load_site_data(args.data)
    print(f"[*] Loaded {len(sites)} sites, {len(proteins)} proteins.")

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
        if mask.sum() == 0:
            print("[!] Warning: Crosstalk filtering removed all sites. Ignoring filter.")
        else:
            sites = [s for s, m in zip(sites, mask) if m]
            positions = positions[mask]
            Y = Y[mask, :]
            prots_used = sorted({s.split("_")[0] for s in sites})
            prot_map = {p: i for i, p in enumerate(prots_used)}
            site_prot_idx = np.array([prot_map[s.split("_")[0]] for s in sites], dtype=int)
            proteins = prots_used
            print(f"[*] Filtered to {len(sites)} sites using crosstalk list.")

    P_scaled, baselines, amplitudes = scale_fc_to_unit_interval(Y)

    if A_data is not None and len(A_data) > 0:
        prot_map = {p: i for i, p in enumerate(proteins)}
        mask_A = [p in prot_map for p in A_proteins]
        A_data = A_data[mask_A]
        A_proteins = A_proteins[mask_A]
        prot_idx_for_A = np.array([prot_map[p] for p in A_proteins], dtype=int)
        A_scaled, A_bases, A_amps = scale_fc_to_unit_interval(A_data)
        print(f"[*] Matched {len(A_scaled)} protein abundance profiles.")
    else:
        A_scaled = np.zeros((0, P_scaled.shape[1]))
        prot_idx_for_A = np.array([], dtype=int)
        A_bases, A_amps = np.array([]), np.array([])

    W_data = 1.0 + 3.0 * ((Y - Y.min()) / (Y.max() - Y.min() + 1e-6))

    print("[*] Building coupling and kinase matrices...")
    Cg, Cl = build_C_matrices_from_db(args.ptm_intra, args.ptm_inter, sites, site_prot_idx, positions, proteins,
                                      args.length_scale)
    Cg, Cl = row_normalize(Cg), row_normalize(Cl)

    if args.kinase_tsv:
        K_site_kin, kinases = load_kinase_site_matrix(args.kinase_tsv, sites)
    elif args.kea_ks_table:
        K_site_kin, kinases = build_kinase_site_from_kea(args.kea_ks_table, sites)
    else:
        print("[!] No kinase data provided. Using Identity matrix.")
        K_site_kin = np.eye(len(sites))
        kinases = [f"K_{i}" for i in range(len(sites))]
    R = K_site_kin.T

    L_alpha = np.zeros((len(kinases), len(kinases)))
    if args.unified_graph_pkl and args.lambda_net > 0:
        L_alpha = build_alpha_laplacian_from_unified_graph(args.unified_graph_pkl, kinases)

    global GLOBAL_K, GLOBAL_M, GLOBAL_N
    GLOBAL_K, GLOBAL_N = len(proteins), len(sites)
    GLOBAL_M = len(kinases)

    K, M, N = GLOBAL_K, GLOBAL_M, GLOBAL_N
    dim = 4 * K + 2 + 3 * M + N

    xl = np.zeros(dim)
    xu = np.zeros(dim)
    idx = 0

    # Bounds
    xl[idx:idx + K] = np.log(1e-5);
    xu[idx:idx + K] = np.log(100.0);
    idx += K
    xl[idx:idx + K] = np.log(1e-5);
    xu[idx:idx + K] = np.log(100.0);
    idx += K
    xl[idx:idx + K] = np.log(1e-5);
    xu[idx:idx + K] = np.log(10.0);
    idx += K
    xl[idx:idx + K] = np.log(1e-5);
    xu[idx:idx + K] = np.log(0.5);
    idx += K  # Deg limitation

    xl[idx] = np.log(1e-5);
    xu[idx] = np.log(10.0);
    idx += 1
    xl[idx] = np.log(1e-5);
    xu[idx] = np.log(10.0);
    idx += 1

    xl[idx:idx + M] = np.log(1e-5);
    xu[idx:idx + M] = np.log(100.0);
    idx += M
    xl[idx:idx + M] = np.log(1e-5);
    xu[idx:idx + M] = np.log(100.0);
    idx += M
    xl[idx:idx + M] = np.log(1e-5);
    xu[idx:idx + M] = np.log(100.0);
    idx += M

    xl[idx:idx + N] = np.log(1e-5);
    xu[idx:idx + N] = np.log(50.0);
    idx += N

    # --- MPI CONTEXT MANAGER ---
    # This block handles the distributed execution
    with MPIPoolExecutor() as executor:
        print(f"[*] MPI Pool Initialized.")

        # runner maps the function calls across the MPI pool
        runner = StarmapParallelization(executor.starmap)

        problem = NetworkOptimizationProblem(
            t, P_scaled, Cg, Cl, site_prot_idx, K_site_kin, R,
            A_scaled, prot_idx_for_A, W_data, L_alpha,
            args.lambda_net, 1e-4,
            xl, xu,
            GLOBAL_K, GLOBAL_M, GLOBAL_N,
            elementwise_runner=runner
        )

        algorithm = CMAES(
            popsize=args.pop_size,
            parallelize=True,
            incpopsize=2,
        )

        termination = DefaultSingleObjectiveTermination(
            xtol=1e-8,
            cvtol=1e-6,
            ftol=1e-6,
            period=20,
            n_max_gen=500,
            n_max_evals=100000
        )

        print(f"[*] Starting Optimization using CMA-ES (pop_size={args.pop_size})...")

        res = minimize(
            problem,
            algorithm,
            termination,
            seed=1,
            verbose=True
        )

        print(f"[*] Optimization finished. Time: {res.exec_time}")
        print(f"    Final Loss (F): {res.F}")

        theta_opt = res.X

        # Post-process (only master rank needs to do this)
        P_sim, A_sim = simulate_p_scipy(t, P_scaled, theta_opt, Cg, Cl, site_prot_idx, K_site_kin, R)

        (k_act, k_deact, s_prod, d_deg, beta_g, beta_l, alpha, kK_act, kK_deact, k_off) = decode_theta(theta_opt)

        np.savez(
            os.path.join(args.outdir, "fitted_params.npz"),
            theta=theta_opt,
            proteins=np.array(proteins),
            sites=np.array(sites),
            kinases=np.array(kinases),
            k_act=k_act, k_deact=k_deact, s_prod=s_prod, d_deg=d_deg,
            beta_g=beta_g, beta_l=beta_l, alpha=alpha,
            kK_act=kK_act, kK_deact=kK_deact, k_off=k_off
        )
        print(f"[*] Saved parameters to {os.path.join(args.outdir, 'fitted_params.npz')}")

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

        A_sim_rescaled = A_sim.copy()
        if A_scaled.size > 0:
            for k, p_idx in enumerate(prot_idx_for_A):
                A_sim_rescaled[p_idx] = A_bases[k] + A_amps[k] * A_sim[p_idx]

        df_prots = pd.DataFrame({
            "Protein": proteins,
            "Residue": "",
            "Type": "ProteinAbundance"
        })
        for j in range(len(t)):
            df_prots[f"sim_t{j}"] = A_sim_rescaled[:, j]
            df_prots[f"data_t{j}"] = np.nan

        if A_data is not None and len(A_data) > 0:
            for k, p_idx in enumerate(prot_idx_for_A):
                for j in range(len(t)):
                    df_prots.at[p_idx, f"data_t{j}"] = A_data[k, j]

        df_out = pd.concat([df_sites, df_prots], ignore_index=True)
        cols = ["Protein", "Residue", "Type"] + [c for c in df_out.columns if c not in ["Protein", "Residue", "Type"]]
        df_out = df_out[cols]

        out_tsv = os.path.join(args.outdir, "fit_timeseries.tsv")
        df_out.to_csv(out_tsv, sep="\t", index=False)
        print(f"[*] Saved time series to {out_tsv}")

        plt.figure(figsize=(10, 10))
        plt.scatter(Y.flatten(), Y_sim_rescaled.flatten(), alpha=0.5, color='blue', label='Phosphosites')
        if A_data is not None and len(A_data) > 0:
            plt.scatter(A_data.flatten(), A_sim_rescaled[prot_idx_for_A, :].flatten(), alpha=0.5, color='green',
                        label='Proteins')
        plt.xlabel("Observed")
        plt.ylabel("Simulated")

        all_min = min(Y.min(), Y_sim_rescaled.min())
        all_max = max(Y.max(), Y_sim_rescaled.max())
        plt.plot([all_min, all_max], [all_min, all_max], 'r--')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(args.outdir, "fit_correlation.png"), dpi=300)
        print("[*] Done.")


if __name__ == "__main__":
    main()