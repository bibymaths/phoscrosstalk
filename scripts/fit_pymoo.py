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
import math
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
from pymoo.parallelization.joblib import JoblibParallelization
from pymoo.termination.default import DefaultMultiObjectiveTermination

import warnings
from scipy.integrate import odeint, solve_ivp
from numba import njit
from scipy.integrate import ODEintWarning

warnings.filterwarnings("ignore", category=ODEintWarning)
warnings.filterwarnings("ignore", message="Excess work done on this call")
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

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


@njit(cache=True)
def clip_scalar(x, lo, hi):
    if x < lo:
        return lo
    elif x > hi:
        return hi
    else:
        return x

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

@njit(cache=True)
def decode_theta(theta, K, M, N):
    idx0 = 0

    # protein rates (arrays) – KEEP np.clip here
    log_k_act   = theta[idx0:idx0 + K]; idx0 += K
    log_k_deact = theta[idx0:idx0 + K]; idx0 += K
    log_s_prod  = theta[idx0:idx0 + K]; idx0 += K
    log_d_deg   = theta[idx0:idx0 + K]; idx0 += K

    # coupling (scalars)
    log_beta_g  = theta[idx0]; idx0 += 1
    log_beta_l  = theta[idx0]; idx0 += 1

    # kinase params (arrays)
    log_alpha    = theta[idx0:idx0 + M]; idx0 += M
    log_kK_act   = theta[idx0:idx0 + M]; idx0 += M
    log_kK_deact = theta[idx0:idx0 + M]; idx0 += M

    # site params (array)
    log_k_off = theta[idx0:idx0 + N]; idx0 += N

    # raw gamma (array of 4 scalars)
    raw_gamma = theta[idx0:idx0 + 4]

    # ---- arrays: np.clip is fine in numba ----
    k_act   = np.exp(np.clip(log_k_act,   -20.0, 10.0))
    k_deact = np.exp(np.clip(log_k_deact, -20.0, 10.0))
    s_prod  = np.exp(np.clip(log_s_prod,  -20.0, 10.0))
    d_deg   = np.exp(np.clip(log_d_deg,   -20.0, 10.0))

    alpha    = np.exp(np.clip(log_alpha,    -20.0, 10.0))
    kK_act   = np.exp(np.clip(log_kK_act,   -20.0, 10.0))
    kK_deact = np.exp(np.clip(log_kK_deact, -20.0, 10.0))
    k_off    = np.exp(np.clip(log_k_off,    -20.0, 10.0))

    # ---- scalars: use clip_scalar + math.exp ----
    beta_g = math.exp(clip_scalar(log_beta_g, -20.0, 10.0))
    beta_l = math.exp(clip_scalar(log_beta_l, -20.0, 10.0))

    # ---- gamma mapping (scalar operations are fine) ----
    gamma_S_p  = 2.0 * np.tanh(raw_gamma[0])
    gamma_A_S  = 2.0 * np.tanh(raw_gamma[1])
    gamma_A_p  = 2.0 * np.tanh(raw_gamma[2])
    gamma_K_net= 2.0 * np.tanh(raw_gamma[3])

    return (k_act, k_deact, s_prod, d_deg,
            beta_g, beta_l, alpha,
            kK_act, kK_deact, k_off,
            gamma_S_p, gamma_A_S, gamma_A_p, gamma_K_net)

@njit(cache=True)
def network_rhs_nb_core(x, t, theta,
                        Cg, Cl, site_prot_idx, K_site_kin, R, L_alpha,
                        kin_to_prot_idx,
                        receptor_mask_prot, receptor_mask_kin,
                        K, M, N):
    """
    Numba-compiled RHS core (distributive mechanism) with:
      - explicit stimulus u(t) (step at t=0)
      - kinase ↔ protein coupling via kin_to_prot_idx
      - phosphorylation depending on protein abundance A
      - receptor vs downstream hierarchy via masks
    """

    # enforce non-negative state
    x = np.maximum(x, 0.0)

    (k_act, k_deact, s_prod, d_deg,
     beta_g, beta_l, alpha,
     kK_act, kK_deact, k_off,
     gamma_S_p, gamma_A_S, gamma_A_p, gamma_K_net) = decode_theta(theta, K, M, N)

    # unpack state
    S    = x[0:K]
    A    = x[K:2 * K]
    Kdyn = x[2 * K:2 * K + M]
    p    = x[2 * K + M:2 * K + M + N]

    # -----------------------------------
    # external stimulus: step at t = 0
    # u(t) = 0 for t < 0 (pre-equilibration),
    # u(t) = 1 for t >= 0 (stimulated)
    # -----------------------------------
    # if t < 0.0:
    #     u = 0.0
    # else:
    #     u = 1.0

    u = 1.0 / (1.0 + np.exp(-(t) / 0.1))

    # ---------- site-level global + local context ----------
    coup_g = beta_g * (Cg @ p)         # (N,)
    coup_l = beta_l * (Cl @ p)         # (N,)
    coup   = np.tanh(coup_g + coup_l)  # (N,)
    coup_pos = np.maximum(coup, 0.0)

    # per-protein aggregates
    num_p  = np.zeros(K)
    num_c  = np.zeros(K)
    den    = np.zeros(K)

    for i in range(N):
        prot = site_prot_idx[i]
        num_p[prot] += p[i]
        num_c[prot] += coup[i]
        den[prot]   += 1.0

    mean_p_per_prot   = np.zeros(K)
    mean_ctx_per_prot = np.zeros(K)
    for k in range(K):
        if den[k] > 0.0:
            mean_p_per_prot[k]   = num_p[k] / den[k]
            mean_ctx_per_prot[k] = num_c[k] / den[k]
        else:
            mean_p_per_prot[k]   = 0.0
            mean_ctx_per_prot[k] = 0.0

    # ---------- 1) S dynamics ----------
    # baseline drive from phospho + context
    D_S = 1.0 + gamma_S_p * mean_p_per_prot + mean_ctx_per_prot

    # add external input only for receptor proteins
    for k in range(K):
        if receptor_mask_prot[k] == 1:
            D_S[k] += u

        if D_S[k] < 0.0:
            D_S[k] = 0.0

    dS = np.empty(K)
    for k in range(K):
        dS[k] = k_act[k] * D_S[k] * (1.0 - S[k]) - k_deact[k] * S[k]

    # ---------- 2) A dynamics ----------
    # abundance modulated by S (activation), but can be extended later
    s_eff = np.empty(K)
    for k in range(K):
        s_eff[k] = s_prod[k] * (1.0 + gamma_A_S * S[k])
        if s_eff[k] < 0.0:
            s_eff[k] = 0.0

    dA = np.empty(K)
    for k in range(K):
        dA[k] = s_eff[k] - d_deg[k] * A[k]

    # ---------- 3) Kdyn dynamics ----------
    # substrate signal
    u_sub = R @ p  # (M,)

    # kinase network / Laplacian
    if L_alpha.shape[0] > 0:
        u_net = -(L_alpha @ Kdyn)
    else:
        u_net = np.zeros(M)

    # base drive from substrate + network
    U = u_sub + gamma_K_net * u_net

    # couple kinase activity to host protein S and A
    for m in range(M):
        prot = kin_to_prot_idx[m]
        if prot >= 0:
            # S- and A-driven modulation (reuse gammas you already have)
            U[m] += gamma_A_S * S[prot] + gamma_A_p * A[prot]

        # external stimulus directly to receptor kinases
        if receptor_mask_kin[m] == 1:
            U[m] += u

    dK = np.empty(M)
    for m in range(M):
        # bounded activation via tanh
        act_term = np.tanh(U[m])
        dK[m] = kK_act[m] * act_term * (1.0 - Kdyn[m]) - kK_deact[m] * Kdyn[m]

    # ---------- 4) p dynamics (distributive, now A-dependent) ----------
    # effective on-rate, combining site→kinase, alpha, and Kdyn
    k_on_eff = K_site_kin @ (alpha * Kdyn)   # (N,)

    dp = np.empty(N)
    for i in range(N):

        # TEMP: ignore abundance dependence to decouple P from A for fitting
        # saturating abundance factor: A/(1+A) to avoid crazy rates
        prot = site_prot_idx[i]
        A_local = A[prot]
        A_factor = A_local / (1.0 + A_local)
        v_on  = k_on_eff[i] * (1.0 + coup_pos[i]) * A_factor * (1.0 - p[i])
        # v_on  = k_on_eff[i] * (1.0 + coup_pos[i]) * (1.0 - p[i])
        v_off = k_off[i] * p[i]
        dp[i] = v_on - v_off

    # pack
    dx = np.empty(2 * K + M + N)
    dx[0:K]                     = dS
    dx[K:2 * K]                 = dA
    dx[2 * K:2 * K + M]         = dK
    dx[2 * K + M:2 * K + M + N] = dp

    return dx

def network_rhs(x, t, theta, Cg, Cl, site_prot_idx, K_site_kin,
                R, L_alpha, kin_to_prot_idx,
                receptor_mask_prot, receptor_mask_kin,
                mech="distributive"):

    K = GLOBAL_K
    M = GLOBAL_M
    N = GLOBAL_N

    return network_rhs_nb_core(
        x, t, theta,
        Cg, Cl, site_prot_idx, K_site_kin, R, L_alpha,
        kin_to_prot_idx,
        receptor_mask_prot, receptor_mask_kin,
        K, M, N
    )

def simulate_p_scipy(t_arr, P_data0, A_data0, theta,
                     Cg, Cl, site_prot_idx,
                     K_site_kin, R,
                     L_alpha, kin_to_prot_idx,
                     receptor_mask_prot, receptor_mask_kin,
                     T_pre=50, n_pre_steps=20):
    """
    Simulate with:
      - pre-equilibration from t = -T_pre to 0 with u(t)=0
      - main simulation from t_arr[0] (should be 0) onwards with u(t)=1

    Default integrator: solve_ivp (LSODA).
    You can quickly switch back to odeint by uncommenting the indicated blocks.
    """

    K = GLOBAL_K
    M = GLOBAL_M
    N = GLOBAL_N
    state_dim = 2 * K + M + N

    # -----------------------------
    # Initial state:
    #   S = 0
    #   A = 0  (or 1, but you currently use 0)
    #   Kdyn = 1
    #   p = first time point of data (scaled)
    # -----------------------------
    x0 = np.zeros((state_dim,))
    x0[K:2 * K] = 0.0
    # x0[K:2 * K] = A_data0[:, 0]
    x0[2 * K:2 * K + M] = 0.0
    x0[2 * K + M:] = P_data0[:, 0]

    # Wrapper for solve_ivp: f(t, x)
    def rhs_ivp(t, x):
        return network_rhs(
            x, t, theta,
            Cg, Cl, site_prot_idx,
            K_site_kin, R, L_alpha, kin_to_prot_idx,
            receptor_mask_prot, receptor_mask_kin,
        )

    # -------------------------------------------------
    # 1) Pre-equilibration: t ∈ [-T_pre, 0]
    # -------------------------------------------------
    if T_pre > 0.0 and n_pre_steps > 1:
        t_pre_eval = np.linspace(-T_pre, 0.0, n_pre_steps)

        # --- DEFAULT: solve_ivp ---
        # sol_pre = solve_ivp(
        #     rhs_ivp,
        #     (-T_pre, 0.0),
        #     x0,
        #     t_eval=t_pre_eval,
        #     method="BDF",           # good for stiff systems & close to odeint
        #     rtol=1e-6,
        #     atol=1e-9,
        # )
        # if not sol_pre.success:
        #     # If integration fails, bail out with NaNs so objective penalizes this theta
        #     xs_pre = np.full((len(t_pre_eval), state_dim), np.nan, dtype=float)
        # else:
        #     xs_pre = sol_pre.y.T
        #
        # x0 = xs_pre[-1]

        # --- ALTERNATIVE: use odeint instead of solve_ivp ---
        xs_pre = odeint(
            network_rhs,
            x0,
            t_pre_eval,
            args=(theta, Cg, Cl, site_prot_idx,
                  K_site_kin, R, L_alpha, kin_to_prot_idx,
                  receptor_mask_prot, receptor_mask_kin),
        )
        x0 = xs_pre[-1]

    # -------------------------------------------------
    # 2) Main simulation: t ∈ [t_arr[0], t_arr[-1]]
    # -------------------------------------------------
    t0 = float(t_arr[0])
    t1 = float(t_arr[-1])

    # --- DEFAULT: solve_ivp ---
    # sol = solve_ivp(
    #     rhs_ivp,
    #     (t0, t1),
    #     x0,
    #     t_eval=t_arr,
    #     method="BDF",               # or "BDF"/"Radau" if very stiff
    #     rtol=1e-6,
    #     atol=1e-9,
    # )
    # if not sol.success:
    #     xs = np.full((len(t_arr), state_dim), np.nan, dtype=float)
    # else:
    #     xs = sol.y.T

    # --- ALTERNATIVE: use odeint instead of solve_ivp ---
    xs = odeint(
        network_rhs,
        x0,
        t_arr,
        args=(theta, Cg, Cl, site_prot_idx,
              K_site_kin, R, L_alpha, kin_to_prot_idx,
              receptor_mask_prot, receptor_mask_kin),
    )

    # -------------------------------------------------
    # 3) Unpack trajectories
    # -------------------------------------------------
    A_sim = xs[:, K:2 * K].T
    P_sim = xs[:, 2 * K + M:].T

    return P_sim, A_sim

# ------------------------------------------------------------
#  PYMOO PROBLEM DEFINITION
# ------------------------------------------------------------
@njit(cache=True)
def compute_objectives_nb(theta,
                          P_data, P_sim,
                          A_scaled, A_sim,
                          W_data, W_data_prot, prot_idx_for_A,
                          L_alpha, lambda_net, reg_lambda,
                          n_p, n_A, n_var,
                          K, M, N):
    """
    Numba-compiled objective computation:

    - f1: phosphosite fit
    - f2: protein abundance fit
    - f3: regularization + network regularization
    """

    # decode parameters (Numba version)
    (k_act, k_deact, s_prod, d_deg,
     beta_g, beta_l, alpha,
     kK_act, kK_deact, k_off,
     gamma_S_p, gamma_A_S, gamma_A_p, gamma_K_net) = decode_theta(theta, K, M, N)

    # -------------------------
    # 1) phosphosite loss f1
    # -------------------------
    N_sites, T = P_data.shape
    loss_p = 0.0

    for i in range(N_sites):
        for j in range(T):
            w = W_data[i, j]
            # sqrt on the fly for numba
            w_sqrt = np.sqrt(w)
            diff = (P_data[i, j] - P_sim[i, j]) * w_sqrt

            # clip (same logic as np.clip)
            if diff > 1e6:
                diff = 1e6
            elif diff < -1e6:
                diff = -1e6

            loss_p += diff * diff

    f1 = loss_p / n_p

    # -------------------------
    # 2) protein loss f2
    # -------------------------
    loss_A = 0.0

    if A_scaled.size > 0:
        K_prot_obs, T_A = A_scaled.shape
        for k in range(K_prot_obs):
            p_idx = prot_idx_for_A[k]
            for j in range(T_A):
                w = W_data_prot[k, j]
                w_sqrt = np.sqrt(w)
                diffA = (A_scaled[k, j] - A_sim[p_idx, j]) * w_sqrt

                if diffA > 1e6:
                    diffA = 1e6
                elif diffA < -1e6:
                    diffA = -1e6

                loss_A += diffA * diffA

        f2 = loss_A / n_A
    else:
        f2 = 0.0

    # -------------------------
    # 3) regularization f3
    # -------------------------
    # L2 on theta
    reg = reg_lambda * np.dot(theta, theta)

    # network regularization on alpha: alpha^T L_alpha alpha
    # y = L_alpha @ alpha
    M_kin = alpha.shape[0]
    y = np.zeros(M_kin)
    for i in range(M_kin):
        acc = 0.0
        for j in range(M_kin):
            acc += L_alpha[i, j] * alpha[j]
        y[i] = acc

    net_quad = 0.0
    for i in range(M_kin):
        net_quad += alpha[i] * y[i]

    reg_net = lambda_net * net_quad

    f3 = (reg + reg_net) / n_var

    return f1, f2, f3

def build_full_A0(K, T, A_scaled, prot_idx_for_A):
    """
    Build a (K, T) abundance matrix from subset A_scaled (K_obs, T)
    using mapping prot_idx_for_A (len = K_obs).

    Unobserved proteins get zeros.
    """
    A0_full = np.zeros((K, T), dtype=float)

    if A_scaled.size > 0:
        for k, p_idx in enumerate(prot_idx_for_A):
            # copy the whole time-course; simulate_p_scipy only uses [:, 0]
            A0_full[p_idx, :] = A_scaled[k, :]

    return A0_full

class NetworkOptimizationProblem(ElementwiseProblem):
    def __init__(self,
                 t, P_data, Cg, Cl, site_prot_idx, K_site_kin, R,
                 A_scaled, prot_idx_for_A, W_data, W_data_prot,
                 L_alpha, kin_to_prot_idx,
                 lambda_net, reg_lambda,
                 receptor_mask_prot, receptor_mask_kin,
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
        self.receptor_mask_prot = receptor_mask_prot
        self.receptor_mask_kin  = receptor_mask_kin

        self.n_p = max(1, self.P_data.size)
        self.n_A = max(1, self.A_scaled.size)
        self.n_var = n_var

    def _evaluate(self, x, out, *args, **kwargs):
        theta = x

        # --- simulate (still Python + odeint) ---
        K = GLOBAL_K
        T = self.P_data.shape[1]

        A0_full = build_full_A0(
            K,
            T,
            self.A_scaled,
            self.prot_idx_for_A
        )

        P_sim, A_sim = simulate_p_scipy(
            self.t, self.P_data, A0_full, theta,
            self.Cg, self.Cl, self.site_prot_idx,
            self.K_site_kin, self.R, self.L_alpha, self.kin_to_prot_idx,
            self.receptor_mask_prot, self.receptor_mask_kin
        )

        # ---- safety check: explosion / NaNs ----
        if (not np.all(np.isfinite(P_sim)) or
            not np.all(np.isfinite(A_sim)) or
            np.max(np.abs(P_sim)) > 1e6 or
            np.max(np.abs(A_sim)) > 1e6 or
            np.max(np.abs(theta)) > 1e3):

            out["F"] = np.array([1e12, 1e12, 1e12])
            return

        # -----------------------
        # JIT'ed objective core
        # -----------------------
        K = GLOBAL_K
        M = GLOBAL_M
        N = GLOBAL_N

        f1, f2, f3 = compute_objectives_nb(
            theta,
            self.P_data, P_sim,
            self.A_scaled, A_sim,
            self.W_data, self.W_data_prot, self.prot_idx_for_A,
            self.L_alpha, self.lambda_net, self.reg_lambda,
            self.n_p, self.n_A, self.n_var,
            K, M, N
        )

        out["F"] = np.array([f1, f2, f3], dtype=float)


@njit(cache=True)
def bio_score_nb(theta, K, M, N):
    (k_act, k_deact, s_prod, d_deg,
     beta_g, beta_l, alpha,
     kK_act, kK_deact, k_off,
     gamma_S_p, gamma_A_S, gamma_A_p, gamma_K_net) = decode_theta(theta, K, M, N)

    t_half_kinase = np.log(2.0) / kK_deact
    t_half_protein = np.log(2.0) / d_deg

    # median in numba: sort + pick middle
    tk = np.sort(t_half_kinase)
    tp = np.sort(t_half_protein)
    median_t_kinase  = tk[len(tk) // 2]
    median_t_protein = tp[len(tp) // 2]

    T_kinase_prior   = 10.0
    T_protein_prior  = 600.0

    term1 = (np.log10(median_t_kinase)  - np.log10(T_kinase_prior))**2
    term2 = (np.log10(median_t_protein) - np.log10(T_protein_prior))**2

    return term1 + term2

def bio_score(theta):
    return float(bio_score_nb(theta, GLOBAL_K, GLOBAL_M, GLOBAL_N))

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
    parser.add_argument("--lambda-net", type=float, default=0.0001, help="Laplacian regularization weight")
    parser.add_argument("--reg-lambda", type=float, default=0.0001, help="L2 regularization weight")
    parser.add_argument("--pop-size", type=int, default=100, help="Population size")
    parser.add_argument("--cores", type=int, default=60, help="Number of cores for multiprocessing")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print(f"[*] Creating output directory at: {args.outdir}")

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

    print(f"[*] Built context matrices Cg and Cl with shape: {Cg.shape}, {Cl.shape}")

    if args.kinase_tsv:
        K_site_kin, kinases = load_kinase_site_matrix(args.kinase_tsv, sites)
    elif args.kea_ks_table:
        K_site_kin, kinases = build_kinase_site_from_kea(args.kea_ks_table, sites)
    else:
        K_site_kin = np.eye(len(sites))
        kinases = [f"K_{i}" for i in range(len(sites))]
    R = K_site_kin.T


    for i, site in enumerate(sites):
        prot = proteins[site_prot_idx[i]]
        kin_for_site = [kinases[m] for m in range(len(kinases)) if K_site_kin[i, m] > 0]
        print(f"    Site: {site}, Protein: {prot}, Kinases: {kin_for_site}")

    print(f"[*] Loaded kinase-site matrix with {len(kinases)} kinases.")

    L_alpha = np.zeros((len(kinases), len(kinases)))
    if args.unified_graph_pkl and args.lambda_net > 0:
        L_alpha = build_alpha_laplacian_from_unified_graph(args.unified_graph_pkl, kinases)

    # After you have `proteins` and `kinases`
    prot_map_all = {p: i for i, p in enumerate(proteins)}
    kin_to_prot_idx = np.array(
        [prot_map_all.get(k, -1) for k in kinases],
        dtype=int
    )

    # After you have `proteins` and `kinases`
    prot_map_all = {p: i for i, p in enumerate(proteins)}
    kin_to_prot_idx = np.array(
        [prot_map_all.get(k, -1) for k in kinases],
        dtype=int
    )

    # --- RECEPTOR / TOP-LAYER HIERARCHY ---
    receptor_kinase_names = {"EGFR", "EPHA2", "ERBB4", "INSR", "RET"}

    receptor_mask_prot = np.array(
        [1 if p in {"EGFR", "ERBB2", "EPHA2", "MET"} else 0 for p in proteins],
        dtype=np.int64
    )

    receptor_mask_kin = np.array(
        [1 if k in receptor_kinase_names else 0 for k in kinases],
        dtype=np.int64
    )

    # Globals
    global GLOBAL_K, GLOBAL_M, GLOBAL_N
    GLOBAL_K, GLOBAL_N = len(proteins), len(sites)
    GLOBAL_M = len(kinases)

    # Params and Bounds
    K, M, N = GLOBAL_K, GLOBAL_M, GLOBAL_N
    dim = 4 * K + 2 + 3 * M + N + 4

    print(f"[*] Problem dimensions: K={K}, M={M}, N={N}, dim={dim}")

    # --- INTELLIGENT BOUNDS SETTING ---
    # We create specific bounds for specific parameter types rather than one global bound.

    xl = np.zeros(dim)
    xu = np.zeros(dim)

    idx = 0

    # 1. Protein Kinetics (K)
    # k_act (Activation): Allow range [1e-5, 100] - Can be fast
    xl[idx:idx + K] = np.log(1e-5);
    xu[idx:idx + K] = np.log(100.0);
    idx += K

    # k_deact (Deactivation): Allow range [1e-5, 100] - Can be fast
    xl[idx:idx + K] = np.log(1e-5);
    xu[idx:idx + K] = np.log(100.0);
    idx += K

    # s_prod (Synthesis): Allow range [1e-5, 10]
    xl[idx:idx + K] = np.log(1e-5);
    xu[idx:idx + K] = np.log(10.0);
    idx += K

    # d_deg (Degradation): *** RESTRICT THIS ***
    # Force degradation to be slow (e.g., max 0.5).
    # This forces the model to use PHOSPHORYLATION dynamics, not abundance destruction.
    xl[idx:idx + K] = np.log(1e-5);
    xu[idx:idx + K] = np.log(0.5);
    idx += K

    # 2. Coupling (2)
    xl[idx] = np.log(1e-5);
    xu[idx] = np.log(10.0);
    idx += 1  # beta_g
    xl[idx] = np.log(1e-5);
    xu[idx] = np.log(10.0);
    idx += 1  # beta_l

    # 3. Kinase Params (M)
    # Alpha (Global Strength): Allow high
    xl[idx:idx + M] = np.log(1e-5);
    xu[idx:idx + M] = np.log(100.0);
    idx += M
    # Kinase Act/Deact
    xl[idx:idx + M] = np.log(1e-5);
    xu[idx:idx + M] = np.log(20.0);
    idx += M
    xl[idx:idx + M] = np.log(1e-5);
    xu[idx:idx + M] = np.log(20.0);
    idx += M

    # 4. Site Params (N)
    # k_off (Phosphatase): Allow range [1e-5, 50] - High but not 100
    xl[idx:idx + N] = np.log(1e-5);
    xu[idx:idx + N] = np.log(10.0);
    idx += N

    # 5. Gamma coupling parameters (4 scalars, raw space)
    # order must match decode_theta: [gamma_S_p, gamma_A_S, gamma_A_p, gamma_K_net]
    xl[idx:idx + 4] = -3.0   # tanh(-3..3) ~ [-0.995, 0.995], then * 2.0 → roughly [-2, 2]
    xu[idx:idx + 4] =  3.0
    idx += 4

    assert idx == dim, f"Bounds indexing mismatch: idx={idx}, dim={dim}"

    # ----------------------------------------------------------
    # WARMP UP - JIT Compile
    # ----------------------------------------------------------
    print("[*] Warming up Numba...")

    K, M, N = GLOBAL_K, GLOBAL_M, GLOBAL_N
    dim = 4 * K + 2 + 3 * M + N + 4

    theta0 = np.zeros(dim, dtype=np.float64)
    state_dim = 2 * K + M + N
    x0 = np.zeros(state_dim, dtype=np.float64)

    _ = decode_theta(theta0, K, M, N)

    dx0 = network_rhs_nb_core(
        x0, 0.0, theta0,
        Cg.astype(np.float64),
        Cl.astype(np.float64),
        site_prot_idx.astype(np.int64),
        K_site_kin.astype(np.float64),
        R.astype(np.float64),
        L_alpha.astype(np.float64),
        kin_to_prot_idx.astype(np.int64),
        receptor_mask_prot.astype(np.int64),
        receptor_mask_kin.astype(np.int64),
        K, M, N
    )

    # Also warm up objective
    P_sim0 = np.zeros_like(P_scaled)
    A_sim0 = np.zeros((K, P_scaled.shape[1]), dtype=np.float64)

    _ = compute_objectives_nb(
        theta0,
        P_scaled, P_sim0,
        A_scaled, A_sim0,
        W_data, W_data_prot, prot_idx_for_A.astype(np.int64),
        L_alpha, args.lambda_net, args.reg_lambda,
        max(1, P_scaled.size),
        max(1, A_scaled.size),
        dim,
        K, M, N
    )

    print("[*] Numba warm-up done.")

    # ----------------------------------------------------------
    # Pymoo Problem Setup
    # ----------------------------------------------------------

    # Initialize Pool
    # print(f"[*] Initializing multiprocessing pool with {args.cores} cores...")
    # pool = multiprocessing.Pool(args.cores)
    # runner = StarmapParallelization(pool.starmap)

    runner = JoblibParallelization(
        n_jobs=-1,  # Use all cores
        backend="loky",  # Robust process-based backend
        batch_size="auto",  # Automatic batching
        pre_dispatch="2*n_jobs",  # Optimal pre-dispatching
    )

    # Initialize Problem
    problem = NetworkOptimizationProblem(
        t, P_scaled, Cg, Cl, site_prot_idx, K_site_kin, R,
        A_scaled, prot_idx_for_A, W_data, W_data_prot, L_alpha,
        kin_to_prot_idx,
        args.lambda_net, args.reg_lambda,
        receptor_mask_prot, receptor_mask_kin,
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
    # pool.close()
    # pool.join()

    print(f"[*] Optimization finished. Time: {res.exec_time}")

    # Objective set
    F = res.F

    # Decision variables
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

    J = d12 + f3n

    best_idx = np.argmin(J)
    theta_opt = X[best_idx]
    F_best = F[best_idx]

    print("[*] Best solution (prioritizing phosphosite + protein fit):")
    print("    f1 (P-sites) =", F_best[0])
    print("    f2 (protein) =", F_best[1])
    print("    f3 (network complexity) =", F_best[2])
    print(f"    Final Loss (F): {F_best}")

    # ----------------------------------------------------------
    #  Pareto front statistics & export
    # ----------------------------------------------------------
    # F is (n_points, 3): [f1 (P-sites), f2 (proteins), f3 (complexity)]
    f1 = F[:, 0]
    f2 = F[:, 1]
    f3 = F[:, 2]

    def summarize(arr, name):
        return {
            "objective": name,
            "min":   float(np.min(arr)),
            "q25":   float(np.percentile(arr, 25)),
            "median":float(np.median(arr)),
            "mean":  float(np.mean(arr)),
            "q75":   float(np.percentile(arr, 75)),
            "max":   float(np.max(arr)),
            "std":   float(np.std(arr)),
        }

    stats_rows = [
        summarize(f1, "f1_P_sites"),
        summarize(f2, "f2_protein"),
        summarize(f3, "f3_complexity"),
    ]

    df_stats = pd.DataFrame(stats_rows)

    print("\n[*] Pareto front objective stats:")
    print(df_stats.to_string(index=False,
                             float_format=lambda x: f"{x:10.4g}"))

    # Save stats and full Pareto front
    df_stats.to_csv(os.path.join(args.outdir, "pareto_stats.tsv"),
                    sep="\t", index=False)

    df_front = pd.DataFrame(
        F,
        columns=["f1_P_sites", "f2_protein", "f3_complexity"]
    )
    df_front.to_csv(os.path.join(args.outdir, "pareto_front.tsv"),
                    sep="\t", index=False)

    # Optional: also save scalarized score J for each solution (same J as below)
    eps = 1e-12
    f1n = (f1 - f1.min()) / (f1.max() - f1.min() + eps)
    f2n = (f2 - f2.min()) / (f2.max() - f2.min() + eps)
    d12 = np.sqrt(f1n ** 2 + f2n ** 2)
    f3n = (f3 - f3.min()) / (f3.max() - f3.min() + eps)
    J_all = d12 + f3n

    df_front["J_scalarized"] = J_all
    df_front.to_csv(os.path.join(args.outdir, "pareto_front_with_J.tsv"),
                    sep="\t", index=False)

    # ----------------------------------------------------------
    #  Pareto Front: summary statistics & exports
    # ----------------------------------------------------------
    obj_names = ["f1_P_sites", "f2_protein", "f3_complexity"]

    # 1) Basic stats for each objective
    df_F = pd.DataFrame(F, columns=obj_names)

    summary = pd.DataFrame({
        "objective": obj_names,
        "min":   df_F.min().values,
        "max":   df_F.max().values,
        "mean":  df_F.mean().values,
        "median": df_F.median().values,
        "std":   df_F.std(ddof=1).values,
    })

    print("\n=== Pareto Front: Objective Summary ===")
    print(summary.to_string(index=False))

    # 2) Index of best point for each single objective
    best_idx_per_obj = {name: int(df_F[name].idxmin()) for name in obj_names}
    print("\n=== Best index per objective (argmin) ===")
    for name in obj_names:
        idx_ = best_idx_per_obj[name]
        print(f"  {name}: idx={idx_}, value={df_F.loc[idx_, name]:.6g}")

    # 3) Pairwise correlations between objectives
    corr = df_F.corr()
    print("\n=== Correlation between objectives ===")
    print(corr.to_string(float_format=lambda x: f"{x: .3f}"))

    # 4) Simple spread metrics (range length in each objective)
    spread = df_F.max() - df_F.min()
    print("\n=== Objective spreads (max - min) ===")
    for name in obj_names:
        print(f"  {name}: spread={spread[name]:.6g}")

    # 5) Compute the same "J" scalar you use for selecting a point
    eps = 1e-12
    f1 = F[:, 0]
    f2 = F[:, 1]
    f3 = F[:, 2]

    f1n = (f1 - f1.min()) / (f1.max() - f1.min() + eps)
    f2n = (f2 - f2.min()) / (f2.max() - f2.min() + eps)
    d12 = np.sqrt(f1n ** 2 + f2n ** 2)

    f3n = (f3 - f3.min()) / (f3.max() - f3.min() + eps)

    J = d12 + f3n

    print("\n=== J metric (combined distance + complexity) ===")
    print(f"  J min:   {J.min():.6g}")
    print(f"  J max:   {J.max():.6g}")
    print(f"  J mean:  {J.mean():.6g}")
    print(f"  J median:{np.median(J):.6g}")

    # 6) Save everything to disk for later analysis
    pareto_front_path = os.path.join(args.outdir, "pareto_front.tsv")
    pareto_stats_path = os.path.join(args.outdir, "pareto_stats.tsv")
    pareto_corr_path  = os.path.join(args.outdir, "pareto_objective_correlation.tsv")
    pareto_npz_path   = os.path.join(args.outdir, "pareto_front.npz")

    # Add J and index to df_F before saving
    df_F_with_J = df_F.copy()
    df_F_with_J["J"] = J
    df_F_with_J["index"] = np.arange(len(df_F_with_J))

    df_F_with_J.to_csv(pareto_front_path, sep="\t", index=False)
    summary.to_csv(pareto_stats_path, sep="\t", index=False)
    corr.to_csv(pareto_corr_path, sep="\t")

    np.savez(
        pareto_npz_path,
        F=F,
        X=X,
        J=J,
        obj_names=np.array(obj_names, dtype=object)
    )

    print(f"\n[*] Saved Pareto front to {pareto_front_path}")
    print(f"[*] Saved Pareto stats  to {pareto_stats_path}")
    print(f"[*] Saved Pareto corr   to {pareto_corr_path}")
    print(f"[*] Saved Pareto NPZ    to {pareto_npz_path}\n")

    # ----------------------------------------------------------
    # Pareto front diagnostics & plots
    # ----------------------------------------------------------
    print("[*] Generating Pareto front diagnostics...")

    # 1) Save raw Pareto front (decision variables + objectives)
    pareto_npz_path = os.path.join(args.outdir, "pareto_front.npz")
    np.savez(pareto_npz_path, X=X, F=F)
    print(f"    Saved raw Pareto front to {pareto_npz_path}")

    # 2) Basic statistics for each objective
    obj_names = ["f1_Psites", "f2_proteins", "f3_complexity"]
    stats_rows = []

    for i, name in enumerate(obj_names):
        vals = F[:, i]
        row = {
            "objective": name,
            "min": float(vals.min()),
            "q25": float(np.quantile(vals, 0.25)),
            "median": float(np.median(vals)),
            "q75": float(np.quantile(vals, 0.75)),
            "max": float(vals.max()),
            "mean": float(vals.mean()),
            "std": float(vals.std()),
        }
        stats_rows.append(row)

        print(
            f"    {name}: "
            f"min={row['min']:.3g}, "
            f"q25={row['q25']:.3g}, "
            f"median={row['median']:.3g}, "
            f"q75={row['q75']:.3g}, "
            f"max={row['max']:.3g}, "
            f"mean={row['mean']:.3g}, "
            f"std={row['std']:.3g}"
        )

    df_stats = pd.DataFrame(stats_rows)
    stats_path = os.path.join(args.outdir, "pareto_stats.tsv")
    df_stats.to_csv(stats_path, sep="\t", index=False)
    print(f"    Saved objective statistics to {stats_path}")

    # 3) Bio score for all Pareto points (kinase/protein half-life prior fit)
    bio_scores = np.array([bio_score(theta) for theta in X], dtype=float)

    df_pf = pd.DataFrame(
        {
            "f1_Psites": f1,
            "f2_proteins": f2,
            "f3_complexity": f3,
            "bio_score": bio_scores,
            "J_selection": J,
        }
    )
    pareto_tsv_path = os.path.join(args.outdir, "pareto_points.tsv")
    df_pf.to_csv(pareto_tsv_path, sep="\t", index=False)
    print(f"    Saved Pareto point table to {pareto_tsv_path}")

    # 4) Scatter plot: f1 vs f2 colored by f3, highlight chosen solution
    plt.figure(figsize=(7, 6))
    sc = plt.scatter(f1, f2, c=f3, cmap="viridis", alpha=0.7)
    plt.colorbar(sc, label="f3 (network complexity)")
    plt.scatter(
        F_best[0],
        F_best[1],
        s=120,
        facecolors="none",
        edgecolors="red",
        linewidths=2,
        label="chosen solution",
    )
    plt.xlabel("f1: phosphosite loss")
    plt.ylabel("f2: protein loss")
    plt.title("Pareto front: f1 vs f2 (colored by f3)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "pareto_f1_f2.png"), dpi=300)
    plt.close()

    # 5) Histograms of each objective with chosen solution marked
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for i, ax in enumerate(axes):
        ax.hist(F[:, i], bins=30, alpha=0.8)
        ax.axvline(F_best[i], color="red", linestyle="--", linewidth=2)
        ax.set_title(obj_names[i])
        ax.set_xlabel("value")
        ax.set_ylabel("count")
    fig.suptitle("Objective distributions across Pareto front", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, "pareto_objective_hists.png"), dpi=300)
    plt.close(fig)

    # 6) Parameter correlation heatmap across Pareto front

    fig, ax = plt.subplots(figsize=(10, 8))
    corr = np.corrcoef(X.T)
    sns.heatmap(
        corr,
        ax=ax,
        cmap="coolwarm",
        center=0.0,
        square=True,
        xticklabels=False,
        yticklabels=False,
    )
    ax.set_title("Parameter correlation heatmap across Pareto front")
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, "pareto_param_corr.png"), dpi=300)
    plt.close(fig)

    # 7) (Optional) Relationship between combined f1/f2 distance and bio_score
    #    – tells you whether more "biological" solutions are also near the knee.
    plt.figure(figsize=(7, 5))
    plt.scatter(d12, bio_scores, alpha=0.6)
    plt.xlabel("d12 (normalized distance to ideal [f1,f2] = [0,0])")
    plt.ylabel("bio_score (half-life prior mismatch)")
    plt.title("Trade-off between phosphosite/protein fit and bio plausibility")
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "pareto_d12_vs_bioscore.png"), dpi=300)
    plt.close()

    print("[*] Pareto diagnostics written to disk.")

    # Final Simulation and Saving
    K = GLOBAL_K
    T = P_scaled.shape[1]

    A0_full = build_full_A0(
        K,
        T,
        A_scaled,
        prot_idx_for_A
    )

    P_sim, A_sim = simulate_p_scipy(
        t, P_scaled, A0_full, theta_opt,
        Cg, Cl, site_prot_idx, K_site_kin, R,
        L_alpha, kin_to_prot_idx,
        receptor_mask_prot, receptor_mask_kin
    )

    # Save
    (k_act, k_deact, s_prod, d_deg,
     beta_g, beta_l, alpha,
     kK_act, kK_deact, k_off,
     gamma_S_p, gamma_A_S, gamma_A_p, gamma_K_net) = decode_theta(
        theta_opt, GLOBAL_K, GLOBAL_M, GLOBAL_N
    )

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
        k_off=k_off,
        gamma_S_p=gamma_S_p,
        gamma_A_S=gamma_A_S,
        gamma_A_p=gamma_A_p,
        gamma_K_net=gamma_K_net,
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