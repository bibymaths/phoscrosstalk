#!/usr/bin/env python3
"""
Global phospho-network model with protein-specific inputs,
abundance dynamics, and global/local coupling from PTM SQLite DBs,
integrated with KEA kinase–substrate information and unified kinase graph.

States (per protein k and site i):

  For each protein k = 0..K-1:
    S_k(t): activation (0..1)
    A_k(t): abundance (>=0)

  For each phosphosite i = 0..N-1:
    p_i(t): phosphorylation fraction (0..1)

  For each kinase m = 0..M-1:
    Kdyn_m(t): kinase activity state (0..1)

Dynamics:

  Protein activation:
    dS_k/dt = k_act_k * (1 - S_k) - k_deact_k * S_k

  Abundance (with optional TF input collapsed into s_prod_k):
    dA_k/dt = s_prod_k - d_deg_k * A_k

  Crosstalk (saturated):
    coup_raw_i = beta_g * (C_g p)_i + beta_l * (C_l p)_i
    coup_i     = sat(coup_raw_i)  (here: tanh)

  Kinase dynamics:
    dKdyn_m/dt = kK_act_m * tanh( (R p)_m ) * (1 - Kdyn_m) - kK_deact_m * Kdyn_m
    where R = K_site_kin^T (kinase sees its substrates)

  Effective kinase on-rate:
    k_on_eff_i = sum_m K_site_kin[i,m] * alpha_m * Kdyn_m

  Phosphosite dynamics:
    dp_i/dt = (k_on_eff_i * S_{prot(i)} * A_{prot(i)} + coup_i) * (1 - p_i)
              - k_off_i * p_i

Observation model (for FC data):
  For each site i:
    y_sim_i(t) = baseline_i + amplitude_i * p_i(t)

Inputs:
  --data              time series CSV/TSV with:
                        (Protein, Residue, v1..v14) OR (GeneID, Psite, x1..x14)
  --ptm-intra         ptm_intra.db   (intra-protein PTM pairs)
  --ptm-inter         ptm_inter.db   (inter-protein PTM pairs)
  --kinase-tsv        optional TSV with per-site kinase mapping
  --kea-ks-table      optional KEA ks_psite_table.tsv to build K_site_kin
  --unified-graph-pkl optional unified_kinase_graph.pkl for α network prior
  --lambda-net        strength of α network prior (default 0)
  --crosstalk-tsv     optional TSV to restrict which sites are fitted
"""

from __future__ import annotations

import argparse
import os
import re
import sqlite3
import pickle

from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import networkx as nx
from mpi4py import MPI
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pymoo.termination.default import DefaultSingleObjectiveTermination
from pymoo.algorithms.soo.nonconvex.ga import GA

from numba import njit
from scipy.integrate import solve_ivp
# from scipy.optimize import minimize
from scipy.sparse import csr_matrix

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

# Global MPI context
COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()
SIZE = COMM.Get_size()


def mpi_print(*args, **kwargs):
    """Print only from rank 0 to avoid SLURM/MPI spam."""
    if RANK == 0:
        print(*args, **kwargs)

# ------------------------------------------------------------
#  TIMEPOINTS & DATA
# ------------------------------------------------------------

DEFAULT_TIMEPOINTS = np.array(
    [0.0, 0.5, 0.75, 1.0, 2.0, 4.0, 8.0, 16.0,
     30.0, 60.0, 120.0, 240.0, 480.0, 960.0]
)


def load_site_data(path, timepoints=DEFAULT_TIMEPOINTS):
    """
    Load time-series from a file that looks like:

        GeneID, Psite, x1..x14

    or the older format:

        Protein, Residue, v1..v14

    Returns
    -------
    sites         : list[str]  e.g. 'ABL2_S620'   (only true sites)
    proteins      : list[str]  unique protein names (with at least one site)
    site_prot_idx : np.array(N,) int (index into proteins for each site)
    positions     : np.array(N,) float (residue positions or NaN)
    t             : np.array(T,)
    Y             : np.array(N,T)  (site-level FC values)
    A_data        : np.array(P,T) or None  (protein-level FC, P rows)
    A_proteins    : np.array(P,) or None   (protein names for A_data rows)
    """
    df = pd.read_csv(path, sep=None, engine="python")

    # ---- identify time-series columns ----
    value_cols = [c for c in df.columns
                  if c.startswith("v") or c.startswith("x")]
    if len(value_cols) != len(timepoints):
        raise ValueError(
            f"Expected {len(timepoints)} value columns (v* or x*), "
            f"found {len(value_cols)}: {value_cols}"
        )

    # ---- unify protein column ----
    if "Protein" in df.columns:
        prot_col = "Protein"
    elif "GeneID" in df.columns:
        prot_col = "GeneID"
    else:
        raise ValueError("Need either 'Protein' or 'GeneID' column in data.")

    # ---- split: site rows vs protein-abundance rows ----
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
        # Old style: e.g. 'Y1172'
        for r in df_sites["Residue"].astype(str):
            residues_raw.append(r)
            m = re.match(r"[A-Z]([0-9]+)", r)
            positions.append(int(m.group(1)) if m else np.nan)

    elif "Psite" in df_sites.columns:
        # New style: e.g. 'S_620'
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
    else:
        raise ValueError("Need either 'Residue' or 'Psite' column in data.")

    positions = np.array(positions, dtype=float)

    # ---- build site IDs 'PROT_RES' ----
    sites = [f"{p}_{r}" for p, r in zip(proteins_raw, residues_raw)]

    # ---- unique proteins & indices (only proteins with at least one site) ----
    proteins = sorted(set(proteins_raw))
    prot_index = {p: k for k, p in enumerate(proteins)}
    site_prot_idx = np.array([prot_index[p] for p in proteins_raw], dtype=int)

    # ---- site-level time-series matrix ----
    Y = df_sites[value_cols].values.astype(float)
    t = np.array(timepoints, dtype=float)

    # ---- protein-level abundance series from rows with empty site ----
    A_data = None
    A_proteins = None

    if not df_prot.empty:
        A_rows = []
        A_prots = []
        for p, sub in df_prot.groupby(prot_col):
            p_str = str(p)
            # only keep proteins that have at least one phosphosite in the model
            if p_str not in prot_index:
                continue
            A_prots.append(p_str)
            A_rows.append(sub[value_cols].values.astype(float).mean(axis=0))

        if A_rows:
            A_data = np.vstack(A_rows)                # (P,T)
            A_proteins = np.array(A_prots, dtype=object)

    return sites, proteins, site_prot_idx, positions, t, Y, A_data, A_proteins


def scale_fc_to_unit_interval(Y, use_log=False, high_percentile=90.0):
    """
    Per site transform FC -> p in [0,1].

    p_i(t) = (y_i(t) - y_i(0)) / (P_y - y_i(0))

    Returns
    -------
    P          : (N,T) scaled p in [0,1]
    baselines  : (N,) y_i(0)
    amplitudes : (N,) P_y - y_i(0)
    """
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


def preprocess_rowwise(Y):
    """
    Generic row-wise preprocessing for raw time-series Y (N x T).

    Here: simple min-max normalisation per row to [0,1].
    """
    eps = 1e-8
    Y_min = Y.min(axis=1, keepdims=True)
    Y_max = Y.max(axis=1, keepdims=True)
    return (Y - Y_min) / (Y_max - Y_min + eps)


# ------------------------------------------------------------
#  C MATRICES FROM PTM SQLITE DBS
# ------------------------------------------------------------

def build_C_matrices_from_db(ptm_intra_path, ptm_inter_path,
                             sites, site_prot_idx, positions,
                             proteins,
                             length_scale=50.0):
    """
    Build global (PTM-based) and local (distance-based) coupling matrices.

    Cg_ij: PTM edges (intra + inter) from DBs, scored as
           0.8 * (r1 + r2) / 200.0, symmetric, only if both sites in 'sites'.

    Cl_ij: exp(-|pos_i-pos_j|/L) if same protein, i != j; else 0.
           Uses residue positions; ignores TF-level / unknown positions.
    """
    N = len(sites)
    idx = {s: i for i, s in enumerate(sites)}

    Cg = np.zeros((N, N), dtype=float)

    # --- INTRA ---
    conn_i = sqlite3.connect(ptm_intra_path)
    cur_i = conn_i.cursor()
    for protein, res1, r1, res2, r2 in cur_i.execute(
        "SELECT protein, residue1, score1, residue2, score2 FROM intra_pairs"
    ):
        s1 = f"{protein}_{res1}"
        s2 = f"{protein}_{res2}"
        if s1 not in idx or s2 not in idx:
            continue
        i = idx[s1]
        j = idx[s2]
        score = 0.8 * (r1 + r2) / 200.0
        if score > Cg[i, j]:
            Cg[i, j] = score
            Cg[j, i] = score
    conn_i.close()

    # --- INTER ---
    conn_e = sqlite3.connect(ptm_inter_path)
    cur_e = conn_e.cursor()
    for p1, res1, r1, p2, res2, r2 in cur_e.execute(
        "SELECT protein1, residue1, score1, protein2, residue2, score2 FROM inter_pairs"
    ):
        s1 = f"{p1}_{res1}"
        s2 = f"{p2}_{res2}"
        if s1 not in idx or s2 not in idx:
            continue
        i = idx[s1]
        j = idx[s2]
        score = 0.8 * (r1 + r2) / 200.0
        if score > Cg[i, j]:
            Cg[i, j] = score
            Cg[j, i] = score
    conn_e.close()

    # --- LOCAL (sequence distance within same protein) ---
    Cl = np.zeros((N, N), dtype=float)
    L = float(length_scale)
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            if site_prot_idx[i] != site_prot_idx[j]:
                continue
            # skip TF-level / unknown positions
            if not (np.isfinite(positions[i]) and np.isfinite(positions[j])):
                continue
            d = abs(positions[i] - positions[j])
            Cl[i, j] = np.exp(-d / L)

    Cg = csr_matrix(Cg)
    Cl = csr_matrix(Cl)
    return Cg, Cl


def row_normalize(C):
    """
    Row-normalise a CSR matrix.
    """
    row_sums = np.array(C.sum(axis=1)).reshape(-1, 1)
    row_sums[row_sums == 0.0] = 1.0
    C_norm = C.multiply(1.0 / row_sums)
    return C_norm


# ------------------------------------------------------------
#  KINASE–SITE MATRIX
# ------------------------------------------------------------

def load_kinase_site_matrix(path, sites):
    """
    Load kinase–site mapping from TSV and build site×kinase matrix.

    Expected TSV columns:
      Site   : string, matching entries in 'sites' (e.g. 'EGFR_Y1172')
      Kinase : kinase name (string)
      weight : optional float column, default 1.0

    Returns
    -------
    K_site_kin : (N,M) np.ndarray
    kinases    : list[str] of length M
    """
    df = pd.read_csv(path, sep="\t")

    if "Site" not in df.columns or "Kinase" not in df.columns:
        raise ValueError(
            f"{path} must contain at least 'Site' and 'Kinase' columns."
        )
    if "weight" not in df.columns:
        df["weight"] = 1.0

    site_to_idx = {s: i for i, s in enumerate(sites)}
    kinases = sorted(df["Kinase"].astype(str).unique())
    kin_index = {k: j for j, k in enumerate(kinases)}

    N = len(sites)
    M = len(kinases)
    K_site_kin = np.zeros((N, M), dtype=float)

    for _, row in df.iterrows():
        s = str(row["Site"])
        k = str(row["Kinase"])
        w = float(row["weight"])
        if s not in site_to_idx or k not in kin_index:
            continue
        i = site_to_idx[s]
        j = kin_index[k]
        if w > K_site_kin[i, j]:
            K_site_kin[i, j] = w

    return K_site_kin, kinases


def build_kinase_site_from_kea(ks_psite_table_path: str | Path,
                               sites: List[str]) -> Tuple[np.ndarray, List[str]]:
    """
    Build K_site_kin from KEA site-level table (ks_psite_table.tsv).

    ks_psite_table.tsv columns:
      kinase, substrate_site, substrate_gene, site, pmid, source

    We aggregate weight per (substrate_site, kinase) as number of unique PMIDs.

    Returns
    -------
    K_site_kin : (N,M) np.ndarray
    kinases    : list[str]
    """
    ks_psite_table_path = Path(ks_psite_table_path)
    df = pd.read_csv(ks_psite_table_path, sep="\t")

    df["substrate_site"] = df["substrate_site"].astype(str).str.upper()
    df["kinase"] = df["kinase"].astype(str).str.upper()

    # Build mapping from sites (model) to index
    sites_upper = [s.upper() for s in sites]
    site_to_idx = {s: i for i, s in enumerate(sites_upper)}

    df = df[df["substrate_site"].isin(site_to_idx.keys())].copy()

    if df.empty:
        raise ValueError("No overlap between ks_psite_table and model sites.")

    grouped = df.groupby(["substrate_site", "kinase"]).agg(
        weight=("pmid", "nunique")
    ).reset_index()

    kinases = sorted(grouped["kinase"].unique())
    kin_index = {k: j for j, k in enumerate(kinases)}

    N = len(sites)
    M = len(kinases)
    K_site_kin = np.zeros((N, M), dtype=float)

    for _, row in grouped.iterrows():
        s = row["substrate_site"]
        k = row["kinase"]
        w = float(row["weight"])
        i = site_to_idx[s]
        j = kin_index[k]
        if w > K_site_kin[i, j]:
            K_site_kin[i, j] = w

    # optional: normalise per-site row to [0,1] or sum=1
    row_sums = K_site_kin.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0.0] = 1.0
    K_site_kin = K_site_kin / row_sums

    return K_site_kin, kinases


# ------------------------------------------------------------
#  NETWORK PRIOR FROM UNIFIED KINASE GRAPH
# ------------------------------------------------------------

def build_alpha_laplacian_from_unified_graph(
        unified_pickle_path: str | Path,
        kinases: List[str],
        weight_attr: str = "weight_mean"
) -> np.ndarray:
    """
    Build Laplacian L (M x M) over the kinases used in K_site_kin,
    based on the unified kinase graph.

    Only kinases present in both the unified graph and in `kinases`
    are kept. L is dense numpy array (aligned with order of `kinases`).
    """
    unified_pickle_path = Path(unified_pickle_path)
    with unified_pickle_path.open("rb") as f:
        G_full = pickle.load(f)

    kin_to_idx = {k: i for i, k in enumerate(kinases)}
    M = len(kinases)
    A = np.zeros((M, M), dtype=float)

    for u, v, data in G_full.edges(data=True):
        if u not in kin_to_idx or v not in kin_to_idx:
            continue
        i = kin_to_idx[u]
        j = kin_to_idx[v]
        if i == j:
            continue
        w = float(data.get(weight_attr, 1.0))
        if w > A[i, j]:
            A[i, j] = w
            A[j, i] = w

    d = A.sum(axis=1)
    L = np.diag(d) - A
    return L


# ------------------------------------------------------------
#  ODE MODEL
# ------------------------------------------------------------

def sat(x):
    """Saturating nonlinearity for crosstalk term (tanh)."""
    return np.tanh(x)


def network_rhs(x, t, params, Cg, Cl, site_prot_idx, K, K_site_kin, R):
    """
    RHS for ODE system:

    x = [ S_0..S_{K-1},
          A_0..A_{K-1},
          Kdyn_0..Kdyn_{M-1},
          p_0..p_{N-1} ]
    """
    N = Cg.shape[0]
    M = K_site_kin.shape[1]

    k_act = params["k_act"]
    k_deact = params["k_deact"]
    s_prod = params["s_prod"]
    d_deg = params["d_deg"]
    beta_g = params["beta_g"]
    beta_l = params["beta_l"]
    alpha = params["alpha"]
    kK_act = params["kK_act"]
    kK_deact = params["kK_deact"]
    k_off = params["k_off"]

    # unpack state
    S = x[:K]
    A = x[K:2 * K]
    Kdyn = x[2 * K:2 * K + M]
    p = x[2 * K + M:2 * K + M + N]

    # --- protein activation ---
    dS = k_act * (1.0 - S) - k_deact * S

    # --- abundance dynamics ---
    dA = s_prod - d_deg * A

    # --- crosstalk terms (global + local, then saturate) ---
    coup_g = beta_g * (Cg @ p)
    coup_l = beta_l * (Cl @ p)
    coup_raw = coup_g + coup_l
    coup = sat(coup_raw)

    # --- kinase dynamics ---
    u = R @ p  # pooled phospho-signal per kinase (M,)
    dK = kK_act * np.tanh(u) * (1.0 - Kdyn) - kK_deact * Kdyn

    # --- effective on-rate per site, modulated by dynamic Kdyn ---
    k_on_eff = K_site_kin @ (alpha * Kdyn)   # (N,)

    S_local = S[site_prot_idx]
    A_local = A[site_prot_idx]

    v_on = (k_on_eff * S_local * A_local + coup) * (1.0 - p)
    v_off = k_off * p
    dp = v_on - v_off

    dx = np.empty(2 * K + M + N, dtype=float)
    dx[:K] = dS
    dx[K:2 * K] = dA
    dx[2 * K:2 * K + M] = dK
    dx[2 * K + M:] = dp
    return dx


@njit(cache=True, fastmath=True)
def decode_theta_core(theta, K, M, N):
    """
    Decode log-parameters from theta.

    theta layout:
      [ log_k_act_k (K),
        log_k_deact_k (K),
        log_s_prod_k (K),
        log_d_deg_k (K),
        log_beta_g,
        log_beta_l,
        log_alpha_m (M),
        log_kK_act_m (M),
        log_kK_deact_m (M),
        log_k_off_i (N) ]

    Returns
    -------
    k_act, k_deact, s_prod, d_deg,
    beta_g, beta_l,
    alpha,
    kK_act, kK_deact,
    k_off
    """
    idx0 = 0

    log_k_act = theta[idx0:idx0 + K]
    idx0 += K

    log_k_deact = theta[idx0:idx0 + K]
    idx0 += K

    log_s_prod = theta[idx0:idx0 + K]
    idx0 += K

    log_d_deg = theta[idx0:idx0 + K]
    idx0 += K

    log_beta_g = theta[idx0]
    idx0 += 1

    log_beta_l = theta[idx0]
    idx0 += 1

    log_alpha = theta[idx0:idx0 + M]
    idx0 += M

    log_kK_act = theta[idx0:idx0 + M]
    idx0 += M

    log_kK_deact = theta[idx0:idx0 + M]
    idx0 += M

    log_k_off = theta[idx0:idx0 + N]

    k_act = np.exp(log_k_act)
    k_deact = np.exp(log_k_deact)
    s_prod = np.exp(log_s_prod)
    d_deg = np.exp(log_d_deg)
    beta_g = np.exp(log_beta_g)
    beta_l = np.exp(log_beta_l)
    alpha = np.exp(log_alpha)
    kK_act = np.exp(log_kK_act)
    kK_deact = np.exp(log_kK_deact)
    k_off = np.exp(log_k_off)

    return (k_act, k_deact,
            s_prod, d_deg,
            beta_g, beta_l,
            alpha,
            kK_act, kK_deact,
            k_off)


def simulate_p(t, Cg, Cl, P_data, theta,
               site_prot_idx, K, K_site_kin):
    """
    Simulate p(t) given theta with dynamic kinase states.

    Returns
    -------
    P_sim  : (N,T) simulated phosphorylation fractions
    A_sim  : (K,T) simulated protein abundance
    params : dict of decoded (non-log) parameters
    """
    N, T = P_data.shape
    M = K_site_kin.shape[1]

    (k_act, k_deact,
     s_prod, d_deg,
     beta_g, beta_l,
     alpha,
     kK_act, kK_deact,
     k_off) = decode_theta_core(theta, K, M, N)

    params = {
        "k_act": k_act,
        "k_deact": k_deact,
        "s_prod": s_prod,
        "d_deg": d_deg,
        "beta_g": float(beta_g),
        "beta_l": float(beta_l),
        "alpha": alpha,
        "kK_act": kK_act,
        "kK_deact": kK_deact,
        "k_off": k_off,
    }
    R = K_site_kin.T

    def _rhs(t_curr, x_curr, params, Cg, Cl, site_prot_idx, K, K_site_kin):
        return network_rhs(x_curr, t_curr, params, Cg, Cl, site_prot_idx, K, K_site_kin, R)

    x0 = np.zeros(2 * K + M + N, dtype=float)
    x0[K:2 * K] = 1.0      # A_k(0)
    x0[2 * K:2 * K + M] = 1.0      # Kdyn_m(0)
    x0[2 * K + M:] = P_data[:, 0]

    sol = solve_ivp(
        fun=_rhs,
        t_span=(float(t[0]), float(t[-1])),
        y0=x0,
        t_eval=t,
        args=(params, Cg, Cl, site_prot_idx, K, K_site_kin),
    )

    X = sol.y.T
    A_sim = X[:, K:2 * K].T          # (K,T)
    P_sim = X[:, 2 * K + M:].T       # (N,T)
    return P_sim, A_sim, params


# ------------------------------------------------------------
#  FITTING
# ------------------------------------------------------------

def objective_slsqp(theta, t, Cg, Cl, P_data,
                    site_prot_idx, K, K_site_kin,
                    A_scaled=None, prot_idx_for_A=None, w_A=1.0,
                    W_data=None,
                    P_flat=None, A_flat=None, W_flat=None,
                    N=None, T=None,
                    reg_lambda=1e-4,
                    L_alpha=None,
                    lambda_net=0.0):
    """
    J(theta) = ||P_sim - P_data||^2
               + w_A * ||A_sim(proteins_with_data) - A_scaled||^2
               + reg_lambda * ||theta||^2
               + lambda_net * alpha^T L_alpha alpha
    """
    if N is None or T is None:
        N, T = P_data.shape

    P_sim, A_sim, _ = simulate_p(
        t, Cg, Cl, P_data, theta, site_prot_idx, K, K_site_kin
    )

    # phosphorylation loss
    if P_flat is None:
        diff_p = (P_sim - P_data).ravel()
        if W_data is None:
            loss_p = diff_p @ diff_p
        else:
            w_flat = W_data.ravel()
            diff_p *= np.sqrt(w_flat)
            loss_p = diff_p @ diff_p
    else:
        diff_p = P_sim.ravel() - P_flat
        if W_flat is None:
            loss_p = diff_p @ diff_p
        else:
            np.multiply(diff_p, np.sqrt(W_flat), out=diff_p)
            loss_p = diff_p @ diff_p

    # abundance loss
    loss_A = 0.0
    if A_scaled is not None and prot_idx_for_A is not None and len(prot_idx_for_A) > 0:
        A_model = A_sim[prot_idx_for_A, :]
        if A_flat is None:
            diff_A = (A_model - A_scaled).ravel()
        else:
            diff_A = A_model.ravel() - A_flat
        loss_A = diff_A @ diff_A

    # L2 reg on theta
    reg = reg_lambda * (theta @ theta)

    # network prior on alpha
    reg_net = 0.0
    if L_alpha is not None and lambda_net > 0.0:
        M = K_site_kin.shape[1]
        (_, _, _, _, _, _, alpha,
         _, _, _) = decode_theta_core(theta, K, M, N)
        reg_net = lambda_net * float(alpha @ (L_alpha @ alpha))

    return loss_p + w_A * loss_A + reg + reg_net

def make_eval_theta(
    t, Cg, Cl, P_data,
    site_prot_idx, K, K_site_kin,
    A_scaled, prot_idx_for_A, w_A,
    W_data,
    P_flat, A_flat, W_flat,
    N, T,
    L_alpha,
    lambda_net,
    reg_lambda=1e-4,
):
    """
    Capture all shared data and return eval_theta(theta) -> scalar J.

    This reuses your existing objective_slsqp, just binding all arguments
    except theta.
    """
    def eval_theta(theta: np.ndarray) -> float:
        return float(
            objective_slsqp(
                theta,
                t, Cg, Cl, P_data,
                site_prot_idx, K, K_site_kin,
                A_scaled, prot_idx_for_A, w_A,
                W_data,
                P_flat, A_flat, W_flat,
                N, T,
                reg_lambda,
                L_alpha,
                lambda_net,
            )
        )
    return eval_theta

def fit_network(t, Cg, Cl, P_data,
                site_prot_idx, K, K_site_kin,
                A_scaled=None, prot_idx_for_A=None, w_A=1.0,
                W_data=None,
                P_flat=None, A_flat=None, W_flat=None,
                N=None, T=None,
                L_alpha=None,
                lambda_net=0.0):
    """
    Fit parameters with SLSQP:

      Per-protein (k = 0..K-1):
        k_act_k, k_deact_k, s_prod_k, d_deg_k

      Global:
        beta_g, beta_l

      Per-kinase (m = 0..M-1):
        alpha_m, kK_act_m, kK_deact_m

      Per-site (i = 0..N-1):
        k_off_i
    """
    if N is None or T is None:
        N, T = P_data.shape
    M = K_site_kin.shape[1]

    k_act0 = np.full(K, 1.0)
    k_deact0 = np.full(K, 0.01)
    s_prod0 = np.full(K, 0.1)
    d_deg0 = np.full(K, 0.01)
    beta_g0 = 0.05
    beta_l0 = 0.05
    alpha0 = np.full(M, 0.1)
    kK_act0 = np.full(M, 0.5)
    kK_deact0 = np.full(M, 0.1)
    k_off0 = np.full(N, 0.05)

    theta0 = np.concatenate([
        np.log(k_act0),
        np.log(k_deact0),
        np.log(s_prod0),
        np.log(d_deg0),
        np.log([beta_g0]),
        np.log([beta_l0]),
        np.log(alpha0),
        np.log(kK_act0),
        np.log(kK_deact0),
        np.log(k_off0),
    ])

    lower = np.log(1e-4) * np.ones_like(theta0)
    upper = np.log(2.0) * np.ones_like(theta0)
    bounds = list(zip(lower, upper))

    res = minimize(
        objective_slsqp,
        theta0,
        args=(t, Cg, Cl, P_data,
              site_prot_idx, K, K_site_kin,
              A_scaled, prot_idx_for_A, w_A,
              W_data,
              P_flat, A_flat, W_flat,
              N, T,
              1e-4,
              L_alpha,
              lambda_net),
        method="SLSQP",
        bounds=bounds,
        options={
            "disp": True,
            "maxiter": 200,
        },
    )

    theta_opt = res.x
    P_sim, A_sim, params_decoded = simulate_p(
        t, Cg, Cl, P_data, theta_opt, site_prot_idx, K, K_site_kin
    )

    return theta_opt, params_decoded, P_sim, res

class PhosProblem(Problem):
    """
    Pymoo problem: minimize J(theta).

    This is *vectorized* over a population X (shape: [pop_size, n_var]).
    """

    def __init__(self, n_var: int,
                 xl: np.ndarray,
                 xu: np.ndarray,
                 eval_theta,
                 **kwargs):
        super().__init__(
            n_var=n_var,
            n_obj=1,
            n_ieq_constr=0,
            xl=xl,
            xu=xu,
            elementwise=False,   # evaluate whole population in one call
            **kwargs
        )
        self.eval_theta = eval_theta

    def _evaluate(self, X, out, *args, **kwargs):
        # X: (pop_size, n_var)
        pop_size = X.shape[0]
        F = np.empty((pop_size, 1), dtype=float)
        for i in range(pop_size):
            F[i, 0] = self.eval_theta(X[i])
        out["F"] = F

def run_pymoo_mpi_ga(
    theta0: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    eval_theta,
    pop_size: int = 64,
    n_gen: int = 200,
):
    """
    Run a GA-based global optimization with pymoo on each MPI rank
    (island model), then pick the globally best theta using MPI.
    """

    # Use global COMM/RANK/SIZE so SLURM-aware behavior is consistent
    comm = COMM
    rank = RANK
    size = SIZE

    n_var = theta0.size

    xl = lower.astype(float)
    xu = upper.astype(float)

    problem = PhosProblem(
        n_var=n_var,
        xl=xl,
        xu=xu,
        eval_theta=eval_theta,
    )

    # Rank-specific seed, but set via numpy global RNG for old pymoo
    seed = 42 + rank
    np.random.seed(seed)

    algorithm = GA(
        pop_size=pop_size,
        eliminate_duplicates=True,
    )

    termination = DefaultSingleObjectiveTermination(
        xtol=1e-8,
        cvtol=1e-6,
        ftol=1e-6,
        period=20,
        n_max_gen=1000,
        n_max_evals=100000
    )

    if rank == 0:
        print(f"[*] MPI+pymoo: starting GA on {size} ranks, pop_size={pop_size}, n_gen={n_gen}")

    def ga_callback(algorithm):
        """Print progress every generation from rank 0 only."""
        gen = algorithm.n_gen
        if RANK == 0 and gen % 5 == 0:  # adjust frequency
            best = algorithm.pop.get("F").min()
            mpi_print(f"[gen {gen}] best J = {best:.4g}")

    res = minimize(
        problem,
        algorithm,
        termination,
        verbose=False,  # disable pymoo’s own noisy print
        callback=ga_callback,  # our MPI-safe progress
    )

    # Local best
    local_best_theta = res.X.copy()
    local_best_J = float(res.F[0, 0])

    # Gather best from all ranks
    all_J = comm.allgather(local_best_J)
    all_theta = comm.allgather(local_best_theta)

    if rank == 0:
        best_rank = int(np.argmin(all_J))
        global_best_J = all_J[best_rank]
        global_best_theta = all_theta[best_rank]

        print(f"[*] MPI+pymoo global best J = {global_best_J:.4g} (from rank {best_rank})")

        return global_best_theta, global_best_J
    else:
        return None, None
# ------------------------------------------------------------
#  HELPERS: CROSSTALK RESTRICTION, PLOTTING
# ------------------------------------------------------------

def load_allowed_sites_from_crosstalk(path):
    """
    Read crosstalk_predictions.tsv and return a set of site IDs
    in format 'PROT_RES', as used in `sites` from load_site_data.

    Expects columns: Protein, Site1, Site2.
    """
    df = pd.read_csv(path, sep="\t")

    required_cols = {"Protein", "Site1", "Site2"}
    if not required_cols.issubset(df.columns):
        raise ValueError(
            f"{path} must contain columns: {required_cols}, "
            f"found: {set(df.columns)}"
        )

    site_ids = set()

    for col in ["Site1", "Site2"]:
        prots = df["Protein"].astype(str).values
        sites = df[col].astype(str).values
        for p, s in zip(prots, sites):
            if pd.isna(p) or pd.isna(s):
                continue
            site_ids.add(f"{p}_{s}")

    return site_ids


def plot_goodness_of_fit(Y_data, Y_sim, outpath):
    """Scatter plot: observed vs simulated FC."""
    plt.figure(figsize=(6, 6))
    plt.scatter(Y_data.flatten(), Y_sim.flatten(), alpha=0.5)
    min_y = min(Y_data.min(), Y_sim.min())
    max_y = max(Y_data.max(), Y_sim.max())
    plt.plot([min_y, max_y], [min_y, max_y], 'r--')
    plt.xlabel('Observed FC')
    plt.ylabel('Simulated FC')
    plt.title('Goodness of Fit: Observed vs Simulated FC')
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()
    print(f"[*] Saved goodness of fit plot to {outpath}")


# ------------------------------------------------------------
#  MAIN
# ------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Fit global phospho network with activation, abundance, "
                    "and kinase–site-driven phosphorylation (KEA-integrated)."
    )
    parser.add_argument(
        "--data", required=True,
        help="CSV/TSV with either (Protein, Residue, v1..v14) "
             "or (GeneID, Psite, x1..x14)"
    )
    parser.add_argument("--ptm-intra", required=True,
                        help="SQLite DB with intra_pairs (ptm_intra.db)")
    parser.add_argument("--ptm-inter", required=True,
                        help="SQLite DB with inter_pairs (ptm_inter.db)")
    parser.add_argument("--outdir", default="network_fit",
                        help="Output directory.")
    parser.add_argument("--length-scale", type=float, default=50.0,
                        help="Length scale L for local coupling exp(-|i-j|/L).")
    parser.add_argument(
        "--crosstalk-tsv",
        help="TSV with columns Protein, Site1, Site2 to restrict which sites are fitted."
    )
    parser.add_argument(
        "--kinase-tsv",
        help="Optional TSV with columns Site, Kinase, [weight] to define "
             "kinase–site matrix. If omitted but --kea-ks-table is given, "
             "KEA-based matrix will be built. Otherwise identity is used."
    )
    parser.add_argument(
        "--kea-ks-table",
        help="KEA ks_psite_table.tsv to automatically build kinase–site matrix "
             "for the sites in --data."
    )
    parser.add_argument(
        "--unified-graph-pkl",
        help="Pickled unified_kinase_graph.pkl (NetworkX Graph) for "
             "α network regularisation."
    )
    parser.add_argument(
        "--lambda-net",
        type=float,
        default=0.0,
        help="Strength of α network prior using unified kinase graph (default 0)."
    )
    # --- HPC/MPI controls ---
    parser.add_argument(
        "--optimizer",
        choices=["ga", "slsqp"],
        default="ga",
        help="Use 'ga' for MPI+pymoo GA (island model) or 'slsqp' for single-rank SciPy SLSQP."
    )
    parser.add_argument(
        "--ga-pop-size",
        type=int,
        default=64,
        help="Population size for GA when --optimizer ga (default 64)."
    )
    parser.add_argument(
        "--ga-n-gen",
        type=int,
        default=200,
        help="Number of generations for GA when --optimizer ga (default 200)."
    )

    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # 1) Load full data (sites + optional protein abundance rows)
    (sites, proteins, site_prot_idx,
     positions, t, Y, A_data, A_proteins) = load_site_data(args.data)
    print(f"[*] Loaded {len(sites)} sites from {len(proteins)} proteins, "
          f"{Y.shape[1]} timepoints.")

    # # Preprocess raw site-level data row-wise to [0,1]
    # Y = preprocess_rowwise(Y)
    #
    # # Preprocess protein-level abundance data
    # if A_data is not None:
    #     A_data = preprocess_rowwise(A_data)

    # Optionally restrict to sites present in crosstalk TSV
    if args.crosstalk_tsv is not None:
        allowed_sites = load_allowed_sites_from_crosstalk(args.crosstalk_tsv)

        mask = np.array([s in allowed_sites for s in sites], dtype=bool)
        n_before = len(sites)
        n_after = mask.sum()

        if n_after == 0:
            raise ValueError(
                "No overlap between time-series data and crosstalk sites. "
                "Check that Protein and Site1/Site2 formats match."
            )

        sites = [s for s, keep in zip(sites, mask) if keep]
        positions = positions[mask]
        Y = Y[mask, :]

        proteins_used = sorted({s.split("_", 1)[0] for s in sites})
        prot_index = {p: k for k, p in enumerate(proteins_used)}
        site_prot_idx = np.array(
            [prot_index[s.split("_", 1)[0]] for s in sites],
            dtype=int
        )
        proteins = proteins_used

        if A_data is not None and A_proteins is not None:
            mask_A = np.array([p in prot_index for p in A_proteins], dtype=bool)
            A_data = A_data[mask_A]
            A_proteins = A_proteins[mask_A]

        print(f"[*] Restricted sites via crosstalk TSV: "
              f"{n_before} -> {n_after} sites, {len(proteins)} proteins.")

    # Map protein-abundance rows to protein indices (after any restriction)
    if A_data is not None and A_proteins is not None and len(A_data) > 0:
        prot_index = {p: k for k, p in enumerate(proteins)}
        mask_A = np.array([p in prot_index for p in A_proteins], dtype=bool)
        A_data = A_data[mask_A]
        A_proteins = A_proteins[mask_A]
        if len(A_data) > 0:
            prot_idx_for_A = np.array([prot_index[p] for p in A_proteins], dtype=int)
        else:
            prot_idx_for_A = None
    else:
        prot_idx_for_A = None

    # 2) Scale FC -> p in [0,1] for sites
    P_scaled, baselines, amplitudes = scale_fc_to_unit_interval(
        Y, use_log=False, high_percentile=90.0
    )
    print("[*] Scaled site-level FC data to p in [0,1].")

    # 2b) Scale protein-level FC -> A_scaled in [0,1] (if present)
    if A_data is not None and prot_idx_for_A is not None and len(A_data) > 0:
        A_scaled, A_baselines, A_amplitudes = scale_fc_to_unit_interval(
            A_data, use_log=False, high_percentile=90.0
        )
        print(f"[*] Scaled protein-level FC data for {A_data.shape[0]} proteins.")
    else:
        A_scaled = None
        A_baselines = None
        A_amplitudes = None

    # 2c) Build weights for phosphorylation loss (emphasise higher FC)
    w_fac = 3.0
    Y_min = Y.min()
    Y_max = Y.max()
    denom = (Y_max - Y_min) if (Y_max > Y_min) else 1.0

    Y_norm = (Y - Y_min) / denom
    W_data = 1.0 + w_fac * Y_norm

    # 3) Build C matrices from DBs
    Cg, Cl = build_C_matrices_from_db(
        args.ptm_intra, args.ptm_inter,
        sites, site_prot_idx, positions, proteins,
        length_scale=args.length_scale,
    )
    Cg = row_normalize(Cg)
    Cl = row_normalize(Cl)
    print("[*] Built and row-normalised C_global and C_local.")

    # 4) Build kinase–site matrix
    if args.kinase_tsv is not None:
        K_site_kin, kinases = load_kinase_site_matrix(args.kinase_tsv, sites)
        print(f"[*] Loaded kinase–site matrix from {args.kinase_tsv}: "
              f"{K_site_kin.shape[0]} sites × {K_site_kin.shape[1]} kinases.")
    elif args.kea_ks_table is not None:
        K_site_kin, kinases = build_kinase_site_from_kea(args.kea_ks_table, sites)
        print(f"[*] Built kinase–site matrix from KEA ks_psite_table: "
              f"{K_site_kin.shape[0]} sites × {K_site_kin.shape[1]} kinases.")
    else:
        N_sites = len(sites)
        K_site_kin = np.eye(N_sites, dtype=float)
        kinases = [f"Kinase_{i}" for i in range(N_sites)]
        print("[*] No --kinase-tsv or --kea-ks-table supplied. "
              "Using identity matrix: one pseudo-kinase per site.")

    # 4b) Build Laplacian over kinases using unified kinase graph (if available)
    L_alpha = None
    if args.unified_graph_pkl and os.path.exists(args.unified_graph_pkl) and args.lambda_net > 0.0:
        print(f"[*] Building α Laplacian from {args.unified_graph_pkl} ...")
        L_alpha = build_alpha_laplacian_from_unified_graph(
            args.unified_graph_pkl, kinases, weight_attr="weight_mean"
        )
        print(f"[*] Built L_alpha for network regularisation (lambda_net={args.lambda_net}).")
    else:
        if args.lambda_net > 0.0 and not args.unified_graph_pkl:
            print("[!] lambda-net > 0 but no --unified-graph-pkl given. "
                  "Network prior will be ignored.")
        else:
            print("[*] No α network prior used.")

    # 5) Fit network
    K = len(proteins)
    N, T = P_scaled.shape

    P_flat = P_scaled.ravel().copy()
    W_flat = W_data.ravel().copy() if W_data is not None else None

    if A_scaled is not None and prot_idx_for_A is not None:
        A_flat = A_scaled.ravel().copy()
    else:
        A_flat = None

    # ---- choose optimizer: SLSQP (old) or MPI+pymoo GA (new) ----
    use_pymoo_mpi = (args.optimizer == "ga")

    if not use_pymoo_mpi:
        # If running under MPI with SLSQP, only rank 0 should do any work.
        if args.optimizer == "slsqp" and SIZE > 1 and RANK != 0:
            mpi_print(f"[rank {RANK}] exiting early (SLSQP runs only on rank 0).")
            return

        # Original SLSQP fit (single rank, enforced above)
        theta_opt, params_decoded, P_sim, result = fit_network(
            t, Cg, Cl, P_scaled, site_prot_idx, K, K_site_kin,
            A_scaled=A_scaled, prot_idx_for_A=prot_idx_for_A, w_A=1.0,
            W_data=W_data,
            P_flat=P_flat, A_flat=A_flat, W_flat=W_flat,
            N=N, T=T,
            L_alpha=L_alpha,
            lambda_net=args.lambda_net,
        )
        final_cost = result.fun
        success = result.success
        msg = result.message

    else:
        # ---- MPI + pymoo GA fit ----
        # 1) Rebuild theta0, lower, upper exactly as in fit_network
        M = K_site_kin.shape[1]

        k_act0   = np.full(K, 1.0)
        k_deact0 = np.full(K, 0.01)
        s_prod0  = np.full(K, 0.1)
        d_deg0   = np.full(K, 0.01)
        beta_g0  = 0.05
        beta_l0  = 0.05
        alpha0   = np.full(M, 0.1)
        kK_act0  = np.full(M, 0.5)
        kK_deact0 = np.full(M, 0.1)
        k_off0   = np.full(N, 0.05)

        theta0 = np.concatenate([
            np.log(k_act0),
            np.log(k_deact0),
            np.log(s_prod0),
            np.log(d_deg0),
            np.log([beta_g0]),
            np.log([beta_l0]),
            np.log(alpha0),
            np.log(kK_act0),
            np.log(kK_deact0),
            np.log(k_off0),
        ])

        lower = np.log(1e-4) * np.ones_like(theta0)
        upper = np.log(2.0)   * np.ones_like(theta0)

        # 2) Build eval_theta wrapper
        eval_theta = make_eval_theta(
            t, Cg, Cl, P_scaled,
            site_prot_idx, K, K_site_kin,
            A_scaled, prot_idx_for_A, 1.0,
            W_data,
            P_flat, A_flat, W_flat,
            N, T,
            L_alpha,
            args.lambda_net,
            reg_lambda=1e-4,
        )

        # 3) Run MPI+pymoo GA
        theta_best, J_best = run_pymoo_mpi_ga(
            theta0=theta0,
            lower=lower,
            upper=upper,
            eval_theta=eval_theta,
            pop_size=args.ga_pop_size,
            n_gen=args.ga_n_gen,
        )

        # Only rank 0 continues with decoding and saving
        comm = MPI.COMM_WORLD
        if comm.Get_rank() != 0:
            # Other ranks should exit cleanly
            return

        theta_opt = theta_best
        final_cost = J_best
        success = True
        msg = "MPI+pymoo GA finished"

        # Decode params and simulate once more to get P_sim
        M = K_site_kin.shape[1]
        (k_act, k_deact,
         s_prod, d_deg,
         beta_g, beta_l,
         alpha,
         kK_act, kK_deact,
         k_off) = decode_theta_core(theta_opt, K, M, N)

        params_decoded = {
            "k_act":   k_act,
            "k_deact": k_deact,
            "s_prod":  s_prod,
            "d_deg":   d_deg,
            "beta_g":  float(beta_g),
            "beta_l":  float(beta_l),
            "alpha":   alpha,
            "kK_act":  kK_act,
            "kK_deact": kK_deact,
            "k_off":   k_off,
        }

        P_sim, A_sim, _ = simulate_p(
            t, Cg, Cl, P_scaled, theta_opt, site_prot_idx, K, K_site_kin
        )
        result = None  # no SciPy result object here

    print("[*] Optimization finished.")
    print(f"    Success: {success}, message: {msg}")
    print(f"    Final cost (J): {final_cost:.4g}")

    if RANK == 0:
        # 6) Save parameters & mappings
        out_params = {
            "proteins": np.array(proteins, dtype=object),
            "sites": np.array(sites, dtype=object),
            "kinases": np.array(kinases, dtype=object),
            "K_site_kin": K_site_kin,
            "site_prot_idx": site_prot_idx,
            "positions": positions,
            "k_act": params_decoded["k_act"],
            "k_deact": params_decoded["k_deact"],
            "s_prod": params_decoded["s_prod"],
            "d_deg": params_decoded["d_deg"],
            "beta_g": params_decoded["beta_g"],
            "beta_l": params_decoded["beta_l"],
            "alpha": params_decoded["alpha"],
            "k_off": params_decoded["k_off"],
            "baselines": baselines,
            "amplitudes": amplitudes,
            "A_proteins": np.array([] if A_proteins is None else A_proteins,
                                   dtype=object),
            "A_baselines": np.array([] if A_baselines is None else A_baselines),
            "A_amplitudes": np.array([] if A_amplitudes is None else A_amplitudes),
        }
        params_path = os.path.join(args.outdir, "fitted_params.npz")
        np.savez(params_path, **out_params)
        mpi_print(f"[*] Saved parameters to {params_path}")

        # 7) Reconstruct FC fits and save TSV
        N_sites, T_points = Y.shape
        Y_sim = np.zeros_like(Y)
        for i in range(N_sites):
            Y_sim[i] = baselines[i] + amplitudes[i] * P_sim[i]

        df_out = pd.DataFrame({
            "Protein": [s.split("_", 1)[0] for s in sites],
            "Residue": [s.split("_", 1)[1] for s in sites],
        })
        for j in range(T_points):
            df_out[f"data_t{j}"] = Y[:, j]
            df_out[f"sim_t{j}"] = Y_sim[:, j]

        tsv_path = os.path.join(args.outdir, "fit_timeseries.tsv")
        df_out.to_csv(tsv_path, sep="\t", index=False)
        mpi_print(f"[*] Saved simulated vs data time series to {tsv_path}")

        # 8) Goodness of fit plot
        gof_path = os.path.join(args.outdir, "goodness_of_fit.png")
        plot_goodness_of_fit(Y, Y_sim, gof_path)
        mpi_print("[*] Done.")


if __name__ == "__main__":
    main()