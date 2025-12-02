#!/usr/bin/env python3
"""
Global phospho-network model with protein-specific inputs,
abundance dynamics, and global/local coupling from PTM SQLite DBs.

States (per protein k and site i):

  For each protein k = 0..K-1:
    S_k(t): activation (0..1)
    A_k(t): abundance (>=0)

  For each phosphosite i = 0..N-1:
    p_i(t): phosphorylation fraction (0..1)
    # q_i(t): optional hyper-phosphorylated state (not yet implemented)

Dynamics:

  Protein activation:
    dS_k/dt = k_act_k * (1 - S_k) - k_deact_k * S_k

  Abundance (with optional TF input collapsed into s_prod_k):
    dA_k/dt = s_prod_k - d_deg_k * A_k

  Crosstalk (saturated):
    coup_raw_i = beta_g * (C_g p)_i + beta_l * (C_l p)_i
    coup_i     = sat(coup_raw_i)  (here: tanh)

  Effective kinase on-rate:
    k_on_eff_i = (K_site_kin @ alpha)_i

  Phosphosite dynamics:
    dp_i/dt = (k_on_eff_i * S_{prot(i)} * A_{prot(i)} + coup_i) * (1 - p_i)
              - k_off_i * p_i

Observation model (for FC data):
  For each site i:
    y_sim_i(t) = baseline_i + amplitude_i * p_i(t)

Inputs:
  --data         time series CSV/TSV with:
                   (Protein, Residue, v1..v14) OR (GeneID, Psite, x1..x14)
  --ptm-intra    ptm_intra.db   (intra-protein PTM pairs)
  --ptm-inter    ptm_inter.db   (inter-protein PTM pairs)
  --kinase-tsv   optional TSV with per-site kinase mapping
  --crosstalk-tsv optional TSV to restrict which sites are fitted
"""

import numpy as np
import pandas as pd
import sqlite3
import argparse
import os
import re

from matplotlib import pyplot as plt
from numba import njit
from scipy.integrate import odeint
from scipy.optimize import minimize
from scipy.sparse import csr_matrix

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
    sites         : list[str]  e.g. 'ABL2_S620'
    proteins      : list[str]  unique protein names (for global model)
    site_prot_idx : np.array(N,) int (index into proteins for each site)
    positions     : np.array(N,) float (residue positions or NaN)
    t             : np.array(T,)
    Y             : np.array(N,T)  (FC values)
    """
    # auto-detect separator so it works for CSV or TSV
    df = pd.read_csv(path, sep=None, engine="python")

    # ---- identify time-series columns ----
    # allow both v1..v14 and x1..x14
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

    proteins_raw = df[prot_col].astype(str).tolist()

    # ---- unify residue / psite column ----
    residues_raw = []
    positions = []

    if "Residue" in df.columns:
        # Old style: e.g. 'Y1172'
        for r in df["Residue"].astype(str):
            residues_raw.append(r)
            m = re.match(r"[A-Z]([0-9]+)", r)
            positions.append(int(m.group(1)) if m else np.nan)

    elif "Psite" in df.columns:
        # New style: e.g. 'S_620' or NaN for TF-level
        for psite in df["Psite"]:
            if pd.isna(psite):
                # TF level / no specific site
                residues_raw.append("TF")
                positions.append(np.nan)
            else:
                # 'S_620' -> 'S620', pos=620
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

    # ---- unique proteins & indices ----
    proteins = sorted(set(proteins_raw))
    prot_index = {p: k for k, p in enumerate(proteins)}
    site_prot_idx = np.array([prot_index[p] for p in proteins_raw], dtype=int)

    # ---- time-series matrix ----
    Y = df[value_cols].values.astype(float)
    t = np.array(timepoints, dtype=float)

    return sites, proteins, site_prot_idx, positions, t, Y


def scale_fc_to_unit_interval(Y):
    """
    Per site transform FC -> p in [0,1].

    p_i(t) = (y_i(t) - y_i(0)) / (max_t y_i(t) - y_i(0))

    Returns
    -------
    P          : (N,T) scaled p in [0,1]
    baselines  : (N,) y_i(0)
    amplitudes : (N,) max_t y_i(t) - y_i(0)
    """
    N, T = Y.shape
    P = np.zeros_like(Y)
    baselines = np.zeros(N)
    amplitudes = np.zeros(N)
    eps = 1e-6

    for i in range(N):
        y = Y[i]
        b = y[0]
        A = y.max() - b
        if A < eps:
            A = 1.0
        p = (y - b) / A
        p = np.clip(p, 0.0, 1.0)
        P[i] = p
        baselines[i] = b
        amplitudes[i] = A

    return P, baselines, amplitudes


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

    # Convert to CSR sparse matrices for speed when N grows
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
        # use max weight if multiple entries
        if w > K_site_kin[i, j]:
            K_site_kin[i, j] = w

    return K_site_kin, kinases


# ------------------------------------------------------------
#  ODE MODEL
# ------------------------------------------------------------

def sat(x):
    """
    Saturating nonlinearity for crosstalk term.
    Here: tanh(x), bounded in (-1,1).
    """
    return np.tanh(x)


def network_rhs(x, t, params, Cg, Cl, site_prot_idx, K, K_site_kin):
    """
    RHS for ODE system:

    x = [ S_0..S_{K-1},
          A_0..A_{K-1},
          p_0..p_{N-1} ]
    """
    N = Cg.shape[0]

    k_act = params["k_act"]
    k_deact = params["k_deact"]
    s_prod = params["s_prod"]
    d_deg = params["d_deg"]
    beta_g = params["beta_g"]
    beta_l = params["beta_l"]
    alpha = params["alpha"]
    k_off = params["k_off"]

    S = x[:K]
    A = x[K:2*K]
    p = x[2*K:2*K + N]

    # Activation dynamics
    dS = k_act * (1.0 - S) - k_deact * S

    # Abundance dynamics (TF input folded into s_prod)
    dA = s_prod - d_deg * A

    # Crosstalk terms (global + local, then saturate)
    coup_g = beta_g * (Cg @ p)
    coup_l = beta_l * (Cl @ p)
    coup_raw = coup_g + coup_l
    coup = sat(coup_raw)

    # Effective kinase on-rate per site via kinase–site matrix
    # k_on_eff_i = (K_site_kin @ alpha)_i
    k_on_eff = K_site_kin @ alpha  # shape (N,)

    # Map protein indices for each site
    S_local = S[site_prot_idx]
    A_local = A[site_prot_idx]

    v_on = (k_on_eff * S_local * A_local + coup) * (1.0 - p)
    v_off = k_off * p
    dp = v_on - v_off

    dx = np.empty_like(x)
    dx[:K] = dS
    dx[K:2*K] = dA
    dx[2*K:2*K + N] = dp
    return dx


@njit(cache=True)
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
        log_k_off_i (N) ]

    Returns
    -------
    k_act, k_deact, s_prod, d_deg,
    beta_g, beta_l,
    alpha, k_off
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

    log_k_off = theta[idx0:idx0 + N]

    k_act = np.exp(log_k_act)
    k_deact = np.exp(log_k_deact)
    s_prod = np.exp(log_s_prod)
    d_deg = np.exp(log_d_deg)
    beta_g = np.exp(log_beta_g)
    beta_l = np.exp(log_beta_l)
    alpha = np.exp(log_alpha)
    k_off = np.exp(log_k_off)

    return (k_act, k_deact,
            s_prod, d_deg,
            beta_g, beta_l,
            alpha, k_off)


def simulate_p(t, Cg, Cl, P_data, theta,
               site_prot_idx, K, K_site_kin):
    """
    Simulate p(t) given theta.

    theta layout:
      [log_k_act_k (K),
       log_k_deact_k (K),
       log_s_prod_k (K),
       log_d_deg_k (K),
       log_beta_g,
       log_beta_l,
       log_alpha_m (M),
       log_k_off_i (N)]

    Returns
    -------
    P_sim  : (N,T) simulated p
    params : dict of decoded (non-log) parameters
    """
    N, T = P_data.shape
    M = K_site_kin.shape[1]

    (k_act, k_deact,
     s_prod, d_deg,
     beta_g, beta_l,
     alpha, k_off) = decode_theta_core(theta, K, M, N)

    params = {
        "k_act": k_act,
        "k_deact": k_deact,
        "s_prod": s_prod,
        "d_deg": d_deg,
        "beta_g": float(beta_g),
        "beta_l": float(beta_l),
        "alpha": alpha,
        "k_off": k_off,
    }

    # Initial conditions:
    #   S_k(0) = 0
    #   A_k(0) = 1
    #   p_i(0) = data at t0
    x0 = np.zeros(2 * K + N, dtype=float)
    x0[K:2*K] = 1.0     # initial abundance
    x0[2*K:] = P_data[:, 0]

    X = odeint(
        network_rhs,
        x0,
        t,
        args=(params, Cg, Cl, site_prot_idx, K, K_site_kin)
    )
    P_sim = X[:, 2*K:].T
    return P_sim, params


# ------------------------------------------------------------
#  FITTING
# ------------------------------------------------------------

def objective_slsqp(theta, t, Cg, Cl, P_data,
                    site_prot_idx, K, K_site_kin,
                    reg_lambda=1e-3):
    """
    Scalar objective for SLSQP:
      J(theta) = ||P_sim - P_data||^2 + reg_lambda * ||theta||^2
    """
    P_sim, _ = simulate_p(
        t, Cg, Cl, P_data, theta, site_prot_idx, K, K_site_kin
    )
    return np.sum((P_data - P_sim) ** 2) + reg_lambda * np.sum(theta ** 2)


def fit_network(t, Cg, Cl, P_data,
                site_prot_idx, K, K_site_kin):
    """
    Fit parameters with SLSQP:

      Per-protein (k = 0..K-1):
        k_act_k, k_deact_k, s_prod_k, d_deg_k

      Global:
        beta_g, beta_l

      Per-kinase (m = 0..M-1):
        alpha_m

      Per-site (i = 0..N-1):
        k_off_i
    """
    N, T = P_data.shape
    M = K_site_kin.shape[1]

    # Initial guesses
    k_act0 = np.full(K, 1.0)
    k_deact0 = np.full(K, 0.01)
    s_prod0 = np.full(K, 0.1)
    d_deg0 = np.full(K, 0.01)
    beta_g0 = 0.05
    beta_l0 = 0.05
    alpha0 = np.full(M, 0.1)
    k_off0 = np.full(N, 0.05)

    theta0 = np.concatenate([
        np.log(k_act0),
        np.log(k_deact0),
        np.log(s_prod0),
        np.log(d_deg0),
        np.log([beta_g0]),
        np.log([beta_l0]),
        np.log(alpha0),
        np.log(k_off0),
    ])

    lower = np.log(1e-4) * np.ones_like(theta0)
    upper = np.log(10.0) * np.ones_like(theta0)
    bounds = list(zip(lower, upper))

    res = minimize(
        objective_slsqp,
        theta0,
        args=(t, Cg, Cl, P_data, site_prot_idx, K, K_site_kin),
        method="SLSQP",
        bounds=bounds,
        options={
            "disp": True,
            "maxiter": 200,
        },
    )

    theta_opt = res.x
    P_sim, params_decoded = simulate_p(
        t, Cg, Cl, P_data, theta_opt, site_prot_idx, K, K_site_kin
    )

    return theta_opt, params_decoded, P_sim, res


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
    """
    Scatter plot: observed vs simulated FC.
    """
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
                    "and kinase–site-driven phosphorylation."
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
             "kinase–site matrix. If omitted, identity matrix is used "
             "(one pseudo-kinase per site)."
    )
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # 1) Load full data
    (sites, proteins, site_prot_idx,
     positions, t, Y) = load_site_data(args.data)
    print(f"[*] Loaded {len(sites)} sites from {len(proteins)} proteins, "
          f"{Y.shape[1]} timepoints.")

    # 1b) Optionally restrict to sites present in crosstalk TSV
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

        # apply mask to site-level quantities
        sites = [s for s, keep in zip(sites, mask) if keep]
        positions = positions[mask]
        Y = Y[mask, :]

        # recompute proteins and site_prot_idx for the filtered set
        proteins_used = sorted({s.split("_", 1)[0] for s in sites})
        prot_index = {p: k for k, p in enumerate(proteins_used)}
        site_prot_idx = np.array(
            [prot_index[s.split("_", 1)[0]] for s in sites],
            dtype=int
        )
        proteins = proteins_used

        print(f"[*] Restricted sites via crosstalk TSV: "
              f"{n_before} -> {n_after} sites, {len(proteins)} proteins.")

    # 2) Scale FC -> p in [0,1]
    P_scaled, baselines, amplitudes = scale_fc_to_unit_interval(Y)
    print("[*] Scaled FC data to p in [0,1].")

    # 3) Build C matrices from DBs (on the filtered site set)
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
    else:
        # Fallback: identity → one pseudo-kinase per site
        N = len(sites)
        K_site_kin = np.eye(N, dtype=float)
        kinases = [f"Kinase_{i}" for i in range(N)]
        print("[*] No --kinase-tsv supplied. Using identity matrix: "
              "one pseudo-kinase per site.")

    # 5) Fit network
    K = len(proteins)
    theta_opt, params_decoded, P_sim, result = fit_network(
        t, Cg, Cl, P_scaled, site_prot_idx, K, K_site_kin
    )

    print("[*] Optimization finished.")
    print(f"    Success: {result.success}, message: {result.message}")
    print(f"    Final cost (J): {result.fun:.4g}")

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
    }
    params_path = os.path.join(args.outdir, "fitted_params.npz")
    np.savez(params_path, **out_params)
    print(f"[*] Saved parameters to {params_path}")

    # 7) Reconstruct FC fits and save a TSV
    N, T = Y.shape
    Y_sim = np.zeros_like(Y)
    for i in range(N):
        Y_sim[i] = baselines[i] + amplitudes[i] * P_sim[i]

    df_out = pd.DataFrame({
        "Protein": [s.split("_", 1)[0] for s in sites],
        "Residue": [s.split("_", 1)[1] for s in sites],
    })
    for j in range(T):
        df_out[f"data_t{j}"] = Y[:, j]
        df_out[f"sim_t{j}"] = Y_sim[:, j]

    tsv_path = os.path.join(args.outdir, "fit_timeseries.tsv")
    df_out.to_csv(tsv_path, sep="\t", index=False)
    print(f"[*] Saved simulated vs data time series to {tsv_path}")

    # 8) Goodness of fit plot
    gof_path = os.path.join(args.outdir, "goodness_of_fit.png")
    plot_goodness_of_fit(Y, Y_sim, gof_path)
    print("[*] Done.")


if __name__ == "__main__":
    main()
