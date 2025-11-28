#!/usr/bin/env python3
"""
Global phospho-network model with protein-specific inputs and
global/local coupling from PTM SQLite DBs.

States:
  S_k(t): protein-specific inputs (k = 0..K-1)
  p_i(t): phosphorylation at site i (i = 0..N-1)

Dynamics:
  dS_k/dt = k_act_k * (1 - S_k) - k_deact_k * S_k

  dp_i/dt = (k_on_i * S_{prot(i)} + [beta_g Cg p + beta_l Cl p]_i)
            * (1 - p_i) - k_off_i * p_i

Inputs:
  --data       time series TSV with Protein, Residue, v1..v14
  --ptm-intra  ptm_intra.db   (intra-protein PTM pairs)
  --ptm-inter  ptm_inter.db   (inter-protein PTM pairs)
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
from scipy.optimize import least_squares, minimize
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
    Load time-series from a file that looks like input1.csv:

        GeneID, Psite, x1..x14

    or the older format:

        Protein, Residue, v1..v14

    Returns
    -------
    sites         : list[str]  e.g. 'ABL2_S620'
    proteins      : list[str]  unique protein names (for global model)
    site_prot_idx : np.array(N,) int (index into proteins for each site)
    positions     : np.array(N,) int or NaN (residue positions)
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
    """
    N = len(sites)
    idx = {s: i for i, s in enumerate(sites)}

    Cg = np.zeros((N, N), dtype=float)

    # --- INTRA ---
    conn_i = sqlite3.connect(ptm_intra_path)
    cur_i = conn_i.cursor()
    for protein, res1, r1, res2, r2 in cur_i.execute(
            "SELECT protein, residue1, score1, residue2, score2 FROM intra_pairs"):
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
            "SELECT protein1, residue1, score1, protein2, residue2, score2 FROM inter_pairs"):
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
    N = len(sites)
    Cl = np.zeros((N, N), dtype=float)
    L = float(length_scale)
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            if site_prot_idx[i] != site_prot_idx[j]:
                continue
            # NEW: skip TF-level / unknown positions
            if not (np.isfinite(positions[i]) and np.isfinite(positions[j])):
                continue
            d = abs(positions[i] - positions[j])
            Cl[i, j] = np.exp(-d / L)

    # Convert to CSR sparse matrices for big speedup when N grows
    Cg = csr_matrix(Cg)
    Cl = csr_matrix(Cl)
    return Cg, Cl

def row_normalize(C):
    # convert to dense row sums, shape (N,1)
    row_sums = np.array(C.sum(axis=1)).reshape(-1, 1)

    # avoid division by zero
    row_sums[row_sums == 0.0] = 1.0

    # elementwise row-normalisation via broadcasting
    C_norm = C.multiply(1.0 / row_sums)

    return C_norm



# ------------------------------------------------------------
#  ODE MODEL: PROTEIN-SPECIFIC INPUTS
# ------------------------------------------------------------

# @njit(parallel=True, fastmath=True, cache=True)
def network_rhs_core(x, K,
                     k_act, k_deact,
                     beta_g, beta_l,
                     k_on, k_off,
                     Cg, Cl,
                     site_prot_idx):
    """
    Numba core for RHS.

    x: [S_0..S_{K-1}, p_0..p_{N-1}]
    """
    N = Cg.shape[0]

    S = x[:K]
    p = x[K:]

    # dS_k/dt
    dS = k_act * (1.0 - S) - k_deact * S

    # coupling
    coup_g = beta_g * (Cg @ p)
    coup_l = beta_l * (Cl @ p)
    total_coup = coup_g + coup_l

    # protein-specific S
    S_local = S[site_prot_idx]

    v_on = (k_on * S_local + total_coup) * (1.0 - p)
    v_off = k_off * p
    dp = v_on - v_off

    dx = np.empty(K + N)
    dx[:K] = dS
    dx[K:] = dp
    return dx

def network_rhs(x, t, params, Cg, Cl, site_prot_idx):
    """
    Python wrapper around numba-compiled core for use with odeint.
    """
    K = len(params["k_act"])
    return network_rhs_core(
        x,
        K,
        params["k_act"],
        params["k_deact"],
        params["beta_g"],
        params["beta_l"],
        params["k_on"],
        params["k_off"],
        Cg,
        Cl,
        site_prot_idx,
    )

@njit(parallel=True, fastmath=True, cache=True)
def decode_theta_core(theta, K, N):
    """
    Numba core: decode log-parameters from theta.
    Returns:
      k_act, k_deact, beta_g, beta_l, k_on, k_off
    """
    idx0 = 0
    log_k_act = theta[idx0:idx0 + K]
    idx0 += K
    log_k_deact = theta[idx0:idx0 + K]
    idx0 += K
    log_beta_g = theta[idx0]
    idx0 += 1
    log_beta_l = theta[idx0]
    idx0 += 1
    log_k_on = theta[idx0:idx0 + N]
    idx0 += N
    log_k_off = theta[idx0:idx0 + N]

    k_act   = np.exp(log_k_act)
    k_deact = np.exp(log_k_deact)
    beta_g  = np.exp(log_beta_g)
    beta_l  = np.exp(log_beta_l)
    k_on    = np.exp(log_k_on)
    k_off   = np.exp(log_k_off)

    return k_act, k_deact, beta_g, beta_l, k_on, k_off

def simulate_p(t, Cg, Cl, P_data, theta, site_prot_idx, K):
    """
    theta =
      [log_k_act_k (K),
       log_k_deact_k (K),
       log_beta_g,
       log_beta_l,
       log_k_on_i (N),
       log_k_off_i (N)]

    Returns
    -------
    P_sim : (N,T)
    params : dict of decoded (non-log) parameters
    """
    N, T = P_data.shape

    # decode in numba core
    (k_act, k_deact,
     beta_g, beta_l,
     k_on, k_off) = decode_theta_core(theta, K, N)

    params = {
        "k_act":   k_act,
        "k_deact": k_deact,
        "beta_g":  float(beta_g),
        "beta_l":  float(beta_l),
        "k_on":    k_on,
        "k_off":   k_off,
    }

    # Initial conditions: S_k(0) = 0, p_i(0) = data at t0
    x0 = np.zeros(K + N, dtype=float)
    x0[K:] = P_data[:, 0]

    X = odeint(network_rhs, x0, t, args=(params, Cg, Cl, site_prot_idx))
    P_sim = X[:, K:].T
    return P_sim, params

# ------------------------------------------------------------
#  FITTING
# ------------------------------------------------------------

def objective_slsqp(theta, t, Cg, Cl, P_data, site_prot_idx, K, reg_lambda=1e-3):
    """
    Scalar objective for SLSQP:
      J(theta) = 0.5 * ||P_sim - P_data||^2 + 0.5 * reg_lambda * ||theta||^2
    """
    P_sim, _ = simulate_p(t, Cg, Cl, P_data, theta, site_prot_idx, K)
    return np.sum((P_data-P_sim) ** 2) + reg_lambda * np.sum(theta ** 2)

def fit_network(t, Cg, Cl, P_data, site_prot_idx, K):
    """
    Fit all parameters with SLSQP:
      k_act_k, k_deact_k (per protein),
      beta_g, beta_l,
      k_on_i, k_off_i (per site)
    """
    N, T = P_data.shape

    # Initial guesses (same as before)
    k_act0 = np.full(K, 1.0)
    k_deact0 = np.full(K, 0.01)
    beta_g0 = 0.05
    beta_l0 = 0.05
    k_on0 = np.full(N, 0.1)
    k_off0 = np.full(N, 0.05)

    theta0 = np.concatenate([
        np.log(k_act0),
        np.log(k_deact0),
        np.log([beta_g0]),
        np.log([beta_l0]),
        np.log(k_on0),
        np.log(k_off0),
    ])

    lower = np.log(1e-4) * np.ones_like(theta0)
    upper = np.log(10.0) * np.ones_like(theta0)

    bounds = list(zip(lower, upper))

    # Simple callback to see progress every few iterations
    iter_counter = {"k": 0}

    def callback(theta):
        iter_counter["k"] += 1
        if iter_counter["k"] % 5 == 0:
            J = objective_slsqp(theta, t, Cg, Cl, P_data, site_prot_idx, K)
            print(f"[SLSQP] iter={iter_counter['k']}, J={J:.4g}")

    res = minimize(
        objective_slsqp,
        theta0,
        args=(t, Cg, Cl, P_data, site_prot_idx, K),
        method="COBYLA",
        bounds=bounds,
        callback=callback,
        options={
            "disp": True,      # print SLSQP messages
            # "maxiter": 50,    # adjust if needed
        },
    )

    theta_opt = res.x
    P_sim, params_decoded = simulate_p(
        t, Cg, Cl, P_data, theta_opt, site_prot_idx, K
    )

    return theta_opt, params_decoded, P_sim, res

def load_allowed_sites_from_crosstalk(path):
    """
    Read crosstalk_predictions.tsv and return a set of site IDs
    in the same format as `sites` from load_site_data, i.e. 'PROT_RES'.

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

    # Site1 and Site2 are residues like 'S18', 'Y70', etc.
    for col in ["Site1", "Site2"]:
        prots = df["Protein"].astype(str).values
        sites = df[col].astype(str).values
        for p, s in zip(prots, sites):
            if pd.isna(p) or pd.isna(s):
                continue
            site_ids.add(f"{p}_{s}")

    return site_ids

# Function to plot goodness of fit as scatter plot of observed vs simulated
def plot_goodness_of_fit(Y_data, Y_sim, outpath):

    plt.figure(figsize=(6, 6))
    plt.scatter(Y_data.flatten(), Y_sim.flatten(), alpha=0.5)
    plt.plot([Y_data.min(), Y_data.max()], [Y_data.min(), Y_data.max()], 'r--')
    plt.xlabel('Observed FC')
    plt.ylabel('Simulated FC')
    plt.title('Goodness of Fit: Observed vs Simulated FC')
    plt.savefig(outpath)
    plt.close()
    print(f"[*] Saved goodness of fit plot to {outpath}")


# ------------------------------------------------------------
#  MAIN
# ------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Fit global phospho network with protein-specific inputs."
    )
    parser.add_argument("--data", required=True,
                        help="CSV/TSV with either (Protein, Residue, v1..v14) "
                             "or (GeneID, Psite, x1..x14)")
    parser.add_argument("--ptm-intra", required=True,
                        help="SQLite DB with intra_pairs (ptm_intra.db)")
    parser.add_argument("--ptm-inter", required=True,
                        help="SQLite DB with inter_pairs (ptm_inter.db)")
    parser.add_argument("--outdir", default="network_fit",
                        help="Output directory.")
    parser.add_argument("--length-scale", type=float, default=50.0,
                        help="Length scale L for local coupling exp(-|i-j|/L).")
    parser.add_argument("--crosstalk-tsv",
                        help="TSV with columns Protein, Site1, Site2 to restrict which sites are fitted.")
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

    # 2) Scale FC
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

    # 4) Fit network
    K = len(proteins)
    theta_opt, params_decoded, P_sim, result = fit_network(
        t, Cg, Cl, P_scaled, site_prot_idx, K
    )

    print("[*] Optimization finished.")
    print(f"    Success: {result.success}, message: {result.message}")
    print(f"    Final cost (J): {result.fun:.4g}")

    # 4) Save parameters & mappings
    out_params = {
        "proteins": np.array(proteins, dtype=object),
        "sites": np.array(sites, dtype=object),
        "site_prot_idx": site_prot_idx,
        "positions": positions,
        "k_act": params_decoded["k_act"],
        "k_deact": params_decoded["k_deact"],
        "beta_g": params_decoded["beta_g"],
        "beta_l": params_decoded["beta_l"],
        "k_on": params_decoded["k_on"],
        "k_off": params_decoded["k_off"],
        "baselines": baselines,
        "amplitudes": amplitudes,
    }
    np.savez(os.path.join(args.outdir, "fitted_params.npz"), **out_params)
    print(f"[*] Saved parameters to {os.path.join(args.outdir, 'fitted_params.npz')}")

    # 5) Optionally: reconstruct FC fits and save a TSV
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

    df_out.to_csv(os.path.join(args.outdir, "fit_timeseries.tsv"),
                  sep="\t", index=False)
    print(f"[*] Saved simulated vs data time series to fit_timeseries.tsv")

    plot_goodness_of_fit(Y, Y_sim, os.path.join(args.outdir, "goodness_of_fit.png"))

    print("[*] Saved goodness of fit plot.")
    print("[*] Done.")

if __name__ == "__main__":
    main()
