#!/usr/bin/env python3
"""
Global phospho-network model with protein-specific inputs and
global/local coupling.

Refactored to support:
1. Pymoo Differential Evolution (Parallelized)
2. Scipy Minimize (SLSQP)
"""

import numpy as np
import pandas as pd
import sqlite3
import argparse
import os
import re
import time
import multiprocessing
from functools import partial

from matplotlib import pyplot as plt
from pymoo.algorithms.soo.nonconvex.nrbo import NRBO
from pymoo.algorithms.soo.nonconvex.pattern import PatternSearch
from scipy.integrate import odeint
from scipy.optimize import least_squares, minimize
from scipy.sparse import csr_matrix

# Pymoo imports

from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.optimize import minimize as pymoo_minimize
from pymoo.termination import get_termination
from pymoo.termination.default import DefaultSingleObjectiveTermination

# ------------------------------------------------------------
#  TIMEPOINTS & DATA
# ------------------------------------------------------------

DEFAULT_TIMEPOINTS = np.array(
    [0.0, 0.5, 0.75, 1.0, 2.0, 4.0, 8.0, 16.0,
     30.0, 60.0, 120.0, 240.0, 480.0, 960.0]
)


def load_site_data(path, timepoints=DEFAULT_TIMEPOINTS):
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

    proteins_raw = df[prot_col].astype(str).tolist()

    residues_raw = []
    positions = []

    if "Residue" in df.columns:
        for r in df["Residue"].astype(str):
            residues_raw.append(r)
            m = re.match(r"[A-Z]([0-9]+)", r)
            positions.append(int(m.group(1)) if m else np.nan)
    elif "Psite" in df.columns:
        for psite in df["Psite"]:
            if pd.isna(psite):
                residues_raw.append("TF")
                positions.append(np.nan)
            else:
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
    sites = [f"{p}_{r}" for p, r in zip(proteins_raw, residues_raw)]
    proteins = sorted(set(proteins_raw))
    prot_index = {p: k for k, p in enumerate(proteins)}
    site_prot_idx = np.array([prot_index[p] for p in proteins_raw], dtype=int)
    Y = df[value_cols].values.astype(float)
    t = np.array(timepoints, dtype=float)

    return sites, proteins, site_prot_idx, positions, t, Y

def load_allowed_sites_from_crosstalk(path):
    """
    Read crosstalk_predictions.tsv and return a set of site IDs
    in the same format as `sites` from load_site_data, i.e. 'PROT_RES'.

    Assumes columns: Protein, Site1, Site2 with Site1/Site2 like 'S18', 'Y70', etc.
    """
    df = pd.read_csv(path, sep="\t")

    required = {"Protein", "Site1", "Site2"}
    if not required.issubset(df.columns):
        raise ValueError(
            f"{path} must contain columns {required}, "
            f"but has {set(df.columns)}"
        )

    allowed = set()
    proteins = df["Protein"].astype(str).values

    for col in ["Site1", "Site2"]:
        residues = df[col].astype(str).values
        for p, r in zip(proteins, residues):
            if pd.isna(p) or pd.isna(r):
                continue
            # 'ABL1' + 'S18' -> 'ABL1_S18'
            allowed.add(f"{p}_{r}")

    return allowed

def scale_fc_to_unit_interval(Y):
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
#  C MATRICES
# ------------------------------------------------------------

def build_C_matrices_from_db(ptm_intra_path, ptm_inter_path,
                             sites, site_prot_idx, positions,
                             proteins, length_scale=50.0):
    N = len(sites)
    idx = {s: i for i, s in enumerate(sites)}
    Cg = np.zeros((N, N), dtype=float)

    # INTRA
    conn_i = sqlite3.connect(ptm_intra_path)
    cur_i = conn_i.cursor()
    # Using try-except to handle potential missing tables if DB is empty/malformed
    try:
        query = "SELECT protein, residue1, score1, residue2, score2 FROM intra_pairs"
        for protein, res1, r1, res2, r2 in cur_i.execute(query):
            s1, s2 = f"{protein}_{res1}", f"{protein}_{res2}"
            if s1 in idx and s2 in idx:
                i, j = idx[s1], idx[s2]
                score = 0.8 * (r1 + r2) / 200.0
                if score > Cg[i, j]:
                    Cg[i, j] = Cg[j, i] = score
    except Exception as e:
        print(f"[!] Warning reading intra_pairs: {e}")
    conn_i.close()

    # INTER
    conn_e = sqlite3.connect(ptm_inter_path)
    cur_e = conn_e.cursor()
    try:
        query = "SELECT protein1, residue1, score1, protein2, residue2, score2 FROM inter_pairs"
        for p1, res1, r1, p2, res2, r2 in cur_e.execute(query):
            s1, s2 = f"{p1}_{res1}", f"{p2}_{res2}"
            if s1 in idx and s2 in idx:
                i, j = idx[s1], idx[s2]
                score = 0.8 * (r1 + r2) / 200.0
                if score > Cg[i, j]:
                    Cg[i, j] = Cg[j, i] = score
    except Exception as e:
        print(f"[!] Warning reading inter_pairs: {e}")
    conn_e.close()

    # LOCAL
    Cl = np.zeros((N, N), dtype=float)
    L = float(length_scale)
    # Vectorized calculation for Cl is possible but loop is safe for now
    for i in range(N):
        for j in range(N):
            if i == j: continue
            if site_prot_idx[i] != site_prot_idx[j]: continue
            if not (np.isfinite(positions[i]) and np.isfinite(positions[j])): continue
            d = abs(positions[i] - positions[j])
            Cl[i, j] = np.exp(-d / L)

    return csr_matrix(Cg), csr_matrix(Cl)


def row_normalize(C):
    row_sums = np.array(C.sum(axis=1)).reshape(-1, 1)
    row_sums[row_sums == 0.0] = 1.0
    return C.multiply(1.0 / row_sums)


# ------------------------------------------------------------
#  ODE MODEL
# ------------------------------------------------------------

def network_rhs(x, t, params, Cg, Cl, site_prot_idx):
    K = len(params["k_act"])
    S = x[:K]
    p = x[K:]

    dS = params["k_act"] * (1.0 - S) - params["k_deact"] * S

    # Sparse matrix multiplication
    total_coup = params["beta_g"] * (Cg.dot(p)) + params["beta_l"] * (Cl.dot(p))

    S_local = S[site_prot_idx]
    v_on = (params["k_on"] * S_local + total_coup) * (1.0 - p)
    v_off = params["k_off"] * p
    dp = v_on - v_off

    return np.concatenate([dS, dp])


def unpack_theta(theta, K, N):
    """Unpacks flat log-parameter vector into dictionary of real values."""
    idx0 = 0
    p = {}
    p["k_act"] = np.exp(theta[idx0:idx0 + K])
    idx0 += K
    p["k_deact"] = np.exp(theta[idx0:idx0 + K])
    idx0 += K
    p["beta_g"] = float(np.exp(theta[idx0]))
    idx0 += 1
    p["beta_l"] = float(np.exp(theta[idx0]))
    idx0 += 1
    p["k_on"] = np.exp(theta[idx0:idx0 + N])
    idx0 += N
    p["k_off"] = np.exp(theta[idx0:idx0 + N])
    return p


def simulate_p(t, Cg, Cl, P_data, theta, site_prot_idx, K):
    N = P_data.shape[0]
    params = unpack_theta(theta, K, N)

    # Init: S=0, p=data[0]
    x0 = np.zeros(K + N, dtype=float)
    x0[K:] = P_data[:, 0]

    # Solving ODE
    X = odeint(network_rhs, x0, t, args=(params, Cg, Cl, site_prot_idx))
    P_sim = X[:, K:].T
    return P_sim


def obj_func_scalar(theta, t, Cg, Cl, P_data, site_prot_idx, K, reg_lambda=1e-3):
    """SSE + L2 Regularization for Minimizers (Scalar output)."""
    P_sim = simulate_p(t, Cg, Cl, P_data, theta, site_prot_idx, K)

    # Sum of Squared Errors
    sse = np.sum((P_data - P_sim) ** 2)

    # L2 Regularization on log-params (keeping them smallish)
    reg = reg_lambda * np.sum(theta ** 2)

    return sse + reg


# ------------------------------------------------------------
#  PYMOO PROBLEM DEFINITION
# ------------------------------------------------------------

class NetworkProblem(ElementwiseProblem):
    def __init__(self, t, Cg, Cl, P_data, site_prot_idx, K, reg_lambda, bounds):
        # n_var calculation
        # K (act) + K (deact) + 1 (bg) + 1 (bl) + N (on) + N (off)
        N = P_data.shape[0]
        n_var = 2 * K + 2 + 2 * N

        self.t = t
        self.Cg = Cg
        self.Cl = Cl
        self.P_data = P_data
        self.site_prot_idx = site_prot_idx
        self.K = K
        self.reg_lambda = reg_lambda

        super().__init__(n_var=n_var, n_obj=1, n_ieq_constr=0, xl=bounds[0], xu=bounds[1])

    def _evaluate(self, x, out, *args, **kwargs):
        # x is a single individual (vector of params)
        cost = obj_func_scalar(x, self.t, self.Cg, self.Cl,
                               self.P_data, self.site_prot_idx,
                               self.K, self.reg_lambda)
        out["F"] = cost


# ------------------------------------------------------------
#  MAIN DRIVERS
# ------------------------------------------------------------

def run_pymoo(t, Cg, Cl, P_data, site_prot_idx, K, bounds, n_cores=4, pop_size=50, n_gen=100):

    print(f"[*] initializing Pymoo DE with {n_cores} cores, pop={pop_size}, gen={n_gen}...")

    # Initialize pool
    pool = multiprocessing.Pool(n_cores)

    # Create problem
    problem = NetworkProblem(t, Cg, Cl, P_data, site_prot_idx, K, 1e-3, bounds)

    # Setup DE Algorithm
    # DE strategy: rand/1/bin is standard, can tune 'CR' and 'F' if needed
    # algorithm = DE(
    #     pop_size=pop_size,
    #     variant="DE/rand/1/bin",
    #     CR=0.7,
    #     F=0.5,
    #     dither="vector",
    #     jitter=False
    # )

    # Initialize the algorithm
    algorithm = PatternSearch()

    # Termination Criterion
    termination = DefaultSingleObjectiveTermination(
        xtol=1e-8,
        cvtol=1e-6,
        ftol=1e-6,
        period=20,
        n_max_gen=1000,
        n_max_evals=100000
    )
    # Run Optimization
    # We use starmap parallelization via the Runner
    start_time = time.time()

    res = pymoo_minimize(
        problem,
        algorithm,
        termination,
        seed=1,
        verbose=True,
        runner=pool.starmap,
    )

    pool.close()
    pool.join()

    print(f"\n[*] Pymoo finished in {time.time() - start_time:.1f}s. Best Score: {res.F[0]:.4f}")
    return res.X


def run_slsqp(t, Cg, Cl, P_data, site_prot_idx, K, theta0, bounds):
    print("[*] Running Scipy SLSQP (Single core)...")

    # SLSQP requires list of (min, max) tuples for bounds
    slsqp_bounds = list(zip(bounds[0], bounds[1]))

    res = minimize(
        obj_func_scalar,
        theta0,
        args=(t, Cg, Cl, P_data, site_prot_idx, K),
        method='SLSQP',
        bounds=slsqp_bounds,
        options={'maxiter': 500, 'disp': True}
    )

    print(f"[*] SLSQP finished. Success: {res.success}. Cost: {res.fun:.4f}")
    return res.x


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--ptm-intra", required=True)
    parser.add_argument("--ptm-inter", required=True)
    parser.add_argument("--outdir", default="network_fit")
    parser.add_argument("--length-scale", type=float, default=50.0)

    # NEW: restrict to sites in crosstalk_predictions
    parser.add_argument("--crosstalk-tsv",
                        help="TSV with Protein, Site1, Site2 specifying which sites to include")

    # Optimizer settings
    parser.add_argument("--method", choices=['pymoo', 'slsqp', 'ls'], default='pymoo',
                        help="pymoo (Parallel DE), slsqp (Scipy Minimize), ls (Scipy Least Squares)")
    parser.add_argument("--cores", type=int, default=os.cpu_count(),
                        help="Number of cores for Pymoo")
    parser.add_argument("--pop", type=int, default=50, help="Population size for DE")
    parser.add_argument("--gen", type=int, default=100, help="Generations for DE")

    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # 1) Load data
    (sites, proteins, site_prot_idx, positions, t, Y) = load_site_data(args.data)

    # 1b) Optionally restrict to sites present in crosstalk TSV
    if args.crosstalk_tsv is not None:
        allowed_sites = load_allowed_sites_from_crosstalk(args.crosstalk_tsv)

        # mask over current dataset
        mask = np.array([s in allowed_sites for s in sites], dtype=bool)
        n_before = len(sites)
        n_after = mask.sum()

        if n_after == 0:
            raise ValueError(
                "No overlap between time-series data and crosstalk sites. "
                "Check Protein / Site formatting (e.g. 'ABL1_S18')."
            )

        # filter site-level arrays
        sites = [s for s, keep in zip(sites, mask) if keep]
        positions = positions[mask]
        Y = Y[mask, :]

        # recompute proteins and site_prot_idx from the filtered sites
        proteins_used = sorted({s.split("_", 1)[0] for s in sites})
        prot_index = {p: k for k, p in enumerate(proteins_used)}
        site_prot_idx = np.array(
            [prot_index[s.split("_", 1)[0]] for s in sites],
            dtype=int
        )
        proteins = proteins_used

        print(f"[*] Restricted via crosstalk TSV: {n_before} -> {n_after} sites, "
              f"{len(proteins)} proteins.")

    P_scaled, baselines, amplitudes = scale_fc_to_unit_interval(Y)

    # 2) Matrices
    Cg, Cl = build_C_matrices_from_db(args.ptm_intra, args.ptm_inter,
                                      sites, site_prot_idx, positions, proteins,
                                      args.length_scale)
    Cg = row_normalize(Cg)
    Cl = row_normalize(Cl)

    # 3) Setup Parameters
    K = len(proteins)
    N = len(sites)
    n_params = 2 * K + 2 + 2 * N

    # Default initial guess (log space)
    k_act0 = np.full(K, 1.0)
    k_deact0 = np.full(K, 0.01)
    beta_g0 = 0.05
    beta_l0 = 0.05
    k_on0 = np.full(N, 0.1)
    k_off0 = np.full(N, 0.05)

    theta0 = np.concatenate([
        np.log(k_act0), np.log(k_deact0),
        np.log([beta_g0]), np.log([beta_l0]),
        np.log(k_on0), np.log(k_off0)
    ])

    # Bounds (log space: 1e-4 to 10.0)
    lower = np.log(1e-4) * np.ones(n_params)
    upper = np.log(10.0) * np.ones(n_params)
    bounds = (lower, upper)

    # 4) Optimization
    theta_opt = None

    if args.method == 'pymoo':
        theta_opt = run_pymoo(t, Cg, Cl, P_scaled, site_prot_idx, K, bounds,
                              n_cores=args.cores, pop_size=args.pop, n_gen=args.gen)
    elif args.method == 'slsqp':
        theta_opt = run_slsqp(t, Cg, Cl, P_scaled, site_prot_idx, K, theta0, bounds)
    else:
        # Fallback to original least_squares
        print("[*] Running Least Squares (Original)...")

        # Define vector residuals wrapper locally
        def residuals_vec(theta, t, Cg, Cl, P_data, site_prot_idx, K):
            P_sim = simulate_p(t, Cg, Cl, P_data, theta, site_prot_idx, K)
            return (P_sim - P_data).ravel()

        res = least_squares(residuals_vec, theta0, bounds=bounds,
                            args=(t, Cg, Cl, P_scaled, site_prot_idx, K), verbose=2)
        theta_opt = res.x

    # 5) Save results
    params_decoded = unpack_theta(theta_opt, K, N)
    P_final = simulate_p(t, Cg, Cl, P_scaled, theta_opt, site_prot_idx, K)

    out_params = {
        "proteins": np.array(proteins, dtype=object),
        "sites": np.array(sites, dtype=object),
        "site_prot_idx": site_prot_idx,
        "positions": positions,
        **params_decoded,
        "baselines": baselines,
        "amplitudes": amplitudes,
    }
    np.savez(os.path.join(args.outdir, "fitted_params.npz"), **out_params)

    # 6) Reconstruct simulated FC
    Y_sim = np.zeros_like(Y)
    for i in range(N):
        Y_sim[i] = baselines[i] + amplitudes[i] * P_final[i]

    df_out = pd.DataFrame({
        "Protein": [s.split("_", 1)[0] for s in sites],
        "Residue": [s.split("_", 1)[1] for s in sites],
    })
    for j in range(len(t)):
        df_out[f"data_t{j}"] = Y[:, j]
        df_out[f"sim_t{j}"] = Y_sim[:, j]

    df_out.to_csv(os.path.join(args.outdir, "fit_timeseries.tsv"), sep="\t", index=False)

    # Plot
    plt.figure(figsize=(6, 6))
    plt.scatter(Y.flatten(), Y_sim.flatten(), alpha=0.5, s=10)
    mx = max(Y.max(), Y_sim.max())
    mn = min(Y.min(), Y_sim.min())
    plt.plot([mn, mx], [mn, mx], 'r--')
    plt.xlabel('Observed FC')
    plt.ylabel('Simulated FC')
    plt.title(f'Fit: {args.method.upper()}')
    plt.savefig(os.path.join(args.outdir, "goodness_of_fit.png"))
    print(f"[*] Done. Results in {args.outdir}")


if __name__ == "__main__":
    # Windows support for multiprocessing
    multiprocessing.freeze_support()
    main()