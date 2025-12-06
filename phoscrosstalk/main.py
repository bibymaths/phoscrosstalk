#!/usr/bin/env python3
"""
main.py
Entry point for the Global Phospho-Network Model orchestration.
"""
import argparse
import os
import multiprocessing
import numpy as np
import pandas as pd

from pymoo.algorithms.moo.unsga3 import UNSGA3
from pymoo.optimize import minimize
from pymoo.termination.default import DefaultMultiObjectiveTermination
from pymoo.parallelization import StarmapParallelization
from pymoo.util.ref_dirs import get_reference_directions

import analysis
import data_loader
from config import ModelDims
from weighting import build_weight_matrices
from optimization import NetworkOptimizationProblem, create_bounds


def main():
    parser = argparse.ArgumentParser(description="Global Phospho-Network Model Fitting")
    parser.add_argument("--data", required=True)
    parser.add_argument("--ptm-intra", required=True)
    parser.add_argument("--ptm-inter", required=True)
    parser.add_argument("--outdir", default="network_fit")
    parser.add_argument("--length-scale", type=float, default=50.0)
    parser.add_argument("--crosstalk-tsv")
    parser.add_argument("--kinase-tsv")
    parser.add_argument("--kea-ks-table")
    parser.add_argument("--unified-graph-pkl")
    parser.add_argument("--lambda-net", type=float, default=0.0001)
    parser.add_argument("--reg-lambda", type=float, default=0.0001)
    parser.add_argument("--gen", type=int, default=500)
    parser.add_argument("--pop-size", type=int, default=200)
    parser.add_argument("--cores", type=int, default=os.cpu_count())
    parser.add_argument("--scale-mode", choices=["minmax", "none", "log-minmax"], default="minmax")
    parser.add_argument("--mechanism", choices=["dist", "seq", "rand"], default="dist")
    parser.add_argument("--weight-scheme",
                        choices=["uniform", "early_emphasis", "early_emphasis_moderate", "flat_no_noise"],
                        default="uniform")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    print(f"[*] Output directory: {args.outdir}")

    # 1. Load Data
    (sites, proteins, site_prot_idx, positions, t, Y, A_data, A_proteins) = data_loader.load_site_data(args.data)

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

    # 2. Scaling
    P_scaled, baselines, amplitudes = data_loader.apply_scaling(Y, mode=args.scale_mode)
    P_scaled = np.nan_to_num(P_scaled, nan=0.0, posinf=0.0, neginf=0.0)

    if A_data is not None and len(A_data) > 0:
        prot_map = {p: i for i, p in enumerate(proteins)}
        mask_A = [p in prot_map for p in A_proteins]
        A_data = A_data[mask_A]
        A_proteins = A_proteins[mask_A]
        prot_idx_for_A = np.array([prot_map[p] for p in A_proteins], dtype=int)
        A_scaled, A_bases, A_amps = data_loader.apply_scaling(A_data, mode=args.scale_mode)
        A_scaled = np.nan_to_num(A_scaled, nan=0.0)
    else:
        A_scaled = np.zeros((0, P_scaled.shape[1]))
        prot_idx_for_A = np.array([], dtype=int)
        A_bases, A_amps = np.array([]), np.array([])

    # 3. Weights
    W_data, W_data_prot = build_weight_matrices(
        t=t,
        Y=Y,
        A_data=A_data,
        scheme=args.weight_scheme
    )

    # 4. Matrices & Graph
    Cg, Cl = data_loader.build_C_matrices_from_db(args.ptm_intra, args.ptm_inter, sites, site_prot_idx, positions,
                                                  proteins, args.length_scale)
    Cg, Cl = data_loader.row_normalize(Cg), data_loader.row_normalize(Cl)

    if args.kinase_tsv:
        K_site_kin, kinases = data_loader.load_kinase_site_matrix(args.kinase_tsv, sites)
    elif args.kea_ks_table:
        K_site_kin, kinases = data_loader.build_kinase_site_from_kea(args.kea_ks_table, sites)
    else:
        K_site_kin = np.eye(len(sites));
        kinases = [f"K_{i}" for i in range(len(sites))]

    R = np.ascontiguousarray(K_site_kin.T)
    L_alpha = np.zeros((len(kinases), len(kinases)))
    if args.unified_graph_pkl and args.lambda_net > 0:
        L_alpha = data_loader.build_alpha_laplacian_from_unified_graph(args.unified_graph_pkl, kinases)

    # 5. Mappings & Masks
    prot_map_all = {p: i for i, p in enumerate(proteins)}
    kin_to_prot_idx = np.array([prot_map_all.get(k, -1) for k in kinases], dtype=int)

    receptor_names = {"EGFR", "ERBB2", "EPHA2", "MET"}
    receptor_kin_names = {"EGFR", "EPHA2", "ERBB4", "INSR", "RET"}
    receptor_mask_prot = np.array([1 if p in receptor_names else 0 for p in proteins], dtype=int)
    receptor_mask_kin = np.array([1 if k in receptor_kin_names else 0 for k in kinases], dtype=int)

    # 6. Global Setup & Bounds
    ModelDims.set_dims(len(proteins), len(kinases), len(sites))
    print(f"[*] K={ModelDims.K}, M={ModelDims.M}, N={ModelDims.N}")

    xl, xu, dim = create_bounds(ModelDims.K, ModelDims.M, ModelDims.N)

    # 7. Optimization
    print(f"[*] Initializing Pool ({args.cores} cores)...")
    pool = multiprocessing.Pool(args.cores)
    runner = StarmapParallelization(pool.starmap)

    problem = NetworkOptimizationProblem(
        t, P_scaled, Cg, Cl, site_prot_idx, K_site_kin, R,
        A_scaled, prot_idx_for_A, W_data, W_data_prot, L_alpha, kin_to_prot_idx,
        args.lambda_net, args.reg_lambda, receptor_mask_prot, receptor_mask_kin,
        args.mechanism, xl, xu, elementwise_runner=runner
    )

    algorithm = UNSGA3(pop_size=args.pop_size, ref_dirs=get_reference_directions("das-dennis", 3, n_partitions=12))
    termination = DefaultMultiObjectiveTermination(xtol=1e-8, cvtol=1e-6, ftol=0.0025, period=30, n_max_gen=args.gen)

    print("[*] Starting Optimization...")
    res = minimize(problem, algorithm, termination, seed=1, verbose=True)
    pool.close();
    pool.join()

    # 8. Analysis & Saving
    F, X = res.F, res.X
    f1, f2, f3 = F[:, 0], F[:, 1], F[:, 2]
    # Normalize for selection
    eps = 1e-12
    J = np.sqrt(
        ((f1 - f1.min()) / (f1.max() - f1.min() + eps)) ** 2 + ((f2 - f2.min()) / (f2.max() - f2.min() + eps)) ** 2) + (
                    (f3 - f3.min()) / (f3.max() - f3.min() + eps))
    best_idx = np.argmin(J)

    analysis.save_pareto_results(args.outdir, F, X, f1, f2, f3, J, F[best_idx])
    analysis.plot_pareto_diagnostics(args.outdir, F, F[best_idx], f1, f2, f3, X)

    analysis.save_fitted_simulation(
        args.outdir, X[best_idx], t, sites, proteins, P_scaled, A_scaled,
        prot_idx_for_A, baselines, amplitudes, A_data, A_bases, A_amps, args.mechanism,
        Cg, Cl, site_prot_idx, K_site_kin, R, L_alpha, kin_to_prot_idx,
        receptor_mask_prot, receptor_mask_kin
    )

    analysis.plot_fitted_simulation(args.outdir)
    analysis.print_parameter_summary(args.outdir, X[best_idx], sites, proteins, kinases)

    print("[*] Done.")


if __name__ == "__main__":
    main()