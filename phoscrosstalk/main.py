#!/usr/bin/env python3
"""
main.py
Entry point for the Global Phospho-Network Model orchestration.
"""

import argparse
import os
import numpy as np
import pandas as pd

from phoscrosstalk import analysis, steadystate, knockouts, hyperparam
from phoscrosstalk import data_loader
from phoscrosstalk.analysis import _save_preopt_snapshot_txt_csv
from phoscrosstalk.config import ModelDims
from phoscrosstalk.equations import generate_equations_report
from phoscrosstalk.multistarts import run_multi_start_optimization
from phoscrosstalk.post_processing import (
    plot_parameter_clustermap,
    plot_residual_heatmap,
    export_network_for_cytoscape,
    save_run_metadata,
)
from phoscrosstalk.sensitivity import run_global_sensitivity, _generate_param_labels

# from phoscrosstalk.debug_main import _sanity_report_data, _sanity_report_C, _coverage_report_K_site_kin, _sanity_report_R,  _sanity_report_weights, _one_shot_sim_check, sim_summary
from phoscrosstalk.weighting import build_weight_matrices
from phoscrosstalk.optimization import (
    NetworkProblem as NetworkOptimizationProblem,
    create_bounds,
)
from phoscrosstalk.logger import get_logger

logger = get_logger()


def main():
    """
    Command-line interface for running the global phospho-network model fitting
    pipeline. This wrapper exposes all major configuration options for data
    loading, model construction, weighting, and optimization.
    """

    parser = argparse.ArgumentParser(
        prog="phoscrosstalk",
        description=(
            "Fit a global phospho-network ODE model using JAX/Diffrax ODE solving "
            "and Optimistix gradient-based optimisation. Supports multiple "
            "phosphorylation mechanisms, flexible weighting schemes, kinase–substrate "
            "network priors, and optional PTM crosstalk filtering."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ------------------------------------------------------------------
    # INPUT DATA
    # ------------------------------------------------------------------
    parser.add_argument(
        "--data",
        required=True,
        help="CSV file containing time-series phosphorylation data.",
    )
    parser.add_argument(
        "--ptm-intra",
        required=True,
        help="SQLite DB containing intra-protein PTM relationships.",
    )
    parser.add_argument(
        "--ptm-inter",
        required=True,
        help="SQLite DB containing inter-protein PTM relationships.",
    )
    parser.add_argument(
        "--crosstalk-tsv",
        help="Optional TSV listing PTM pairs to keep (crosstalk filtering).",
    )

    # ------------------------------------------------------------------
    # KINASE–SUBSTRATE MAPPING
    # ------------------------------------------------------------------
    parser.add_argument(
        "--kinase-tsv",
        help="TSV file mapping sites → kinases (weight column optional).",
    )
    parser.add_argument(
        "--kea-ks-table",
        help="Alternative KEA/KS mapping table if --kinase-tsv is not provided.",
    )
    parser.add_argument(
        "--unified-graph-pkl",
        help="Pickled networkx graph of kinase–kinase relationships "
        "used to build Laplacian regularizers.",
    )

    # ------------------------------------------------------------------
    # EXTERNAL STIMULI SPECIFIC RECEPTORS & KINASES
    # ------------------------------------------------------------------
    parser.add_argument(
        "--receptors",
        nargs="*",  # Accepts zero or more arguments
        default=[],
        help="List of proteins that act as receptors (receive external u(t)).",
    )

    parser.add_argument(
        "--receptor-kinases",
        nargs="*",
        default=[],
        help="List of kinases that act as receptors.",
    )
    # ------------------------------------------------------------------
    # OUTPUT DIRECTORY
    # ------------------------------------------------------------------
    parser.add_argument(
        "--outdir",
        default="network_fit",
        help="Directory where results, logs, and output files are saved.",
    )

    # ------------------------------------------------------------------
    # MODEL CONFIGURATION
    # ------------------------------------------------------------------
    parser.add_argument(
        "--length-scale",
        type=float,
        default=50.0,
        help="Length-scale for exponential local PTM decay on the protein domain.",
    )
    parser.add_argument(
        "--scale-mode",
        choices=["minmax", "none", "log-minmax"],
        default="minmax",
        help="Method used to normalize/scale the experimental FC values.",
    )
    parser.add_argument(
        "--mechanism",
        choices=["dist", "seq", "rand"],
        default="dist",
        help="Phosphorylation mechanism: distributive, sequential, or random/cooperative.",
    )
    parser.add_argument(
        "--weight-scheme",
        choices=[
            "uniform",
            "early_emphasis",
            "early_emphasis_moderate",
            "flat_no_noise",
        ],
        default="uniform",
        help="Weighting strategy for time-series and site-level importance.",
    )

    # ------------------------------------------------------------------
    # REGULARIZATION
    # ------------------------------------------------------------------
    parser.add_argument(
        "--lambda-net",
        type=float,
        default=0.0001,
        help="Laplacian regularization strength on kinase–kinase α parameters.",
    )
    parser.add_argument(
        "--reg-lambda",
        type=float,
        default=0.0001,
        help="L2 regularization strength for all model parameters.",
    )

    # ------------------------------------------------------------------
    # OPTIMIZATION SETTINGS
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # OPTIMISATION SETTINGS
    #
    # Note: The traditional pymoo flags (--gen, --pop-size, --algorithm) have
    # been superseded by JAX/Optimistix-native options.  We retain the old
    # names as deprecated aliases for one release cycle.  New scripts should
    # use --n-starts and --max-steps to control the number of multi-start
    # initialisations and the maximum optimiser iterations, respectively.

    parser.add_argument(
        "--n-starts",
        type=int,
        default=None,
        help=(
            "Number of multi-start initialisations. Replaces the deprecated "
            "--pop-size argument."
        ),
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help=(
            "Maximum number of optimisation steps per run. Replaces the deprecated "
            "--gen argument."
        ),
    )

    # Deprecated flags kept for backward compatibility.  These are accepted
    # silently and mapped onto the new parameters.  They remain in the help
    # output so that existing scripts do not break, but users are encouraged
    # to migrate to --n-starts/--max-steps and remove these flags entirely.
    parser.add_argument(
        "--gen",
        type=int,
        default=None,
        help=("[DEPRECATED] Max optimiser steps per run.  Use --max-steps instead."),
    )
    parser.add_argument(
        "--pop-size",
        type=int,
        default=None,
        help=("[DEPRECATED] Number of multi-start restarts.  Use --n-starts instead."),
    )
    parser.add_argument(
        "--cores",
        type=int,
        default=None,
        help=(
            "[DEPRECATED] Number of CPU cores.  The JAX backend manages its own "
            "parallelism; this flag is ignored.  Control JAX threading via "
            "environment variables such as XLA_FLAGS."
        ),
    )
    parser.add_argument(
        "--algorithm",
        default=None,
        help=(
            "[DEPRECATED] Legacy NSGA2/UNSGA3 algorithm selector.  Ignored in the "
            "Optimistix backend."
        ),
    )
    parser.add_argument(
        "--optimizer",
        default=None,
        help=(
            "[DEPRECATED] Optimistix optimiser name.  Only BFGS is supported at "
            "present.  Ignored."
        ),
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-6,
        help="Relative tolerance for the Diffrax ODE solver.",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-9,
        help="Absolute tolerance for the Diffrax ODE solver.",
    )
    parser.add_argument(
        "--loss-weight-phospho",
        type=float,
        default=1.0,
        dest="loss_weight_phospho",
        help="Scalarisation weight for the phosphosite error objective.",
    )
    parser.add_argument(
        "--loss-weight-abundance",
        type=float,
        default=1.0,
        dest="loss_weight_abundance",
        help="Scalarisation weight for the protein abundance error objective.",
    )
    parser.add_argument(
        "--loss-weight-reg",
        type=float,
        default=1.0,
        dest="loss_weight_reg",
        help="Scalarisation weight for the regularisation objective.",
    )

    # ------------------------------------------------------------------
    # MODES
    # ------------------------------------------------------------------
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Run hyperparameter scanning before optimization.",
    )
    parser.add_argument(
        "--run-steadystate",
        action="store_true",
        help="Run post-optimization steady state simulation.",
    )
    parser.add_argument(
        "--run-knockouts",
        action="store_true",
        help="Run systematic in-silico knockout screening.",
    )
    parser.add_argument(
        "--run-sensitivity",
        action="store_true",
        help="Run global sensitivity analysis.",
    )
    # ------------------------------------------------------------------
    # META OPTIONS
    # ------------------------------------------------------------------
    parser.add_argument(
        "--version",
        action="version",
        version="Phospho-Network Model Fitting 1.0",
    )
    # ------------------------------------------------------------------
    # PARSE ARGUMENTS
    # ------------------------------------------------------------------

    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    results_dir = os.path.join(args.outdir)
    logger.header(f"[*] Output directory: {args.outdir}")

    # ------------------------------------------------------------------
    # Backwards-compatibility mapping for deprecated optimisation flags
    #
    # Prefer the new --n-starts and --max-steps flags when provided.  If
    # unspecified, fall back to the deprecated counterparts to avoid
    # breaking older scripts.  Warn users when both old and new flags are
    # supplied so they are aware of the precedence.

    # Determine n_starts
    if args.n_starts is None:
        if args.pop_size is not None:
            args.n_starts = args.pop_size
        else:
            args.n_starts = 5
    else:
        if args.pop_size is not None:
            logger.warning(
                "[!] Both --n-starts and --pop-size were provided; "
                "using --n-starts and ignoring the deprecated --pop-size flag."
            )

    # Determine max_steps
    if args.max_steps is None:
        if args.gen is not None:
            args.max_steps = args.gen
        else:
            args.max_steps = 200
    else:
        if args.gen is not None:
            logger.warning(
                "[!] Both --max-steps and --gen were provided; "
                "using --max-steps and ignoring the deprecated --gen flag."
            )

    # Normalise types and ensure positive values
    args.n_starts = max(1, int(args.n_starts))
    args.max_steps = max(1, int(args.max_steps))

    # Warn that --cores, --algorithm and --optimizer are ignored
    if args.cores is not None:
        logger.warning(
            "[!] The --cores flag no longer controls parallelism.  "
            "Set JAX threading via environment variables (e.g., "
            "XLA_FLAGS=--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=<N>) "
            "prior to running this script."
        )
    if args.algorithm:
        logger.warning(
            f"[!] --algorithm {args.algorithm!r} is ignored by the Optimistix backend."
        )
    if args.optimizer:
        logger.warning(
            f"[!] --optimizer {args.optimizer!r} is ignored.  Only BFGS is currently supported."
        )

    # 1. Load Data
    (sites, proteins, site_prot_idx, positions, t, Y, A_data, A_proteins) = (
        data_loader.load_site_data(args.data)
    )

    logger.success(f"[*] Loaded {len(sites)} sites, {len(proteins)} proteins.")

    # Crosstalk filtering
    if args.crosstalk_tsv:

        def load_allowed(path):
            df = pd.read_csv(path, sep="\t")
            s = set()
            for c in ["Site1", "Site2"]:
                for p, site in zip(df["Protein"], df[c]):
                    if pd.notna(p) and pd.notna(site):
                        s.add(f"{p}_{site}")
            return s

        allowed = load_allowed(args.crosstalk_tsv)
        mask = np.array([s in allowed for s in sites], dtype=bool)
        sites = [s for s, m in zip(sites, mask) if m]
        positions = positions[mask]
        Y = Y[mask, :]
        site_prot_idx = site_prot_idx[mask]
        prots_used = sorted({s.split("_", 1)[0] for s in sites})
        prot_map = {p: i for i, p in enumerate(prots_used)}
        site_prot_idx = np.array(
            [prot_map[s.split("_", 1)[0]] for s in sites], dtype=int
        )
        proteins = prots_used
        logger.info(f"[*] Filtered to {len(sites)} sites.")

    # 2. Scaling
    P_scaled, baselines, amplitudes = data_loader.apply_scaling(Y, mode=args.scale_mode)
    P_scaled = np.nan_to_num(P_scaled, nan=0.0, posinf=0.0, neginf=0.0)

    if A_data is not None and len(A_data) > 0:
        prot_map = {p: i for i, p in enumerate(proteins)}
        mask_A = [p in prot_map for p in A_proteins]
        A_data = A_data[mask_A]
        A_proteins = A_proteins[mask_A]
        prot_idx_for_A = np.array([prot_map[p] for p in A_proteins], dtype=int)
        A_scaled, A_bases, A_amps = data_loader.apply_scaling(
            A_data, mode=args.scale_mode
        )
        A_scaled = np.nan_to_num(A_scaled, nan=0.0)
    else:
        A_scaled = np.zeros((0, P_scaled.shape[1]))
        prot_idx_for_A = np.array([], dtype=int)
        A_bases, A_amps = np.array([]), np.array([])

    # _sanity_report_data(P_scaled, Y, t)

    # 3. Weights
    W_data, W_data_prot = build_weight_matrices(
        t=t, Y=Y, A_data=A_data, scheme=args.weight_scheme
    )

    # _sanity_report_weights(W_data, W_data_prot)

    # 4. Matrices & Graph
    Cg, Cl = data_loader.build_C_matrices_from_db(
        args.ptm_intra,
        args.ptm_inter,
        sites,
        site_prot_idx,
        positions,
        proteins,
        args.length_scale,
    )
    Cg, Cl = data_loader.row_normalize(Cg), data_loader.row_normalize(Cl)

    # _sanity_report_C(Cg, Cl, N=len(sites))

    if args.kinase_tsv:
        K_site_kin, kinases = data_loader.load_kinase_site_matrix(
            args.kinase_tsv, sites
        )
    elif args.kea_ks_table:
        K_site_kin, kinases = data_loader.build_kinase_site_from_kea(
            args.kea_ks_table, sites
        )
    else:
        K_site_kin = np.eye(len(sites))
        kinases = [f"K_{i}" for i in range(len(sites))]

    # _coverage_report_K_site_kin(K_site_kin, sites, kinases)

    # Set the dimensions globally for the model
    ModelDims.set_dims(len(proteins), len(kinases), len(sites))

    # Transpose kinase - site matrix & normalize
    R = np.ascontiguousarray(K_site_kin.T)
    rs = R.sum(axis=1)
    nz = rs > 0
    R[nz] /= rs[nz, None]

    # _sanity_report_R(R, N=len(sites), M=len(kinases))

    L_alpha = np.zeros((len(kinases), len(kinases)))
    if args.unified_graph_pkl and args.lambda_net > 0:
        L_alpha = data_loader.build_alpha_laplacian_from_unified_graph(
            args.unified_graph_pkl, kinases
        )

    # 5. Mappings & Masks
    prot_map_all = {p: i for i, p in enumerate(proteins)}
    kin_to_prot_idx = np.array([prot_map_all.get(k, -1) for k in kinases], dtype=int)

    # receptor_names = {"EGFR", "ERBB2", "EPHA2", "MET"}
    # receptor_kin_names = {"EGFR", "EPHA2", "ERBB4", "INSR", "RET"}

    # Convert list args to sets for fast lookup
    receptor_names = set(args.receptors)
    receptor_kin_names = set(args.receptor_kinases)

    # Build masks (will be all zeros if args.receptors is empty)
    receptor_mask_prot = np.array(
        [1 if p in receptor_names else 0 for p in proteins], dtype=int
    )
    receptor_mask_kin = np.array(
        [1 if k in receptor_kin_names else 0 for k in kinases], dtype=int
    )

    if len(receptor_names) == 0:
        logger.warning(
            "[!] No Receptors defined. External stimulus u(t) will be ignored."
        )

    # --- HYPERPARAMETER TUNING ---
    if args.tune:
        best_params = hyperparam.run_hyperparameter_scan(
            args.outdir,
            t,
            P_scaled,
            sites,
            site_prot_idx,
            positions,
            proteins,
            args.ptm_intra,
            args.ptm_inter,
            Cg,
            K_site_kin,
            R,
            L_alpha,
            kin_to_prot_idx,
            A_scaled,
            prot_idx_for_A,
            W_data,
            W_data_prot,
            receptor_mask_prot,
            receptor_mask_kin,
            args.mechanism,
        )
        # Apply Best Params
        args.length_scale = best_params["length_scale"]
        args.lambda_net = best_params["lambda_net"]
        args.reg_lambda = best_params["reg_lambda"]
        logger.success(
            f"[*] Applied Tuned Params: LS={args.length_scale}, LN={args.lambda_net}, Reg={args.reg_lambda}"
        )

    logger.info(f"[*] Building final matrices with Length Scale {args.length_scale}...")
    _, Cl = data_loader.build_C_matrices_from_db(
        args.ptm_intra,
        args.ptm_inter,
        sites,
        site_prot_idx,
        positions,
        proteins,
        length_scale=args.length_scale,
    )
    Cl = data_loader.row_normalize(Cl)

    # _sanity_report_C(Cg, Cl, N=len(sites))

    # 6. Global Setup & Bounds
    logger.header(f"[*] K={ModelDims.K}, M={ModelDims.M}, N={ModelDims.N}")

    xl, xu, dim = create_bounds(ModelDims.K, ModelDims.M, ModelDims.N)

    _save_preopt_snapshot_txt_csv(
        args.outdir,
        t=t,
        sites=sites,
        proteins=proteins,
        kinases=kinases,
        positions=positions,
        P_scaled=P_scaled,
        Y=Y,
        A_scaled=A_scaled,
        A_data=A_data,
        A_proteins=A_proteins,
        W_data=W_data,
        W_data_prot=W_data_prot,
        Cg=Cg,
        Cl=Cl,
        site_prot_idx=site_prot_idx,
        K_site_kin=K_site_kin,
        R=R,
        L_alpha=L_alpha,
        kin_to_prot_idx=kin_to_prot_idx,
        receptor_mask_prot=receptor_mask_prot,
        receptor_mask_kin=receptor_mask_kin,
        xl=xl,
        xu=xu,
        args=args,
    )

    # 7. Optimisation

    logger.info(
        f"[*] Initialising Optimistix problem ({args.n_starts} starts, max_steps={args.max_steps})..."
    )

    problem = NetworkOptimizationProblem(
        t,
        P_scaled,
        Cg,
        Cl,
        site_prot_idx,
        K_site_kin,
        R,
        A_scaled,
        prot_idx_for_A,
        W_data,
        W_data_prot,
        L_alpha,
        kin_to_prot_idx,
        args.lambda_net,
        args.reg_lambda,
        receptor_mask_prot,
        receptor_mask_kin,
        args.mechanism,
        xl,
        xu,
    )

    res, best_idx, total_losses = run_multi_start_optimization(problem, args, P_scaled)

    # 8. Analysis & Saving
    # Find best solution by minimum total loss (single-objective criterion)
    F, X = res.F, res.X
    f1, f2, f3 = F[:, 0], F[:, 1], F[:, 2]

    analysis.save_run_results(args.outdir, F, X, f1, f2, f3, total_losses, F[best_idx])
    analysis.plot_run_diagnostics(args.outdir, F, F[best_idx], f1, f2, f3, X)

    analysis.save_fitted_simulation(
        args.outdir,
        X[best_idx],
        t,
        sites,
        proteins,
        P_scaled,
        A_scaled,
        prot_idx_for_A,
        baselines,
        amplitudes,
        Y,
        A_data,
        A_bases,
        A_amps,
        args.mechanism,
        Cg,
        Cl,
        site_prot_idx,
        K_site_kin,
        R,
        L_alpha,
        kin_to_prot_idx,
        receptor_mask_prot,
        receptor_mask_kin,
    )

    analysis.plot_fitted_simulation(args.outdir)
    analysis.print_parameter_summary(args.outdir, X[best_idx], proteins, kinases, sites)
    analysis.print_biological_scores(args.outdir, X)
    analysis.plot_biological_scores(args.outdir, X, F)
    analysis.plot_goodness_of_fit(f"{results_dir}/fit_timeseries.tsv", args.outdir)

    if args.run_steadystate:
        steadystate.run_steadystate_analysis(
            args.outdir, problem, X[best_idx], sites, proteins, kinases
        )

    if args.run_knockouts:
        knockouts.run_knockout_screen(
            args.outdir, problem, X[best_idx], sites, proteins, kinases
        )

    if args.run_sensitivity:
        bounds = (xl, xu)

        run_global_sensitivity(
            args.outdir,
            problem,
            bounds,
            proteins=proteins,
            kinases=kinases,
            sites=sites,
        )

    # 1. Save Run Configuration (Provenance)
    save_run_metadata(args.outdir, args)

    # 2. Export Cytoscape Network
    # Requires theta_opt (best solution)
    export_network_for_cytoscape(
        args.outdir, X[best_idx], proteins, kinases, sites, K_site_kin, site_prot_idx
    )

    # 3. Residual Heatmap
    # Re-simulate best solution one last time to get P_sim
    P_best = problem.simulate(X[best_idx])
    plot_residual_heatmap(args.outdir, P_scaled, P_best, sites, t)

    # 4. Parameter Correlations (using the X population from multi_start)
    # We reuse the label generator from sensitivity if available, or generate generic ones
    p_labels = _generate_param_labels(
        ModelDims.K, ModelDims.M, ModelDims.N, proteins, kinases, sites
    )

    plot_parameter_clustermap(args.outdir, X, p_labels, top_n=50)

    generate_equations_report(
        args.outdir,
        X[best_idx],
        proteins,
        kinases,
        sites,
        Cg,
        Cl,
        site_prot_idx,
        K_site_kin,
        R,
        L_alpha,
        kin_to_prot_idx,
        receptor_mask_prot,
        receptor_mask_kin,
        args.mechanism,
    )

    logger.success("[*] Done.")


if __name__ == "__main__":
    main()


def cli():
    main()
