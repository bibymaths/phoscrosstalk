#!/usr/bin/env python3
"""
main.py
Entry point for the Global Phospho-Network Model orchestration.
"""

import argparse
import os

import numpy as np
import pandas as pd

from phoscrosstalk import analysis, data_loader, hyperparam, knockouts, steadystate
from phoscrosstalk.analysis import _save_preopt_snapshot_txt_csv
from phoscrosstalk.config import ModelDims, load_config
from phoscrosstalk.derived_rates import make_k_act_fn, make_s_prod_fn
from phoscrosstalk.equations import generate_equations_report
from phoscrosstalk.logger import get_logger
from phoscrosstalk.multistarts import run_multi_start_optimization
from phoscrosstalk.optimization import (
    NetworkProblem as NetworkOptimizationProblem,
)
from phoscrosstalk.optimization import (
    create_bounds,
)
from phoscrosstalk.post_processing import (
    export_network_for_cytoscape,
    plot_parameter_clustermap,
    plot_residual_heatmap,
    save_run_metadata,
)
from phoscrosstalk.sensitivity import _generate_param_labels, run_global_sensitivity
from phoscrosstalk.weighting import build_weight_matrices

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
    # CONFIGURATION FILE
    # ------------------------------------------------------------------
    parser.add_argument(
        "--config",
        default="./config.toml",
        help="Path to config.toml file with tuneable parameters.",
    )

    # ------------------------------------------------------------------
    # INPUT DATA
    # ------------------------------------------------------------------
    parser.add_argument(
        "--data",
        required=True,
        help=(
            "CSV file containing protein/phosphosite time-series data. "
            "Expected columns: Protein (or GeneID), Psite (or Residue), "
            "and value columns (x1..xN or v1..vN). "
            "Rows without a Psite/Residue value are treated as protein abundance rows. "
            "Example: data_timeseries/filtered_input1.csv"
        ),
    )
    parser.add_argument(
        "--ptm-intra",
        required=True,
        help=(
            "SQLite database containing intra-protein PTM crosstalk pairs "
            "(table: intra_pairs). "
            "Example: data_curated/processed/ptm_intra.db"
        ),
    )
    parser.add_argument(
        "--ptm-inter",
        required=True,
        help=(
            "SQLite database containing inter-protein PTM crosstalk pairs "
            "(table: inter_pairs). "
            "Example: data_curated/processed/ptm_inter.db"
        ),
    )
    parser.add_argument(
        "--crosstalk-tsv",
        help=(
            "Optional TSV listing specific PTM pairs to keep (crosstalk filtering). "
            "Rows not in this list are removed from the model. "
            "Columns: Protein, Site1, Site2."
        ),
    )
    parser.add_argument(
        "--rna-data",
        default=None,
        dest="rna_data",
        help=(
            "CSV file with mRNA time-series data. "
            "Expected columns: GeneID, x1, x2, ..., x9 "
            "(mapped to time points [4, 8, 15, 30, 60, 120, 240, 480, 960] min). "
            "When provided, mRNA state R(t) is included in the ODE system and "
            "an mRNA loss term (f4) is added to the objective. "
            "Example: data_timeseries/filtered_input3.csv"
        ),
    )
    parser.add_argument(
        "--tf-net",
        default=None,
        dest="tf_net",
        help=(
            "CSV file describing the TF-to-mRNA regulatory network. "
            "Required columns (case-insensitive): Source, Target, Weight. "
            "Source = TF gene symbol; Target = regulated gene symbol. "
            "Used to construct k_act(t) as a derived rate from mRNA signals. "
            "Example: data_interactions/tf_mrna.csv"
        ),
    )

    # ------------------------------------------------------------------
    # KINASE–SUBSTRATE MAPPING
    # ------------------------------------------------------------------
    parser.add_argument(
        "--kinase-tsv",
        help=(
            "TSV file mapping phosphosites to kinases (Kinase-Substrate prior). "
            "Required columns: Site, Kinase; optional: weight. "
            "At least one of --kinase-tsv or --kea-ks-table must be provided; "
            "if neither is given, the model falls back to an identity mapping. "
            "Example: data_interactions/kinase_sites.tsv"
        ),
    )
    parser.add_argument(
        "--kea-ks-table",
        help=(
            "Alternative KEA/KS kinase-substrate mapping table in TSV format. "
            "Required columns: substrate_site, kinase, pmid. "
            "Used when --kinase-tsv is not available. "
            "Example: data_curated/processed/ks_psite_table.tsv"
        ),
    )
    parser.add_argument(
        "--unified-graph-pkl",
        help=(
            "Pickled NetworkX graph of kinase-kinase regulatory relationships. "
            "Used to build the Laplacian regularizer L_alpha. "
            "Optional; if absent, L_alpha is set to zero. "
            "Example: data_curated/processed/unified_kinase_graph.gpickle"
        ),
    )

    # ------------------------------------------------------------------
    # EXTERNAL STIMULI SPECIFIC RECEPTORS & KINASES
    # ------------------------------------------------------------------
    parser.add_argument(
        "--receptors",
        nargs="*",
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
        default=None,
        help=(
            "Directory where all results, plots, and output files are saved. "
            "Overrides config.toml [paths] output_dir. "
            "Created automatically if it does not exist."
        ),
    )

    # ------------------------------------------------------------------
    # MODEL CONFIGURATION
    # ------------------------------------------------------------------
    parser.add_argument(
        "--mechanism",
        choices=["dist", "seq", "rand"],
        default=None,
        help=(
            "Phosphorylation mechanism for the kinetic model. "
            "dist = distributive (default), seq = sequential, "
            "rand = random/cooperative. "
            "Overrides config.toml [model] mechanism."
        ),
    )

    # ------------------------------------------------------------------
    # OPTIMIZATION SETTINGS (override TOML)
    # ------------------------------------------------------------------
    parser.add_argument(
        "--n-starts",
        type=int,
        default=None,
        help=(
            "Number of multi-start optimization initialisations. "
            "The run with the lowest total loss is selected as the best result. "
            "Use --n-starts 1 --max-steps 50 for a quick smoke test. "
            "Overrides config.toml [optimisation] n_starts."
        ),
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help=(
            "Maximum number of gradient-based optimization steps per start. "
            "Larger values improve convergence but increase runtime. "
            "Overrides config.toml [optimisation] max_steps."
        ),
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
        help=(
            "Run post-optimization steady-state simulation. "
            "Results are saved to <outdir>/steadystate/."
        ),
    )
    parser.add_argument(
        "--run-knockouts",
        action="store_true",
        help=(
            "Run systematic in-silico kinase knockout screening. "
            "Each kinase is zeroed out and the dynamics compared to baseline. "
            "Results are saved to <outdir>/knockouts/."
        ),
    )
    parser.add_argument(
        "--run-sensitivity",
        action="store_true",
        help=(
            "Run global sensitivity analysis (Sobol indices) over the parameter space. "
            "Results are saved to <outdir>/sensitivity/."
        ),
    )

    # ------------------------------------------------------------------
    # META OPTIONS
    # ------------------------------------------------------------------
    parser.add_argument(
        "--version",
        action="version",
        version="Phospho-Network Model Fitting 2.0",
    )

    # ------------------------------------------------------------------
    # PARSE ARGUMENTS & LOAD CONFIG
    # ------------------------------------------------------------------
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # DEFENSIVE VALIDATION
    # ------------------------------------------------------------------
    # Validate input file existence before doing any heavy computation.
    _missing_files = []
    for _flag, _path in [
        ("--data", args.data),
        ("--ptm-intra", args.ptm_intra),
        ("--ptm-inter", args.ptm_inter),
        ("--crosstalk-tsv", args.crosstalk_tsv),
        ("--rna-data", args.rna_data),
        ("--tf-net", args.tf_net),
        ("--kinase-tsv", args.kinase_tsv),
        ("--kea-ks-table", args.kea_ks_table),
        ("--unified-graph-pkl", args.unified_graph_pkl),
    ]:
        if _path is not None and not os.path.exists(_path):
            _missing_files.append(f"  {_flag}: {_path}")
    if _missing_files:
        print(
            "ERROR: The following input files do not exist:\n"
            + "\n".join(_missing_files),
            flush=True,
        )
        raise SystemExit(1)

    cfg = load_config(args.config)

    # Merge CLI overrides into config (CLI wins when explicitly provided)
    outdir = args.outdir if args.outdir is not None else cfg.paths.output_dir
    mechanism = args.mechanism if args.mechanism is not None else cfg.model.mechanism
    n_starts = args.n_starts if args.n_starts is not None else cfg.optimisation.n_starts
    max_steps = (
        args.max_steps if args.max_steps is not None else cfg.optimisation.max_steps
    )

    # Expose merged values back on args namespace for compatibility with
    # run_multi_start_optimization which reads from args
    args.outdir = outdir
    args.mechanism = mechanism
    args.n_starts = n_starts
    args.max_steps = max_steps

    # Validate numeric settings.
    if args.n_starts < 1:
        print(f"ERROR: --n-starts must be >= 1; got {args.n_starts}", flush=True)
        raise SystemExit(1)
    if args.max_steps < 1:
        print(f"ERROR: --max-steps must be >= 1; got {args.max_steps}", flush=True)
        raise SystemExit(1)

    # Pull numeric settings from config
    args.length_scale = cfg.model.length_scale
    args.scale_mode = cfg.model.scale_mode
    args.weight_scheme = cfg.model.weight_scheme
    args.lambda_net = cfg.optimisation.lambda_net
    args.reg_lambda = cfg.optimisation.reg_lambda
    args.loss_weight_phospho = cfg.loss_weights.phospho
    args.loss_weight_abundance = cfg.loss_weights.abundance
    args.loss_weight_reg = cfg.loss_weights.reg
    args.rtol = cfg.solver.rtol
    args.atol = cfg.solver.atol

    interp_mode = cfg.time.interpolation
    s_prod_fn_type = cfg.derived_rates.s_prod_fn

    os.makedirs(outdir, exist_ok=True)

    logger.header(f"[*] Output directory: {outdir}")

    # 1. Load primary phospho data
    (sites, proteins, site_prot_idx, positions, t, Y, A_data, A_proteins) = (
        data_loader.load_site_data(args.data)
    )
    logger.success(f"[*] Loaded {len(sites)} sites, {len(proteins)} proteins.")

    # 2. Load optional mRNA data and TF network
    gene_ids = None
    t_rna = None
    rna_matrix = None
    tf_prot_weights = None

    if args.rna_data:
        gene_ids, t_rna, rna_matrix = data_loader.load_rna_data(args.rna_data)
        logger.success(
            f"[*] Loaded mRNA data: {len(gene_ids)} genes x {len(t_rna)} time points."
        )

    if args.tf_net:
        tf_net_df = data_loader.load_tf_network(args.tf_net, gene_ids=gene_ids)
        logger.success(f"[*] Loaded TF network: {len(tf_net_df)} edges.")
        if gene_ids is not None:
            tf_prot_weights = data_loader.build_tf_prot_weights(
                tf_net_df, gene_ids, proteins
            )
    elif args.rna_data:
        logger.warning(
            "[!] --rna-data provided but --tf-net is absent; "
            "k_act will default to constant 1.0."
        )

    # 3. Crosstalk filtering
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

    # 4. Scaling
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

    # 5. Weights
    W_data, W_data_prot = build_weight_matrices(
        t=t, Y=Y, A_data=A_data, scheme=args.weight_scheme
    )

    # 6. Matrices & Graph
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

    # Set the dimensions globally for the model
    ModelDims.set_dims(len(proteins), len(kinases), len(sites))

    # Transpose kinase-site matrix & normalize
    R = np.ascontiguousarray(K_site_kin.T)
    rs = R.sum(axis=1)
    nz = rs > 0
    R[nz] /= rs[nz, None]

    L_alpha = np.zeros((len(kinases), len(kinases)))
    if args.unified_graph_pkl and args.lambda_net > 0:
        L_alpha = data_loader.build_alpha_laplacian_from_unified_graph(
            args.unified_graph_pkl, kinases
        )

    # 7. Mappings & Masks
    prot_map_all = {p: i for i, p in enumerate(proteins)}
    kin_to_prot_idx = np.array([prot_map_all.get(k, -1) for k in kinases], dtype=int)

    receptor_names = set(args.receptors)
    receptor_kin_names = set(args.receptor_kinases)

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

    # 8. Build derived rate functions (k_act_fn, s_prod_fn)
    K = len(proteins)
    M = len(kinases)

    k_act_fn = make_k_act_fn(
        t_rna=t_rna,
        rna_data=rna_matrix,
        tf_prot_weights=tf_prot_weights,
        K=K,
        interp_mode=interp_mode,
    )

    s_prod_fn = make_s_prod_fn(
        t_protein=t,
        Y_data=P_scaled,
        R_kin_site=R,
        kin_to_prot_idx=kin_to_prot_idx,
        K=K,
        M=M,
        s_prod_fn_type=s_prod_fn_type,
        interp_mode=interp_mode,
    )
    logger.info("[*] Built derived rate closures k_act_fn and s_prod_fn.")

    # --- HYPERPARAMETER TUNING ---
    if args.tune:
        best_params = hyperparam.run_hyperparameter_scan(
            outdir,
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
            mechanism,
        )
        args.length_scale = best_params["length_scale"]
        args.lambda_net = best_params["lambda_net"]
        args.reg_lambda = best_params["reg_lambda"]
        logger.success(
            f"[*] Applied Tuned Params: LS={args.length_scale}, "
            f"LN={args.lambda_net}, Reg={args.reg_lambda}"
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

    # 9. Global Setup & Bounds
    logger.header(f"[*] K={ModelDims.K}, M={ModelDims.M}, N={ModelDims.N}")
    xl, xu, dim = create_bounds(ModelDims.K, ModelDims.M, ModelDims.N)

    _save_preopt_snapshot_txt_csv(
        outdir,
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

    # 10. Optimisation
    logger.info(
        f"[*] Initialising Optimistix problem ({n_starts} starts, "
        f"max_steps={max_steps})..."
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
        mechanism,
        xl,
        xu,
        k_act_fn=k_act_fn,
        s_prod_fn=s_prod_fn,
        t_rna=t_rna if rna_matrix is not None else None,
    )

    # Build RNA-to-model-protein mapping (when RNA data is available)
    rna_fit_genes = []
    rna_obs_matched = None
    rna_model_prot_idx = None
    R_data0 = None

    if rna_matrix is not None and gene_ids is not None:
        (
            rna_fit_genes,
            rna_obs_matched_raw,
            rna_model_prot_idx_raw,
            rna_obs_idx_raw,
        ) = data_loader.match_rna_to_model_proteins(gene_ids, rna_matrix, proteins)

        if len(rna_fit_genes) > 0:
            rna_obs_matched = rna_obs_matched_raw
            rna_model_prot_idx = rna_model_prot_idx_raw
            # Build full R_rna initial condition vector for all K proteins
            R_data0 = data_loader.build_full_R0(K, gene_ids, rna_matrix, proteins)
            problem.rna_obs_matched = rna_obs_matched
            problem.rna_model_prot_idx = rna_model_prot_idx
            problem.rna_obs_idx = rna_obs_idx_raw
            problem.rna_fit_genes = rna_fit_genes
            problem.R_data0 = R_data0
            logger.info(
                f"[*] RNA-to-model mapping: {len(rna_fit_genes)} matched genes/proteins."
            )
        else:
            logger.warning(
                "[!] No RNA genes matched model proteins. RNA loss disabled."
            )

    res, best_idx, total_losses = run_multi_start_optimization(problem, args, P_scaled)

    # 11. Analysis & Saving
    F, X = res.F, res.X
    f1, f2, f3 = F[:, 0], F[:, 1], F[:, 2]
    # f4 (RNA loss) is in column 3 when present
    f4 = F[:, 3] if F.shape[1] > 3 else np.zeros(len(f1))

    theta_best = X[best_idx]

    analysis.save_derived_rates(
        outdir=outdir,
        proteins=proteins,
        t_protein=t,
        k_act_fn=k_act_fn,
        s_prod_fn=s_prod_fn,
        t_rna=t_rna,
    )

    analysis.save_run_results(outdir, F, X, f1, f2, f3, total_losses, F[best_idx])
    analysis.plot_run_diagnostics(outdir, F, F[best_idx], f1, f2, f3, X)

    analysis.save_fitted_simulation(
        outdir,
        theta_best,
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
        mechanism,
        Cg,
        Cl,
        site_prot_idx,
        K_site_kin,
        R,
        L_alpha,
        kin_to_prot_idx,
        receptor_mask_prot,
        receptor_mask_kin,
        k_act_fn=k_act_fn,
        s_prod_fn=s_prod_fn,
        R_data0=R_data0,
    )

    analysis.plot_fitted_simulation(outdir)
    analysis.print_parameter_summary(outdir, theta_best, proteins, kinases, sites)
    analysis.print_biological_scores(outdir, X)
    analysis.plot_biological_scores(outdir, X, F)
    analysis.plot_goodness_of_fit(f"{outdir}/fit_timeseries.tsv", outdir)

    # mRNA outputs (only when RNA data was provided and RNA matched model proteins)
    if rna_matrix is not None and gene_ids is not None and len(rna_fit_genes) > 0:
        # Use full simulation to get R_sim_rna at RNA time points
        sim_full = problem.simulate_full(theta_best)
        R_sim_all = sim_full.get("R_sim_rna")  # (K, T_rna)
        if R_sim_all is not None and rna_model_prot_idx is not None:
            R_sim_matched = R_sim_all[rna_model_prot_idx, :]  # (n_match, T_rna)
            analysis.save_mrna_outputs(
                outdir=outdir,
                gene_ids=rna_fit_genes,
                t_rna=t_rna,
                rna_data_obs=rna_obs_matched,
                rna_simulated=R_sim_matched,
            )

    if args.run_steadystate:
        steadystate.run_steadystate_analysis(
            outdir, problem, theta_best, sites, proteins, kinases
        )

    if args.run_knockouts:
        knockouts.run_knockout_screen(
            outdir, problem, theta_best, sites, proteins, kinases
        )

    if args.run_sensitivity:
        bounds = (xl, xu)
        run_global_sensitivity(
            outdir,
            problem,
            bounds,
            proteins=proteins,
            kinases=kinases,
            sites=sites,
        )

    # 12. Provenance & exports
    save_run_metadata(outdir, args)
    export_network_for_cytoscape(
        outdir, theta_best, proteins, kinases, sites, K_site_kin, site_prot_idx
    )

    P_best = problem.simulate(theta_best)
    plot_residual_heatmap(outdir, P_scaled, P_best, sites, t)

    p_labels = _generate_param_labels(
        ModelDims.K, ModelDims.M, ModelDims.N, proteins, kinases, sites
    )
    plot_parameter_clustermap(outdir, X, p_labels, top_n=50)

    generate_equations_report(
        outdir,
        theta_best,
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
        mechanism,
    )

    logger.success("[*] Done.")


if __name__ == "__main__":
    main()


def cli():
    main()
