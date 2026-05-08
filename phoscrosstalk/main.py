#!/usr/bin/env python3
"""
main.py
Entry point for the Global Phospho-Network Model orchestration.
"""

# ---------------------------------------------------------------------------
# CPU / XLA environment setup — MUST happen before any JAX import.
#
# 1. Read the optional ``[runtime] cpu_threads`` value directly from the TOML
#    file (stdlib-only, no JAX) so the thread count can be config-driven.
# 2. Call setup_cpu_env() which sets JAX_PLATFORMS, XLA_FLAGS, and all
#    BLAS/OpenMP thread-cap variables.
#
# phoscrosstalk/__init__.py no longer imports JAX-dependent symbols at the
# top level (it uses PEP-562 lazy __getattr__), so importing runtime_env here
# is safe — it only uses ``os`` and ``sys``.
# ---------------------------------------------------------------------------

import os
import pathlib

from phoscrosstalk.config import _read_runtime_config
from phoscrosstalk.utils import sync_gitignore, print_config_summary, save_model_table

from phoscrosstalk.runtime_env import (  # noqa: E402
    enable_x64,
    log_env_summary,
    plan_cpu_runtime,
    setup_cpu_env,
)
from phoscrosstalk.utils.helpers import namespace_to_dict

_runtime_cfg = _read_runtime_config()
_cpu_plan = plan_cpu_runtime(
    cpu_threads=_runtime_cfg["cpu_threads"],
    n_starts=_runtime_cfg["n_starts"],
    parallel_starts=_runtime_cfg["parallel_starts"],
    threads_per_start=_runtime_cfg["threads_per_start"],
    use_physical_cores=_runtime_cfg["use_physical_cores"],
    reserve_cores=_runtime_cfg["reserve_cores"],
)
_n_cpu_threads = setup_cpu_env(n_threads=_cpu_plan.threads_per_run)

enable_x64()  # Must be before any JAX import

# ---------------------------------------------------------------------------
# Standard library and third-party imports (JAX enters here via phoscrosstalk
# submodule imports below — env vars are already set at this point).
# ---------------------------------------------------------------------------

import argparse
from types import SimpleNamespace

import numpy as np
import pandas as pd

from phoscrosstalk import analysis, data_loader, hyperparam, knockouts, steadystate
from phoscrosstalk.analysis import _save_preopt_snapshot_txt_csv
from phoscrosstalk.config import ModelDims, _opt, load_config, validate_config, _read_runtime_config
from phoscrosstalk.derived_rates import make_k_act_fn, make_s_prod_fn
from phoscrosstalk.equations import generate_equations_report
from phoscrosstalk.logger import get_logger
from phoscrosstalk.logo import print_logo
from phoscrosstalk.multistarts import run_multi_start_optimization
from phoscrosstalk.optimization import (
    NetworkProblem as NetworkOptimizationProblem, bounds_to_original_scale,
)
from phoscrosstalk.optimization import (
    create_bounds,
    make_residuals_fn,
    validate_problem_shapes,
)
from phoscrosstalk.post_processing import (
    export_network_for_cytoscape,
    plot_parameter_clustermap,
    plot_residual_heatmap,
    save_run_metadata,
)
from phoscrosstalk.data_loader import (
    _build_network_allow_sets,
    _prefilter_phospho_csv,
    _prefilter_rna_csv,
)


from phoscrosstalk.sensitivity import _generate_param_labels, run_global_sensitivity
from phoscrosstalk.weighting import build_weight_matrices

logger = get_logger("logs/pipeline.log", timestamp=True)

def main():
    """
    Entry point for PhosCrosstalk.

    The CLI accepts ``--config <path>`` (plus ``--help`` and ``--version``).
    All runtime options are read from the TOML configuration file.
    """

    parser = argparse.ArgumentParser(
        prog="phoscrosstalk",
        description=(
            "Fit a global phospho-network ODE model using JAX/Diffrax ODE solving "
            "and Optimistix gradient-based optimisation. "
            "All options are configured via config.toml; "
            "see docs/running.md for details."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--config",
        default="./config.toml",
        help="Path to config.toml.  All runtime settings live here.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version="Phospho-Network Model Fitting 2.0",
    )

    args = parser.parse_args()
    config_path = args.config

    # ------------------------------------------------------------------
    # LOAD AND VALIDATE CONFIG
    # ------------------------------------------------------------------
    if not os.path.exists(config_path):
        logger.error(
            "Config file not found: %r\n"
            "Create a config.toml file or pass --config <path>.\n"
            "See docs/running.md for a complete example.",
            config_path,
        )
        raise SystemExit(1)

    cfg = load_config(config_path)
    validate_config(cfg, config_path)

    # ------------------------------------------------------------------
    # EXTRACT ALL SETTINGS FROM CONFIG
    # ------------------------------------------------------------------
    # Paths
    data_path = cfg.paths.data
    ptm_intra_path = cfg.paths.ptm_intra
    ptm_inter_path = cfg.paths.ptm_inter
    outdir = cfg.paths.output_dir
    crosstalk_tsv_path = _opt(cfg.paths.crosstalk_tsv)
    rna_data_path = _opt(cfg.paths.rna_data)
    tf_net_path = _opt(cfg.paths.tf_net)
    kinase_tsv_path = _opt(cfg.paths.kinase_tsv)
    kea_ks_table_path = _opt(cfg.paths.kea_ks_table)
    unified_graph_pkl_path = _opt(cfg.paths.unified_graph_pkl)

    # Model
    mechanism = cfg.model.mechanism
    include_tfs_as_proteins = bool(cfg.model.include_tfs_as_proteins)
    receptor_names = set(list(cfg.model.receptors))
    receptor_kin_names = set(list(cfg.model.receptor_kinases))

    # Optimisation / solver / misc – expose on a namespace for run_multi_start_optimization  # noqa: E501
    args = SimpleNamespace(
        data=data_path,
        ptm_intra=ptm_intra_path,
        ptm_inter=ptm_inter_path,
        outdir=outdir,
        crosstalk_tsv=crosstalk_tsv_path,
        rna_data=rna_data_path,
        tf_net=tf_net_path,
        kinase_tsv=kinase_tsv_path,
        kea_ks_table=kea_ks_table_path,
        unified_graph_pkl=unified_graph_pkl_path,
        mechanism=mechanism,
        include_tfs_as_proteins=include_tfs_as_proteins,
        receptors=list(receptor_names),
        receptor_kinases=list(receptor_kin_names),
        # optimisation
        n_starts=cfg.optimisation.n_starts,
        max_steps=cfg.optimisation.max_steps,
        lambda_net=cfg.optimisation.lambda_net,
        reg_lambda=cfg.optimisation.reg_lambda,
        # Backend selection — resolved once here; propagated into multistarts.
        # Default "optimistix" keeps full backward-compatible behaviour.
        optimizer_backend=getattr(cfg.optimisation, "optimizer_backend", "optimistix"),
        optimizer_backend_kwargs=namespace_to_dict(
            getattr(cfg.optimisation, "optimizer_backend_kwargs", None)
        ),
        # Optimistix least-squares controls
        ls_solver=getattr(cfg.optimisation, "ls_solver", "lm"),
        optx_adjoint=getattr(cfg.optimisation, "optx_adjoint", "implicit"),
        jac_mode=getattr(cfg.optimisation, "jac_mode", "fwd"),
        # Diffrax ODE solver controls
        ode_solver=getattr(cfg.solver, "ode_solver", "tsit5"),
        ode_adjoint=getattr(cfg.solver, "ode_adjoint", "forward"),
        ode_dt0=getattr(cfg.solver, "dt0", 0.01),
        ode_root_find_max_steps=getattr(cfg.solver, "root_find_max_steps", 10),
        # Optimistix solver settings (separate from ODE solver tolerances)
        opt_rtol=getattr(cfg.optimisation, "rtol", 1e-8),
        opt_atol=getattr(cfg.optimisation, "atol", 1e-8),
        opt_verbose=getattr(cfg.optimisation, "verbose", False),
        opt_log_every=getattr(cfg.optimisation, "log_every", 100),
        # model tuning
        scale_mode=cfg.model.scale_mode,
        length_scale=cfg.model.length_scale,
        weight_scheme=cfg.model.weight_scheme,
        # loss weights (read by run_multi_start_optimization via getattr)
        loss_weight_phospho=cfg.loss_weights.phospho,
        loss_weight_abundance=cfg.loss_weights.abundance,
        loss_weight_reg=cfg.loss_weights.reg,
        loss_weight_mrna=cfg.loss_weights.mrna,
        # solver
        rtol=cfg.solver.rtol,
        atol=cfg.solver.atol,
        solver_max_steps=cfg.solver.max_steps,
        # analysis flags
        tune=getattr(cfg.analysis, "tune", False),
        run_steadystate=getattr(cfg.analysis, "run_steadystate", False),
        run_knockouts=getattr(cfg.analysis, "run_knockouts", False),
        run_sensitivity=getattr(cfg.analysis, "run_sensitivity", False),
        # CPU parallelism (read by run_multi_start_optimization)
        cpu_threads=getattr(getattr(cfg, "runtime", None), "cpu_threads", "auto"),
        parallel_starts=getattr(
            getattr(cfg, "runtime", None), "parallel_starts", "auto"
        ),
        threads_per_start=getattr(
            getattr(cfg, "runtime", None), "threads_per_start", "auto"
        ),
        use_physical_cores=getattr(
            getattr(cfg, "runtime", None), "use_physical_cores", True
        ),
        reserve_cores=getattr(getattr(cfg, "runtime", None), "reserve_cores", 0),
        parallel_frechet=getattr(
            getattr(cfg, "runtime", None), "parallel_frechet", "auto"
        ),
        # Debug: when cfg.debug.save_jaxpr_reports is True, this path is set
        # so that multistarts.py and optimization.py can capture jaxpr reports.
        _jaxpr_out_dir=(
            str(pathlib.Path(outdir) / "jaxpr_reports")
            if getattr(getattr(cfg, "debug", None), "save_jaxpr_reports", False)
            else None
        ),
    )

    interp_mode = cfg.time.interpolation
    s_prod_fn_type = cfg.derived_rates.s_prod_fn

    # ------------------------------------------------------------------
    # PRINT LOGO
    # ------------------------------------------------------------------

    print_logo(
        name="PhosCrossTalk",
        version="alpha",
        tagline=(
            "Global phospho-network ODE modeling with PTM crosstalk, "
            "kinase-site priors, and TF/mRNA integration"
        ),
        author="Abhinav Mishra",
        email="mishraabhinav36@gmail.com",
        orcid="0009-0005-3179-7408",
        website="https://bibymaths.github.io",
        font="slant",
        color="bright_green",
        animate=False,
    )

    # ------------------------------------------------------------------
    # PRINT CONFIG SUMMARY AND RUNTIME ENVIRONMENT
    # ------------------------------------------------------------------
    sync_gitignore()
    # Create output directory
    os.makedirs(outdir, exist_ok=True)
    logger.success(f"[*] Results will be saved here : {outdir}")
    print_config_summary(cfg, config_path)
    log_env_summary(logger, plan=_cpu_plan)
    logger.info(
        "[runtime_env] CPU threads configured: %d per run "
        "  (parallel_runs=%d, source: SLURM_CPUS_PER_TASK=%s, config cpu_threads=%s)",
        _n_cpu_threads,
        _cpu_plan.n_parallel_runs,
        os.environ.get("SLURM_CPUS_PER_TASK", "unset"),
        getattr(getattr(cfg, "runtime", None), "cpu_threads", "auto"),
    )

    logger.header("[*] Loading and preprocessing timeseries data")

    (
        _net_allowed_sites,
        _net_allowed_kinases,
        _net_tf_sources,
        _net_tf_targets,
    ) = _build_network_allow_sets(
        kinase_tsv_path,
        tf_net_path,
        include_tfs_as_proteins,
    )

    _filtered_data_path = _prefilter_phospho_csv(
        data_path=args.data,
        allowed_sites=_net_allowed_sites,
        allowed_kinases=_net_allowed_kinases,
        tf_sources=_net_tf_sources,
        tf_targets=_net_tf_targets,
        include_tfs_as_proteins=include_tfs_as_proteins,
        logger_=logger,
    )

    _tmp_files_to_cleanup = []
    if _filtered_data_path != args.data:
        _tmp_files_to_cleanup.append(_filtered_data_path)

    try:
        # 1. Load primary phospho data
        (sites, proteins, site_prot_idx, positions, t, Y, A_data, A_proteins) = (
            data_loader.load_site_data(_filtered_data_path)
        )
    finally:
        # Delete the phospho temp file as soon as the loader has consumed it.
        for _p in _tmp_files_to_cleanup:
            try:
                os.unlink(_p)
            except OSError:
                pass
        _tmp_files_to_cleanup.clear()

    # 2. Load optional mRNA data and TF network
    gene_ids = None
    t_rna = None
    rna_matrix = None
    tf_prot_weights = None
    tf_net_df = None

    if args.rna_data:
        _filtered_rna_path = _prefilter_rna_csv(
            rna_path=args.rna_data,
            model_proteins=proteins,  # defined in step 1 above
            tf_sources=_net_tf_sources,
            tf_targets=_net_tf_targets,
            logger_=logger,
        )
        _rna_tmp = _filtered_rna_path if _filtered_rna_path != args.rna_data else None
        try:
            gene_ids, t_rna, rna_matrix = data_loader.load_rna_data(_filtered_rna_path)
        finally:
            # H1: delete RNA temp file immediately after loader returns.
            if _rna_tmp is not None:
                try:
                    os.unlink(_rna_tmp)
                except OSError:
                    pass
        logger.success(
            f"[*] Loaded mRNA data: {len(gene_ids)} genes x {len(t_rna)} time points."
        )

    logger.success(f"[*] Loaded {len(sites)} phosphorylation sites & {len(proteins)} proteins.")

    if args.tf_net:
        tf_net_df = data_loader.load_tf_network(args.tf_net, gene_ids=gene_ids)
        logger.success(f"[*] Loaded TF network: {len(tf_net_df)} edges.")
        if gene_ids is not None:
            tf_prot_weights = data_loader.build_tf_prot_weights(
                tf_net_df, gene_ids, proteins
            )
    elif args.rna_data:
        logger.warning(
            "[!] rna_data is set but tf_net is absent; "
            "k_act will default to constant 1.0."
        )

    # 3. Crosstalk filtering
    if args.crosstalk_tsv:

        def load_allowed(path):
            df = pd.read_csv(path, sep="\t")
            s = set()
            for c in ["Site1", "Site2"]:
                for p, site in zip(df["Protein"], df[c], strict=False):
                    if pd.notna(p) and pd.notna(site):
                        s.add(f"{p}_{site}")
            return s

        allowed = load_allowed(args.crosstalk_tsv)
        mask = np.array([s in allowed for s in sites], dtype=bool)
        sites = [s for s, m in zip(sites, mask, strict=False) if m]
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
        mask_A = np.array([p in prot_map for p in A_proteins], dtype=bool)
        n_A_before = int(mask_A.shape[0])
        A_data = A_data[mask_A]
        A_proteins = A_proteins[mask_A]
        n_A_dropped = n_A_before - len(A_proteins)
        if n_A_dropped > 0:
            # H4: log when abundance rows are discarded (e.g. after crosstalk filter)
            logger.info(
                f"[*] Abundance data: dropped {n_A_dropped}/{n_A_before} protein row(s) "
                "not present in filtered protein list."
            )
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
    logger.info(f"[*] Building weight matrices")
    W_data, W_data_prot, W_data_mrna = build_weight_matrices(
        t=t,
        Y=Y,
        A_data=A_data,
        t_mrna=t_rna,
        rna_data=rna_matrix,
        scheme=args.weight_scheme,
    )

    logger.success(f"[*] Weight matrices built successfully")

    # H3: Validate biological inputs (non-negative finite data) before fitting.
    from phoscrosstalk.optimization import validate_biological_inputs  # noqa: PLC0415

    validate_biological_inputs(
        P_data=P_scaled,
        A_scaled=A_scaled if A_scaled.size > 0 else None,
        rna_data_scaled=None,  # RNA data validated later after matching
        W_data=W_data,
        W_data_prot=W_data_prot,
    )

    # 6. Matrices & Graph
    logger.info(f"[*] Building matrices and graph")
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
    logger.success(f"[*] Matrices and graph built successfully")

    if args.kinase_tsv:
        logger.info(f"[*] Loading kinase-site matrix from {args.kinase_tsv}")
        K_site_kin, kinases = data_loader.load_kinase_site_matrix(
            args.kinase_tsv, sites
        )
        logger.success(f"[*] Kinase-site matrix loaded successfully")
    elif args.kea_ks_table:
        logger.info(f"[*] Building kinase-site matrix from KEA table {args.kea_ks_table}")
        K_site_kin, kinases = data_loader.build_kinase_site_from_kea(
            args.kea_ks_table, sites
        )
        logger.success(f"[*] Kinase-site matrix built successfully")
    else:
        logger.info("[*] No kinase-site data provided; using identity matrix (each site regulated by its own kinase).")
        K_site_kin = np.eye(len(sites))
        kinases = [f"K_{i}" for i in range(len(sites))]
        logger.success(f"[*] Kinase-site matrix initialized successfully")

    # Transpose kinase-site matrix & normalize
    R = np.ascontiguousarray(K_site_kin.T)
    rs = R.sum(axis=1)
    nz = rs > 0
    R[nz] /= rs[nz, None]

    L_alpha = np.zeros((len(kinases), len(kinases)))
    if args.unified_graph_pkl and args.lambda_net > 0:
        logger.info(f"[*] Building alpha Laplacian from unified graph {args.unified_graph_pkl}")
        L_alpha = data_loader.build_alpha_laplacian_from_unified_graph(
            args.unified_graph_pkl, kinases
        )
        logger.success(f"[*] Alpha Laplacian built successfully")
        logger.info(f"[*] Alpha Laplacian shape: {L_alpha.shape}")


    # 7. Mappings & Masks
    prot_map_all = {p: i for i, p in enumerate(proteins)}
    kin_to_prot_idx = np.array([prot_map_all.get(k, -1) for k in kinases], dtype=int)
    dims = ModelDims.from_data(
        P_data=P_scaled,
        A_data=A_scaled if A_scaled.size > 0 else None,
        kin_to_prot_idx=kin_to_prot_idx,
    )
    logger.success(f"[*] Model dimensions initialized successfully")

    logger.header("[*] Model network universe")
    # receptor_names / receptor_kin_names come from cfg.model
    receptor_mask_prot = np.array(
        [1 if p in receptor_names else 0 for p in proteins], dtype=int
    )
    receptor_mask_kin = np.array(
        [1 if k in receptor_kin_names else 0 for k in kinases], dtype=int
    )

    if len(receptor_names) == 0:
        logger.warning(
            "[!] No receptors defined in [model] receptors. "
            "External stimulus u(t) will be ignored."
        )

    if receptor_names:
        logger.info(
            "[*] Config receptors (%d): %s",
            len(receptor_names),
            ", ".join(sorted(receptor_names)),
        )
    else:
        logger.warning(
            "[!] Config receptors list is empty: [model] receptors = []."
        )

    if receptor_kin_names:
        logger.info(
            "[*] Config receptor kinases (%d): %s",
            len(receptor_kin_names),
            ", ".join(sorted(receptor_kin_names)),
        )
    else:
        logger.warning(
            "[!] Config receptor_kinases list is empty: [model] receptor_kinases = []."
        )

    # 7b. Build entity metadata masks (prior support, TF membership, RNA presence)
    logger.info(f"[*] Building entity metadata masks")
    entity_masks = data_loader.build_protein_entity_masks(
        proteins=proteins,
        sites=sites,
        site_prot_idx=site_prot_idx,
        K_site_kin=K_site_kin,
        tf_net_df=tf_net_df,
        tf_prot_weights=tf_prot_weights,
        gene_ids=gene_ids,
        include_tfs_as_proteins=include_tfs_as_proteins,
    )
    logger.success(f"[*] Entity metadata masks built successfully")

    # Log TF network overlap counts
    logger.info(
        f"[*] TF network: {entity_masks['n_tf_sources']} sources, "
        f"{entity_masks['n_tf_targets']} targets. "
        f"Sources in RNA: {entity_masks['n_sources_in_rna']}, "
        f"Targets in RNA: {entity_masks['n_targets_in_rna']}. "
        f"Sources in model proteins: {entity_masks['n_sources_in_proteins']}, "
        f"Targets in model proteins: {entity_masks['n_targets_in_proteins']}."
    )
    n_no_kinase = int((~entity_masks["protein_has_kinase_prior"]).sum())
    n_no_tf = int((~entity_masks["protein_has_tf_input"]).sum())
    _n_proteins = len(proteins)
    logger.info(
        f"[*] Proteins without kinase priors: {n_no_kinase}/{_n_proteins}; "
        f"without TF upstream: {n_no_tf}/{_n_proteins}."
    )
    if include_tfs_as_proteins:
        n_ext = entity_masks["n_included_by_extended"]
        logger.info(
            f"[*] Extended mode: {n_ext} protein(s) included only due to "
            "include_tfs_as_proteins = true."
        )

    # Save model_entities.tsv
    save_model_table(
        outdir=outdir,
        proteins=proteins,
        sites=sites,
        site_prot_idx=site_prot_idx,
        entity_masks=entity_masks,
        gene_ids=gene_ids,
        A_proteins=A_proteins,
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
        protein_self_rna_idx=entity_masks["protein_self_rna_idx"],
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
    logger.success(f"[*] Derived rate closures k_act_fn and s_prod_fn built successfully")

    # --- HYPERPARAMETER TUNING ---
    if args.tune:
        best_params = hyperparam.run_hyperparameter_scan(
            dims,
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

    logger.info(f"[*] Building Cl (local) & Cg (global) with Length Scale {args.length_scale} from database")
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
    logger.success(f"[*] Final C matrices built successfully with tuned Length Scale.")


    # 9. Global Setup & Bounds
    logger.header("[*] Model dimension summary")

    n_tf_sources = int(entity_masks.get("n_tf_sources", 0))
    n_tf_targets = int(entity_masks.get("n_tf_targets", 0))
    n_sources_in_rna = int(entity_masks.get("n_sources_in_rna", 0))
    n_targets_in_rna = int(entity_masks.get("n_targets_in_rna", 0))
    n_sources_in_proteins = int(entity_masks.get("n_sources_in_proteins", 0))
    n_targets_in_proteins = int(entity_masks.get("n_targets_in_proteins", 0))

    logger.info("[*] Dimensions entering create_bounds:")
    logger.info("    K = %d proteins", dims.K)
    logger.info("    M = %d kinases", dims.M)
    logger.info("    N = %d phosphorylation sites", dims.N)

    logger.info("[*] Transcription factor network:")
    logger.info("    TF sources = %d", n_tf_sources)
    logger.info("    TF targets = %d", n_tf_targets)
    logger.info("    Sources in RNA = %d", n_sources_in_rna)
    logger.info("    Targets in RNA = %d", n_targets_in_rna)
    logger.info("    Sources in model proteins = %d", n_sources_in_proteins)
    logger.info("    Targets in model proteins = %d", n_targets_in_proteins)

    # logger.warning("[DEBUG] Stopping execution here.")
    # raise SystemExit(0)

    xl, xu, dim = create_bounds(
        dims.K,
        dims.M,
        dims.N,
        bounds=getattr(cfg, "bounds", None),
    )

    if xl is None or xu is None:
        raise RuntimeError(
            f"create_bounds() returned None for xl={xl} or xu={xu}. "
            f"ModelDims are K={dims.K}, M={dims.M}, N={dims.N}. "
            "This usually means ModelDims.set_dims() was called with zero dimensions — "
            "check that kinase_tsv/kea_ks_table loaded at least one site and one kinase."
        )

    xl = np.asarray(xl, dtype=float)
    xu = np.asarray(xu, dtype=float)

    assert np.all(np.isfinite(xl)) and np.all(np.isfinite(xu)), (
        f"create_bounds() returned non-finite values: xl={xl}, xu={xu}"
    )

    param_labels = _generate_param_labels(
        dims.K,
        dims.M,
        dims.N,
        proteins,
        kinases,
        sites,
    )

    xl_orig, xu_orig = bounds_to_original_scale(
        xl,
        xu,
        dims.K,
        dims.M,
        dims.N,
    )

    bounds_df = pd.DataFrame(
        {
            "parameter": param_labels,
            "optimizer_lower": xl,
            "optimizer_upper": xu,
            "original_lower": xl_orig,
            "original_upper": xu_orig,
        }
    )

    logger.header("Model parameters")

    logger.info(
        "[*] create_bounds → dim=%d\n%s",
        dim,
        bounds_df.to_string(index=False),
    )

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

    logger.success(f"[*] Saved snapshot pre-optimization")

    # 10. Build RNA-to-model-protein mapping (must happen BEFORE NetworkProblem)
    rna_fit_genes = []
    rna_obs_matched = None
    rna_model_prot_idx = None
    rna_obs_idx = None
    W_data_mrna_matched = None
    R_data0 = None

    if rna_matrix is not None and gene_ids is not None:
        # H5: Guard against shape mismatch between weight matrix and gene list.
        assert W_data_mrna.shape[0] == len(gene_ids), (
            f"W_data_mrna.shape[0]={W_data_mrna.shape[0]} != len(gene_ids)={len(gene_ids)}. "
            "This indicates a mismatch between the RNA weight matrix and the gene list."
        )

        (
            rna_fit_genes,
            rna_obs_matched_raw,
            rna_model_prot_idx_raw,
            rna_obs_idx_raw,
        ) = data_loader.match_rna_to_model_proteins(gene_ids, rna_matrix, proteins)

        if len(rna_fit_genes) > 0:
            rna_obs_matched = rna_obs_matched_raw
            rna_model_prot_idx = rna_model_prot_idx_raw
            rna_obs_idx = rna_obs_idx_raw
            W_data_mrna_matched = W_data_mrna[rna_obs_idx_raw, :]
            R_data0 = data_loader.build_full_R0(K, gene_ids, rna_matrix, proteins)
            logger.info(
                f"[*] RNA-to-model mapping: {len(rna_fit_genes)} matched genes/proteins."  # noqa: E501
            )
        else:
            logger.warning(
                "[!] No RNA genes matched model proteins. RNA loss disabled."
            )

    # 11. Optimisation
    logger.info(
        "[*] Optimizer backend: %s",
        getattr(args, "optimizer_backend", "optimistix"),
    )
    logger.info(
        f"[*] Initialising optimisation problem ({args.n_starts} starts, "
        f"max_steps={args.max_steps})"
    )

    problem = NetworkOptimizationProblem(
        dims,
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
        rna_obs_matched=rna_obs_matched,
        rna_model_prot_idx=rna_model_prot_idx,
        rna_obs_idx=rna_obs_idx,
        rna_fit_genes=rna_fit_genes,
        loss_weight_rna=args.loss_weight_mrna,
        R_data0=R_data0,
        rna_relax=cfg.derived_rates.rna_relax,
        W_data_mrna=W_data_mrna_matched if len(rna_fit_genes) > 0 else None,
        ode_solver_kind=args.ode_solver,
        ode_dt0=args.ode_dt0,
        ode_root_find_max_steps=args.ode_root_find_max_steps,
        ode_adjoint_kind=args.ode_adjoint,
        rtol=args.rtol,
        atol=args.atol,
        max_steps=args.solver_max_steps,
    )

    logger.success(f"[*] Network problem initialized successfully")

    # Store picklable rebuild kwargs on problem so that parallel multi-start
    # workers can reconstruct k_act_fn / s_prod_fn inside the worker process
    # without receiving non-picklable JAX closures across process boundaries.
    # Keys mirror the parameter names of make_k_act_fn() / make_s_prod_fn()
    # in derived_rates.py exactly; changing those signatures requires updating
    # these dicts to match.
    problem._k_act_rebuild_kwargs = {
        "t_rna": t_rna,
        "rna_data": rna_matrix,  # (n_genes, T_rna) or None
        "tf_prot_weights": tf_prot_weights,  # (K, n_genes) or None
        "K": K,
        "interp_mode": interp_mode,
        "protein_self_rna_idx": entity_masks["protein_self_rna_idx"],
    }
    problem._s_prod_rebuild_kwargs = {
        "t_protein": t,  # (T,) protein time points
        "Y_data": P_scaled,  # (N_sites, T) phospho data
        "R_kin_site": R,  # (M, N_sites) kinase-to-site weights
        "kin_to_prot_idx": kin_to_prot_idx,  # (M,) kinase→protein mapping
        "K": K,
        "M": M,
        "s_prod_fn_type": s_prod_fn_type,
        "interp_mode": interp_mode,
    }

    # Validate problem shapes before starting optimization
    try:
        validate_problem_shapes(problem)
        logger.success("[*] Problem shape validation passed.")
    except ValueError as e:
        logger.warning(f"[!] Problem shape validation warnings:\n{e}")

    # ------------------------------------------------------------------
    # Optimisation: multi-start LM via run_multi_start_optimization
    # ------------------------------------------------------------------
    res, best_idx, total_losses = run_multi_start_optimization(
        problem, args, P_scaled
    )

    F, X = res.F, res.X
    theta_best = X[best_idx]

    # 11. Analysis & Saving
    f1 = F[:, 0]
    f2 = F[:, 1]
    f3 = F[:, 2]
    # L1: f4 (RNA loss) is in column 3 when present; assign it properly.
    f4 = F[:, 3] if F.shape[1] > 3 else np.zeros(len(f1))

    analysis.save_derived_rates(
        outdir=outdir,
        proteins=proteins,
        t_protein=t,
        k_act_fn=k_act_fn,
        s_prod_fn=s_prod_fn,
        t_rna=t_rna,
    )
    logger.success(f"[*] Derived rates saved successfully")

    analysis.save_run_results(outdir, F, X, f1, f2, f3, total_losses, F[best_idx], f4=f4)
    logger.success(f"[*] Run results saved successfully")
    analysis.plot_run_diagnostics(outdir, F, F[best_idx], f1, f2, f3, X, f4=f4)
    logger.success(f"[*] Run diagnostics plotted successfully")

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
        dims=dims,
        k_act_fn=k_act_fn,
        s_prod_fn=s_prod_fn,
        R_data0=R_data0,
        kinases=kinases,
        simulation_cfg=getattr(cfg, "simulation", None),
        data_interpolation_cfg=getattr(cfg, "data_interpolation", None),
        t_rna=t_rna,
    )
    logger.success(f"[*] Fitted simulation saved successfully")

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

    analysis.plot_fitted_simulation(outdir)
    analysis.print_parameter_summary(outdir, theta_best, proteins, kinases, sites, dims=dims)
    analysis.print_biological_scores(outdir, X)
    analysis.plot_biological_scores(outdir, X, F)
    analysis.plot_goodness_of_fit(f"{outdir}/fit_timeseries.tsv", outdir)

    if args.run_steadystate:
        _ss = getattr(cfg, "steadystate", None)
        steadystate.run_steadystate_analysis(
            outdir=outdir,
            dims=dims,
            problem=problem,
            theta_opt=theta_best,
            sites=sites,
            proteins=proteins,
            kinases=kinases,
            t_end=getattr(_ss, "t_end", 2000.0),
            early_end=getattr(_ss, "early_end", 100.0),
            n_early=getattr(_ss, "n_early", 100),
            n_late=getattr(_ss, "n_late", 80),
            late_grid=getattr(_ss, "late_grid", "geomspace"),
            rtol=getattr(_ss, "rtol", 1e-6),
            atol=getattr(_ss, "atol", 1e-8),
            dt0=getattr(_ss, "dt0", 0.1),
            max_steps=getattr(_ss, "max_steps", 131072),
            top_n=getattr(_ss, "top_n", 10),
            skip_plots_on_nonfinite=getattr(_ss, "skip_plots_on_nonfinite", True),
            strict=getattr(_ss, "strict", False),
            use_event=getattr(_ss, "use_event", True),
            event_rtol=getattr(_ss, "event_rtol", None),
            event_atol=getattr(_ss, "event_atol", None),
        )

    if args.run_knockouts:
        knockouts.run_knockout_screen(
            outdir, dims, problem, theta_best, sites, proteins, kinases
        )

    if args.run_sensitivity:
        bounds = (xl, xu)
        run_global_sensitivity(
            outdir,
            dims,
            problem,
            bounds,
            proteins=proteins,
            kinases=kinases,
            sites=sites,
        )

    # 12. Provenance & exports
    save_run_metadata(outdir, dims, args)
    export_network_for_cytoscape(
        outdir, dims, theta_best, proteins, kinases, sites, K_site_kin, site_prot_idx
    )

    P_best = problem.simulate(theta_best)
    plot_residual_heatmap(outdir, P_scaled, P_best, sites, t)

    p_labels = _generate_param_labels(dims.K, dims.M, dims.N, proteins, kinases, sites)
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

    # 13. Optional post-fit neural latent-rate refinement
    _neural_cfg = getattr(cfg, "neural_ode", None)
    if _neural_cfg is not None and getattr(_neural_cfg, "enabled", False):
        from phoscrosstalk.neuralODE import run_neural_latent_rate_refinement, save_neural_ode_plots  # noqa: PLC0415

        logger.header("[*] Running post-fit neural latent-rate refinement")
        jaxpr_out_dir = None
        if getattr(cfg.debug, "save_jaxpr_reports", False):
            jaxpr_out_dir = str(pathlib.Path(outdir) / "jaxpr_reports")

        _neural_ts, _neural_ys, _neural_model, _neural_loss_hist, _neural_time_hist = run_neural_latent_rate_refinement(
            dims=dims,
            problem=problem,
            theta_best=theta_best,
            k_act_fn=k_act_fn,
            s_prod_fn=s_prod_fn,
            t=t,
            P_scaled=P_scaled,
            A_scaled=A_scaled,
            prot_idx_for_A=prot_idx_for_A,
            W_data=W_data,
            W_data_prot=W_data_prot,
            proteins=proteins,
            sites=sites,
            kinases=kinases,
            t_rna=t_rna if rna_matrix is not None else None,
            rna_obs_matched=rna_obs_matched,
            rna_model_prot_idx=rna_model_prot_idx,
            W_data_mrna_matched=W_data_mrna_matched if len(rna_fit_genes) > 0 else None,
            outdir=outdir,
            neural_cfg=_neural_cfg,
            mechanism=mechanism,
            rna_relax=cfg.derived_rates.rna_relax,
            abundance_max=getattr(getattr(cfg, "bounds", None), "abundance_max", 5.0),
            R_data0=R_data0,
            jaxpr_out_dir=jaxpr_out_dir,
        )
        save_neural_ode_plots(
            outdir=os.path.join(outdir, "neural_ode"),
            ts=_neural_ts,
            ys=_neural_ys,
            model=_neural_model,
            loss_history=_neural_loss_hist,
            time_history=_neural_time_hist,
        )
        logger.info("[*] Neural ODE visualisations saved to %s/neural_ode", outdir)

    logger.success("[*] Done.")


if __name__ == "__main__":
    main()


def cli():
    main()
