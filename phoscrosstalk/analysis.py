"""
analysis.py
Post-optimization analysis, file export, and plotting.
"""

import os

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from phoscrosstalk.config import DEFAULT_TIMEPOINTS, ModelDims
from phoscrosstalk.mechanisms import decode_theta
from phoscrosstalk.logger import get_logger
from phoscrosstalk.optimization import bio_score, build_full_A0
from phoscrosstalk.simulation import simulate, simulate_dense

logger = get_logger(__name__)


def save_run_results(outdir, F, X, f1, f2, f3, J, F_best, f4=None):
    """
    Save multi-start optimization results, including loss component statistics and
    total loss scores, to disk.

    Args:
        outdir (str): Path to the output directory.
        F (np.ndarray): Array of loss components for all solutions
            (shape: n_solutions x 3 or n_solutions x 4 when RNA loss is included).
        X (np.ndarray): Array of parameter values for all solutions.
        f1 (np.ndarray): Phosphosite relative-signal loss component for each solution.
        f2 (np.ndarray): Protein abundance error component for each solution.
        f3 (np.ndarray): Regularization component for each solution.
        J (np.ndarray): Total loss per solution, used for model selection.
        F_best (np.ndarray): The loss components for the best solution.
        f4 (np.ndarray | None): mRNA / R_rna state loss component for each solution.
            Unused directly (f4 is already embedded in column 3 of *F* when present)
            but accepted for a consistent call-site signature.

    Returns:
        None: Files are written to `outdir`.  Output filenames are preserved for
              backward compatibility with external readers (pareto_stats.tsv,
              pareto_front_with_J.tsv, pareto_points.tsv, pareto_front.npz).
    """
    F_arr = np.asarray(F)
    n_obj = F_arr.shape[1] if F_arr.ndim == 2 else 3
    if n_obj == 4:
        obj_names = ["f1_P_sites", "f2_protein", "f3_complexity", "f4_mrna"]
    else:
        obj_names = ["f1_P_sites", "f2_protein", "f3_complexity"]

    # DataFrame stats
    df_F = pd.DataFrame(F, columns=obj_names)
    summary = pd.DataFrame(
        {
            "objective": obj_names,
            "min": df_F.min().values,
            "mean": df_F.mean().values,
            "median": df_F.median().values,
            "std": df_F.std(ddof=1).values,
        }
    )

    summary.to_csv(os.path.join(outdir, "pareto_stats.tsv"), sep="\t", index=False)

    df_front = df_F.copy()
    df_front["J_scalarized"] = J
    df_front.to_csv(
        os.path.join(outdir, "pareto_front_with_J.tsv"), sep="\t", index=False
    )

    bio_scores = np.array([bio_score(theta) for theta in X])
    df_front["bio_score"] = bio_scores
    df_front.to_csv(os.path.join(outdir, "pareto_points.tsv"), sep="\t", index=False)

    np.savez(os.path.join(outdir, "pareto_front.npz"), F=F, X=X, J=J)


# Backward-compatibility alias
save_pareto_results = save_run_results


def plot_run_diagnostics(outdir, F, F_best, f1, f2, f3, X, f4=None):
    """
    Plot diagnostic visualizations for the multi-start optimization results.

    Generates a scatter plot of loss component space (f1 vs f2 colored by f3)
    with the selected best solution highlighted, and a heatmap of parameter
    correlations across all runs.  When *f4* is provided (RNA loss enabled),
    an additional panel showing f1 vs f4 is saved alongside.

    Args:
        outdir (str): Path to the output directory.
        F (np.ndarray): Loss components for all solutions.
        F_best (np.ndarray): Loss components of the selected best solution.
        f1 (np.ndarray): Phosphosite relative-signal loss component for each solution.
        f2 (np.ndarray): Protein abundance error component for each solution.
        f3 (np.ndarray): Regularization component for each solution.
        X (np.ndarray): Parameter values for all solutions.
        f4 (np.ndarray | None): mRNA / R_rna state loss component for each solution.
            When *None* (or all-zero), the RNA panel is skipped.

    Returns:
        None: Saves 'pareto_f1_f2.png', 'pareto_param_corr.png', and
              (when f4 is non-trivial) 'pareto_f1_f4.png' to `outdir`.
              Output filenames are preserved for backward compatibility.
    """
    # F1 vs F2
    plt.figure(figsize=(7, 6))
    sc = plt.scatter(f1, f2, c=f3, cmap="viridis", alpha=0.7)
    plt.colorbar(sc, label="f3")
    plt.scatter(
        F_best[0], F_best[1], s=120, facecolors="none", edgecolors="red", linewidths=2
    )
    plt.title("Optimization Results: f1 vs f2")
    plt.savefig(os.path.join(outdir, "pareto_f1_f2.png"), dpi=300)
    plt.close()

    # f1 vs f4 panel (only when RNA data were used, i.e. f4 is non-trivially non-zero)
    f4_arr = np.asarray(f4) if f4 is not None else None
    if f4_arr is not None and np.any(f4_arr > 0):
        plt.figure(figsize=(7, 6))
        sc4 = plt.scatter(f1, f4_arr, c=f3, cmap="viridis", alpha=0.7)
        plt.colorbar(sc4, label="f3 (regularisation)")
        if len(F_best) > 3:
            plt.scatter(
                F_best[0],
                F_best[3],
                s=120,
                facecolors="none",
                edgecolors="red",
                linewidths=2,
            )
        plt.xlabel("f1 (phosphosite loss)")
        plt.ylabel("f4 (mRNA loss)")
        plt.title("Optimization Results: f1 vs f4 (RNA)")
        plt.savefig(os.path.join(outdir, "pareto_f1_f4.png"), dpi=300)
        plt.close()

    # Param Correlation
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(np.corrcoef(X.T), ax=ax, cmap="coolwarm", center=0.0, square=True)
    fig.savefig(os.path.join(outdir, "pareto_param_corr.png"), dpi=300)
    plt.close(fig)


# Backward-compatibility alias
plot_pareto_diagnostics = plot_run_diagnostics


def print_parameter_summary(outdir, theta_opt, proteins, kinases, sites):
    """
    Decode optimized parameters and export summaries for proteins, kinases, and sites.

    Args:
        outdir (str): Path to the output directory.
        theta_opt (np.ndarray): The optimized parameter vector.
        proteins (list): List of protein names.
        kinases (list): List of kinase names.
        sites (list): List of phosphorylation site names.

    Returns:
        None: Writes summary TSV/TXT files to `outdir` and prints summaries to console.
    """
    K, M, N = ModelDims.K, ModelDims.M, ModelDims.N
    params_decoded = decode_theta(theta_opt, K, M, N)

    # Protein-specific parameters (k_act and s_prod are now derived, not fitted)
    df_prot = pd.DataFrame(
        {
            "Protein": proteins,
            "k_deact (Deactivation)": params_decoded[0],
            "d_deg (Degradation)": params_decoded[1],
        }
    )
    df_prot.to_csv(
        os.path.join(outdir, "parameter_summary_proteins.tsv"), sep="\t", index=False
    )

    # Kinase-specific parameters
    df_kin = pd.DataFrame(
        {
            "Kinase": kinases,
            "Alpha (Global Str)": params_decoded[4],
            "kK_act (Kinase Act)": params_decoded[5],
            "kK_deact (Kinase Deact)": params_decoded[6],
        }
    )
    df_kin.to_csv(
        os.path.join(outdir, "parameter_summary_kinases.tsv"), sep="\t", index=False
    )

    # Site-specific parameters
    df_site = pd.DataFrame(
        {"Site": sites, "k_off (Phosphatase Rate)": params_decoded[7]}
    )
    df_site.to_csv(
        os.path.join(outdir, "parameter_summary_sites.tsv"), sep="\t", index=False
    )

    # Global parameters
    with open(os.path.join(outdir, "parameter_summary_global.txt"), "w") as f:
        f.write("=== Global Coupling Parameters ===\n")
        f.write(f"beta_g (Global Coupling): {params_decoded[2]:.5f}\n")
        f.write(f"beta_l (Local Coupling):  {params_decoded[3]:.5f}\n")
        f.write("-" * 40 + "\n")
        f.write(
            "Note: k_act and s_prod are derived from TF/kinase signals, not fitted.\n"
        )

    # Print the summary to console as well
    logger.info("=== Parameter Summary ===")
    logger.header("\n--- Protein-specific Parameters ---")
    logger.info(df_prot.to_string(index=False))
    logger.header("\n--- Kinase-specific Parameters ---")
    logger.info(df_kin.to_string(index=False))
    logger.header("\n--- Site-specific Parameters ---")
    logger.info(df_site.to_string(index=False))
    logger.header("\n--- Global Coupling Parameters ---")
    logger.info(f"beta_g (Global Coupling): {params_decoded[2]:.5f}")
    logger.info(f"beta_l (Local Coupling):  {params_decoded[3]:.5f}")
    logger.info("-" * 40 + "\n")


def _save_dense_simulation(
    outdir,
    theta_opt,
    t_obs,
    sites,
    proteins,
    P_scaled,
    A_scaled,
    prot_idx_for_A,
    Cg,
    Cl,
    site_prot_idx,
    K_site_kin,
    R,
    L_alpha,
    kin_to_prot_idx,
    mask_p,
    mask_k,
    mechanism,
    k_act_fn=None,
    s_prod_fn=None,
    R_data0=None,
    n_dense: int = 200,
    interpolation_label: str = "diffrax_dense",
    data_interp_P=None,
    data_interp_A=None,
    prot_idx_for_A_full=None,
):
    """Run a dense-grid simulation and save long-format output for visualisation.

    Runs an additional forward ODE simulation over a uniform grid of *n_dense*
    points spanning ``[0, max(t_obs)]`` using :func:`simulate_dense` and writes
    ``fit_timeseries_dense.tsv`` in a long format suitable for the dashboard.

    This function is called by ``save_fitted_simulation`` and does **not** affect
    the optimisation objective or loss values.  It is best-effort; failures are
    caught and warned rather than propagated.

    When *data_interp_P* or *data_interp_A* are provided (callables built by
    :func:`~phoscrosstalk.derived_rates.build_data_interpolations`), the dense
    output also includes ``series_type = "observed_interpolated_dense"`` rows for
    diagnostic comparison.  These rows are clearly labelled and must not be
    confused with measured data.

    Output columns: entity_type, entity, site, protein, time, value, series_type,
                    source, interpolation_method

    Recommended ``series_type`` values used here:

    * ``"simulated_dense"``           – ODE simulation on dense grid (model output)
    * ``"observed_interpolated_dense"`` – interpolated from sparse observed data
                                          (diagnostic only, not training data)
    """
    K, M, N = ModelDims.K, ModelDims.M, ModelDims.N
    t_max = float(np.nanmax(t_obs)) if len(t_obs) > 0 else 1.0
    t_dense = np.linspace(0.0, t_max, n_dense)

    # Build initial protein abundance matrix for the dense simulation.
    # We only need the initial column (t=0) for the IC.
    if A_scaled is not None and np.asarray(A_scaled).size > 0:
        A_scaled_initial = np.asarray(A_scaled, dtype=float)[:, :1]
    else:
        A_scaled_initial = np.zeros((0, 1), dtype=float)

    A0_full = build_full_A0(K, 1, A_scaled_initial, prot_idx_for_A)

    dense_result = simulate_dense(
        t_dense=t_dense,
        P_data0=P_scaled,
        A_data0=A0_full,
        theta=theta_opt,
        Cg=Cg,
        Cl=Cl,
        site_prot_idx=site_prot_idx,
        K_site_kin=K_site_kin,
        R=R,
        L_alpha=L_alpha,
        kin_to_prot_idx=kin_to_prot_idx,
        receptor_mask_prot=mask_p,
        receptor_mask_kin=mask_k,
        mechanism=mechanism,
        k_act_fn=k_act_fn,
        s_prod_fn=s_prod_fn,
        R_data0=R_data0,
    )

    P_sim_d = dense_result.get("P_sim", np.full((len(sites), n_dense), np.nan))
    A_sim_d = dense_result.get("A_sim", np.full((K, n_dense), np.nan))
    R_sim_d = dense_result.get("R_sim", np.full((K, n_dense), np.nan))

    rows = []

    # Phosphosite dense simulation
    for i, site in enumerate(sites):
        parts = site.split("_", 1)
        prot = parts[0]
        s = parts[1] if len(parts) > 1 else ""
        for j, ti in enumerate(t_dense):
            rows.append(
                {
                    "entity_type": "Phosphosite",
                    "entity": site,
                    "site": s,
                    "protein": prot,
                    "time": float(ti),
                    "value": float(P_sim_d[i, j]),
                    "series_type": "simulated_dense",
                    "source": "model",
                    "interpolation_method": interpolation_label,
                }
            )

    # Protein abundance dense simulation
    for p_idx in range(K):
        prot = proteins[p_idx]
        for j, ti in enumerate(t_dense):
            rows.append(
                {
                    "entity_type": "ProteinAbundance",
                    "entity": prot,
                    "site": "",
                    "protein": prot,
                    "time": float(ti),
                    "value": float(A_sim_d[p_idx, j]),
                    "series_type": "simulated_dense",
                    "source": "model",
                    "interpolation_method": interpolation_label,
                }
            )

    # Observed interpolated dense rows (diagnostic only, not training data).
    # Only included when data interpolation callables are provided.
    if data_interp_P is not None:
        for i, site in enumerate(sites):
            parts = site.split("_", 1)
            prot = parts[0]
            s = parts[1] if len(parts) > 1 else ""
            try:
                # data_interp_P(t_dense) returns (N_sites, T_dense) when t_dense is
                # an array, so we index as interp_vals[i, :].
                interp_vals = np.asarray(data_interp_P(t_dense), dtype=float)
                if interp_vals.ndim == 2:
                    site_vals = interp_vals[i, :]
                else:
                    # Fallback for callables returning flat arrays (single-site edge case)
                    site_vals = interp_vals
                for j, ti in enumerate(t_dense):
                    rows.append(
                        {
                            "entity_type": "Phosphosite",
                            "entity": site,
                            "site": s,
                            "protein": prot,
                            "time": float(ti),
                            "value": float(site_vals[j]),
                            "series_type": "observed_interpolated_dense",
                            "source": "data_interpolation",
                            "interpolation_method": "data_interp",
                        }
                    )
            except Exception:
                pass

    if data_interp_A is not None and prot_idx_for_A_full is not None:
        try:
            # data_interp_A(t_dense) returns (K_obs, T_dense) when t_dense is an array
            interp_A_vals = np.asarray(data_interp_A(t_dense), dtype=float)
            for k_obs, p_idx in enumerate(prot_idx_for_A_full):
                prot = proteins[p_idx]
                if interp_A_vals.ndim == 2:
                    row_vals = interp_A_vals[k_obs, :]
                else:
                    row_vals = interp_A_vals
                for j, ti in enumerate(t_dense):
                    rows.append(
                        {
                            "entity_type": "ProteinAbundance",
                            "entity": prot,
                            "site": "",
                            "protein": prot,
                            "time": float(ti),
                            "value": float(row_vals[j]),
                            "series_type": "observed_interpolated_dense",
                            "source": "data_interpolation",
                            "interpolation_method": "data_interp",
                        }
                    )
        except Exception:
            pass

    df_dense = pd.DataFrame(rows)
    df_dense.to_csv(
        os.path.join(outdir, "fit_timeseries_dense.tsv"), sep="\t", index=False
    )
    logger.info(f"[*] Dense simulation output saved ({len(t_dense)} time points).")

    # Dense mRNA / R_rna output (model only, simulated_dense series_type).
    # Saved to mrna_fit_timeseries_dense.tsv when R_sim is available.
    try:
        if R_sim_d is not None and np.any(np.isfinite(R_sim_d)):
            mrna_rows = []
            for p_idx in range(K):
                prot = proteins[p_idx]
                for j, ti in enumerate(t_dense):
                    mrna_rows.append(
                        {
                            "gene": prot,
                            "time": float(ti),
                            "value": float(R_sim_d[p_idx, j]),
                            "series_type": "simulated_dense",
                            "source": "model",
                            "interpolation_method": interpolation_label,
                        }
                    )
            df_mrna_dense = pd.DataFrame(mrna_rows)
            df_mrna_dense.to_csv(
                os.path.join(outdir, "mrna_fit_timeseries_dense.tsv"),
                sep="\t",
                index=False,
            )
            logger.info("[*] Dense mRNA simulation output saved.")
    except Exception as exc:
        logger.warning(f"[!] Dense mRNA output skipped: {exc}")


def save_fitted_simulation(
    outdir,
    theta_opt,
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
    mask_p,
    mask_k,
    k_act_fn=None,
    s_prod_fn=None,
    R_data0=None,
    kinases=None,
    simulation_cfg=None,
    data_interpolation_cfg=None,
):
    """
    Run a simulation with optimized parameters,
    rescale outputs, and save comparison data.

    This function simulates the model using `theta_opt`, rescales the results to match
    experimental data units (using baselines/amplitudes), and aggregates both simulated
    and observed data into a single long-format time-series file.

    Args:
        outdir (str): Output directory path.
        theta_opt (np.ndarray): Optimized parameter vector.
        t (np.ndarray): Time points vector.
        sites (list): List of phosphosite labels.
        proteins (list): List of protein labels.
        P_scaled (np.ndarray): Scaled phosphodata (input/reference).
        A_scaled (np.ndarray): Scaled protein data (input/reference).
        prot_idx_for_A (list): Indices of proteins that have abundance data.
        baselines (np.ndarray): Baselines for rescaling phosphosites.
        amplitudes (np.ndarray): Amplitudes for rescaling phosphosites.
        Y (np.ndarray): Original phosphosite data (used for direct comparison).
        A_data (np.ndarray): Original protein abundance data.
        A_bases (np.ndarray): Baselines for rescaling proteins.
        A_amps (np.ndarray): Amplitudes for rescaling proteins.
        mechanism (str): The specific kinetic mechanism used in simulation.
        Cg, Cl (np.ndarray): Global and Local connectivity matrices.
        site_prot_idx (np.ndarray): Mapping of sites to proteins.
        K_site_kin (np.ndarray): Kinase-site interaction matrix.
        R (np.ndarray): Receptor/Input matrix.
        L_alpha (np.ndarray): Laplacian or interaction matrix for alpha term.
        kin_to_prot_idx (np.ndarray): Mapping of kinases to protein indices.
        mask_p, mask_k (np.ndarray): Boolean masks for proteins and kinases.
        kinases (list | None): List of kinase names.  When provided, Kdyn_sim
            rows in ``internal_states.tsv`` use real kinase names instead of
            generic ``Kinase_0, Kinase_1, …`` labels.
        simulation_cfg (SimpleNamespace | None): Optional ``[simulation]`` config
            section.  Controls ``dense_n_points``, ``save_dense``, and
            ``dense_interpolation`` label.  Defaults are used when None.
        data_interpolation_cfg (SimpleNamespace | None): Optional
            ``[data_interpolation]`` config section.  When
            ``data_interpolation_cfg.enabled`` is True, diagnostic interpolated
            observed curves are added to ``fit_timeseries_dense.tsv``.  The
            original sparse observed arrays in the loss are never modified.

    Returns:
        None: Saves 'fitted_params.npz' and 'fit_timeseries.tsv' to `outdir`.
    """
    K, M, N = ModelDims.K, ModelDims.M, ModelDims.N

    # Save Params – k_act and s_prod are derived quantities, not fitted
    params_decoded = decode_theta(theta_opt, K, M, N)
    param_names = [
        "k_deact",
        "d_deg",
        "beta_g",
        "beta_l",
        "alpha",
        "kK_act",
        "kK_deact",
        "k_off",
        "gamma_S_p",
        "gamma_A_S",
        "gamma_A_p",
        "gamma_K_net",
    ]
    save_dict = {
        "theta": theta_opt,
        "proteins": np.array(proteins),
        "sites": np.array(sites),
    }
    for name, val in zip(param_names, params_decoded, strict=True):
        save_dict[name] = val
    np.savez(os.path.join(outdir, "fitted_params.npz"), **save_dict)

    # Re-simulate
    A0_full = build_full_A0(K, len(t), A_scaled, prot_idx_for_A)

    P_sim, A_sim, S_sim, Kdyn_sim = simulate(
        t,
        P_scaled,
        A0_full,
        theta_opt,
        Cg,
        Cl,
        site_prot_idx,
        K_site_kin,
        R,
        L_alpha,
        kin_to_prot_idx,
        mask_p,
        mask_k,
        mechanism,
        full_output=True,
        k_act_fn=k_act_fn,
        s_prod_fn=s_prod_fn,
        R_data0=R_data0,
    )

    # Rescale Sites (model)
    Y_sim_rescaled = np.zeros_like(P_sim)
    for i in range(len(sites)):
        Y_sim_rescaled[i] = baselines[i] + amplitudes[i] * P_sim[i]

    # Rescale Sites (data) – from original Y or from P_scaled if you prefer
    Y_data_rescaled = Y  # if Y is already in FC units
    # or:
    # Y_data_rescaled = np.zeros_like(P_scaled)
    # for i in range(len(sites)):
    #     Y_data_rescaled[i] = baselines[i] + amplitudes[i] * P_scaled[i]

    # Rescale Proteins
    A_sim_rescaled = A_sim.copy()
    if A_scaled.size > 0:
        for k, p_idx in enumerate(prot_idx_for_A):
            A_sim_rescaled[p_idx] = A_bases[k] + A_amps[k] * A_sim[p_idx]

    # Column names – USE INDICES, NOT int(time)
    T = len(t)
    sim_cols = [f"sim_t{j}" for j in range(T)]
    data_cols = [f"data_t{j}" for j in range(T)]

    records = []

    # Phosphosites
    for i, site in enumerate(sites):
        prot, residue = site.split("_", 1)
        record = {
            "Type": "Phosphosite",
            "Protein": prot,
            "Residue": residue,
        }
        for j in range(T):
            record[sim_cols[j]] = Y_sim_rescaled[i, j]
            record[data_cols[j]] = Y_data_rescaled[i, j]
        records.append(record)

    # Proteins
    if A_data is not None and A_data.size > 0:
        for k, p_idx in enumerate(prot_idx_for_A):
            prot = proteins[p_idx]
            record = {
                "Type": "ProteinAbundance",
                "Protein": prot,
                "Residue": "",
            }
            for j in range(T):
                record[sim_cols[j]] = A_sim_rescaled[p_idx, j]
                record[data_cols[j]] = A_data[k, j]
            records.append(record)

    df_out = pd.DataFrame.from_records(records)
    df_out.to_csv(os.path.join(outdir, "fit_timeseries.tsv"), sep="\t", index=False)

    # Dense continuous output for smooth dashboard visualisation.
    # Runs a separate forward simulation over a fine time grid [0, t_max] and
    # saves the result in long format to fit_timeseries_dense.tsv.
    # This does NOT affect optimisation or loss computation.
    _sim_cfg = simulation_cfg
    _do_dense = _sim_cfg is None or getattr(_sim_cfg, "save_dense", True)
    _n_dense = int(getattr(_sim_cfg, "dense_n_points", 200)) if _sim_cfg else 200
    _dense_label = getattr(_sim_cfg, "dense_interpolation", "diffrax_dense") if _sim_cfg else "diffrax_dense"

    if _do_dense:
        # Optionally build diagnostic data interpolation callables.
        _data_interp_P = None
        _data_interp_A = None
        _di_cfg = data_interpolation_cfg
        if _di_cfg is not None and getattr(_di_cfg, "enabled", False):
            try:
                from phoscrosstalk.derived_rates import build_data_interpolations
                _di_method = getattr(_di_cfg, "method", "linear")
                _di_fwd = getattr(_di_cfg, "fill_forward_nans_at_end", False)
                _di_start = getattr(_di_cfg, "replace_nans_at_start", None)
                _interp_result = build_data_interpolations(
                    t_obs=t,
                    P_data=P_scaled,
                    A_data=A_scaled if (A_scaled is not None and np.asarray(A_scaled).size > 0) else None,
                    method=_di_method,
                    fill_forward_nans_at_end=_di_fwd,
                    replace_nans_at_start=_di_start,
                )
                _data_interp_P = _interp_result.get("P_interp")
                _data_interp_A = _interp_result.get("A_interp")
                for msg in _interp_result.get("nan_fill_log", []):
                    logger.info(f"[data_interp]{msg}")
            except Exception as exc:
                logger.warning(f"[!] Data interpolation build failed: {exc}")

        try:
            _save_dense_simulation(
                outdir=outdir,
                theta_opt=theta_opt,
                t_obs=t,
                sites=sites,
                proteins=proteins,
                P_scaled=P_scaled,
                A_scaled=A_scaled,
                prot_idx_for_A=prot_idx_for_A,
                Cg=Cg,
                Cl=Cl,
                site_prot_idx=site_prot_idx,
                K_site_kin=K_site_kin,
                R=R,
                L_alpha=L_alpha,
                kin_to_prot_idx=kin_to_prot_idx,
                mask_p=mask_p,
                mask_k=mask_k,
                mechanism=mechanism,
                k_act_fn=k_act_fn,
                s_prod_fn=s_prod_fn,
                R_data0=R_data0,
                n_dense=_n_dense,
                interpolation_label=_dense_label,
                data_interp_P=_data_interp_P,
                data_interp_A=_data_interp_A,
                prot_idx_for_A_full=prot_idx_for_A,
            )
        except Exception as exc:  # pragma: no cover – dense output is best-effort
            logger.warning(f"[!] Dense simulation output skipped: {exc}")

    records_internal = []
    T = len(t)
    cols = [f"t{j}" for j in range(T)]

    # S_sim (Protein Activity)
    for k, prot in enumerate(proteins):
        rec = {"Type": "S_sim", "ID": prot}
        for j in range(T):
            rec[cols[j]] = S_sim[k, j]
        records_internal.append(rec)

    # Kdyn_sim (Kinase Activity) – use real kinase names when available
    for m in range(M):
        kin_name = (
            kinases[m] if (kinases is not None and m < len(kinases)) else f"Kinase_{m}"
        )
        rec = {"Type": "Kdyn_sim", "ID": kin_name}
        for j in range(T):
            rec[cols[j]] = Kdyn_sim[m, j]
        records_internal.append(rec)

    df_int = pd.DataFrame.from_records(records_internal)
    df_int.to_csv(os.path.join(outdir, "internal_states.tsv"), sep="\t", index=False)

    # Plot the internal states
    plot_internal_states(outdir, t, S_sim, Kdyn_sim, proteins, kinases=kinases)


def plot_internal_states(outdir, t, S_sim, Kdyn_sim, proteins, kinases=None):
    """
    Plot the internal states (S_sim and Kdyn_sim) for comparison with experimental data.

    Args:
        outdir (str): Output directory for saving plots.
        t (np.ndarray): Time points.
        S_sim (np.ndarray): Simulated protein activity.
        Kdyn_sim (np.ndarray): Simulated kinase activity.
        proteins (list): List of protein names.
        kinases (list | None): List of kinase names.  When provided, legend
            labels use real kinase names instead of generic ``Kinase_N``.

    Returns:
        None
    """

    fig, (axS, axK) = plt.subplots(1, 2, figsize=(16, 6), sharex=True)

    # Panel 1: S_sim (Substrate Active Fraction)
    # Plot top changing ones to avoid clutter
    s_range = np.ptp(S_sim, axis=1)
    top_s = np.argsort(s_range)[-10:]  # Top 10 dynamic

    for idx in top_s:
        axS.plot(t, S_sim[idx], label=proteins[idx], lw=2, alpha=0.8)

    axS.set_title("Protein Active Fraction ($S_{sim}$)")
    axS.set_xlabel("Time (min)")
    axS.set_ylabel("Fraction Active [0-1]")
    axS.legend(fontsize=8, loc="upper right")
    axS.grid(alpha=0.3)

    # Panel 2: Kdyn_sim (Kinase Active Fraction)
    k_range = np.ptp(Kdyn_sim, axis=1)
    top_k = np.argsort(k_range)[-10:]

    for idx in top_k:
        kin_label = (
            kinases[idx]
            if (kinases is not None and idx < len(kinases))
            else f"Kinase_{idx}"
        )
        axK.plot(t, Kdyn_sim[idx], label=kin_label, lw=2, alpha=0.8)

    axK.set_title("Kinase Active Fraction ($K_{dyn}$)")
    axK.set_xlabel("Time (min)")
    axK.set_ylabel("Fraction Active [0-1]")
    axK.legend(fontsize=8, loc="upper right")
    axK.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "fitted_internal_states.png"), dpi=300)
    plt.close()


def plot_fitted_simulation(outdir):
    """
    Generate per-protein plots comparing simulated trajectories to experimental data.

    Reads ``fit_timeseries.tsv`` from *outdir*. When ``mrna_fit_timeseries.tsv``
    is also present, creates three-panel figures (mRNA / protein abundance /
    phosphosites); otherwise creates two-panel figures (protein abundance /
    phosphosites).

    Args:
        outdir (str): Directory containing output TSV files.

    Returns:
        None: Saves per-protein PNG files to *outdir*.
    """

    # Load protein/phosphosite fit data
    ts_path = os.path.join(outdir, "fit_timeseries.tsv")
    if not os.path.exists(ts_path):
        logger.warning(f"[!] fit_timeseries.tsv not found in {outdir}; skipping.")
        return

    df = pd.read_csv(ts_path, sep="\t")
    proteins = sorted(df["Protein"].unique())
    logger.info(f"[*] Found {len(proteins)} proteins")
    df_sites = df[df["Type"] == "Phosphosite"].reset_index(drop=True)
    df_prots = df[df["Type"] == "ProteinAbundance"].reset_index(drop=True)

    sim_cols = [col for col in df.columns if col.startswith("sim_t")]
    data_cols = [col for col in df.columns if col.startswith("data_t")]
    t_vals = DEFAULT_TIMEPOINTS

    if len(sim_cols) != len(t_vals) or len(data_cols) != len(t_vals):
        t_vals = np.arange(len(sim_cols), dtype=float)

    # Optionally load mRNA fit data
    mrna_path = os.path.join(outdir, "mrna_fit_timeseries.tsv")
    has_rna_data = os.path.exists(mrna_path)
    df_mrna = None
    if has_rna_data:
        try:
            df_mrna = pd.read_csv(mrna_path, sep="\t")
        except Exception:
            df_mrna = None
            has_rna_data = False
    if has_rna_data and (df_mrna is None or df_mrna.empty):
        has_rna_data = False

    # Plot per protein
    for prot in proteins:
        logger.info(f"   → Plotting {prot}")

        # Determine number of panels
        has_rna_for_prot = (
            has_rna_data and df_mrna is not None and prot in df_mrna["gene"].values
        )
        n_panels = 3 if has_rna_for_prot else 2
        color = plt.cm.tab10(proteins.index(prot) % 10)

        fig, axes = plt.subplots(
            1,
            n_panels,
            figsize=(9 * n_panels, 7),
            gridspec_kw={"wspace": 0.12},
            constrained_layout=True,
        )
        axes = list(axes)

        panel_idx = 0

        # ---------------------------------------------------------------
        # PANEL 0 (optional): mRNA R(t)
        # ---------------------------------------------------------------
        if has_rna_for_prot:
            axR = axes[panel_idx]
            panel_idx += 1
            rna_sub = df_mrna[df_mrna["gene"] == prot].sort_values("time")
            t_rna_vals = rna_sub["time"].values
            # Support both "fitted" and "simulated" column names
            if "fitted" in rna_sub.columns:
                y_rna_fit = rna_sub["fitted"].values
            elif "simulated" in rna_sub.columns:
                y_rna_fit = rna_sub["simulated"].values
            else:
                y_rna_fit = np.full(len(t_rna_vals), np.nan)
            y_rna_obs = rna_sub["observed"].values
            axR.plot(
                t_rna_vals, y_rna_fit, "-", lw=3, color=color, label="mRNA (model)"
            )
            axR.scatter(
                t_rna_vals, y_rna_obs, s=50, color=color, zorder=5, label="mRNA (obs)"
            )
            axR.set_title("mRNA / R(t)", fontsize=12, fontweight="bold")
            axR.set_xlabel("Time (min)")
            axR.set_ylabel("mRNA fold-change / R(t)")
            axR.legend(fontsize=9)
            axR.grid(alpha=0.25)

        # ---------------------------------------------------------------
        # PANEL 1: Protein abundance
        # ---------------------------------------------------------------
        axP = axes[panel_idx]
        panel_idx += 1
        row_prot = df_prots[df_prots["Protein"] == prot]
        if not row_prot.empty:
            row_prot = row_prot.iloc[0]
            y_sim = row_prot[sim_cols].values.astype(float)
            y_dat = row_prot[data_cols].values.astype(float)
            mask_dat = np.isfinite(y_dat)
            axP.plot(t_vals, y_sim, "-", lw=4, color=color, label="Protein (model)")
            if bool(np.any(mask_dat)):
                axP.plot(
                    np.asarray(t_vals)[mask_dat],
                    y_dat[mask_dat],
                    "-",
                    lw=2,
                    alpha=0.35,
                    color=color,
                    label="Protein (data)",
                )
                axP.scatter(
                    np.asarray(t_vals)[mask_dat],
                    y_dat[mask_dat],
                    marker="s",
                    s=55,
                    alpha=0.6,
                    color=color,
                    edgecolors="none",
                )
        else:
            axP.text(
                0.5,
                0.5,
                "No protein abundance row",
                transform=axP.transAxes,
                ha="center",
                va="center",
                fontsize=10,
                alpha=0.7,
            )
        axP.set_title("Protein abundance", fontsize=12, fontweight="bold")
        axP.set_xlabel("Time (min)")
        axP.set_ylabel("Protein abundance / A(t)")
        axP.legend(fontsize=9)
        axP.grid(alpha=0.25)

        # ---------------------------------------------------------------
        # PANEL 2: Phosphosites
        # ---------------------------------------------------------------
        axS = axes[panel_idx]
        sub = df_sites[df_sites["Protein"] == prot]
        if sub.empty:
            axS.text(
                0.5,
                0.5,
                "No phosphosites",
                transform=axS.transAxes,
                ha="center",
                va="center",
                fontsize=10,
                alpha=0.7,
            )
        else:
            cmap = plt.cm.tab20
            for i, (_, row) in enumerate(sub.iterrows()):
                res = row.get("Residue", "")
                pos = row.get("Position", row.get("Pos", row.get("SitePos", "")))
                site_label = (
                    f"{res}_{pos}" if (pd.notna(pos) and str(pos) != "") else f"{res}"
                )
                y_sim = row[sim_cols].values.astype(float)
                y_dat = row[data_cols].values.astype(float)
                mask_dat = np.isfinite(y_dat)
                c = cmap(i % 20)
                axS.plot(
                    t_vals, y_sim, "-", lw=4, color=c, label=f"{site_label} (model)"
                )
                if bool(np.any(mask_dat)):
                    axS.plot(
                        np.asarray(t_vals)[mask_dat],
                        y_dat[mask_dat],
                        "-",
                        lw=2,
                        alpha=0.35,
                        color=c,
                        label=f"{site_label} (data)",
                    )
                    axS.scatter(
                        np.asarray(t_vals)[mask_dat],
                        y_dat[mask_dat],
                        marker="s",
                        s=45,
                        alpha=0.6,
                        color=c,
                        edgecolors="none",
                    )
        axS.set_title("Phosphosites", fontsize=12, fontweight="bold")
        axS.set_xlabel("Time (min)")
        axS.set_ylabel("Relative phosphosite signal p(t)")
        axS.legend(
            fontsize=8,
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            borderaxespad=0.0,
            frameon=True,
        )
        axS.grid(alpha=0.25)

        fig.suptitle(f"{prot}", fontsize=14, fontweight="bold", y=1.01)
        plt.savefig(
            os.path.join(outdir, f"fit_{prot}.png"), dpi=300, bbox_inches="tight"
        )
        plt.close(fig)


def print_biological_scores(outdir, X):
    """
    Calculate and save biological plausibility scores for a set of parameters.

    Args:
        outdir (str): Output directory path.
        X (np.ndarray): Matrix of parameter vectors (n_points x n_params).

    Returns:
        None: Writes 'biological_scores.tsv' to disk and prints scores to console.
    """
    bio_scores = np.array([bio_score(theta) for theta in X])

    with open(os.path.join(outdir, "biological_scores.tsv"), "w") as f:
        f.write("Index\tBio_Score\n")
        for i, score in enumerate(bio_scores):
            f.write(f"{i}\t{score:.6f}\n")

    logger.info("[*] Biological Scores:")
    for i, score in enumerate(bio_scores):
        logger.info(f"   → Point {i}: Bio Score = {score:.6f}")


def plot_biological_scores(outdir, X, F):
    """
    Visualize the distribution of biological scores across the Pareto front.

    Args:
        outdir (str): Output directory path.
        X (np.ndarray): Parameter values.
        F (np.ndarray): Objective function values (f1, f2, f3).

    Returns:
        None: Saves 'biological_scores.png' to `outdir`.
    """
    bio_scores = np.array([bio_score(theta) for theta in X])
    plt.figure(figsize=(7, 6))
    f1 = F[:, 0]
    f2 = F[:, 1]
    sc = plt.scatter(f1, f2, c=bio_scores, cmap="plasma", alpha=0.7)
    plt.colorbar(sc, label="Biological Score")
    plt.title("Biological Scores across Pareto Points")
    plt.xlabel("f1")
    plt.ylabel("f2")
    plt.savefig(os.path.join(outdir, "biological_scores.png"), dpi=300)
    plt.close()


def plot_goodness_of_fit(file, outdir):
    """
    Generate a global goodness-of-fit scatter plot (Observed vs. Simulated).

    Calculates R-squared, MSE, and MAE metrics, and visualizes the data with
    an identity line and a 95% confidence interval band. Outliers are labeled.

    Args:
        file (str): Path to the time-series TSV file (output of `save_fitted_simulation`).
        outdir (str): Directory to save the plot.

    Returns:
        None: Saves 'goodness_of_fit.png' to `outdir`.
    """  # noqa: E501
    df = pd.read_csv(file, sep="\t")

    sim_cols = [c for c in df.columns if c.startswith("sim_t")]
    data_cols = [c for c in df.columns if c.startswith("data_t")]

    if len(sim_cols) == 0 or len(data_cols) == 0:
        logger.warning(
            f"[!] Skipping goodness-of-fit plot: no sim_t*/data_t* columns found in {file}"
        )
        return

    # Construct labels
    labels = []
    for _, row in df.iterrows():
        if row.get("Type", "") == "Phosphosite":
            # Residue can be missing; keep robust
            residue = row.get("Residue", "")
            if pd.notna(residue) and str(residue) != "":
                labels.append(f"{row.get('Protein', 'NA')}_{residue}")
            else:
                labels.append(f"{row.get('Protein', 'NA')}_site")
        else:
            labels.append(f"{row.get('Protein', 'NA')}_Abundance")
    df["Label"] = labels

    # --- Flatten all points for global stats ---
    data_all = df[data_cols].to_numpy(dtype=float).reshape(-1)
    sim_all = df[sim_cols].to_numpy(dtype=float).reshape(-1)
    mask_all = np.isfinite(data_all) & np.isfinite(sim_all)

    if mask_all.sum() < 3:
        logger.warning(
            "[!] Skipping goodness-of-fit plot: fewer than 3 finite observed/simulated "
            "points are available."
        )
        return

    x = data_all[mask_all]
    y = sim_all[mask_all]
    resid = y - x

    # Global metrics
    mse = float(np.mean((y - x) ** 2))
    mae = float(np.mean(np.abs(y - x)))

    # R^2 (safe)
    y_mean = float(np.mean(y))
    ss_res = float(
        np.sum((y - x) ** 2)
    )  # here "residual" is y-x (since identity is the target)
    ss_tot = float(np.sum((y - y_mean) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

    # --- 95% CI band: parallel to identity line ---
    # Interpret as 95% of residuals around identity (robust to heavy tails):
    # Use empirical 97.5th percentile of |residual| as band half-width.
    abs_resid = np.abs(resid)
    delta = float(np.quantile(abs_resid, 0.975))

    # Identify outside-band points at the per-item level (row label)
    # We'll label only those with any timepoint outside band, and choose the worst deviation.  # noqa: E501
    outside_items = []
    outside_points = []  # (x, y, label, dev)

    for _, row in df.iterrows():
        sim_vals = row[sim_cols].values.astype(float)
        data_vals = row[data_cols].values.astype(float)
        m = np.isfinite(sim_vals) & np.isfinite(data_vals)
        if not np.any(m):
            continue
        rv = sim_vals[m] - data_vals[m]
        ar = np.abs(rv)
        if np.any(ar > delta):
            # pick the worst point for labeling
            j = int(np.argmax(ar))
            outside_items.append(row["Label"])
            outside_points.append(
                (
                    float(data_vals[m][j]),
                    float(sim_vals[m][j]),
                    row["Label"],
                    float(ar[j]),
                )
            )

    # Sort by deviation and limit labels to avoid unreadable plot
    outside_points.sort(key=lambda t: t[3], reverse=True)
    max_labels = 25
    outside_points = outside_points[:max_labels]

    # --- Plot ---
    plt.figure(figsize=(10, 10))

    # Plot scatter points grouped by Type (so legend is meaningful and not cluttered)
    # Phosphosite
    for idx, row in df[df["Type"] == "Phosphosite"].iterrows():
        sim_vals = row[sim_cols].values.astype(float)
        data_vals = row[data_cols].values.astype(float)
        m = np.isfinite(sim_vals) & np.isfinite(data_vals)
        if not np.any(m):
            continue
        plt.scatter(
            data_vals[m],
            sim_vals[m],
            alpha=0.35,
            color="green",
            s=40,
            label="Phosphosite"
            if idx == df[df["Type"] == "Phosphosite"].index[0]
            else None,
        )

    # Abundance (protein / kinases, depending on your file semantics)
    for idx, row in df[df["Type"] != "Phosphosite"].iterrows():
        sim_vals = row[sim_cols].values.astype(float)
        data_vals = row[data_cols].values.astype(float)
        m = np.isfinite(sim_vals) & np.isfinite(data_vals)
        if not np.any(m):
            continue
        plt.scatter(
            data_vals[m],
            sim_vals[m],
            alpha=0.55,
            color="blue",
            s=40,
            label="Abundance"
            if idx == df[df["Type"] != "Phosphosite"].index[0]
            else None,
        )

    # Identity line and CI band
    max_val = float(np.nanmax(np.r_[x, y]))
    min_val = float(np.nanmin(np.r_[x, y]))
    pad = 0.05 * (max_val - min_val + 1e-12)
    lo = min_val - pad
    hi = max_val + pad

    xx = np.array([lo, hi], dtype=float)
    plt.plot(xx, xx, "r--", lw=2, label="Identity (y=x)")
    plt.plot(xx, xx + delta, "k:", lw=1.5, label="95% band (parallel)")
    plt.plot(xx, xx - delta, "k:", lw=1.5)

    # Label outside-band points (top deviators only)
    for px, py, lab, _dev in outside_points:
        plt.scatter(
            [px], [py], s=70, facecolors="none", edgecolors="black", linewidths=1.5
        )
        plt.text(px, py, f"  {lab}", fontsize=9, va="center")

    # Metrics box
    txt = (
        f"N={mask_all.sum()}\n"
        f"R²={r2:.4f}\n"
        f"MSE={mse:.4g}\n"
        f"MAE={mae:.4g}\n"
        f"95% band: |sim-obs| ≤ {delta:.4g}\n"
        f"Outside band (items): {len(set(outside_items))}"
    )
    plt.gca().text(
        0.02,
        0.98,
        txt,
        transform=plt.gca().transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="gray"),
    )

    plt.xlabel("Observed")
    plt.ylabel("Simulated")
    plt.title("Goodness of Fit: Observed vs Simulated")
    plt.xlim(lo, hi)
    plt.ylim(lo, hi)
    plt.legend(loc="lower right")

    plt.tight_layout()
    os.makedirs(outdir, exist_ok=True)
    plt.savefig(f"{outdir}/goodness_of_fit.png", dpi=300)
    plt.close()


def _save_txt(path: str, text: str) -> None:
    """
    Helper function to save a string to a text file, ensuring the directory exists.

    Args:
        path (str): Full file path.
        text (str): Content to write.

    Returns:
        None
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text if text.endswith("\n") else text + "\n")


def _save_matrix_tsv(path: str, mat: np.ndarray) -> None:
    """
    Helper function to save a 2D numpy array as a tab-separated file.

    Args:
        path (str): Full file path.
        mat (np.ndarray): 2D Matrix to save.

    Returns:
        None
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savetxt(path, np.asarray(mat, dtype=float), delimiter="\t")


def _save_vector_tsv(path: str, vec: np.ndarray) -> None:
    """
    Helper function to save a 1D numpy array as a column in a tab-separated file.

    Args:
        path (str): Full file path.
        vec (np.ndarray): 1D Vector to save.

    Returns:
        None
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savetxt(path, np.asarray(vec, dtype=float).reshape(-1, 1), delimiter="\t")


def _save_index_tsv(path: str, vec: np.ndarray) -> None:
    """
    Helper function to save an integer array as a column in a tab-separated file.

    Args:
        path (str): Full file path.
        vec (np.ndarray): Integer vector to save.

    Returns:
        None
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savetxt(
        path, np.asarray(vec, dtype=int).reshape(-1, 1), fmt="%d", delimiter="\t"
    )


def _save_preopt_snapshot_txt_csv(
    outdir,
    *,
    t,
    sites,
    proteins,
    kinases,
    positions,
    P_scaled,
    Y,
    A_scaled,
    A_data,
    A_proteins,
    W_data,
    W_data_prot,
    Cg,
    Cl,
    site_prot_idx,
    K_site_kin,
    R,
    L_alpha,
    kin_to_prot_idx,
    receptor_mask_prot,
    receptor_mask_kin,
    xl,
    xu,
    args,
) -> None:
    """
    Save a comprehensive snapshot of all model inputs and configuration before optimization.

    This acts as a provenance step, dumping input matrices, configuration arguments,
    and metadata to a 'preopt_snapshot' subdirectory.

    Args:
        outdir (str): Output root directory.
        t (np.ndarray): Time points.
        sites (list/array): Phosphosite labels.
        proteins (list/array): Protein labels.
        kinases (list/array): Kinase labels.
        positions (np.ndarray): Site positions.
        P_scaled, Y (np.ndarray): Phosphorylation data matrices.
        A_scaled, A_data (np.ndarray): Protein abundance data matrices.
        A_proteins (list): Proteins with abundance data.
        W_data, W_data_prot (np.ndarray): Weight matrices.
        Cg, Cl (np.ndarray): Topology matrices.
        site_prot_idx, kin_to_prot_idx (np.ndarray): Index mapping arrays.
        K_site_kin (np.ndarray): Kinase-substrate relationship matrix.
        R (np.ndarray): Receptor input.
        L_alpha (np.ndarray): Laplacian matrix.
        receptor_mask_prot, receptor_mask_kin (np.ndarray): Receptor boolean masks.
        xl, xu (np.ndarray): Lower and upper bounds for parameters.
        args (Namespace): Parsed command-line arguments or configuration object.

    Returns:
        None: Creates a 'preopt_snapshot' folder containing metadata and data files.
    """  # noqa: E501
    snap_dir = os.path.join(outdir, "preopt_snapshot")
    os.makedirs(snap_dir, exist_ok=True)

    # --- metadata (txt) ---
    lines = []
    lines.append("preopt_snapshot")
    lines.append(f"n_sites\t{len(sites)}")
    lines.append(f"n_proteins\t{len(proteins)}")
    lines.append(f"n_kinases\t{len(kinases)}")
    lines.append(f"scale_mode\t{args.scale_mode}")
    lines.append(f"mechanism\t{args.mechanism}")
    lines.append(f"weight_scheme\t{args.weight_scheme}")
    lines.append(f"length_scale\t{args.length_scale}")
    lines.append(f"lambda_net\t{args.lambda_net}")
    lines.append(f"reg_lambda\t{args.reg_lambda}")
    lines.append("")
    lines.append("shapes")

    def _shape(name, arr):
        a = np.asarray(arr)
        return f"{name}\t{tuple(a.shape)}"

    lines.extend(
        [
            _shape("t", t),
            _shape("positions", positions),
            _shape("Y", Y),
            _shape("P_scaled", P_scaled),
            _shape("A_data", A_data if A_data is not None else np.zeros((0, 0))),
            _shape("A_scaled", A_scaled),
            _shape("W_data", W_data),
            _shape("W_data_prot", W_data_prot),
            _shape("Cg", Cg),
            _shape("Cl", Cl),
            _shape("K_site_kin", K_site_kin),
            _shape("R", R),
            _shape("L_alpha", L_alpha),
            _shape("site_prot_idx", site_prot_idx),
            _shape("kin_to_prot_idx", kin_to_prot_idx),
            _shape("receptor_mask_prot", receptor_mask_prot),
            _shape("receptor_mask_kin", receptor_mask_kin),
            _shape("xl", xl),
            _shape("xu", xu),
        ]
    )
    _save_txt(os.path.join(snap_dir, "meta.txt"), "\n".join(lines))

    # --- labels (txt) ---
    _save_txt(os.path.join(snap_dir, "sites.txt"), "\n".join(map(str, sites)))
    _save_txt(os.path.join(snap_dir, "proteins.txt"), "\n".join(map(str, proteins)))
    _save_txt(os.path.join(snap_dir, "kinases.txt"), "\n".join(map(str, kinases)))
    if A_proteins is not None:
        _save_txt(
            os.path.join(snap_dir, "A_proteins.txt"),
            "\n".join(map(str, list(A_proteins))),
        )
    else:
        _save_txt(os.path.join(snap_dir, "A_proteins.txt"), "")

    # --- numeric arrays (tsv) ---
    _save_vector_tsv(os.path.join(snap_dir, "t.tsv"), t)
    _save_vector_tsv(os.path.join(snap_dir, "positions.tsv"), positions)

    _save_matrix_tsv(os.path.join(snap_dir, "Y.tsv"), Y)
    _save_matrix_tsv(os.path.join(snap_dir, "P_scaled.tsv"), P_scaled)
    _save_matrix_tsv(os.path.join(snap_dir, "A_scaled.tsv"), A_scaled)

    if A_data is not None and np.asarray(A_data).size > 0:
        _save_matrix_tsv(os.path.join(snap_dir, "A_data.tsv"), A_data)
    else:
        _save_matrix_tsv(os.path.join(snap_dir, "A_data.tsv"), np.zeros((0, 0)))

    _save_matrix_tsv(os.path.join(snap_dir, "W_data.tsv"), W_data)
    _save_matrix_tsv(os.path.join(snap_dir, "W_data_prot.tsv"), W_data_prot)

    _save_matrix_tsv(os.path.join(snap_dir, "Cg.tsv"), Cg)
    _save_matrix_tsv(os.path.join(snap_dir, "Cl.tsv"), Cl)

    _save_index_tsv(os.path.join(snap_dir, "site_prot_idx.tsv"), site_prot_idx)
    _save_matrix_tsv(os.path.join(snap_dir, "K_site_kin.tsv"), K_site_kin)
    _save_matrix_tsv(os.path.join(snap_dir, "R.tsv"), R)
    _save_matrix_tsv(os.path.join(snap_dir, "L_alpha.tsv"), L_alpha)

    _save_index_tsv(os.path.join(snap_dir, "kin_to_prot_idx.tsv"), kin_to_prot_idx)
    _save_index_tsv(
        os.path.join(snap_dir, "receptor_mask_prot.tsv"), receptor_mask_prot
    )
    _save_index_tsv(os.path.join(snap_dir, "receptor_mask_kin.tsv"), receptor_mask_kin)

    _save_vector_tsv(os.path.join(snap_dir, "xl.tsv"), xl)
    _save_vector_tsv(os.path.join(snap_dir, "xu.tsv"), xu)

    # Also write a consolidated NPZ for faster dashboard loading
    save_preopt_snapshot_npz(
        snap_dir,
        t=t,
        Y=Y,
        P_scaled=P_scaled,
        A_data=A_data,
        A_scaled=A_scaled,
        Cg=Cg,
        Cl=Cl,
        K_site_kin=K_site_kin,
        R=R,
        L_alpha=L_alpha,
        W_data=W_data,
        W_data_prot=W_data_prot,
        site_prot_idx=site_prot_idx,
        kin_to_prot_idx=kin_to_prot_idx,
        receptor_mask_prot=receptor_mask_prot,
        receptor_mask_kin=receptor_mask_kin,
        positions=positions,
        xl=xl,
        xu=xu,
    )

    # Write entity labels NPZ
    _labels_npz = os.path.join(snap_dir, "entity_labels.npz")
    if not os.path.exists(_labels_npz):
        _a_prots = list(A_proteins) if A_proteins is not None else []
        np.savez(
            _labels_npz,
            sites=np.array(list(sites), dtype=object),
            proteins=np.array(list(proteins), dtype=object),
            kinases=np.array(list(kinases), dtype=object),
            A_proteins=np.array(_a_prots, dtype=object),
        )


def save_preopt_snapshot_npz(
    snap_dir,
    *,
    t,
    Y,
    P_scaled,
    A_data,
    A_scaled,
    Cg,
    Cl,
    K_site_kin,
    R,
    L_alpha,
    W_data,
    W_data_prot,
    site_prot_idx,
    kin_to_prot_idx,
    receptor_mask_prot,
    receptor_mask_kin,
    positions,
    xl,
    xu,
) -> None:
    """
    Save a consolidated machine-readable NPZ of all pre-optimisation inputs.

    Written to ``snap_dir/preopt_snapshot.npz``.  Existing files are not
    overwritten – delete the file first if a refresh is needed.

    Args:
        snap_dir (str): Path to the ``preopt_snapshot/`` directory.
        t, Y, P_scaled, A_data, A_scaled, Cg, Cl, K_site_kin, R, L_alpha,
        W_data, W_data_prot, site_prot_idx, kin_to_prot_idx,
        receptor_mask_prot, receptor_mask_kin, positions, xl, xu:
            Model input arrays.
    """
    out_path = os.path.join(snap_dir, "preopt_snapshot.npz")
    if os.path.exists(out_path):
        return

    def _safe(arr):
        if arr is None:
            return np.empty(0)
        a = np.asarray(arr)
        return a if a.size > 0 else np.empty(0)

    np.savez(
        out_path,
        t=_safe(t),
        Y=_safe(Y),
        P_scaled=_safe(P_scaled),
        A_data=_safe(A_data),
        A_scaled=_safe(A_scaled),
        Cg=_safe(Cg),
        Cl=_safe(Cl),
        K_site_kin=_safe(K_site_kin),
        R=_safe(R),
        L_alpha=_safe(L_alpha),
        W_data=_safe(W_data),
        W_data_prot=_safe(W_data_prot),
        site_prot_idx=_safe(site_prot_idx),
        kin_to_prot_idx=_safe(kin_to_prot_idx),
        receptor_mask_prot=_safe(receptor_mask_prot),
        receptor_mask_kin=_safe(receptor_mask_kin),
        positions=_safe(positions),
        xl=_safe(xl),
        xu=_safe(xu),
    )
    logger.info(f"[*] Saved preopt_snapshot.npz to {out_path}")


def save_mrna_outputs(outdir, gene_ids, t_rna, rna_data_obs, rna_simulated):
    """
    Save mRNA fit time-series and per-gene diagnostics.

    Requires both observed and simulated mRNA data.  Raises ``ValueError`` if
    *rna_simulated* is None – do not use this function to save fake zero-residual
    outputs.  Always pass real simulated R(t) from the ODE.

    Emits a warning when the simulated maximum is more than 10× the observed
    maximum, which indicates a numerical scaling problem in the RNA ODE.

    Args:
        outdir (str): Output directory.
        gene_ids (list[str]): Gene identifiers (matched model proteins only).
        t_rna (np.ndarray): mRNA time points (T_rna,).
        rna_data_obs (np.ndarray): Observed mRNA fold-change matrix (n_genes, T_rna).
        rna_simulated (np.ndarray): Model-simulated R(t) (n_genes, T_rna).
            Must not be None; must match shape of rna_data_obs.

    Returns:
        None: Writes ``mrna_fit_timeseries.tsv`` and ``mrna_diagnostics.tsv``
        to *outdir*.
    """
    if rna_simulated is None:
        logger.warning(
            "[!] save_mrna_outputs skipped: rna_simulated is None. "
            "Pass real simulated R(t) values from the model; fake zero-residual "
            "mRNA outputs will not be written."
        )
        return

    os.makedirs(outdir, exist_ok=True)

    rna_data_obs = np.asarray(rna_data_obs, dtype=float)
    rna_simulated = np.asarray(rna_simulated, dtype=float)

    if rna_data_obs.shape != rna_simulated.shape:
        logger.warning(
            "[!] save_mrna_outputs skipped: shape mismatch: "
            f"observed {rna_data_obs.shape} vs simulated {rna_simulated.shape}. "
            "Ensure both use matched gene rows."
        )
        return

    # Clip simulated RNA to non-negative before saving (fold-change is always ≥ 0)
    rna_simulated = np.clip(rna_simulated, 0.0, None)

    # --- Global scale diagnostics ---
    obs_finite = rna_data_obs[np.isfinite(rna_data_obs)]
    sim_finite = rna_simulated[np.isfinite(rna_simulated)]
    obs_min = float(np.min(obs_finite)) if obs_finite.size > 0 else float("nan")
    obs_max = float(np.max(obs_finite)) if obs_finite.size > 0 else float("nan")
    obs_mean = float(np.mean(obs_finite)) if obs_finite.size > 0 else float("nan")
    sim_min = float(np.min(sim_finite)) if sim_finite.size > 0 else float("nan")
    sim_max = float(np.max(sim_finite)) if sim_finite.size > 0 else float("nan")
    sim_mean = float(np.mean(sim_finite)) if sim_finite.size > 0 else float("nan")

    logger.info(
        f"[*] mRNA scale diagnostics – observed: min={obs_min:.3f}, "
        f"mean={obs_mean:.3f}, max={obs_max:.3f} | "
        f"fitted R(t): min={sim_min:.3f}, mean={sim_mean:.3f}, max={sim_max:.3f}"
    )

    if not np.isfinite(sim_max):
        logger.warning(
            "[!] mRNA WARNING: fitted R(t) contains non-finite values. "
            "The RNA ODE may be numerically unstable."
        )
    elif np.isfinite(obs_max) and obs_max > 0 and sim_max > 10.0 * obs_max:
        logger.warning(
            f"[!] mRNA WARNING: max(R_sim)={sim_max:.2f} > "
            f"10 × max(R_obs)={obs_max:.2f}. "
            "Fitted R(t) is far off the observed fold-change scale. "
            "Check RNA ODE and initial conditions."
        )

    records = []
    diag_records = []

    for i, gene in enumerate(gene_ids):
        obs = rna_data_obs[i]
        fit = rna_simulated[i]
        for t_idx, t_val in enumerate(t_rna):
            records.append(
                {
                    "gene": gene,
                    "time": float(t_val),
                    "observed": float(obs[t_idx]),
                    "fitted": float(fit[t_idx]),
                }
            )

        # Per-gene diagnostics – computed from real residuals
        resid = obs - fit
        gene_obs_min = float(np.nanmin(obs))
        gene_obs_max = float(np.nanmax(obs))
        gene_obs_mean = float(np.nanmean(obs))
        gene_sim_min = float(np.nanmin(fit))
        gene_sim_max = float(np.nanmax(fit))
        gene_sim_mean = float(np.nanmean(fit))
        rmse = float(np.sqrt(np.mean(resid**2)))
        ss_res = float(np.sum(resid**2))
        ss_tot = float(np.sum((obs - np.mean(obs)) ** 2))
        r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else float("nan")
        diag_records.append(
            {
                "gene": gene,
                "rmse": rmse,
                "r2": r2,
                "mean_residual": float(np.mean(resid)),
                "std_residual": float(np.std(resid)),
                "max_abs_residual": float(np.max(np.abs(resid))),
                "obs_min": gene_obs_min,
                "obs_max": gene_obs_max,
                "obs_mean": gene_obs_mean,
                "sim_min": gene_sim_min,
                "sim_max": gene_sim_max,
                "sim_mean": gene_sim_mean,
            }
        )

    pd.DataFrame(records).to_csv(
        os.path.join(outdir, "mrna_fit_timeseries.tsv"), sep="\t", index=False
    )
    pd.DataFrame(diag_records).to_csv(
        os.path.join(outdir, "mrna_diagnostics.tsv"), sep="\t", index=False
    )
    logger.info(
        f"[*] mRNA outputs written to {outdir}: "
        "mrna_fit_timeseries.tsv, mrna_diagnostics.tsv"
    )


def plot_mrna_fit(outdir, gene_ids=None, max_genes=12):
    """
    Plot observed vs fitted mRNA trajectories (dots vs line per gene).

    Reads ``mrna_fit_timeseries.tsv`` from *outdir*.

    Args:
        outdir (str): Directory containing ``mrna_fit_timeseries.tsv``.
        gene_ids (list[str] | None): Subset of genes to plot; ``None`` plots
            up to *max_genes* with the largest dynamic range.
        max_genes (int): Maximum number of genes to include in one figure.

    Returns:
        None: Saves ``mrna_fit_panel.png`` to *outdir*.
    """
    tsv_path = os.path.join(outdir, "mrna_fit_timeseries.tsv")
    if not os.path.exists(tsv_path):
        logger.warning(
            f"[!] mrna_fit_timeseries.tsv not found in {outdir}; skipping plot."
        )
        return

    df = pd.read_csv(tsv_path, sep="\t")
    all_genes = df["gene"].unique().tolist()

    if gene_ids is not None:
        plot_genes = [g for g in gene_ids if g in all_genes]
    else:
        # Pick top-N by dynamic range of observed values
        ranges = (
            df.groupby("gene")["observed"]
            .apply(lambda v: float(v.max() - v.min()))
            .sort_values(ascending=False)
        )
        plot_genes = ranges.head(max_genes).index.tolist()

    n = len(plot_genes)
    if n == 0:
        logger.warning("[!] No genes to plot for mRNA fit panel.")
        return

    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False
    )
    axes_flat = axes.flatten()

    for ax_idx, gene in enumerate(plot_genes):
        ax = axes_flat[ax_idx]
        sub = df[df["gene"] == gene].sort_values("time")
        ax.plot(sub["time"], sub["fitted"], "-", lw=2, label="fitted")
        ax.scatter(sub["time"], sub["observed"], s=40, zorder=5, label="observed")
        ax.set_title(gene, fontsize=10, fontweight="bold")
        ax.set_xlabel("Time")
        ax.set_ylabel("mRNA FC")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)

    # Hide unused axes
    for ax in axes_flat[n:]:
        ax.set_visible(False)

    fig.suptitle("mRNA Fit: Observed vs Fitted", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "mrna_fit_panel.png"), dpi=300)
    plt.close(fig)
    logger.info(f"[*] mRNA fit panel saved to {outdir}/mrna_fit_panel.png")


def save_derived_rates(
    outdir,
    proteins,
    t_protein,
    k_act_fn=None,
    s_prod_fn=None,
    t_rna=None,
):
    """
    Save derived k_act and s_prod trajectories.

    k_act and s_prod are not optimized parameters. They are time-dependent
    derived rates computed from network/input closures.

    Outputs
    -------
    derived_rates.npz
        Compact NumPy archive.

    derived_rates_long.tsv
        Long-format table for inspection and plotting.
    """
    os.makedirs(outdir, exist_ok=True)

    if k_act_fn is None and s_prod_fn is None:
        logger.warning(
            "[!] No derived rate functions provided; skipping derived rate export."
        )
        return

    proteins = np.asarray(proteins, dtype=object)

    if t_rna is not None:
        t_k = np.asarray(t_rna, dtype=float)
    else:
        t_k = np.asarray(t_protein, dtype=float)

    t_s = np.asarray(t_protein, dtype=float)

    save_dict = {
        "proteins": proteins,
        "t_k_act": t_k,
        "t_s_prod": t_s,
        # Shape semantics metadata
        "entity_type_k_act": np.array(["protein"], dtype=object),
        "entity_type_s_prod": np.array(["protein_aggregated"], dtype=object),
    }

    rows = []

    if k_act_fn is not None:
        k_act = np.vstack([np.asarray(k_act_fn(float(ti))) for ti in t_k]).T
        save_dict["k_act"] = k_act
        logger.info(
            f"[*] k_act(t): protein-level transcriptional drive, shape {k_act.shape}"
        )

        for i, protein in enumerate(proteins):
            for j, time in enumerate(t_k):
                rows.append(
                    {
                        "rate_type": "k_act",
                        "entity_type": "protein",
                        "protein": protein,
                        "site": "",
                        "entity": protein,
                        "time": float(time),
                        "value": float(k_act[i, j]),
                    }
                )

    if s_prod_fn is not None:
        s_prod = np.vstack([np.asarray(s_prod_fn(float(ti))) for ti in t_s]).T
        save_dict["s_prod"] = s_prod
        logger.info(
            f"[*] s_prod(t): protein-level aggregated phosphorylation drive, shape {s_prod.shape}"
        )

        for i, protein in enumerate(proteins):
            for j, time in enumerate(t_s):
                rows.append(
                    {
                        "rate_type": "s_prod",
                        "entity_type": "protein_aggregated",
                        "protein": protein,
                        "site": "",
                        "entity": protein,
                        "time": float(time),
                        "value": float(s_prod[i, j]),
                    }
                )

    np.savez(os.path.join(outdir, "derived_rates.npz"), **save_dict)

    if rows:
        pd.DataFrame(rows).to_csv(
            os.path.join(outdir, "derived_rates_long.tsv"),
            sep="\t",
            index=False,
        )

    logger.success("[*] Saved derived_rates.npz and derived_rates_long.tsv")
