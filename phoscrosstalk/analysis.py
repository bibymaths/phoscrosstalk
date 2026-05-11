"""
PhosCrossTalk Post-optimization analysis, file export, and plotting.

This module provides functions for analyzing the results of PhosCrossTalk optimization,
exporting simulation results to files, and generating visualizations.
"""

import os

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

from phoscrosstalk.config import ModelDims
from phoscrosstalk.mechanisms import decode_theta
from phoscrosstalk.logger import get_logger
from phoscrosstalk.optimization import bio_score, build_full_A0
from phoscrosstalk.simulation import simulate, simulate_dense

logger = get_logger()


def save_run_results(outdir, F, X, f1, f2, f3, J, F_best, f4=None, dims: ModelDims | None = None):
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
        (None): Files are written to `outdir`.  Output filenames are preserved for
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

    if dims is not None:
        bio_scores = np.array([bio_score(theta, dims) for theta in X])
    else:
        bio_scores = np.full(len(X), np.nan, dtype=float)
    df_front["bio_score"] = bio_scores
    df_front.to_csv(os.path.join(outdir, "pareto_points.tsv"), sep="\t", index=False)

    np.savez(os.path.join(outdir, "pareto_front.npz"), F=F, X=X, J=J)


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
        (None): Saves 'pareto_f1_f2.png', 'pareto_param_corr.png', and
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

def print_parameter_summary(outdir, theta_opt, proteins, kinases, sites, dims: ModelDims | None = None):
    """
    Decode optimized parameters and export summaries for proteins, kinases, and sites.

    Args:
        outdir (str): Path to the output directory.
        theta_opt (np.ndarray): The optimized parameter vector.
        proteins (list): List of protein names.
        kinases (list): List of kinase names.
        sites (list): List of phosphorylation site names.

    Returns:
        (None): Writes summary TSV/TXT files to `outdir` and prints summaries to console.
    """
    if dims is None:
        dims = ModelDims.set_dims(len(proteins), len(kinases), len(sites))
    K, M, N = dims.K, dims.M, dims.N
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
        dims: ModelDims | None = None,
        k_act_fn=None,
        s_prod_fn=None,
        R_data0=None,
        n_dense: int = 200,
        interpolation_label: str = "diffrax_dense",
        data_interp_P=None,
        data_interp_A=None,
        data_interp_R=None,
        prot_idx_for_A_full=None,
):
    """Run a dense-grid simulation and save long-format output for visualisation.

    Runs an additional forward ODE simulation over a uniform grid of *n_dense*
    points spanning ``[0, max(t_obs)]`` using :func:`simulate_dense` and writes
    ``protein_fit_timeseries_dense.tsv`` in a long format suitable for the dashboard.

    This function is called by ``save_fitted_simulation`` and does **not** affect
    the optimisation objective or loss values.  It is best-effort; failures are
    caught and warned rather than propagated.

    When *data_interp_P*, *data_interp_A*, or *data_interp_R* are provided
    (callables built by
    :func:`~phoscrosstalk.derived_rates.build_data_interpolations`), the dense
    output also includes ``series_type = "observed_interpolated_dense"`` rows for
    diagnostic comparison.  These rows are clearly labelled and must not be
    confused with measured data.  *data_interp_R* is built from the RNA-specific
    time axis (``t_rna``) and must not be confused with the phospho/protein time
    axis.

    Output columns: entity_type, entity, site, protein, time, value, series_type,
                    source, interpolation_method

    Recommended ``series_type`` values used here:

    * ``"simulated_dense"``           – ODE simulation on dense grid (model output)
    * ``"observed_interpolated_dense"`` – interpolated from sparse observed data
                                          (diagnostic only, not training data)
    """
    if dims is None:
        dims = ModelDims.set_dims(len(proteins), len(kin_to_prot_idx), len(sites))
    K, M, N = dims.K, dims.M, dims.N
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
        dims=dims,
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

    if data_interp_A is not None and prot_idx_for_A_full is not None:
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

    # Interpolated mRNA (observed) — diagnostic only. NOT training data.
    # series_type is clearly labelled to avoid confusion with simulated_dense rows.
    if data_interp_R is not None:
        interp_R_vals = np.asarray(data_interp_R(t_dense), dtype=float)
        # data_interp_R(t_dense) returns shape (K, n_dense) for array input
        for p_idx in range(K):
            prot = proteins[p_idx]
            row_vals = interp_R_vals[p_idx, :] if interp_R_vals.ndim == 2 else interp_R_vals
            for j, ti in enumerate(t_dense):
                rows.append(
                    {
                        "entity_type": "mRNA",
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

    df_dense = pd.DataFrame(rows)
    df_dense.to_csv(
        os.path.join(outdir, "protein_fit_timeseries_dense.tsv"), sep="\t", index=False
    )
    logger.info(f"[*] Dense simulation output saved ({len(t_dense)} time points).")

    # Dense mRNA / R_rna output (model only, simulated_dense series_type).
    # Saved to mrna_fit_timeseries_dense.tsv when R_sim is available.
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
        dims: ModelDims | None = None,
        k_act_fn=None,
        s_prod_fn=None,
        R_data0=None,
        rna_data=None,
        kinases=None,
        simulation_cfg=None,
        data_interpolation_cfg=None,
        t_rna=None,
        sim_full_override=None,
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
        Cg (np.ndarray): Global connectivity matrix.
        Cl (np.ndarray): Local connectivity matrix.
        site_prot_idx (np.ndarray): Mapping of sites to proteins.
        K_site_kin (np.ndarray): Kinase-site interaction matrix.
        R (np.ndarray): Receptor/Input matrix.
        L_alpha (np.ndarray): Laplacian or interaction matrix for alpha term.
        kin_to_prot_idx (np.ndarray): Mapping of kinases to protein indices.
        mask_p (np.ndarray): Boolean mask for proteins.
        mask_k (np.ndarray): Boolean mask for kinases.
        kinases (list | None): List of kinase names.  When provided, Kdyn_sim
            rows in ``internal_states.tsv`` use real kinase names instead of
            generic ``Kinase_0, Kinase_1, …`` labels.
        simulation_cfg (SimpleNamespace | None): Optional ``[simulation]`` config
            section.  Controls ``dense_n_points``, ``save_dense``, and
            ``dense_interpolation`` label.  Defaults are used when None.
        data_interpolation_cfg (SimpleNamespace | None): Optional
            ``[data_interpolation]`` config section.  When
            ``data_interpolation_cfg.enabled`` is True, diagnostic interpolated
            observed curves are added to ``protein_fit_timeseries_dense.tsv``.  The
            original sparse observed arrays in the loss are never modified.
        t_rna (np.ndarray | None): RNA-specific time vector.  Must be provided
            when *R_data0* is not None and RNA data uses a different time axis
            than the phospho/protein observations (*t*).  Used exclusively for
            the diagnostic mRNA interpolation; never conflated with *t*.
        sim_full_override (bool | None): Override the default simulation mode
            (dense vs sparse) for PINN simulation. If None, the default mode is
            used.
        R_data0 (np.ndarray | None): RNA initial condition/state used by the simulator.
            This is not necessarily a full RNA time-series matrix.
        rna_data (np.ndarray | None): Observed RNA time-series matrix for diagnostic
            interpolation only. Expected shape is (n_genes, len(t_rna)).

    Returns:
        (None): Saves 'fitted_params.npz' and 'protein_fit_timeseries.tsv' to `outdir`.
    """
    if dims is None:
        dims = ModelDims.set_dims(len(proteins), len(kin_to_prot_idx), len(sites))
    K, M, N = dims.K, dims.M, dims.N

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

    # Re-simulate, or use externally supplied PINN/full simulation output.
    if sim_full_override is not None:
        sim_full = sim_full_override

        P_sim = np.asarray(sim_full["P_sim"], dtype=float)
        A_sim = np.asarray(sim_full["A_sim"], dtype=float)
        S_sim = np.asarray(sim_full["S_sim"], dtype=float)
        Kdyn_sim = np.asarray(sim_full["Kdyn_sim"], dtype=float)

    else:
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
            dims=dims,
        )

    # Rescale Sites (model)
    Y_sim_rescaled = np.zeros_like(P_sim, dtype=float)
    for i in range(len(sites)):
        Y_sim_rescaled[i] = baselines[i] + amplitudes[i] * P_sim[i]

    # Rescale Sites (data)
    # Y is expected to be the original phosphosite data in FC/original units.
    # If Y is unavailable, reconstruct the original-scale data from P_scaled.
    if Y is not None:
        Y_data_rescaled = np.asarray(Y, dtype=float)
    else:
        Y_data_rescaled = np.zeros_like(P_scaled, dtype=float)
        for i in range(len(sites)):
            Y_data_rescaled[i] = baselines[i] + amplitudes[i] * P_scaled[i]

    if Y_data_rescaled.shape != Y_sim_rescaled.shape:
        raise ValueError(
            "Y_data_rescaled and Y_sim_rescaled must have the same shape. "
            f"Got data={Y_data_rescaled.shape}, sim={Y_sim_rescaled.shape}."
        )

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
    df_out.to_csv(os.path.join(outdir, "protein_fit_timeseries.tsv"), sep="\t", index=False)

    # Dense continuous output for smooth dashboard visualisation.
    # Runs a separate forward simulation over a fine time grid [0, t_max] and
    # saves the result in long format to protein_fit_timeseries_dense.tsv.
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

        # Separately build RNA interpolation using the RNA-specific time axis.
        # This must NOT reuse t (phospho/protein time axis) — t_rna may differ.
        #
        # IMPORTANT:
        #   R_data0 is simulation initial state / RNA state input.
        #   rna_data is observed RNA time-series data for interpolation.
        #   Do not use R_data0 here unless it has explicitly been passed as rna_data.
        _data_interp_R = None
        if _di_cfg is not None and getattr(_di_cfg, "enabled", False):
            if rna_data is not None and t_rna is not None:
                from phoscrosstalk.derived_rates import build_data_interpolations as _bdi

                _di_method = getattr(_di_cfg, "method", "linear")
                _di_fwd = getattr(_di_cfg, "fill_forward_nans_at_end", False)
                _di_start = getattr(_di_cfg, "replace_nans_at_start", None)

                _rna_arr = np.asarray(rna_data, dtype=float)
                _t_rna_arr = np.asarray(t_rna, dtype=float)

                if _t_rna_arr.ndim != 1:
                    raise ValueError(
                        "[data_interp/rna] t_rna must be a 1-D time vector. "
                        f"Got shape {_t_rna_arr.shape}."
                    )

                # Allow a single RNA trajectory only when it is truly time-indexed.
                if _rna_arr.ndim == 1:
                    if _rna_arr.shape[0] != len(_t_rna_arr):
                        raise ValueError(
                            "[data_interp/rna] rna_data is 1-D but does not match t_rna. "
                            f"rna_data length={_rna_arr.shape[0]}, "
                            f"len(t_rna)={len(_t_rna_arr)}. "
                            "Expected either (len(t_rna),) or (n_genes, len(t_rna))."
                        )
                    _rna_arr = _rna_arr.reshape(1, -1)

                if _rna_arr.ndim != 2:
                    raise ValueError(
                        "[data_interp/rna] rna_data must be 2-D with shape "
                        f"(n_genes, len(t_rna)). Got shape {_rna_arr.shape}."
                    )

                if _rna_arr.shape[1] != len(_t_rna_arr):
                    raise ValueError(
                        "[data_interp/rna] RNA time-series axis mismatch. "
                        f"rna_data.shape={_rna_arr.shape}, len(t_rna)={len(_t_rna_arr)}. "
                        "Expected rna_data.shape[1] == len(t_rna)."
                    )

                _rna_interp_result = _bdi(
                    t_obs=_t_rna_arr,
                    rna_data=_rna_arr,
                    method=_di_method,
                    fill_forward_nans_at_end=_di_fwd,
                    replace_nans_at_start=_di_start,
                )

                _data_interp_R = _rna_interp_result.get("rna_interp")

                for msg in _rna_interp_result.get("nan_fill_log", []):
                    logger.info("[data_interp/rna] %s", msg)

        if _do_dense and sim_full_override is not None:
            _save_dense_simulation(
                outdir=outdir,
                dims=dims,
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
                data_interp_R=_data_interp_R,
                prot_idx_for_A_full=prot_idx_for_A,
            )

        elif _do_dense and sim_full_override is not None:
            logger.info(
                "[pinn] Skipping mechanistic _save_dense_simulation(); dense PINN outputs should come from run_pinn_pipeline().")

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
        (None): Saves plots to disk.
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

    Reads ``protein_fit_timeseries.tsv`` from *outdir*. When ``mrna_fit_timeseries.tsv``
    is also present, creates three-panel figures (mRNA / protein abundance /
    phosphosites); otherwise creates two-panel figures (protein abundance /
    phosphosites).

    The time axis is read from ``time_axes.json`` in *outdir* (written by
    ``main.py`` during the fitting run).  The function returns early with an
    error log if ``time_axes.json`` is absent or does not contain
    ``phosphosite_time_points``.

    Args:
        outdir (str): Directory containing output TSV files.

    Returns:
        (None): Saves per-protein PNG files to *outdir*.
    """

    # Load protein/phosphosite fit data
    ts_path = os.path.join(outdir, "protein_fit_timeseries.tsv")
    if not os.path.exists(ts_path):
        logger.warning(f"[!] protein_fit_timeseries.tsv not found in {outdir}; skipping.")
        return

    df = pd.read_csv(ts_path, sep="\t")
    proteins = sorted(df["Protein"].unique())
    logger.info(f"[*] Found {len(proteins)} proteins")
    df_sites = df[df["Type"] == "Phosphosite"].reset_index(drop=True)
    df_prots = df[df["Type"] == "ProteinAbundance"].reset_index(drop=True)

    sim_cols = [col for col in df.columns if col.startswith("sim_t")]
    data_cols = [col for col in df.columns if col.startswith("data_t")]

    # Guard against corrupted/legacy outputs where sim and data column counts differ.
    if len(sim_cols) != len(data_cols):
        logger.warning(
            "[!] plot_fitted_simulation: sim_t column count (%d) != data_t column count (%d) "
            "in %s. Skipping plot to avoid length mismatch.",
            len(sim_cols),
            len(data_cols),
            ts_path,
        )
        return

    # --- Resolve time axis from time_axes.json (required) ---
    _time_axes_path = os.path.join(outdir, "time_axes.json")
    if not os.path.exists(_time_axes_path):
        logger.error(
            "[!] plot_fitted_simulation: time_axes.json not found in %s. "
            "This file is written by main.py during the fitting run. "
            "Cannot plot without explicit time axis.",
            outdir,
        )
        return

    try:
        import json as _json
        with open(_time_axes_path) as _fh:
            _time_axes_cfg = _json.load(_fh)
    except (OSError, ValueError) as _exc:
        logger.error("[!] Could not read time_axes.json: %s", _exc)
        return

    t_vals = np.asarray(_time_axes_cfg.get("phosphosite_time_points", []), dtype=float)
    if t_vals.size == 0:
        logger.error(
            "[!] plot_fitted_simulation: time_axes.json in %s does not contain "
            "phosphosite_time_points. Cannot plot without explicit time axis.",
            outdir,
        )
        return
    if len(sim_cols) != len(t_vals):
        logger.error(
            "[!] plot_fitted_simulation: time_axes.json phosphosite_time_points length (%d) "
            "does not match sim_t column count (%d) in %s. Cannot plot.",
            len(t_vals),
            len(sim_cols),
            outdir,
        )
        return

    # Optionally load mRNA fit data
    # Optionally load mRNA fit data
    mrna_path = os.path.join(outdir, "mrna_fit_timeseries.tsv")
    has_rna_data = os.path.exists(mrna_path)
    df_mrna = None
    if has_rna_data:
        try:
            df_mrna = pd.read_csv(mrna_path, sep="\t")
        except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
            logger.warning("[!] Could not read mRNA fit timeseries %s: %s", mrna_path, exc)
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

        figsize = (16, 5) if n_panels == 2 else (22, 5)

        fig, axes = plt.subplots(
            1,
            n_panels,
            figsize=figsize,
        )

        fig.set_layout_engine(
            "constrained",
            w_pad=0.04,
            h_pad=0.12,
            hspace=0.28,
            wspace=0.12,
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

        fig.suptitle(f"{prot}", fontsize=16, fontweight="bold")

        plt.savefig(
            os.path.join(outdir, f"fit_{prot}.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)


def print_biological_scores(outdir, X):
    """
    Calculate and save biological plausibility scores for a set of parameters.

    Args:
        outdir (str): Output directory path.
        X (np.ndarray): Matrix of parameter vectors (n_points x n_params).

    Returns:
        (None): Writes 'biological_scores.tsv' to disk and prints scores to console.
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
        (None): Saves 'biological_scores.png' to `outdir`.
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


def plot_goodness_of_fit(protein_fit_timeseries_path, outdir, mrna_fit_timeseries_path=None):
    """
    Generate a global goodness-of-fit scatter plot: observed vs simulated/fitted.

    Reads protein/phosphosite fit data from a wide-format TSV file containing
    sim_t*/data_t* columns. Optionally also reads mRNA fit data from
    mrna_fit_timeseries.tsv.

    Args:
        protein_fit_timeseries_path (str): Path to the main time-series TSV file.
        outdir (str): Directory to save the plot.
        mrna_fit_timeseries_path (str | None): Optional path to
            mrna_fit_timeseries.tsv. If None, the function looks for this file
            inside outdir.

    Returns:
        None: Saves 'goodness_of_fit.png' to outdir.
    """
    df = pd.read_csv(protein_fit_timeseries_path, sep="\t")

    sim_cols = [c for c in df.columns if c.startswith("sim_t")]
    data_cols = [c for c in df.columns if c.startswith("data_t")]

    if len(sim_cols) == 0 or len(data_cols) == 0:
        logger.warning(
            f"[!] Skipping goodness-of-fit plot: no sim_t*/data_t* columns found in {protein_fit_timeseries_path}"
        )
        return

    if len(sim_cols) != len(data_cols):
        logger.warning(
            "[!] Skipping goodness-of-fit plot: number of sim_t* and data_t* "
            f"columns differs: sim={len(sim_cols)}, data={len(data_cols)}"
        )
        return

    # ------------------------------------------------------------------
    # Build row labels for protein/phosphosite data
    # ------------------------------------------------------------------
    labels = []
    for _, row in df.iterrows():
        row_type = row.get("Type", "")

        if row_type == "Phosphosite":
            residue = row.get("Residue", "")
            if pd.notna(residue) and str(residue) != "":
                labels.append(f"{row.get('Protein', 'NA')}_{residue}")
            else:
                labels.append(f"{row.get('Protein', 'NA')}_site")
        elif row_type == "ProteinAbundance":
            labels.append(f"{row.get('Protein', 'NA')}_Abundance")
        else:
            labels.append(f"{row.get('Protein', 'NA')}_{row_type}")

    df["Label"] = labels

    # ------------------------------------------------------------------
    # Flatten main wide-format fit data
    # ------------------------------------------------------------------
    plot_records = []

    for _, row in df.iterrows():
        row_type = row.get("Type", "Unknown")
        label = row.get("Label", "Unknown")

        sim_vals = row[sim_cols].values.astype(float)
        data_vals = row[data_cols].values.astype(float)

        for obs, sim in zip(data_vals, sim_vals, strict=True):
            if np.isfinite(obs) and np.isfinite(sim):
                plot_records.append(
                    {
                        "Observed": float(obs),
                        "Simulated": float(sim),
                        "Type": row_type,
                        "Label": label,
                    }
                )

    # ------------------------------------------------------------------
    # Optionally append mRNA long-format fit data
    # ------------------------------------------------------------------
    if mrna_fit_timeseries_path is None:
        mrna_fit_timeseries_path = os.path.join(outdir, "mrna_fit_timeseries.tsv")

    if os.path.exists(mrna_fit_timeseries_path):
        try:
            df_mrna = pd.read_csv(mrna_fit_timeseries_path, sep="\t")
        except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
            raise RuntimeError(
                f"[!] Failed to read mRNA goodness-of-fit file "
                f"{mrna_fit_timeseries_path}: {exc}"
            ) from exc

        required_cols = {"gene", "observed"}
        if not required_cols.issubset(df_mrna.columns):
            logger.warning(
                "[!] Skipping mRNA goodness-of-fit points: "
                f"missing required columns in {mrna_fit_timeseries_path}. "
                f"Required at least {required_cols}; found {set(df_mrna.columns)}"
            )
        else:
            # Prefer "fitted"; fall back to "simulated" if needed.
            if "fitted" in df_mrna.columns:
                fit_col = "fitted"
            elif "simulated" in df_mrna.columns:
                fit_col = "simulated"
            else:
                fit_col = None

            if fit_col is None:
                logger.warning(
                    "[!] Skipping mRNA goodness-of-fit points: neither "
                    "'fitted' nor 'simulated' column found in "
                    f"{mrna_fit_timeseries_path}"
                )
            else:
                for _, row in df_mrna.iterrows():
                    obs = row.get("observed", np.nan)
                    sim = row.get(fit_col, np.nan)
                    gene = row.get("gene", "NA")

                    if np.isfinite(obs) and np.isfinite(sim):
                        plot_records.append(
                            {
                                "Observed": float(obs),
                                "Simulated": float(sim),
                                "Type": "mRNA",
                                "Label": f"{gene}_mRNA",
                            }
                        )

                logger.info(
                    f"[*] Added mRNA goodness-of-fit points from "
                    f"{mrna_fit_timeseries_path}"
                )
    else:
        logger.info(
            f"[*] No mRNA goodness-of-fit file found at {mrna_fit_timeseries_path}; "
            "plotting protein/phosphosite data only."
        )

    if len(plot_records) < 3:
        logger.warning(
            "[!] Skipping goodness-of-fit plot: fewer than 3 finite observed/simulated "
            "points are available."
        )
        return

    df_plot = pd.DataFrame.from_records(plot_records)

    x = df_plot["Observed"].to_numpy(dtype=float)
    y = df_plot["Simulated"].to_numpy(dtype=float)
    resid = y - x

    # ------------------------------------------------------------------
    # Global metrics
    # ------------------------------------------------------------------
    mse = float(np.mean((y - x) ** 2))
    mae = float(np.mean(np.abs(y - x)))

    # R² against identity target: observed is target, simulated is prediction.
    x_mean = float(np.mean(x))
    ss_res = float(np.sum((y - x) ** 2))
    ss_tot = float(np.sum((x - x_mean) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

    # Empirical 95% residual band around identity.
    abs_resid = np.abs(resid)
    delta = float(np.quantile(abs_resid, 0.975))

    # ------------------------------------------------------------------
    # Label worst outside-band items
    # ------------------------------------------------------------------
    outside_items = []
    outside_points = []

    for label, sub in df_plot.groupby("Label"):
        sub_x = sub["Observed"].to_numpy(dtype=float)
        sub_y = sub["Simulated"].to_numpy(dtype=float)
        ar = np.abs(sub_y - sub_x)

        if np.any(ar > delta):
            j = int(np.argmax(ar))
            outside_items.append(label)
            outside_points.append(
                (
                    float(sub_x[j]),
                    float(sub_y[j]),
                    label,
                    float(ar[j]),
                )
            )

    outside_points.sort(key=lambda t: t[3], reverse=True)
    outside_points = outside_points[:25]

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 10))

    type_styles = {
        "Phosphosite": {
            "color": "green",
            "alpha": 0.35,
            "s": 40,
            "label": "Phosphosite",
        },
        "ProteinAbundance": {
            "color": "blue",
            "alpha": 0.55,
            "s": 40,
            "label": "Protein abundance",
        },
        "mRNA": {
            "color": "purple",
            "alpha": 0.55,
            "s": 40,
            "label": "mRNA",
        },
    }

    plotted_labels = set()

    for row_type, sub in df_plot.groupby("Type"):
        style = type_styles.get(
            row_type,
            {
                "color": "gray",
                "alpha": 0.45,
                "s": 35,
                "label": str(row_type),
            },
        )

        legend_label = style["label"]
        if legend_label in plotted_labels:
            legend_label = None
        else:
            plotted_labels.add(style["label"])

        ax.scatter(
            sub["Observed"],
            sub["Simulated"],
            alpha=style["alpha"],
            color=style["color"],
            s=style["s"],
            label=legend_label,
        )

    # Identity line and residual band
    max_val = float(np.nanmax(np.r_[x, y]))
    min_val = float(np.nanmin(np.r_[x, y]))
    pad = 0.05 * (max_val - min_val + 1e-12)

    lo = min_val - pad
    hi = max_val + pad
    xx = np.array([lo, hi], dtype=float)

    ax.plot(xx, xx, "r--", lw=2, label="Identity (y=x)")
    ax.plot(xx, xx + delta, "k:", lw=1.5, label="95% band")
    ax.plot(xx, xx - delta, "k:", lw=1.5)

    # Label top outside-band points
    for px, py, lab, _dev in outside_points:
        ax.scatter(
            [px],
            [py],
            s=70,
            facecolors="none",
            edgecolors="black",
            linewidths=1.5,
        )
        ax.text(px, py, f"  {lab}", fontsize=9, va="center")

    # Metrics box
    type_counts = df_plot["Type"].value_counts().to_dict()
    type_counts_txt = "\n".join(
        f"{k}: {v}" for k, v in sorted(type_counts.items())
    )

    txt = (
        f"N={len(df_plot)}\n"
        f"R²={r2:.4f}\n"
        f"MSE={mse:.4g}\n"
        f"MAE={mae:.4g}\n"
        f"95% band: |sim-obs| ≤ {delta:.4g}\n"
        f"Outside band items: {len(set(outside_items))}\n"
        f"\n{type_counts_txt}"
    )

    ax.text(
        0.02,
        0.98,
        txt,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox=dict(
            boxstyle="round",
            facecolor="white",
            alpha=0.8,
            edgecolor="gray",
        ),
    )

    ax.set_xlabel("Observed")
    ax.set_ylabel("Simulated / fitted")
    ax.set_title("Goodness of Fit: Observed vs Simulated/Fitted")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.25)

    fig.tight_layout()

    os.makedirs(outdir, exist_ok=True)
    fig.savefig(os.path.join(outdir, "goodness_of_fit.png"), dpi=300)
    plt.close(fig)


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
        W_data (np.ndarray): Phosphosite weight matrix.
        W_data_prot (np.ndarray): Protein weight matrix.
        Cg (np.ndarray): Global connectivity matrix.
        Cl (np.ndarray): Local connectivity matrix.
        site_prot_idx (np.ndarray): Site-to-protein index mapping.
        kin_to_prot_idx (np.ndarray): Kinase-to-protein index mapping.
        K_site_kin (np.ndarray): Kinase-substrate relationship matrix.
        R (np.ndarray): Receptor input.
        L_alpha (np.ndarray): Laplacian matrix.
        receptor_mask_prot (np.ndarray): Receptor boolean mask for proteins.
        receptor_mask_kin (np.ndarray): Receptor boolean mask for kinases.
        xl (np.ndarray): Lower bounds for parameters.
        xu (np.ndarray): Upper bounds for parameters.
        args (Namespace): Parsed command-line arguments or configuration object.

    Returns:
        (None): Creates a 'preopt_snapshot' folder containing metadata and data files.
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
        t (np.ndarray): Pre-optimisation time vector.
        Y (np.ndarray): Phosphosite data matrix.
        P_scaled (np.ndarray): Scaled phosphosite data.
        A_data (np.ndarray): Protein abundance data.
        A_scaled (np.ndarray): Scaled protein abundance data.
        Cg (np.ndarray): Global connectivity matrix.
        Cl (np.ndarray): Local connectivity matrix.
        K_site_kin (np.ndarray): Kinase-site interaction matrix.
        R (np.ndarray): Receptor/input matrix.
        L_alpha (np.ndarray): Laplacian matrix for alpha term.
        W_data (np.ndarray): Phosphosite weight matrix.
        W_data_prot (np.ndarray): Protein weight matrix.
        site_prot_idx (np.ndarray): Site-to-protein index mapping.
        kin_to_prot_idx (np.ndarray): Kinase-to-protein index mapping.
        receptor_mask_prot (np.ndarray): Receptor mask for proteins.
        receptor_mask_kin (np.ndarray): Receptor mask for kinases.
        positions (np.ndarray): Initial positions array.
        xl (np.ndarray): Lower bounds for parameters.
        xu (np.ndarray): Upper bounds for parameters.
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
        (None): Writes ``mrna_fit_timeseries.tsv`` and ``mrna_diagnostics.tsv``
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
        rmse = float(np.sqrt(np.mean(resid ** 2)))
        ss_res = float(np.sum(resid ** 2))
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
        (None): Saves ``mrna_fit_panel.png`` to *outdir*.
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

    Args:
        outdir (str): Directory to save derived rate data.
        proteins (list[str]): List of protein names.
        t_protein (list[float]): Time points for protein data.
        k_act_fn (callable | None): Function to compute k_act.
        s_prod_fn (callable | None): Function to compute s_prod.
        t_rna (list[float] | None): Time points for RNA data.

    Returns
        None
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


# ---------------------------------------------------------------------------
# Neural ODE analysis helpers
# ---------------------------------------------------------------------------


def plot_neural_ode_overlay(
        outdir: str,
        *,
        ts: np.ndarray,
        ys: dict,
        proteins: list,
        sites: list,
        P_scaled: np.ndarray | None = None,
        A_scaled: np.ndarray | None = None,
        prot_idx_for_A: np.ndarray | None = None,
        t_protein: np.ndarray | None = None,
        t_rna: np.ndarray | None = None,
        rna_obs_matched: np.ndarray | None = None,
        rna_model_prot_idx: np.ndarray | None = None,
        k_act_init_vals: np.ndarray | None = None,
        s_prod_init_vals: np.ndarray | None = None,
        k_hats_obs: np.ndarray | None = None,
        s_hats_obs: np.ndarray | None = None,
) -> None:
    """Plot observed values and mechanistic + neural fit curves as overlays.

    For each protein, saves ``neural_overlay_{prot}.png`` in *outdir* showing:

    * Observed data (scatter markers).
    * Neural ODE fitted curve (solid line).
    * Mechanistic prior trajectory (dashed line, when *k_act_init_vals* /
      *s_prod_init_vals* are provided).
    * Learned k_act and s_prod curves in a bottom strip below the ODE state panels.

    Args:
        outdir:           Directory to write PNG files.
        ts:               Neural ODE evaluation time points (protein scale).
        ys:               Dict with ``"P_sim"`` ``(N, T)`` and ``"A_sim"`` ``(K, T)``.
        proteins:         List of protein names.
        sites:            List of phosphosite names.
        P_scaled:         Observed phosphosite data ``(N, T_prot)``.
        A_scaled:         Observed abundance data ``(n_obs, T_prot)``.
        prot_idx_for_A:   Protein indices for *A_scaled* rows.
        t_protein:        Protein time points (defaults to *ts*).
        t_rna:            RNA time points (optional).
        rna_obs_matched:  Matched RNA observations ``(n_matched, T_rna)``.
        rna_model_prot_idx: Protein indices for RNA gene rows.
        k_act_init_vals:  Mechanistic k_act ``(K, T_obs)``.
        s_prod_init_vals: Mechanistic s_prod ``(K, T_obs)``.
        k_hats_obs:       Neural k_hat ``(T_obs, K)``.
        s_hats_obs:       Neural s_hat ``(T_obs, K)``.

    Returns:
        None: PNG files are written to *outdir*.
    """
    os.makedirs(outdir, exist_ok=True)

    ts_arr = np.asarray(ts)
    t_prot = np.asarray(t_protein) if t_protein is not None else ts_arr
    P_sim = np.asarray(ys.get("P_sim", np.empty((0, len(ts_arr)))))
    A_sim = np.asarray(ys.get("A_sim", np.empty((0, len(ts_arr)))))

    has_mech_priors = k_act_init_vals is not None and s_prod_init_vals is not None
    has_neural_rates = k_hats_obs is not None and s_hats_obs is not None

    prot_to_obs_k: dict[int, int] = {}
    if prot_idx_for_A is not None and A_scaled is not None:
        for k, p_idx in enumerate(prot_idx_for_A):
            prot_to_obs_k[int(p_idx)] = k

    prot_to_rna_k: dict[int, int] = {}
    has_rna_global = False
    t_rna_arr = None
    if rna_model_prot_idx is not None and rna_obs_matched is not None and t_rna is not None and len(t_rna) > 0:
        for k, p_idx in enumerate(rna_model_prot_idx):
            prot_to_rna_k[int(p_idx)] = k
        has_rna_global = len(prot_to_rna_k) > 0
        t_rna_arr = np.asarray(t_rna)

    site_to_prot: dict[str, str] = {}
    for s_name in sites:
        parts = s_name.split("_", 1)
        site_to_prot[s_name] = parts[0]

    # Precompute O(1) site→index lookup to avoid O(N_sites) list.index() calls
    # inside the per-protein/per-site loop (which would be O(N_sites²) overall).
    site_to_index: dict[str, int] = {s: i for i, s in enumerate(sites)}

    cmap10 = plt.cm.tab10

    for p_idx, prot in enumerate(proteins):
        color = cmap10(p_idx % 10)
        prot_sites = [s for s in sites if site_to_prot.get(s) == prot]
        has_rna_for_prot = has_rna_global and (p_idx in prot_to_rna_k)

        # Number of ODE state panels + bottom parameter strip
        n_state_panels = (1 if has_rna_for_prot else 0) + 2  # mRNA + abundance + phospho
        n_rate_panels = 2 if (has_mech_priors or has_neural_rates) else 0
        n_rows = 2 if n_rate_panels > 0 else 1

        if n_rate_panels > 0:
            fig, axes = plt.subplots(
                n_rows, max(n_state_panels, n_rate_panels),
                figsize=(9 * max(n_state_panels, n_rate_panels), 12),
                gridspec_kw={"height_ratios": [3, 1], "wspace": 0.15, "hspace": 0.35},
                constrained_layout=False,
            )
            state_axes = axes[0, :n_state_panels]
            rate_axes = axes[1, :n_rate_panels]
            # Hide unused subplots in rate row
            for ax in axes[1, n_rate_panels:]:
                ax.set_visible(False)
        else:
            fig, axes_1d = plt.subplots(1, n_state_panels, figsize=(9 * n_state_panels, 7),
                                         gridspec_kw={"wspace": 0.15}, constrained_layout=True)
            state_axes = [axes_1d] if n_state_panels == 1 else list(axes_1d)
            rate_axes = []

        panel_idx = 0

        # mRNA panel
        if has_rna_for_prot and t_rna_arr is not None:
            ax_rna = state_axes[panel_idx]
            panel_idx += 1
            rna_k = prot_to_rna_k[p_idx]
            y_obs_rna = np.asarray(rna_obs_matched[rna_k], dtype=float)
            ax_rna.scatter(t_rna_arr, y_obs_rna, s=50, color=color, zorder=5, label="mRNA (obs)")
            ax_rna.set_title("mRNA / R(t)", fontsize=12, fontweight="bold")
            ax_rna.set_xlabel("Time (min)")
            ax_rna.set_ylabel("mRNA fold-change")
            ax_rna.legend(fontsize=9)
            ax_rna.grid(alpha=0.25)

        # Abundance panel
        ax_prot = state_axes[panel_idx]
        panel_idx += 1
        if p_idx < A_sim.shape[0]:
            ax_prot.plot(ts_arr, A_sim[p_idx], "-", lw=3, color=color, label="Abundance (neural)")
        obs_k = prot_to_obs_k.get(p_idx)
        if obs_k is not None and A_scaled is not None:
            y_obs_A = np.asarray(A_scaled[obs_k], dtype=float)
            m = np.isfinite(y_obs_A)
            if np.any(m):
                ax_prot.plot(t_prot[m], y_obs_A[m], "--", lw=2, alpha=0.6, color=color, label="Abundance (obs)")
                ax_prot.scatter(t_prot[m], y_obs_A[m], marker="s", s=55, alpha=0.7, color=color, edgecolors="none")
        ax_prot.set_title("Protein abundance / A(t)", fontsize=12, fontweight="bold")
        ax_prot.set_xlabel("Time (min)")
        ax_prot.set_ylabel("Protein abundance")
        ax_prot.legend(fontsize=9)
        ax_prot.grid(alpha=0.25)

        # Phosphosites panel
        ax_sites = state_axes[panel_idx]
        cmap20 = plt.cm.tab20
        for si, site in enumerate(prot_sites):
            s_idx = site_to_index.get(site, -1)
            if s_idx < 0:
                continue
            if s_idx >= P_sim.shape[0]:
                continue
            c = cmap20(si % 20)
            residue = site.split("_", 1)[1] if "_" in site else site
            ax_sites.plot(ts_arr, P_sim[s_idx], "-", lw=3, color=c, label=f"{residue} (neural)")
            if P_scaled is not None and s_idx < P_scaled.shape[0]:
                y_obs_p = np.asarray(P_scaled[s_idx], dtype=float)
                m = np.isfinite(y_obs_p)
                if np.any(m):
                    ax_sites.plot(t_prot[m], y_obs_p[m], "--", lw=2, alpha=0.45, color=c)
                    ax_sites.scatter(t_prot[m], y_obs_p[m], marker="s", s=45, alpha=0.6, color=c, edgecolors="none")
        if not prot_sites:
            ax_sites.text(0.5, 0.5, "No phosphosites", transform=ax_sites.transAxes,
                          ha="center", va="center", fontsize=10, alpha=0.7)
        ax_sites.set_title("Phosphosites", fontsize=12, fontweight="bold")
        ax_sites.set_xlabel("Time (min)")
        ax_sites.set_ylabel("Relative signal p(t)")
        ax_sites.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1.0),
                        borderaxespad=0.0, frameon=True)
        ax_sites.grid(alpha=0.25)

        # --- Bottom rate strip ---
        if n_rate_panels > 0:
            # k_act panel
            ax_k = rate_axes[0]
            if has_mech_priors and p_idx < k_act_init_vals.shape[0]:
                ax_k.plot(ts_arr, k_act_init_vals[p_idx], "--", lw=1.5, color="gray", label="k_act mech")
            if has_neural_rates and p_idx < k_hats_obs.shape[1]:
                ax_k.plot(ts_arr, k_hats_obs[:, p_idx], "-", lw=1.5, color=color, label="k_act neural")
            ax_k.set_title("k_act(t)", fontsize=10)
            ax_k.set_xlabel("Time (min)")
            ax_k.set_ylabel("k_act")
            ax_k.legend(fontsize=8)
            ax_k.grid(alpha=0.2)

            # s_prod panel
            ax_s = rate_axes[1]
            if has_mech_priors and p_idx < s_prod_init_vals.shape[0]:
                ax_s.plot(ts_arr, s_prod_init_vals[p_idx], "--", lw=1.5, color="gray", label="s_prod mech")
            if has_neural_rates and p_idx < s_hats_obs.shape[1]:
                ax_s.plot(ts_arr, s_hats_obs[:, p_idx], "-", lw=1.5, color=color, label="s_prod neural")
            ax_s.set_title("s_prod(t)", fontsize=10)
            ax_s.set_xlabel("Time (min)")
            ax_s.set_ylabel("s_prod")
            ax_s.legend(fontsize=8)
            ax_s.grid(alpha=0.2)

        fig.suptitle(f"{prot} — Neural ODE overlay", fontsize=14, fontweight="bold", y=1.01)
        _path = os.path.join(outdir, f"neural_overlay_{prot}.png")
        fig.savefig(_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info("[neural_ode] Saved overlay plot %s", _path)


def save_neural_ode_residuals(
        outdir: str,
        *,
        ts: np.ndarray,
        ys: dict,
        proteins: list,
        sites: list,
        P_scaled: np.ndarray | None = None,
        A_scaled: np.ndarray | None = None,
        prot_idx_for_A: np.ndarray | None = None,
        t_protein: np.ndarray | None = None,
        t_rna: np.ndarray | None = None,
        rna_obs_matched: np.ndarray | None = None,
        rna_model_prot_idx: np.ndarray | None = None,
        mech_P_sim: np.ndarray | None = None,
        mech_A_sim: np.ndarray | None = None,
        mech_R_sim: np.ndarray | None = None,
        mech_t: np.ndarray | None = None,
        mech_t_rna: np.ndarray | None = None,
) -> None:
    """Save per-row time-wise residuals for each ODE state.

    Writes ``neural_residuals.tsv`` to *outdir* in long format.  Each row
    records the entity, time, neural residual (neural_fit − observed), and —
    when mechanistic simulation arrays are provided — the mechanistic residual
    (mech_fit − observed) for direct comparison.

    Time scales:

    * mRNA / R(t): uses *t_rna* (RNA-specific time axis).
    * Protein abundance / A(t): uses *t_protein* (protein time axis).
    * Phosphosite / P(t): uses *t_protein* (protein time axis).

    Args:
        outdir:         Directory to write ``neural_residuals.tsv``.
        ts:             Neural ODE evaluation time points (protein scale).
        ys:             Dict with ``"P_sim"`` ``(N, T)`` and ``"A_sim"`` ``(K, T)``.
        proteins:       List of protein names.
        sites:          List of phosphosite names.
        P_scaled:       Observed phosphosite data ``(N, T_prot)``.
        A_scaled:       Observed abundance data ``(n_obs, T_prot)``.
        prot_idx_for_A: Protein indices for *A_scaled* rows.
        t_protein:      Protein time points (defaults to *ts*).
        t_rna:          RNA time points.
        rna_obs_matched: Matched RNA observations ``(n_matched, T_rna)``.
        rna_model_prot_idx: Protein indices for each RNA gene row.
        mech_P_sim:     Mechanistic phosphosite simulation ``(N, T_mech)``.
        mech_A_sim:     Mechanistic abundance simulation ``(K, T_mech)``.
        mech_R_sim:     Mechanistic mRNA simulation ``(K, T_mech_rna)`` — must be
                        paired with the RNA time grid, not the protein time grid.
        mech_t:         Time points for mechanistic P/A simulation (protein grid;
                        defaults to *t_protein*).
        mech_t_rna:     Time points for mechanistic mRNA simulation (RNA grid).
                        When provided its length must equal ``mech_R_sim.shape[1]``.
                        If omitted, ``t_rna`` is used as a fallback.

    Returns:
        None: ``neural_residuals.tsv`` is written to *outdir*.
    """
    os.makedirs(outdir, exist_ok=True)

    ts_arr = np.asarray(ts)
    t_prot = np.asarray(t_protein) if t_protein is not None else ts_arr
    t_mech = np.asarray(mech_t) if mech_t is not None else t_prot
    t_mech_rna = np.asarray(mech_t_rna) if mech_t_rna is not None else None
    P_sim = np.asarray(ys.get("P_sim", np.empty((0, len(ts_arr)))))
    A_sim = np.asarray(ys.get("A_sim", np.empty((0, len(ts_arr)))))

    prot_to_obs_k: dict[int, int] = {}
    if prot_idx_for_A is not None and A_scaled is not None:
        for k, p_idx in enumerate(prot_idx_for_A):
            prot_to_obs_k[int(p_idx)] = k

    prot_to_rna_k: dict[int, int] = {}
    t_rna_arr = None
    if rna_model_prot_idx is not None and rna_obs_matched is not None and t_rna is not None and len(t_rna) > 0:
        for k, p_idx in enumerate(rna_model_prot_idx):
            prot_to_rna_k[int(p_idx)] = k
        t_rna_arr = np.asarray(t_rna)

    site_to_prot: dict[str, str] = {}
    for s_name in sites:
        parts = s_name.split("_", 1)
        site_to_prot[s_name] = parts[0]

    rows: list[dict] = []

    # Phosphosite residuals
    for s_idx, site in enumerate(sites):
        if s_idx >= P_sim.shape[0]:
            continue
        y_neural = P_sim[s_idx]  # (T_neural,)
        y_obs = np.asarray(P_scaled[s_idx], dtype=float) if P_scaled is not None and s_idx < P_scaled.shape[0] else None

        for ti_idx, t_val in enumerate(ts_arr):
            # Map neural time point to the nearest observed (t_prot) index.
            ti_prot = int(np.argmin(np.abs(t_prot - t_val))) if len(t_prot) > 0 else ti_idx
            obs_val = float(y_obs[ti_prot]) if (y_obs is not None and ti_prot < len(y_obs)) else float("nan")
            neural_val = float(y_neural[ti_idx])
            neural_resid = neural_val - obs_val if np.isfinite(obs_val) else float("nan")

            # Mechanistic residual at the nearest time index (reuse mech_ti below).
            mech_resid = float("nan")
            mech_val_out = float("nan")
            if mech_P_sim is not None and s_idx < mech_P_sim.shape[0]:
                mech_ti = int(np.argmin(np.abs(t_mech - t_val))) if len(t_mech) > 0 else 0
                mech_val_out = float(mech_P_sim[s_idx, mech_ti])
                mech_resid = mech_val_out - obs_val if np.isfinite(obs_val) else float("nan")

            rows.append({
                "entity_type": "phosphosite",
                "entity": site,
                "protein": site_to_prot.get(site, ""),
                "time": float(t_val),
                "value_observed": obs_val,
                "value_neural": neural_val,
                "residual_neural": neural_resid,
                "value_mechanistic": mech_val_out,
                "residual_mechanistic": mech_resid,
            })

    # Abundance residuals
    for p_idx, prot in enumerate(proteins):
        if p_idx >= A_sim.shape[0]:
            continue
        y_neural = A_sim[p_idx]
        obs_k = prot_to_obs_k.get(p_idx)
        y_obs = np.asarray(A_scaled[obs_k], dtype=float) if (obs_k is not None and A_scaled is not None) else None

        for ti_idx, t_val in enumerate(ts_arr):
            # Map neural time point to the nearest observed (t_prot) index.
            ti_prot = int(np.argmin(np.abs(t_prot - t_val))) if len(t_prot) > 0 else ti_idx
            obs_val = float(y_obs[ti_prot]) if (y_obs is not None and ti_prot < len(y_obs)) else float("nan")
            neural_val = float(y_neural[ti_idx])
            neural_resid = neural_val - obs_val if np.isfinite(obs_val) else float("nan")

            mech_resid = float("nan")
            mech_val_out = float("nan")
            if mech_A_sim is not None and p_idx < mech_A_sim.shape[0]:
                mech_ti = int(np.argmin(np.abs(t_mech - t_val))) if len(t_mech) > 0 else 0
                mech_val_out = float(mech_A_sim[p_idx, mech_ti])
                mech_resid = mech_val_out - obs_val if np.isfinite(obs_val) else float("nan")

            rows.append({
                "entity_type": "abundance",
                "entity": prot,
                "protein": prot,
                "time": float(t_val),
                "value_observed": obs_val,
                "value_neural": neural_val,
                "residual_neural": neural_resid,
                "value_mechanistic": mech_val_out,
                "residual_mechanistic": mech_resid,
            })

    # mRNA residuals — populate value_neural / residual_neural from R_sim when available.
    R_sim_ys = ys.get("R_sim") if ys is not None else None
    R_sim_arr = np.asarray(R_sim_ys) if R_sim_ys is not None else None

    if t_rna_arr is not None:
        for p_idx, prot in enumerate(proteins):
            rna_k = prot_to_rna_k.get(p_idx)
            if rna_k is None:
                continue
            y_obs_rna = np.asarray(rna_obs_matched[rna_k], dtype=float)

            for ti_idx, t_val in enumerate(t_rna_arr):
                obs_val = float(y_obs_rna[ti_idx]) if ti_idx < len(y_obs_rna) else float("nan")

                # Neural mRNA from R_sim (shape K × T_rna when available).
                if R_sim_arr is not None and p_idx < R_sim_arr.shape[0] and ti_idx < R_sim_arr.shape[1]:
                    neural_rna_val = float(R_sim_arr[p_idx, ti_idx])
                    neural_rna_resid = neural_rna_val - obs_val if np.isfinite(obs_val) else float("nan")
                else:
                    neural_rna_val = float("nan")
                    neural_rna_resid = float("nan")

                mech_val_out = float("nan")
                mech_resid = float("nan")
                if mech_R_sim is not None and p_idx < mech_R_sim.shape[0]:
                    # mech_R_sim is on the RNA grid — must NOT use the protein-grid
                    # t_mech for indexing.  Resolve the correct RNA reference axis:
                    if t_mech_rna is not None and len(t_mech_rna) == mech_R_sim.shape[1]:
                        t_ref = t_mech_rna
                    elif t_rna_arr is not None and len(t_rna_arr) == mech_R_sim.shape[1]:
                        t_ref = t_rna_arr
                    else:
                        raise ValueError(
                            "save_neural_ode_residuals: mech_R_sim has shape "
                            f"{mech_R_sim.shape} but no compatible RNA time axis was "
                            f"supplied. len(t_rna)={len(t_rna_arr) if t_rna_arr is not None else None}, "
                            f"len(mech_t)={len(t_mech)}, "
                            f"len(mech_t_rna)={len(t_mech_rna) if t_mech_rna is not None else None}. "
                            "Pass mech_t_rna matching the RNA grid to save_neural_ode_residuals."
                        )
                    mech_ti = int(np.argmin(np.abs(t_ref - t_val))) if len(t_ref) > 0 else 0
                    if mech_ti >= mech_R_sim.shape[1]:
                        raise IndexError(
                            f"save_neural_ode_residuals: mech_ti={mech_ti} is out of bounds "
                            f"for mech_R_sim axis 1 with size {mech_R_sim.shape[1]}. "
                            f"t_ref has length {len(t_ref)}, t_val={t_val}."
                        )
                    mech_val_out = float(mech_R_sim[p_idx, mech_ti])
                    mech_resid = mech_val_out - obs_val if np.isfinite(obs_val) else float("nan")

                rows.append({
                    "entity_type": "mrna",
                    "entity": prot,
                    "protein": prot,
                    "time": float(t_val),
                    "value_observed": obs_val,
                    "value_neural": neural_rna_val,
                    "residual_neural": neural_rna_resid,
                    "value_mechanistic": mech_val_out,
                    "residual_mechanistic": mech_resid,
                })

    df_resid = pd.DataFrame(rows)
    resid_path = os.path.join(outdir, "neural_residuals.tsv")
    df_resid.to_csv(resid_path, sep="\t", index=False)
    logger.info("[neural_ode] Saved residuals to %s (%d rows)", resid_path, len(rows))


def plot_neural_residuals(
        outdir: str,
        *,
        residuals_tsv: str | None = None,
) -> None:
    """Visualise and compare neural vs mechanistic residuals.

    Reads ``neural_residuals.tsv`` (either from *residuals_tsv* or from
    *outdir*/neural_residuals.tsv) and generates:

    * ``neural_residuals_heatmap_phospho.png`` – heatmap of neural phosphosite
      residuals (sites × time).
    * ``neural_vs_mech_residuals.png`` – scatter of neural residual vs
      mechanistic residual for all entity types, coloured by entity type.

    Args:
        outdir:         Directory to save PNG files (and to find the TSV when
                        *residuals_tsv* is None).
        residuals_tsv:  Path to ``neural_residuals.tsv``.  Defaults to
                        ``outdir/neural_residuals.tsv``.

    Returns:
        None: PNG files are written to *outdir*.
    """
    os.makedirs(outdir, exist_ok=True)

    tsv_path = residuals_tsv or os.path.join(outdir, "neural_residuals.tsv")
    if not os.path.exists(tsv_path):
        logger.warning("[neural_ode] neural_residuals.tsv not found at %s; skipping plot.", tsv_path)
        return

    df = pd.read_csv(tsv_path, sep="\t")

    # ------------------------------------------------------------------ #
    # 1. Neural phosphosite residuals heatmap                             #
    # ------------------------------------------------------------------ #
    df_p = df[df["entity_type"] == "phosphosite"].copy()
    if not df_p.empty and "residual_neural" in df_p.columns:
        try:
            import seaborn as sns  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "[neural_ode] seaborn is required for the phosphosite residual heatmap. "
                "Install it with: pip install seaborn"
            ) from exc
        pivot = df_p.pivot_table(index="entity", columns="time", values="residual_neural", aggfunc="mean")
        n_sites = len(pivot)
        fig_h = max(6, min(40, n_sites * 0.35))
        fig, ax = plt.subplots(figsize=(max(8, len(pivot.columns) * 1.2), fig_h))
        sns.heatmap(pivot, ax=ax, cmap="vlag", center=0,
                    xticklabels=[f"{c:.0f}" for c in pivot.columns],
                    yticklabels=True, linewidths=0)
        ax.set_title("Neural ODE — Phosphosite Residuals (neural − observed)", fontsize=12)
        ax.set_xlabel("Time (min)")
        ax.set_ylabel("Phosphosite")
        plt.tight_layout()
        _path = os.path.join(outdir, "neural_residuals_heatmap_phospho.png")
        fig.savefig(_path, dpi=300)
        plt.close(fig)
        logger.info("[neural_ode] Saved %s", _path)

    # ------------------------------------------------------------------ #
    # 2. Neural vs mechanistic residuals scatter                          #
    # ------------------------------------------------------------------ #
    df_valid = df.dropna(subset=["residual_neural", "residual_mechanistic"])
    if len(df_valid) >= 3:
        fig, ax = plt.subplots(figsize=(8, 8))
        entity_types = df_valid["entity_type"].unique()
        colors = plt.cm.tab10(np.arange(len(entity_types)) / len(entity_types))
        for et, col in zip(entity_types, colors):
            sub = df_valid[df_valid["entity_type"] == et]
            ax.scatter(sub["residual_mechanistic"], sub["residual_neural"],
                       s=20, alpha=0.6, color=col, label=et)
        lim = float(max(df_valid[["residual_neural", "residual_mechanistic"]].abs().max().max(), 1e-6))
        ax.axline((0, 0), slope=1, color="gray", lw=1.5, linestyle="--", label="identity")
        ax.axhline(0, color="gray", lw=0.8, alpha=0.5)
        ax.axvline(0, color="gray", lw=0.8, alpha=0.5)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_xlabel("Mechanistic residual (mech − observed)")
        ax.set_ylabel("Neural residual (neural − observed)")
        ax.set_title("Neural ODE vs Mechanistic Residuals")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.25)
        plt.tight_layout()
        _path = os.path.join(outdir, "neural_vs_mech_residuals.png")
        fig.savefig(_path, dpi=300)
        plt.close(fig)
        logger.info("[neural_ode] Saved %s", _path)
