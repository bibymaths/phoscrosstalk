"""
analysis.py
Post-optimization analysis, file export, and plotting.
"""
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from phoscrosstalk.core_mechanisms import decode_theta
from phoscrosstalk.config import ModelDims, DEFAULT_TIMEPOINTS
from phoscrosstalk.simulation import simulate_p_scipy
from phoscrosstalk.optimization import build_full_A0, bio_score


def save_pareto_results(outdir, F, X, f1, f2, f3, J, F_best):
    """
    Save Pareto front optimization results, including statistics and scalarized scores, to disk.

    Args:
        outdir (str): Path to the output directory.
        F (np.ndarray): Array of objective values for the Pareto front (shape: n_points x 3).
        X (np.ndarray): Array of parameter values for the Pareto front.
        f1 (np.ndarray): Array of values for objective 1 (Phosphosite error).
        f2 (np.ndarray): Array of values for objective 2 (Protein error).
        f3 (np.ndarray): Array of values for objective 3 (Complexity/Regularization).
        J (np.ndarray): Array of scalarized objective values.
        F_best (np.ndarray): The objective values corresponding to the best scalarized solution.

    Returns:
        None: Files are written to `outdir` (pareto_stats.tsv, pareto_front_with_J.tsv, etc.).
    """
    obj_names = ["f1_P_sites", "f2_protein", "f3_complexity"]

    # DataFrame stats
    df_F = pd.DataFrame(F, columns=obj_names)
    summary = pd.DataFrame({
        "objective": obj_names,
        "min": df_F.min().values, "mean": df_F.mean().values,
        "median": df_F.median().values, "std": df_F.std(ddof=1).values
    })

    summary.to_csv(os.path.join(outdir, "pareto_stats.tsv"), sep="\t", index=False)

    df_front = df_F.copy()
    df_front["J_scalarized"] = J
    df_front.to_csv(os.path.join(outdir, "pareto_front_with_J.tsv"), sep="\t", index=False)

    bio_scores = np.array([bio_score(theta) for theta in X])
    df_front["bio_score"] = bio_scores
    df_front.to_csv(os.path.join(outdir, "pareto_points.tsv"), sep="\t", index=False)

    np.savez(os.path.join(outdir, "pareto_front.npz"), F=F, X=X, J=J)


def plot_pareto_diagnostics(outdir, F, F_best, f1, f2, f3, X):
    """
    Plot diagnostic visualizations for the Pareto front analysis.

    Generates a scatter plot of objective space (f1 vs f2 colored by f3) and
    a heatmap of parameter correlations.

    Args:
        outdir (str): Path to the output directory.
        F (np.ndarray): Pareto front objective values.
        F_best (np.ndarray): Objective values of the best selected point.
        f1 (np.ndarray): Values for objective function 1.
        f2 (np.ndarray): Values for objective function 2.
        f3 (np.ndarray): Values for objective function 3.
        X (np.ndarray): Parameter values associated with the Pareto front.

    Returns:
        None: Saves 'pareto_f1_f2.png' and 'pareto_param_corr.png' to `outdir`.
    """
    # F1 vs F2
    plt.figure(figsize=(7, 6))
    sc = plt.scatter(f1, f2, c=f3, cmap="viridis", alpha=0.7)
    plt.colorbar(sc, label="f3")
    plt.scatter(F_best[0], F_best[1], s=120, facecolors="none", edgecolors="red", linewidths=2)
    plt.title("Pareto Front: f1 vs f2")
    plt.savefig(os.path.join(outdir, "pareto_f1_f2.png"), dpi=300)
    plt.close()

    # Param Correlation
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(np.corrcoef(X.T), ax=ax, cmap="coolwarm", center=0.0, square=True)
    fig.savefig(os.path.join(outdir, "pareto_param_corr.png"), dpi=300)
    plt.close(fig)


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

    # Protein-specific parameters
    df_prot = pd.DataFrame({
        'Protein': proteins,
        'k_act (Activation)': params_decoded[0],
        'k_deact (Deactivation)': params_decoded[1],
        's_prod (Synthesis)': params_decoded[2],
        'd_deg (Degradation)': params_decoded[3]
    })
    df_prot.to_csv(os.path.join(outdir, "parameter_summary_proteins.tsv"), sep="\t", index=False)

    # Kinase-specific parameters
    df_kin = pd.DataFrame({
        'Kinase': kinases,
        'Alpha (Global Str)': params_decoded[6],
        'kK_act (Kinase Act)': params_decoded[7],
        'kK_deact (Kinase Deact)': params_decoded[8]
    })
    df_kin.to_csv(os.path.join(outdir, "parameter_summary_kinases.tsv"), sep="\t", index=False)

    # Site-specific parameters
    df_site = pd.DataFrame({
        'Site': sites,
        'k_off (Phosphatase Rate)': params_decoded[9]
    })
    df_site.to_csv(os.path.join(outdir, "parameter_summary_sites.tsv"), sep="\t", index=False)

    # Global parameters
    with open(os.path.join(outdir, "parameter_summary_global.txt"), "w") as f:
        f.write("=== Global Coupling Parameters ===\n")
        f.write(f"beta_g (Global Coupling): {params_decoded[4]:.5f}\n")
        f.write(f"beta_l (Local Coupling):  {params_decoded[5]:.5f}\n")
        f.write("-" * 40 + "\n")

    # Print the summary to console as well
    print("=== Parameter Summary ===")
    print("\n--- Protein-specific Parameters ---")
    print(df_prot.to_string(index=False))
    print("\n--- Kinase-specific Parameters ---")
    print(df_kin.to_string(index=False))
    print("\n--- Site-specific Parameters ---")
    print(df_site.to_string(index=False))
    print("\n--- Global Coupling Parameters ---")
    print(f"beta_g (Global Coupling): {params_decoded[4]:.5f}")
    print(f"beta_l (Local Coupling):  {params_decoded[5]:.5f}")
    print("-" * 40 + "\n")


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
        A_data, A_bases, A_amps,
        mechanism,
        Cg, Cl, site_prot_idx, K_site_kin, R, L_alpha, kin_to_prot_idx,
        mask_p, mask_k,
):
    """
    Run a simulation with optimized parameters, rescale outputs, and save comparison data.

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

    Returns:
        None: Saves 'fitted_params.npz' and 'fit_timeseries.tsv' to `outdir`.
    """
    K, M, N = ModelDims.K, ModelDims.M, ModelDims.N

    # Save Params (unchanged)
    params_decoded = decode_theta(theta_opt, K, M, N)
    param_names = [
        "k_act", "k_deact", "s_prod", "d_deg",
        "beta_g", "beta_l",
        "alpha", "kK_act", "kK_deact", "k_off",
        "gamma_S_p", "gamma_A_S", "gamma_A_p", "gamma_K_net",
    ]
    save_dict = {"theta": theta_opt, "proteins": np.array(proteins), "sites": np.array(sites)}
    for name, val in zip(param_names, params_decoded):
        save_dict[name] = val
    np.savez(os.path.join(outdir, "fitted_params.npz"), **save_dict)

    # Re-simulate
    A0_full = build_full_A0(K, len(t), A_scaled, prot_idx_for_A)
    P_sim, A_sim = simulate_p_scipy(
        t, P_scaled, A0_full, theta_opt,
        Cg, Cl, site_prot_idx,
        K_site_kin, R, L_alpha, kin_to_prot_idx,
        mask_p, mask_k,
        mechanism,
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


def plot_fitted_simulation(outdir):
    """
    Generate per-protein plots comparing simulated trajectories to experimental data.

    Reads the 'fit_timeseries.tsv' generated by `save_fitted_simulation` and creates
    a panel plot for every protein, showing protein abundance (if available) and
    associated phosphosite dynamics.

    Args:
        outdir (str): Directory containing the 'fit_timeseries.tsv' file.

    Returns:
        None: Saves individual PNG files (e.g., 'fit_ProteinName.png') to `outdir`.
    """

    # Load Data
    df = pd.read_csv(os.path.join(outdir, "fit_timeseries.tsv"), sep="\t")
    proteins = sorted(df["Protein"].unique())
    print(f"[*] Found {len(proteins)} proteins")
    df_sites = df[df["Type"] == "Phosphosite"].reset_index(drop=True)
    df_prots = df[df["Type"] == "ProteinAbundance"].reset_index(drop=True)

    sim_cols = [col for col in df.columns if col.startswith("sim_t")]
    data_cols = [col for col in df.columns if col.startswith("data_t")]
    t_vals = DEFAULT_TIMEPOINTS

    # Defensive: ensure time axis matches column count
    # (keeps logic; only guards mismatch)
    if len(sim_cols) != len(t_vals) or len(data_cols) != len(t_vals):
        # fall back to numeric range to avoid hard crash
        t_vals = np.arange(len(sim_cols), dtype=float)

    # Plot per Protein
    for prot in proteins:
        print(f"   → Plotting {prot}")

        fig, (axP, axS) = plt.subplots(
            1, 2, figsize=(18, 7), sharex=True, sharey=True,
            gridspec_kw={"wspace": 0.10}
        )

        # -------------------------
        # LEFT PANEL: Protein abundance
        # -------------------------
        row_prot = df_prots[df_prots["Protein"] == prot]
        if not row_prot.empty:
            row_prot = row_prot.iloc[0]

            y_sim = row_prot[sim_cols].values.astype(float)
            y_dat = row_prot[data_cols].values.astype(float)
            mask_dat = np.isfinite(y_dat)
            has_data = bool(np.any(mask_dat))

            # One color for this protein trajectory
            # (use tab10 index derived from protein order for stability)
            color = plt.cm.tab10(proteins.index(prot) % 10)

            # Model: thick line, no markers
            axP.plot(
                t_vals, y_sim,
                "-", lw=4.0, alpha=1.0, color=color,
                label="Protein (model)"
            )

            # Data: lighter line + square markers, same color
            if has_data:
                axP.plot(
                    np.asarray(t_vals)[mask_dat], y_dat[mask_dat],
                    "-", lw=2.0, alpha=0.35, color=color,
                    label="Protein (data)"
                )
                axP.scatter(
                    np.asarray(t_vals)[mask_dat], y_dat[mask_dat],
                    marker="s", s=55, alpha=0.6, color=color, edgecolors="none"
                )
        else:
            axP.text(
                0.5, 0.5, "No protein abundance row",
                transform=axP.transAxes, ha="center", va="center", fontsize=10, alpha=0.7
            )

        # -------------------------
        # RIGHT PANEL: Phosphosites
        # -------------------------
        sub = df_sites[df_sites["Protein"] == prot]

        if sub.empty:
            axS.text(
                0.5, 0.5, "No phosphosites",
                transform=axS.transAxes, ha="center", va="center", fontsize=10, alpha=0.7
            )
        else:
            # Colors per site within the protein (consistent between model & data)
            # If many sites, tab20 will repeat; still usable.
            cmap = plt.cm.tab20
            n_sites = len(sub)

            for i, (_, row) in enumerate(sub.iterrows()):
                res = row.get("Residue", "")
                pos = row.get("Position", row.get("Pos", row.get("SitePos", "")))  # robust lookup
                # Label uses both residue and position if available
                if pd.notna(pos) and str(pos) != "":
                    site_label = f"{res}_{pos}"
                else:
                    site_label = f"{res}"

                y_sim = row[sim_cols].values.astype(float)
                y_dat = row[data_cols].values.astype(float)
                mask_dat = np.isfinite(y_dat)
                has_data = bool(np.any(mask_dat))

                c = cmap(i % 20)

                # Model: thick line, no markers
                axS.plot(
                    t_vals, y_sim,
                    "-", lw=4.0, alpha=1.0, color=c,
                    label=f"{site_label} (model)"
                )

                # Data: lighter line + square markers, same color
                if has_data:
                    axS.plot(
                        np.asarray(t_vals)[mask_dat], y_dat[mask_dat],
                        "-", lw=2.0, alpha=0.35, color=c,
                        label=f"{site_label} (data)"
                    )
                    axS.scatter(
                        np.asarray(t_vals)[mask_dat], y_dat[mask_dat],
                        marker="s", s=45, alpha=0.6, color=c, edgecolors="none"
                    )

        # -------------------------
        # Formatting
        # -------------------------
        fig.suptitle(f"{prot}", fontsize=14, fontweight="bold", y=0.98)

        axP.set_title("Protein abundance", fontsize=12, fontweight="bold")
        axS.set_title("Phosphosites (Residue + Position)", fontsize=12, fontweight="bold")

        for ax in (axP, axS):
            ax.grid(alpha=0.25)
            ax.set_xlabel("Time (min)")

        axP.set_ylabel("FC / Scaled abundance")

        # Legends: keep readable, separate per panel
        # Protein panel small legend; site panel can be large -> place outside
        axP.legend(fontsize=9, loc="best")

        # For phosphosites, put legend outside to avoid covering curves
        axS.legend(
            fontsize=8,
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            borderaxespad=0.0,
            frameon=True
        )

        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"fit_{prot}.png"), dpi=300, bbox_inches="tight")
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

    print("[*] Biological Scores:")
    for i, score in enumerate(bio_scores):
        print(f"   → Point {i}: Bio Score = {score:.6f}")


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
    """
    df = pd.read_csv(file, sep="\t")

    sim_cols = [c for c in df.columns if c.startswith("sim_t")]
    data_cols = [c for c in df.columns if c.startswith("data_t")]

    if len(sim_cols) == 0 or len(data_cols) == 0:
        raise ValueError(f"No sim_t*/data_t* columns found in {file}")

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
        raise ValueError("Not enough finite points to compute fit metrics.")

    x = data_all[mask_all]
    y = sim_all[mask_all]
    resid = y - x

    # Global metrics
    mse = float(np.mean((y - x) ** 2))
    mae = float(np.mean(np.abs(y - x)))

    # R^2 (safe)
    y_mean = float(np.mean(y))
    ss_res = float(np.sum((y - x) ** 2))  # here "residual" is y-x (since identity is the target)
    ss_tot = float(np.sum((y - y_mean) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

    # --- 95% CI band: parallel to identity line ---
    # Interpret as 95% of residuals around identity (robust to heavy tails):
    # Use empirical 97.5th percentile of |residual| as band half-width.
    abs_resid = np.abs(resid)
    delta = float(np.quantile(abs_resid, 0.975))

    # Identify outside-band points at the per-item level (row label)
    # We'll label only those with any timepoint outside band, and choose the worst deviation.
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
            outside_points.append((float(data_vals[m][j]), float(sim_vals[m][j]), row["Label"], float(ar[j])))

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
            label="Phosphosite" if idx == df[df["Type"] == "Phosphosite"].index[0] else None,
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
            label="Abundance" if idx == df[df["Type"] != "Phosphosite"].index[0] else None,
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
    for (px, py, lab, dev) in outside_points:
        plt.scatter([px], [py], s=70, facecolors="none", edgecolors="black", linewidths=1.5)
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
        0.02, 0.98, txt,
        transform=plt.gca().transAxes,
        va="top", ha="left",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="gray")
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
    np.savetxt(path, np.asarray(vec, dtype=int).reshape(-1, 1), fmt="%d", delimiter="\t")


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
    """
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

    lines.extend([
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
    ])
    _save_txt(os.path.join(snap_dir, "meta.txt"), "\n".join(lines))

    # --- labels (txt) ---
    _save_txt(os.path.join(snap_dir, "sites.txt"), "\n".join(map(str, sites)))
    _save_txt(os.path.join(snap_dir, "proteins.txt"), "\n".join(map(str, proteins)))
    _save_txt(os.path.join(snap_dir, "kinases.txt"), "\n".join(map(str, kinases)))
    if A_proteins is not None:
        _save_txt(os.path.join(snap_dir, "A_proteins.txt"), "\n".join(map(str, list(A_proteins))))
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
    _save_index_tsv(os.path.join(snap_dir, "receptor_mask_prot.tsv"), receptor_mask_prot)
    _save_index_tsv(os.path.join(snap_dir, "receptor_mask_kin.tsv"), receptor_mask_kin)

    _save_vector_tsv(os.path.join(snap_dir, "xl.tsv"), xl)
    _save_vector_tsv(os.path.join(snap_dir, "xu.tsv"), xu)
