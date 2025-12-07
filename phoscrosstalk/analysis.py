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
    Y,             # <--- add this: original site data (FC)
    A_data, A_bases, A_amps,
    mechanism,
    Cg, Cl, site_prot_idx, K_site_kin, R, L_alpha, kin_to_prot_idx,
    mask_p, mask_k,
):
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
    sim_cols  = [f"sim_t{j}"  for j in range(T)]
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
            record[sim_cols[j]]  = Y_sim_rescaled[i, j]
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
                record[sim_cols[j]]  = A_sim_rescaled[p_idx, j]
                record[data_cols[j]] = A_data[k, j]
            records.append(record)

    df_out = pd.DataFrame.from_records(records)
    df_out.to_csv(os.path.join(outdir, "fit_timeseries.tsv"), sep="\t", index=False)


def plot_fitted_simulation(outdir):
    # Load Data
    df = pd.read_csv(os.path.join(outdir, "fit_timeseries.tsv"), sep="\t")
    proteins = sorted(df["Protein"].unique())
    print(f"[*] Found {len(proteins)} proteins")
    df_sites = df[df["Type"] == "Phosphosite"].reset_index(drop=True)
    df_prots = df[df["Type"] == "ProteinAbundance"].reset_index(drop=True)

    sim_cols = [col for col in df.columns if col.startswith("sim_t")]
    data_cols = [col for col in df.columns if col.startswith("data_t")]
    t_vals = DEFAULT_TIMEPOINTS

    # Plot per Protein
    for prot in proteins:
        print(f"   → Plotting {prot}")

        plt.figure(figsize=(12, 8))
        ax = plt.gca()

        # ---- Protein abundance (if exists)
        row_prot = df_prots[df_prots["Protein"] == prot]
        if not row_prot.empty:
            row_prot = row_prot.iloc[0]

            y_sim = row_prot[sim_cols].values.astype(float)
            y_dat = row_prot[data_cols].values.astype(float)
            has_data = np.any(np.isfinite(y_dat))

            ax.plot(t_vals, y_sim, "-", lw=3, color="blue", label="Protein (model)")
            ax.scatter(t_vals, y_sim, color="blue", s=30)

            if has_data:
                ax.plot(t_vals, y_dat, "k--", lw=2, label="Protein (data)")
                ax.scatter(t_vals, y_dat, color="black", s=35)

        # ---- Phosphosites
        sub = df_sites[df_sites["Protein"] == prot]

        for _, row in sub.iterrows():
            res = row["Residue"]
            y_sim = row[sim_cols].values.astype(float)
            y_dat = row[data_cols].values.astype(float)
            has_data = np.any(np.isfinite(y_dat))

            ax.plot(t_vals, y_sim, "-", alpha=0.7, lw=1.7, label=f"{res} (model)")
            ax.scatter(t_vals, y_sim, s=20, alpha=0.7)

            if has_data:
                ax.plot(t_vals, y_dat, "o--", ms=4, alpha=0.7, label=f"{res} (data)")

        # ---- Format plot
        ax.set_title(f"{prot}", fontsize=14, weight="bold")
        ax.set_xscale("log")
        ax.grid(alpha=0.3)
        ax.set_xlabel("Time (min)")
        ax.set_ylabel("FC / Scaled abundance")
        ax.legend(fontsize=8, loc="upper right", bbox_to_anchor=(1.25, 1.0))
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"fit_{prot}.png"), dpi=300)
        plt.close()

def print_biological_scores(outdir, X):
    bio_scores = np.array([bio_score(theta) for theta in X])

    with open(os.path.join(outdir, "biological_scores.tsv"), "w") as f:
        f.write("Index\tBio_Score\n")
        for i, score in enumerate(bio_scores):
            f.write(f"{i}\t{score:.6f}\n")

    print("[*] Biological Scores:")
    for i, score in enumerate(bio_scores):
        print(f"   → Point {i}: Bio Score = {score:.6f}")

def plot_biological_scores(outdir, X, F):
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

    df = pd.read_csv(file, sep="\t")

    sim_cols  = [c for c in df.columns if c.startswith("sim_t")]
    data_cols = [c for c in df.columns if c.startswith("data_t")]

    # Construct labels
    labels = []
    for _, row in df.iterrows():
        if row["Type"] == "Phosphosite":
            labels.append(f"{row['Protein']}_{row['Residue']}")
        else:
            labels.append(f"{row['Protein']}_Abundance")

    df["Label"] = labels

    # --- Prepare scatter plot ---
    plt.figure(figsize=(8, 8))

    # Scatter by item
    for idx, row in df.iterrows():
        sim_vals  = row[sim_cols].values.astype(float)
        data_vals = row[data_cols].values.astype(float)

        # Remove NaNs if protein abundances have missing time points
        mask = np.isfinite(sim_vals) & np.isfinite(data_vals)

        if row["Type"] == "Phosphosite":
            color = "green"
            alpha = 0.35
        else:
            color = "blue"
            alpha = 0.55

        plt.scatter(
            data_vals[mask],
            sim_vals[mask],
            label=row["Label"] if idx < 15 else None,  # avoid clutter
            alpha=alpha,
            color=color,
            s=30,
        )

    # identity line
    all_data = df[data_cols].values.flatten()
    all_sim  = df[sim_cols].values.flatten()
    max_val  = np.nanmax([all_data, all_sim])
    plt.plot([0, max_val], [0, max_val], 'r--', lw=2)

    plt.xlabel("Observed")
    plt.ylabel("Simulated")
    plt.title("Goodness of Fit: Observed vs Simulated")

    plt.tight_layout()
    plt.savefig(f'{outdir}/goodness_of_fit', dpi=300)
    plt.close()