"""
analysis.py
Post-optimization analysis, file export, and plotting.
"""
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from core_mechanisms import decode_theta
from config import ModelDims
from simulation import simulate_p_scipy
from optimization import build_full_A0, bio_score


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


def save_fitted_simulation(outdir, theta_opt, t, sites, proteins, P_scaled, A_scaled,
                           prot_idx_for_A, baselines, amplitudes,
                           A_data, A_bases, A_amps, mechanism,
                           Cg, Cl, site_prot_idx, K_site_kin, R, L_alpha, kin_to_prot_idx,
                           mask_p, mask_k):
    K, M, N = ModelDims.K, ModelDims.M, ModelDims.N

    # Save Params
    params_decoded = decode_theta(theta_opt, K, M, N)
    param_names = ["k_act", "k_deact", "s_prod", "d_deg", "beta_g", "beta_l", "alpha",
                   "kK_act", "kK_deact", "k_off", "gamma_S_p", "gamma_A_S", "gamma_A_p", "gamma_K_net"]

    save_dict = {"theta": theta_opt, "proteins": np.array(proteins), "sites": np.array(sites)}
    for name, val in zip(param_names, params_decoded):
        save_dict[name] = val
    np.savez(os.path.join(outdir, "fitted_params.npz"), **save_dict)

    # Re-simulate
    A0_full = build_full_A0(K, len(t), A_scaled, prot_idx_for_A)
    P_sim, A_sim = simulate_p_scipy(t, P_scaled, A0_full, theta_opt, Cg, Cl, site_prot_idx,
                                    K_site_kin, R, L_alpha, kin_to_prot_idx, mask_p, mask_k, mechanism)

    # Rescale Sites
    Y_sim_rescaled = np.zeros_like(P_sim)
    for i in range(len(sites)):
        Y_sim_rescaled[i] = baselines[i] + amplitudes[i] * P_sim[i]

    # Rescale Proteins
    A_sim_rescaled = A_sim.copy()
    if A_scaled.size > 0:
        for k, p_idx in enumerate(prot_idx_for_A):
            A_sim_rescaled[p_idx] = A_bases[k] + A_amps[k] * A_sim[p_idx]

    # Create DF
    df_sites = pd.DataFrame({"Protein": [s.split("_")[0] for s in sites],
                             "Residue": [s.split("_")[1] for s in sites], "Type": "Phosphosite"})
    for j in range(len(t)): df_sites[f"sim_t{j}"] = Y_sim_rescaled[:, j]

    df_prots = pd.DataFrame({"Protein": proteins, "Residue": "", "Type": "ProteinAbundance"})
    for j in range(len(t)): df_prots[f"sim_t{j}"] = A_sim_rescaled[:, j]

    pd.concat([df_sites, df_prots], ignore_index=True).to_csv(os.path.join(outdir, "fit_timeseries.tsv"), sep="\t",
                                                              index=False)