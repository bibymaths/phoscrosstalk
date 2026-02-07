"""
post_processing.py
Advanced diagnostics, network topology export, and provenance tracking.
"""
import os
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx

from phoscrosstalk.config import ModelDims
from phoscrosstalk.core_mechanisms import decode_theta
from phoscrosstalk.logger import get_logger

logger = get_logger()


def save_run_metadata(outdir, args, execution_time=None):
    """
    Saves a JSON file containing all configuration arguments and model dimensions.
    Crucial for reproducibility.
    """
    meta_path = os.path.join(outdir, "run_config.json")

    # Convert args namespace to dict
    config_dict = vars(args).copy()

    # Add model dimensions
    config_dict["ModelDims"] = {
        "K (Proteins)": ModelDims.K,
        "M (Kinases)": ModelDims.M,
        "N (Sites)": ModelDims.N
    }

    if execution_time:
        config_dict["execution_time_seconds"] = execution_time

    with open(meta_path, "w") as f:
        json.dump(config_dict, f, indent=4, sort_keys=True)

    logger.info(f"[*] Saved run metadata to {meta_path}")


def export_network_for_cytoscape(outdir, theta_opt, proteins, kinases, sites,
                                 K_site_kin, site_prot_idx):
    """
    Exports the fitted network topology as a Cytoscape-compatible Edge List (SIF/CSV).

    Edges are weighted by the *optimized* kinetic parameters.
    1. Kinase -> Site (Weight = K_site_kin_ij * alpha_j * kK_act_j)
    2. Site -> Protein (Weight = 1.0, just structural)
    """
    logger.info("[*] Exporting Network Topology for Cytoscape...")

    # Decode parameters to get alpha (Kinase strength)
    # theta structure: [Prot params]...[Beta]...[Alpha]...
    K, M, N = ModelDims.K, ModelDims.M, ModelDims.N
    (_, _, _, _, _, _, alpha, kK_act, _, _, _, _, _, _) = decode_theta(theta_opt, K, M, N)

    edges = []

    # 1. Kinase -> Site Edges
    # Edge weight represents the "Potential V_max" contribution
    for i in range(N):  # Site i
        for j in range(M):  # Kinase j
            weight_base = K_site_kin[i, j]
            if weight_base > 1e-6:
                # Effective strength = Connectivity * Kinase_Global_Alpha * Kinase_Activity_Rate
                eff_weight = weight_base * alpha[j] * kK_act[j]

                edges.append({
                    "Source": kinases[j],
                    "Target": sites[i],
                    "Interaction": "phosphorylates",
                    "Weight_Fitted": eff_weight,
                    "Weight_Prior": weight_base,
                    "Type": "Kinase-Site"
                })

    # 2. Site -> Protein Edges (Structural mapping)
    for i, s in enumerate(sites):
        p_idx = site_prot_idx[i]
        p_name = proteins[p_idx]
        edges.append({
            "Source": s,
            "Target": p_name,
            "Interaction": "part_of",
            "Weight_Fitted": 1.0,
            "Weight_Prior": 1.0,
            "Type": "Site-Protein"
        })

    df_edges = pd.DataFrame(edges)
    df_edges.to_csv(os.path.join(outdir, "network_cytoscape_edges.csv"), index=False)

    # Save Node Attributes (e.g. Total Activity) can be added here if simulated data is passed
    logger.success(f"    -> Saved {len(edges)} edges to network_cytoscape_edges.csv")


def plot_residual_heatmap(outdir, P_obs, P_sim, sites, t):
    """
    Generates a heatmap of residuals (Observed - Simulated) to identify
    systematic biases in time or specific sites.
    """
    logger.info("[*] Generating Residual Heatmap...")

    # Calculate Residuals
    residuals = P_obs - P_sim

    # Sort sites by Total Absolute Error to put worst fits at the top
    total_error = np.sum(np.abs(residuals), axis=1)
    sort_idx = np.argsort(total_error)[::-1]

    residuals_sorted = residuals[sort_idx]
    sites_sorted = [sites[i] for i in sort_idx]

    plt.figure(figsize=(12, 10))
    # Diverging colormap: Red = Model Underestimates, Blue = Model Overestimates
    sns.heatmap(residuals_sorted, center=0, cmap="vlag",
                xticklabels=[f"{x:.1f}" for x in t],
                yticklabels=sites_sorted)

    plt.title("Residuals (Observed - Simulated)\nRed = Model Underestimates | Blue = Model Overestimates")
    plt.xlabel("Time (min)")
    plt.ylabel("Phosphosites (Sorted by Error)")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "residuals_heatmap.png"), dpi=300)
    plt.close()


def plot_parameter_clustermap(outdir, X_population, param_labels, top_n=50):
    """
    Plots a correlation matrix of parameters from the top N solutions.
    Helps identify 'stiff' parameter combinations (identifiability analysis).
    """
    logger.info("[*] Generating Parameter Correlation Clustermap...")

    # Take top N solutions
    X_subset = X_population[:top_n]

    # Create DataFrame
    df_params = pd.DataFrame(X_subset, columns=param_labels)

    # Drop columns with zero variance (fixed parameters)
    df_params = df_params.loc[:, (df_params != df_params.iloc[0]).any()]

    if df_params.shape[1] < 2:
        logger.warning("    -> Not enough varying parameters for correlation analysis.")
        return

    # Correlation Matrix
    corr = df_params.corr()

    # Plot
    plt.figure(figsize=(14, 14))
    g = sns.clustermap(corr, center=0, cmap="coolwarm",
                       vmin=-1, vmax=1,
                       linewidths=0.5,
                       figsize=(12, 12))

    g.fig.suptitle("Parameter Correlations (Top Solutions)", y=1.02)
    plt.savefig(os.path.join(outdir, "parameter_correlation_clustermap.png"), dpi=300)
    plt.close()