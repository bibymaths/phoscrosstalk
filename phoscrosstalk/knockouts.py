"""
knockouts.py
Systematic in-silico knockout screens (Kinase, Protein, and Phosphosite KO).
"""
import os
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from tqdm import tqdm
from phoscrosstalk.config import ModelDims
from phoscrosstalk.simulation import simulate_p_scipy, build_full_A0
from phoscrosstalk.logger import get_logger

logger = get_logger()


def run_knockout_screen(outdir, problem, theta_opt, sites, proteins, kinases):
    """
    Perform a systematic in-silico knockout screen for kinases, proteins, and phosphosites.

    This function iterates through every component in the network, virtually "deleting" it
    by modifying parameters (setting rates to zero) or topology matrices, and simulating
    the new steady-state. It calculates the Log2 Fold Change compared to the Wild Type (WT)
    baseline and generates a clustered heatmap of the results.

    Perturbation types:
    - **Kinase KO**: Sets global strength (alpha) to ~0.
    - **Protein KO**: Sets synthesis rate (s_prod) to ~0.
    - **Site KO**: Removes all upstream kinase inputs (simulating Alanine mutation).

    Args:
        outdir (str): Output directory for TSV and PNG files.
        problem (NetworkOptimizationProblem): The initialized optimization problem object containing matrices and config.
        theta_opt (np.ndarray): The optimized parameter vector (Wild Type).
        sites (list): List of phosphosite IDs.
        proteins (list): List of protein IDs.
        kinases (list): List of kinase IDs.

    Returns:
        None: Saves 'knockout_l2fc.tsv' and 'knockout_clustermap.png' to disk.
    """
    logger.info("\n[*] Running Systematic Knockout Screen (Fold Change)...")
    ko_dir = os.path.join(outdir, "knockouts")
    os.makedirs(ko_dir, exist_ok=True)

    K, M, N = ModelDims.K, ModelDims.M, ModelDims.N

    # 1. Establish Wild Type (WT) Baseline
    t_eval = np.array([0, 240.0])

    def get_steady_state(theta_in, K_mat_in=None):
        if K_mat_in is None: K_mat_in = problem.K_site_kin

        A0 = build_full_A0(K, len(t_eval), problem.A_scaled[:, 0:1] if problem.A_scaled.size > 0 else np.array([]),
                           problem.prot_idx_for_A)

        P, _, S, Kdyn = simulate_p_scipy(
            t_eval, problem.P_data, A0, theta_in,
            problem.Cg, problem.Cl, problem.site_prot_idx,
            K_mat_in, problem.R, problem.L_alpha, problem.kin_to_prot_idx,
            problem.receptor_mask_prot, problem.receptor_mask_kin,
            problem.mechanism,
            full_output=True
        )
        return P[:, -1], S[:, -1], Kdyn[:, -1]

    wt_P, wt_S, wt_Kdyn = get_steady_state(theta_opt)

    # Storage for results (Fold Change)
    res_P, res_S, res_Kdyn = {}, {}, {}

    # 2. Kinase Knockouts (Set Alpha -> 0)
    idx_alpha = 4 * K + 2
    for m, kin_name in enumerate(tqdm(kinases, desc="Kinase KOs")):
        theta_ko = theta_opt.copy()
        theta_ko[idx_alpha + m] = -20.0  # Effectively zero in log space

        ko_P, ko_S, ko_Kdyn = get_steady_state(theta_ko)

        # Calculate Fold Change: (KO / WT)
        # Add epsilon to denominator to avoid division by zero
        res_P[f"KO_Kin_{kin_name}"] = (ko_P + 1e-9) / (wt_P + 1e-9)
        res_S[f"KO_Kin_{kin_name}"] = (ko_S + 1e-9) / (wt_S + 1e-9)
        res_Kdyn[f"KO_Kin_{kin_name}"] = (ko_Kdyn + 1e-9) / (wt_Kdyn + 1e-9)

    # --- Helper to Save & Plot FC ---
    def process_and_save(res_dict, cols, name):
        df = pd.DataFrame(res_dict).T
        df.columns = cols

        # Filter: Keep rows where at least one value deviates significantly from 1.0
        # e.g., < 0.9 or > 1.1
        mask = ((df < 0.95) | (df > 1.05)).any(axis=1)
        df = df.loc[mask]

        if not df.empty:
            df.to_csv(os.path.join(ko_dir, f"knockout_fc_{name}.tsv"), sep="\t")
            # Plot
            try:
                # For FC, center is 1.0.
                # We determine vmax to handle high fold changes gracefully.
                vals = df.values.flatten()
                vmax = np.percentile(vals, 98)
                if vmax < 1.5: vmax = 1.5  # Minimum contrast

                g = sns.clustermap(
                    df,
                    cmap="vlag",
                    center=1.0,  # Center white at 1.0 (No Change)
                    vmin=0.0,  # Floor at 0.0
                    vmax=vmax,
                    figsize=(10, 10),
                    cbar_kws={'label': 'Fold Change (KO/WT)'}
                )
                g.fig.suptitle(f"Fold Change {name} upon Knockout")
                plt.savefig(os.path.join(ko_dir, f"clustermap_{name}_fc.png"), dpi=300)
                plt.close()
            except Exception as e:
                logger.warning(f"Clustermap {name} failed: {e}")

    process_and_save(res_S, proteins, "S_sim")
    process_and_save(res_Kdyn, kinases, "Kdyn_sim")

    # 4. Phosphosite Knockouts (Remove Input Edges)
    logger.info("   -> Simulating Phosphosite Knockouts (Alanine Scanning)...")
    for i, site_name in enumerate(tqdm(sites)):
        K_site_kin_ko = problem.K_site_kin.copy()
        K_site_kin_ko[i, :] = 0.0  # Remove all kinase inputs

        # Fix: unpacking tuple, using correct baseline (wt_P)
        ko_P_vals, _, _ = get_steady_state(theta_opt, K_mat_in=K_site_kin_ko)

        # Calculate FC
        fc_val = (ko_P_vals + 1e-9) / (wt_P + 1e-9)
        res_P[f"KO_Site_{site_name}"] = fc_val

    # 5. Compile Main P Results
    df_res = pd.DataFrame(res_P).T
    df_res.columns = sites

    # Filter: Drop perturbations close to 1.0 (no change)
    mask_rows = ((df_res < 0.9) | (df_res > 1.1)).any(axis=1)
    df_filtered = df_res.loc[mask_rows]

    if df_filtered.empty:
        logger.warning("[!] No perturbations caused significant changes > 10% FC.")
        return

    df_filtered.to_csv(os.path.join(ko_dir, "knockout_fc.tsv"), sep="\t")

    # 6. Plotting Main Clustermap
    logger.info("   -> Generating Clustermap...")
    try:
        vals = df_filtered.values.flatten()
        vmax = np.percentile(vals, 98)
        if vmax < 2.0: vmax = 2.0

        g = sns.clustermap(
            df_filtered,
            cmap="vlag",
            center=1.0,  # White at 1.0
            vmin=0.0,
            vmax=vmax,
            figsize=(14, 14),
            xticklabels=False,
            yticklabels=True,
            dendrogram_ratio=(.1, .2),
            cbar_pos=(0.02, 0.8, 0.03, 0.18),
            cbar_kws={'label': 'Fold Change (KO/WT)'}
        )
        g.ax_heatmap.set_xlabel("Downstream Phosphosites")
        g.ax_heatmap.set_ylabel("Perturbation (KO)")
        plt.savefig(os.path.join(ko_dir, "knockout_clustermap_fc.png"), dpi=300)
        plt.close()
    except Exception as e:
        logger.critical(f"[!] Clustermap generation failed: {e}")
