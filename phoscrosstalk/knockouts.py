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
    logger.info("\n[*] Running Systematic Knockout Screen...")
    ko_dir = os.path.join(outdir, "knockouts")
    os.makedirs(ko_dir, exist_ok=True)

    K, M, N = ModelDims.K, ModelDims.M, ModelDims.N

    # 1. Establish Wild Type (WT) Baseline
    # We use a single timepoint for comparison: The final steady state (t=240 or longer)
    # Using t=240 as a proxy for "result"
    t_eval = np.array([0, 240.0])

    def get_steady_state(theta_in, K_mat_in=None):
        if K_mat_in is None: K_mat_in = problem.K_site_kin

        # Build A0 dummy
        A0 = build_full_A0(K, len(t_eval), problem.A_scaled[:, 0:1] if problem.A_scaled.size > 0 else np.array([]),
                           problem.prot_idx_for_A)

        P, _ = simulate_p_scipy(
            t_eval, problem.P_data, A0, theta_in,
            problem.Cg, problem.Cl, problem.site_prot_idx,
            K_mat_in, problem.R, problem.L_alpha, problem.kin_to_prot_idx,
            problem.receptor_mask_prot, problem.receptor_mask_kin,
            problem.mechanism
        )
        return P[:, -1]  # Return last timepoint

    wt_ss = get_steady_state(theta_opt)

    # Storage for results: Rows = Perturbations, Cols = Phosphosites
    # We calculate Log2 Fold Change: log2((KO + eps) / (WT + eps))
    results = {}

    # 2. Kinase Knockouts (Set Alpha -> 0)
    # Parameter mapping: alpha starts at index 4K+2 in decoded,
    # but we modify the encoded theta vector directly.
    # We need to know where alpha indices are in the flat theta.
    # Helper to find indices:
    idx_start_alpha = 4 * K + 2

    logger.info("   -> Simulating Kinase Knockouts...")
    for m, kin_name in enumerate(tqdm(kinases)):
        theta_ko = theta_opt.copy()
        # Set alpha (log scale) to very small number ~ exp(-20) approx 0
        theta_ko[idx_start_alpha + m] = -20.0

        ko_ss = get_steady_state(theta_ko)
        l2fc = np.log2((ko_ss + 1e-4) / (wt_ss + 1e-4))
        results[f"KO_Kinase_{kin_name}"] = l2fc

    # 3. Protein Knockouts (Set Synthesis -> 0)
    # s_prod is at index 2*K
    idx_start_sprod = 2 * K
    logger.info("   -> Simulating Protein Knockouts...")
    for k, prot_name in enumerate(tqdm(proteins)):
        theta_ko = theta_opt.copy()
        theta_ko[idx_start_sprod + k] = -20.0

        ko_ss = get_steady_state(theta_ko)
        l2fc = np.log2((ko_ss + 1e-4) / (wt_ss + 1e-4))
        results[f"KO_Protein_{prot_name}"] = l2fc

    # 4. Phosphosite Knockouts (Remove Input Edges)
    # Physically, mutating S->A means no kinase can phosphorylate it.
    # We zero out the row in K_site_kin for that site.
    logger.info("   -> Simulating Phosphosite Knockouts (Alanine Scanning)...")
    for i, site_name in enumerate(tqdm(sites)):
        K_site_kin_ko = problem.K_site_kin.copy()
        K_site_kin_ko[i, :] = 0.0  # Remove all kinase inputs to this site

        # We assume phosphatase still works (k_off), dragging it to 0
        ko_ss = get_steady_state(theta_opt, K_mat_in=K_site_kin_ko)
        l2fc = np.log2((ko_ss + 1e-4) / (wt_ss + 1e-4))
        results[f"KO_Site_{site_name}"] = l2fc

    # 5. Compile Results
    df_res = pd.DataFrame(results).T  # (Perturbations x Sites)
    df_res.columns = sites

    # Filter: Drop perturbations that did nothing (all zeros)
    # A perturbation is interesting if it changes at least one site by > 0.1 Log2FC
    mask_rows = (df_res.abs() > 0.1).any(axis=1)
    df_filtered = df_res.loc[mask_rows]

    if df_filtered.empty:
        logger.critical("[!] No perturbations caused significant changes > 0.1 Log2FC.")
        return

    df_filtered.to_csv(os.path.join(ko_dir, "knockout_l2fc.tsv"), sep="\t")

    # 6. Clever Plotting: Clustermap
    # Limits color range to visible contrast
    vmax = np.percentile(np.abs(df_filtered.values), 95)

    logger.info("   -> Generating Clustermap...")
    try:
        g = sns.clustermap(
            df_filtered,
            cmap="vlag",
            center=0,
            vmax=vmax, vmin=-vmax,
            figsize=(14, 14),
            xticklabels=False,  # Too many sites usually
            yticklabels=True,
            dendrogram_ratio=(.1, .2),
            cbar_pos=(0.02, 0.8, 0.03, 0.18),
            cbar_kws={'label': 'Log2 Fold Change (KO/WT)'}
        )
        g.ax_heatmap.set_xlabel("Downstream Phosphosites")
        g.ax_heatmap.set_ylabel("Perturbation (KO)")
        plt.savefig(os.path.join(ko_dir, "knockout_clustermap.png"), dpi=300)
        plt.close()
    except Exception as e:
        logger.critical(f"[!] Clustermap generation failed (matrix likely too large or singular): {e}")
