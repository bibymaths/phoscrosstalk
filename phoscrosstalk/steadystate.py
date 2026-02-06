"""
steadystate.py
Simulates the network to steady state (long-term behavior) and visualizes convergence.
"""
import os
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from phoscrosstalk.config import ModelDims
from phoscrosstalk.simulation import simulate_p_scipy, build_full_A0
from phoscrosstalk.logger import get_logger

logger = get_logger()

def run_steadystate_analysis(outdir, problem, theta_opt, sites, proteins, kinases):
    """
    Simulates the network over a long time horizon to analyze convergence and steady-state behavior.



    Extends the simulation to T=10,000 using a log-spaced time vector to capture both rapid initial
    dynamics and long-term asymptotic behavior. Outputs include convergence metrics, raw data tables
    for sites and proteins, and visualizations of the trajectories.

    Args:
        outdir (str): Directory to save steady-state results.
        problem (NetworkOptimizationProblem): The optimization problem instance containing model matrices and configuration.
        theta_opt (np.ndarray): The optimized parameter vector.
        sites (list): List of phosphosite IDs.
        proteins (list): List of protein IDs.
        kinases (list): List of kinase IDs.

    Returns:
        None: Saves 'steadystate_sites.tsv', 'steadystate_proteins.tsv', and heatmap/trajectory plots to disk.
    """
    logger.info("\n[*] Running Steady State Analysis...")
    ss_dir = os.path.join(outdir, "steadystate")
    os.makedirs(ss_dir, exist_ok=True)

    # 1. Setup Long Time Vector (Log-spaced to capture early dynamics and long tail)
    T_end = 10000.0
    # 100 points from 0 to 100, 50 points from 100 to 10000
    t1 = np.linspace(0, 100, 100)
    t2 = np.linspace(101, T_end, 50)
    t_long = np.concatenate([t1, t2])

    # 2. Simulate
    K, M, N = ModelDims.K, ModelDims.M, ModelDims.N

    # We need to rebuild A0 for the long timeline.
    # We assume protein abundance stimulus (if any) persists at the last observed level
    # or stays constant. Here we extend the last column of A_scaled.
    if problem.A_scaled.size > 0:
        # Create a new A_scaled that matches t_long length by repeating last col
        # However, simulate_p_scipy takes A0 (initial condition) mostly.
        # The internal A(t) logic depends on s_prod/d_deg.
        # We pass A_scaled just for initial conditions in the wrapper.
        pass

        # We manually call simulate_p_scipy with t_long
    # Need to construct a dummy A0_full for the initial condition extractor
    A0_initial = build_full_A0(K, 1, problem.A_scaled[:, 0:1] if problem.A_scaled.size > 0 else np.array([]),
                               problem.prot_idx_for_A)

    P_ss, A_ss = simulate_p_scipy(
        t_long,
        problem.P_data,  # For Initial Conditions
        A0_initial,
        theta_opt,
        problem.Cg, problem.Cl, problem.site_prot_idx,
        problem.K_site_kin, problem.R, problem.L_alpha, problem.kin_to_prot_idx,
        problem.receptor_mask_prot, problem.receptor_mask_kin,
        problem.mechanism
    )

    # 3. Analyze Convergence
    # Check if derivatives are close to zero at the end
    delta_P = np.abs(P_ss[:, -1] - P_ss[:, -2])
    converged_sites = np.mean(delta_P) < 1e-4
    logger.info(f"   -> System convergence metric (mean delta P): {np.mean(delta_P):.6e}")

    # 4. Save Data
    df_P = pd.DataFrame(P_ss, index=sites, columns=[f"t_{t:.1f}" for t in t_long])
    df_P.to_csv(os.path.join(ss_dir, "steadystate_sites.tsv"), sep="\t")

    df_A = pd.DataFrame(A_ss, index=proteins, columns=[f"t_{t:.1f}" for t in t_long])
    df_A.to_csv(os.path.join(ss_dir, "steadystate_proteins.tsv"), sep="\t")

    # 5. Plotting
    _plot_convergence_heatmap(ss_dir, P_ss, t_long, "Phosphosites")
    _plot_convergence_heatmap(ss_dir, A_ss, t_long, "Proteins")
    _plot_trajectories(ss_dir, P_ss, t_long, sites, "Top_Changing_Sites")


def _plot_convergence_heatmap(outdir, data, t, label):
    """
    Generates a heatmap of trajectories sorted by their final steady-state value.

    Args:
        outdir (str): Output directory path.
        data (np.ndarray): Matrix of time-series data (Rows=Entities, Cols=Time).
        t (np.ndarray): Time vector.
        label (str): Label for the entities (e.g., "Phosphosites", "Proteins") used in titles.

    Returns:
        None: Saves a heatmap image to `outdir`.
    """
    # Sort by final value
    idx = np.argsort(data[:, -1])[::-1]
    sorted_data = data[idx, :]

    plt.figure(figsize=(10, 8))
    # Use log scale for time axis in imshow is tricky, we just plot indices and label ticks
    sns.heatmap(sorted_data, cmap="viridis", xticklabels=False, yticklabels=False)
    plt.xlabel(f"Time (0 to {t[-1]:.0f})")
    plt.ylabel(f"{label} (Sorted by High SS)")
    plt.title(f"{label} Approach to Steady State")
    plt.savefig(os.path.join(outdir, f"heatmap_convergence_{label}.png"), dpi=300)
    plt.close()


def _plot_trajectories(outdir, data, t, names, filename_suffix):
    """
    Plots time-series line graphs for the top 10 entities with the highest dynamic range.

    Uses a symmetric logarithmic scale (SymLog) for the time axis to visualize both early
    transient phases and late steady-state approaches effectively.

    Args:
        outdir (str): Output directory path.
        data (np.ndarray): Matrix of time-series data.
        t (np.ndarray): Time vector.
        names (list): List of names corresponding to the rows in `data`.
        filename_suffix (str): Suffix for the output filename.

    Returns:
        None: Saves a line plot image to `outdir`.
    """
    dynamic_range = np.max(data, axis=1) - np.min(data, axis=1)
    top_indices = np.argsort(dynamic_range)[-10:]  # Top 10

    plt.figure(figsize=(12, 6))
    for idx in top_indices:
        plt.plot(t, data[idx], label=names[idx], linewidth=2, alpha=0.8)

    plt.xscale("symlog", linthresh=10)  # Log scale helps see early and late dynamics
    plt.xlabel("Time (SymLog Scale)")
    plt.ylabel("Activity / Abundance")
    plt.title("Dynamics to Steady State (Top 10 High Variance)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"trajectories_{filename_suffix}.png"), dpi=300)
    plt.close()
