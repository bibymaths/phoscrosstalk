"""
sensitivity.py
Global Sensitivity Analysis (GSA) using SALib with labeled parameters
and full perturbation data export.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from SALib.sample import saltelli
from SALib.analyze import sobol
from joblib import Parallel, delayed
from tqdm import tqdm

from phoscrosstalk.config import ModelDims
from phoscrosstalk.simulation import simulate_p_scipy, build_full_A0
from phoscrosstalk.logger import get_logger

logger = get_logger()

def _generate_param_labels(K, M, N, proteins, kinases, sites):
    """
    Generates human-readable labels for the flattened parameter vector theta.

    Order matches core_mechanisms.decode_theta:
    1. Protein Kinetics (4 * K): k_act, k_deact, s_prod, d_deg
    2. Coupling (2): beta_g, beta_l
    3. Kinase Kinetics (3 * M): alpha, kK_act, kK_deact
    4. Site Kinetics (1 * N): k_off
    5. Gammas (4): S_p, A_S, A_p, K_net
    """
    labels = []

    # 1. Proteins
    for tag in ["k_act", "k_deact", "s_prod", "d_deg"]:
        for p in proteins:
            labels.append(f"{tag}_{p}")

    # 2. Coupling
    labels.extend(["beta_g", "beta_l"])

    # 3. Kinases
    for tag in ["alpha", "kK_act", "kK_deact"]:
        for k in kinases:
            labels.append(f"{tag}_{k}")

    # 4. Sites
    for s in sites:
        # shorten site name if needed, e.g. EGFR_Y1068 -> k_off_EGFR_Y1068
        labels.append(f"k_off_{s}")

    # 5. Gammas
    labels.extend(["gamma_S_p", "gamma_A_S", "gamma_A_p", "gamma_K_net"])

    return labels

def _evaluate_single_sample(i, theta, problem, K, M, N, sites, proteins, kinases):
    """
    Helper function to evaluate one sample in parallel.
    Returns the MSE and the formatted string of rows for the output file.
    """
    # CRITICAL FIX: Re-initialize global dimensions inside the worker process
    # because static class variables like ModelDims are lost (None) in new processes.
    ModelDims.set_dims(K, M, N)

    # Reconstruct A0 (needed for simulation wrapper)
    # Note: problem object usually handles this inside _evaluate, we replicate here.
    A0_full = build_full_A0(K, len(problem.t), problem.A_scaled, problem.prot_idx_for_A)

    P_sim, A_sim, S_sim, Kdyn_sim = simulate_p_scipy(
        problem.t, problem.P_data, A0_full, theta,
        problem.Cg, problem.Cl, problem.site_prot_idx,
        problem.K_site_kin, problem.R, problem.L_alpha, problem.kin_to_prot_idx,
        problem.receptor_mask_prot, problem.receptor_mask_kin,
        problem.mechanism, full_output=True
    )

    # B. Calculate Metric for Sobol (MSE on Phosphosites)
    # Filter NaNs if simulation exploded
    if not np.all(np.isfinite(P_sim)):
        mse = 1e6 # penalty
    else:
        diff = (problem.P_data - P_sim)
        mse = np.mean(diff ** 2)

    # C. Format Data for Table
    # NOTE: In Tidy format, we do NOT repeat parameters here.
    # We only write Sample_ID and the observed values.

    rows_buffer = []
    t_pts = problem.t

    # Helper to append rows locally
    def add_rows_local(matrix, type_label, entity_names):
        # matrix shape: (N_entities, T)
        for r_idx in range(matrix.shape[0]):
            entity = entity_names[r_idx]
            for t_idx in range(matrix.shape[1]):
                val = matrix[r_idx, t_idx]
                t_val = t_pts[t_idx]
                # Row: Sample_ID, Type, Entity, Time, Value
                row_str = f"{i}\t{type_label}\t{entity}\t{t_val:.2f}\t{val:.5g}\n"
                rows_buffer.append(row_str)

    # 1. Phosphosites
    add_rows_local(P_sim, "Phosphosite", sites)
    # 2. Proteins (Abundance)
    add_rows_local(A_sim, "Protein_Abundance", proteins)
    # 3. Proteins (Activity S)
    add_rows_local(S_sim, "Protein_Activity_S", proteins)
    # 4. Kinases (Activity Kdyn)
    add_rows_local(Kdyn_sim, "Kinase_Activity_Kdyn", kinases)

    return mse, "".join(rows_buffer)

def run_global_sensitivity(outdir, problem, param_bounds,
                           proteins, kinases, sites,
                           samples=64):
    """
    Performs Sobol GSA using SALib, plots labeled sensitivities, and exports
    full perturbation trajectories.

    Args:
        outdir (str): Output directory.
        problem (NetworkOptimizationProblem): Pymoo problem instance.
        param_bounds (tuple): (xl, xu) bounds.
        proteins (list): List of protein names.
        kinases (list): List of kinase names.
        sites (list): List of site names.
        samples (int): N for Saltelli sampler (total runs = N * (2D + 2)).
    """
    logger.info(f"\n[*] Running SALib Sensitivity Analysis (N={samples})...")

    sens_dir = os.path.join(outdir, "sensitivity")
    os.makedirs(sens_dir, exist_ok=True)

    xl, xu = param_bounds
    dim = len(xl)
    K, M, N = ModelDims.K, ModelDims.M, ModelDims.N

    # 1. Generate Labeled Problem Spec
    param_names = _generate_param_labels(K, M, N, proteins, kinases, sites)

    if len(param_names) != dim:
        logger.warning(f"Label count ({len(param_names)}) != Dim ({dim}). Fallback to generic.")
        param_names = [f"p_{i}" for i in range(dim)]

    problem_spec = {
        'num_vars': dim,
        'names': param_names,
        'bounds': list(zip(xl, xu))
    }

    # 2. Generate Samples
    # calc_second_order=False keeps total runs manageable
    # Total runs = N * (D + 2)
    logger.info("    -> Generating Saltelli samples...")
    param_values = saltelli.sample(problem_spec, samples, calc_second_order=False)
    n_evals = param_values.shape[0]
    logger.info(f"    -> Total Evaluations: {n_evals}")

    # 2b. Save Parameters Table (Tidy Format Part 1)
    # Links Sample_ID to Parameter Values
    logger.info("    -> Saving Parameters Table...")
    param_file = os.path.join(sens_dir, "perturbation_params.tsv")

    # Create DataFrame: Sample_ID, Param1, Param2...
    df_params = pd.DataFrame(param_values, columns=param_names)
    df_params.insert(0, "Sample_ID", range(n_evals))
    df_params.to_csv(param_file, sep="\t", index=False)

    # 3. Evaluate & Store Full Trajectories
    # We need to store:
    #   - MSE (scalar) for Sobol analysis
    #   - Full trajectories (huge) for the "perturbation data table"

    Y_mse = np.zeros(n_evals)

    # We will buffer data and write to CSV in chunks to avoid OOM for large N
    perturbation_file = os.path.join(sens_dir, "perturbation_data.tsv")

    # Write Header
    # Tidy format: No parameter columns here. Link via Sample_ID.
    header_cols = ["Sample_ID", "Type", "Entity", "Time", "Value"]

    logger.info("    -> Simulating and streaming data to disk (Parallel)...")

    # Use all available cores except one
    n_jobs = -1

    # Run parallel simulations
    # 1. Create a generator for the delayed tasks
    tasks = (delayed(_evaluate_single_sample)(i, theta, problem, K, M, N, sites, proteins, kinases)
             for i, theta in enumerate(param_values))

    # 2. Execute using return_as="generator" to stream results to tqdm
    # verbose=0 prevents joblib's native prints from breaking the tqdm bar
    results_gen = Parallel(n_jobs=n_jobs, verbose=0, return_as="generator")(tasks)

    # 3. Collect results with progress bar
    results = [
        res for res in tqdm(results_gen, total=len(param_values), desc="       Progress", unit="sim")
    ]

    # Unpack results and write to file
    with open(perturbation_file, "w") as f_out:
        # Write header
        f_out.write("\t".join(header_cols) + "\n")

        for i, (mse, row_data) in enumerate(results):
            Y_mse[i] = mse
            f_out.write(row_data)

    print("") # clear line
    logger.success(f"    -> Saved perturbation data to {perturbation_file}")
    logger.success(f"    -> Saved perturbation params to {param_file}")

    # 4. Analyze Sobol Indices
    logger.info("    -> Calculating Sobol Indices (MSE metric)...")
    Si = sobol.analyze(problem_spec, Y_mse, calc_second_order=False, print_to_console=False)

    df_sens = pd.DataFrame({
        "Parameter": param_names,
        "Total_Order": Si['ST'],
        "First_Order": Si['S1'],
        "Confidence_Total": Si['ST_conf']
    })

    # Sort
    df_sens = df_sens.sort_values("Total_Order", ascending=False)
    df_sens.to_csv(os.path.join(sens_dir, "sobol_indices_labeled.tsv"), sep="\t", index=False)

    # 5. Plot Top 20
    logger.info("    -> Plotting Sensitivity Ranks...")
    top_20 = df_sens.head(20)

    plt.figure(figsize=(10, 8))
    sns.barplot(data=top_20, x="Total_Order", y="Parameter", color="skyblue", label="Total Effect")
    # Overlay first order?
    # Usually easier to just plot Total for ranking, but we can do a dodged bar or overlay.
    # Simple bar chart is clearer for labels.

    plt.title(f"Top 20 Parameters driving Model Error (Sobol ST)")
    plt.xlabel("Total-Order Sensitivity Index")
    plt.ylabel("Parameter")
    plt.tight_layout()
    plt.savefig(os.path.join(sens_dir, "sensitivity_ranking_labeled.png"), dpi=300)
    plt.close()

    logger.success("[*] Sensitivity Analysis Complete.")