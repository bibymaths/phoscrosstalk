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

    # 3. Evaluate & Store Full Trajectories
    # We need to store:
    #   - MSE (scalar) for Sobol analysis
    #   - Full trajectories (huge) for the "perturbation data table"

    Y_mse = np.zeros(n_evals)

    # We will buffer data and write to CSV in chunks to avoid OOM for large N
    perturbation_file = os.path.join(sens_dir, "perturbation_data.tsv")

    # Write Header
    # Cols: Sample_ID, Type, Entity, Time, Value, [All Parameters...]
    # NOTE: Writing all parameters in every row is extremely redundant/large.
    # A standard "Tidy" format usually links Sample_ID to a separate Parameters table.
    # However, per request, we will dump "all parameters in the columns".

    # Prepare header
    # We'll include parameters as columns.
    header_cols = ["Sample_ID", "Type", "Entity", "Time", "Value"] + param_names

    logger.info("    -> Simulating and streaming data to disk...")

    with open(perturbation_file, "w") as f_out:
        # Write header
        f_out.write("\t".join(header_cols) + "\n")

        for i, theta in enumerate(param_values):
            if i % 50 == 0:
                print(f"       Sim {i}/{n_evals}...", end="\r")

            # A. Simulate (Getting full outputs: P, A, S, Kdyn)
            # We need to use simulate_p_scipy directly to get S and Kdyn,
            # or extend problem.simulate to return them.
            # Calling simulate_p_scipy manually here:

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
            Y_mse[i] = mse

            # C. Format Data for Table
            # To save space, we format the parameter string once per sample
            param_str = "\t".join([f"{x:.5g}" for x in theta])

            rows = []
            t_pts = problem.t

            # Helper to append rows
            def add_rows(matrix, type_label, entity_names):
                # matrix shape: (N_entities, T)
                for r_idx in range(matrix.shape[0]):
                    entity = entity_names[r_idx]
                    for t_idx in range(matrix.shape[1]):
                        val = matrix[r_idx, t_idx]
                        t_val = t_pts[t_idx]
                        # Row: Sample, Type, Entity, Time, Value, ...Params
                        row_str = f"{i}\t{type_label}\t{entity}\t{t_val:.2f}\t{val:.5g}\t{param_str}\n"
                        f_out.write(row_str)

            # 1. Phosphosites
            add_rows(P_sim, "Phosphosite", sites)
            # 2. Proteins (Abundance)
            add_rows(A_sim, "Protein_Abundance", proteins)
            # 3. Proteins (Activity S)
            add_rows(S_sim, "Protein_Activity_S", proteins)
            # 4. Kinases (Activity Kdyn)
            add_rows(Kdyn_sim, "Kinase_Activity_Kdyn", kinases)

    print("") # clear line
    logger.success(f"    -> Saved perturbation data to {perturbation_file}")

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