"""
hyperparam.py
Hyperparameter tuning scan using Optimistix (replaces the pymoo UNSGA3 backend).

Performs a grid search over (length_scale, lambda_net, reg_lambda) using a
short Optimistix optimisation for each combination, evaluated by Fréchet distance.
"""

import itertools

import numpy as np
import pandas as pd

from phoscrosstalk import data_loader
from phoscrosstalk.config import ModelDims
from phoscrosstalk.fretchet import frechet_distance
from phoscrosstalk.logger import get_logger
from phoscrosstalk.optimization import (
    NetworkProblem,
    create_bounds,
    make_residuals_fn,
    run_single_optimisation,
)

logger = get_logger()

# --- Bounds Configuration ---
BOUNDS_CONFIG = {
    "k_act": (1e-5, 10.0),
    "k_deact": (1e-5, 10.0),
    "s_prod": (1e-5, 10.0),
    "d_deg": (1e-5, 0.5),
    "beta": (1e-5, 10.0),
    "alpha": (1e-5, 10.0),
    "kK_act": (1e-5, 3.0),
    "kK_deact": (1e-5, 3.0),
    "k_off": (1e-5, 5.0),
    "gamma": (-3.0, 3.0),
}


def run_hyperparameter_scan(
    outdir,
    t,
    P_scaled,
    sites,
    site_prot_idx,
    positions,
    proteins,
    ptm_intra_path,
    ptm_inter_path,
    Cg,
    K_site_kin,
    R,
    L_alpha,
    kin_to_prot_idx,
    A_scaled,
    prot_idx_for_A,
    W_data,
    W_data_prot,
    receptor_mask_prot,
    receptor_mask_kin,
    mechanism,
):
    """
    Grid search over (length_scale, lambda_net, reg_lambda) using short Optimistix runs.

    Evaluates each combination with a Fréchet distance score and returns the best params.

    Returns:
        dict: Best hyperparameter combination (length_scale, lambda_net, reg_lambda, score).
    """
    logger.info("\n" + "=" * 60)
    logger.header("[*] STARTING HYPERPARAMETER TUNING SCAN")
    logger.info("=" * 60)

    grid = {
        "length_scale": [25.0, 50.0, 100.0],
        "lambda_net": [0.0, 1e-4, 1e-2],
        "reg_lambda": [1e-4, 1e-2],
    }

    keys, values = zip(*grid.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    logger.info(f"[*] Total combinations to test: {len(combinations)}")
    logger.info("[*] Using 'Coarse' settings: max_steps=40")

    best_score = np.inf
    best_params = None
    results_log = []

    xl, xu, _ = create_bounds(ModelDims.K, ModelDims.M, ModelDims.N)
    dim = len(xl)

    for i, combo in enumerate(combinations):
        ls = combo["length_scale"]
        ln = combo["lambda_net"]
        rl = combo["reg_lambda"]

        logger.info(
            f"\n--- Combo {i + 1}/{len(combinations)}: "
            f"LS={ls}, LambdaNet={ln}, Reg={rl} ---"
        )

        # Rebuild Cl for this length scale
        N_sites = len(sites)
        Cl_new = np.zeros((N_sites, N_sites), dtype=float)
        for r in range(N_sites):
            for c in range(N_sites):
                if r == c or site_prot_idx[r] != site_prot_idx[c]:
                    continue
                if np.isfinite(positions[r]) and np.isfinite(positions[c]):
                    d = abs(positions[r] - positions[c])
                    Cl_new[r, c] = np.exp(-d / ls)
        Cl_new = data_loader.row_normalize(Cl_new)
        # Build objective functions for this combo
        try:
            residuals_fn = make_residuals_fn(
                t=t,
                P_data=P_scaled,
                A_scaled=A_scaled,
                prot_idx_for_A=prot_idx_for_A,
                W_data=W_data,
                W_data_prot=W_data_prot,
                Cg=Cg,
                Cl=Cl_new,
                site_prot_idx=site_prot_idx,
                K_site_kin=K_site_kin,
                R=R,
                L_alpha=L_alpha,
                kin_to_prot_idx=kin_to_prot_idx,
                receptor_mask_prot=receptor_mask_prot,
                receptor_mask_kin=receptor_mask_kin,
                mechanism=mechanism,
                lambda_net=ln,
                reg_lambda=rl,
            )

            rng = np.random.default_rng(1)
            theta0 = xl + rng.random(dim) * (xu - xl)

            theta_best, *_ = run_single_optimisation(
                residuals_fn,
                theta0,
                max_steps=40,
            )

        except Exception as e:
            logger.warning(f"    -> Combo failed: {e}")
            return np.inf

        # Evaluate Fréchet distance
        problem_tmp = NetworkProblem(
            t=t,
            P_data=P_scaled,
            Cg=Cg,
            Cl=Cl_new,
            site_prot_idx=site_prot_idx,
            K_site_kin=K_site_kin,
            R=R,
            A_scaled=A_scaled,
            prot_idx_for_A=prot_idx_for_A,
            W_data=W_data,
            W_data_prot=W_data_prot,
            L_alpha=L_alpha,
            kin_to_prot_idx=kin_to_prot_idx,
            lambda_net=ln,
            reg_lambda=rl,
            receptor_mask_prot=receptor_mask_prot,
            receptor_mask_kin=receptor_mask_kin,
            mechanism=mechanism,
            xl=xl,
            xu=xu,
        )

        P_pred = problem_tmp.simulate(theta_best)
        true_c = np.ascontiguousarray(P_scaled, dtype=np.float64)
        pred_c = np.ascontiguousarray(P_pred, dtype=np.float64)
        score = frechet_distance(true_c, pred_c)

        logger.info(f"    -> Fréchet Score: {score:.4f}")
        combo["score"] = score
        results_log.append(combo)

        if score < best_score:
            best_score = score
            best_params = combo
            logger.success("    [!] New Best Found!")

    # Save scan results
    df_scan = pd.DataFrame(results_log).sort_values("score")
    df_scan.to_csv(f"{outdir}/hyperparameter_scan_results.tsv", sep="\t", index=False)

    logger.info("\n" + "=" * 60)
    logger.success(f"[*] TUNING COMPLETE. Best Params: {best_params}")
    logger.info("=" * 60 + "\n")

    return best_params
