"""
hyperparam.py
Centralizes parameter bounds and handles hyperparameter tuning scans.
"""
import itertools
import numpy as np
import pandas as pd
import multiprocessing
from pymoo.algorithms.moo.unsga3 import UNSGA3
from pymoo.optimize import minimize
from pymoo.termination.default import DefaultMultiObjectiveTermination
from pymoo.parallelization import StarmapParallelization
from pymoo.util.ref_dirs import get_reference_directions

from phoscrosstalk.config import ModelDims
from phoscrosstalk import data_loader
from phoscrosstalk.optimization import NetworkOptimizationProblem
from phoscrosstalk.fretchet import frechet_distance

# --- Bounds Configuration ---
BOUNDS_CONFIG = {
    "k_act": (1e-5, 10.0),  # Protein activation
    "k_deact": (1e-5, 10.0),  # Protein deactivation
    "s_prod": (1e-5, 10.0),  # Protein synthesis
    "d_deg": (1e-5, 0.5),  # Protein degradation (restricted)
    "beta": (1e-5, 10.0),  # Global/Local coupling strengths
    "alpha": (1e-5, 10.0),  # Kinase global strength
    "kK_act": (1e-5, 3.0),  # Kinase activation
    "kK_deact": (1e-5, 3.0),  # Kinase deactivation
    "k_off": (1e-5, 5.0),  # Phosphosite phosphatase rate
    "gamma": (-3.0, 3.0)  # Tanh shape params (linear scale)
}


def create_bounds(K=None, M=None, N=None):
    """Generates lower (xl) and upper (xu) bounds vectors."""
    if K is None: K = ModelDims.K
    if M is None: M = ModelDims.M
    if N is None: N = ModelDims.N

    dim = 4 * K + 2 + 3 * M + N + 4
    xl = np.zeros(dim)
    xu = np.zeros(dim)
    idx = 0

    # Protein Kinetics
    for key in ["k_act", "k_deact", "s_prod"]:
        low, high = np.log(BOUNDS_CONFIG[key])
        xl[idx:idx + K] = low;
        xu[idx:idx + K] = high;
        idx += K

    # Degradation
    low, high = np.log(BOUNDS_CONFIG["d_deg"])
    xl[idx:idx + K] = low;
    xu[idx:idx + K] = high;
    idx += K

    # Coupling
    low, high = np.log(BOUNDS_CONFIG["beta"])
    xl[idx] = low;
    xu[idx] = high;
    idx += 1  # beta_g
    xl[idx] = low;
    xu[idx] = high;
    idx += 1  # beta_l

    # Kinase Params
    for key in ["alpha", "kK_act", "kK_deact"]:
        low, high = np.log(BOUNDS_CONFIG[key])
        xl[idx:idx + M] = low;
        xu[idx:idx + M] = high;
        idx += M

    # Phosphosite Params
    low, high = np.log(BOUNDS_CONFIG["k_off"])
    xl[idx:idx + N] = low;
    xu[idx:idx + N] = high;
    idx += N

    # Gammas
    low, high = BOUNDS_CONFIG["gamma"]
    xl[idx:idx + 4] = low;
    xu[idx:idx + 4] = high;
    idx += 4

    return xl, xu, dim


# --- Hyperparameter Scanning ---

def run_hyperparameter_scan(
        outdir,
        # Data Context
        t, P_scaled, sites, site_prot_idx, positions, proteins,
        ptm_intra_path, ptm_inter_path,
        # Static Matrices (invariant to hyperparameters)
        Cg, K_site_kin, R, L_alpha, kin_to_prot_idx,
        A_scaled, prot_idx_for_A, W_data, W_data_prot,
        receptor_mask_prot, receptor_mask_kin,
        # Config
        mechanism, cores
):
    """
    Performs a grid search over key hyperparameters by running short optimizations.
    Returns the dictionary of the best hyperparameter set.
    """
    print("\n" + "=" * 60)
    print("[*] STARTING HYPERPARAMETER TUNING SCAN")
    print("=" * 60)

    # 1. Define Search Grid
    # Customize these ranges based on your domain knowledge
    grid = {
        "length_scale": [25.0, 50.0, 100.0],
        "lambda_net": [0.0, 1e-4, 1e-2],
        "reg_lambda": [1e-4, 1e-2]
    }

    keys, values = zip(*grid.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    print(f"[*] Total combinations to test: {len(combinations)}")
    print("[*] Using 'Coarse' Optimization settings: Gen=40, Pop=100")

    best_score = np.inf
    best_params = None
    results_log = []

    # Prepare Multiprocessing Pool
    pool = multiprocessing.Pool(cores)
    runner = StarmapParallelization(pool.starmap)

    # Reference directions for NSGA3
    ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=8)

    # 2. Iterate Grid
    for i, combo in enumerate(combinations):
        ls = combo["length_scale"]
        ln = combo["lambda_net"]
        rl = combo["reg_lambda"]

        print(f"\n--- Combo {i + 1}/{len(combinations)}: LS={ls}, LambdaNet={ln}, Reg={rl} ---")

        # A. Rebuild Cl (Dependent on length_scale)
        # Note: We assume Cg is static and passed in. Cl needs rebuilding.
        # We reuse the logic from data_loader but we need to do it here manually
        # or call a helper. To avoid circular imports, we implement a lightweight builder here
        # or rely on data_loader being imported.

        # Rebuild Cl locally
        N_sites = len(sites)
        Cl_new = np.zeros((N_sites, N_sites), dtype=float)
        for r in range(N_sites):
            for c in range(N_sites):
                if r == c: continue
                if site_prot_idx[r] != site_prot_idx[c]: continue
                if np.isfinite(positions[r]) and np.isfinite(positions[c]):
                    d = abs(positions[r] - positions[c])
                    Cl_new[r, c] = np.exp(-d / ls)
        Cl_new = data_loader.row_normalize(Cl_new)

        # B. Setup Problem
        xl, xu, _ = create_bounds(ModelDims.K, ModelDims.M, ModelDims.N)

        problem = NetworkOptimizationProblem(
            t, P_scaled, Cg, Cl_new, site_prot_idx, K_site_kin, R,
            A_scaled, prot_idx_for_A, W_data, W_data_prot, L_alpha, kin_to_prot_idx,
            ln, rl, receptor_mask_prot, receptor_mask_kin,
            mechanism, xl, xu, elementwise_runner=runner
        )

        # C. Run Short Optimization
        algorithm = UNSGA3(pop_size=100, ref_dirs=ref_dirs)
        termination = DefaultMultiObjectiveTermination(
            xtol=1e-4, cvtol=1e-4, ftol=0.01, period=10,
            n_max_gen=40,  # Short run
            n_max_evals=10000
        )

        res = minimize(problem, algorithm, termination, seed=1, verbose=False)

        # D. Evaluate Best Solution (Fréchet Distance)
        # We simulate the "best" individual (e.g., knee point or min sum of objs)
        if len(res.F) > 0:
            # Simple scalarization to pick one representative solution from Pareto front
            # Normalize objectives roughly
            ptp = np.ptp(res.F, axis=0)
            F_norm = (res.F - res.F.min(axis=0)) / (ptp + 1e-9)
            best_idx_run = np.argmin(np.sum(F_norm, axis=1))
            theta_best = res.X[best_idx_run]

            P_pred = problem.simulate(theta_best)

            # Compute Fréchet Distance
            true_coords = np.ascontiguousarray(P_scaled, dtype=np.float64)
            pred_coords = np.ascontiguousarray(P_pred, dtype=np.float64)
            score = frechet_distance(true_coords, pred_coords)
        else:
            score = np.inf

        print(f"    -> Fréchet Score: {score:.4f}")

        combo["score"] = score
        results_log.append(combo)

        if score < best_score:
            best_score = score
            best_params = combo
            print(f"    [!] New Best Found!")

    pool.close()
    pool.join()

    # 3. Save Scan Results
    df_scan = pd.DataFrame(results_log)
    df_scan = df_scan.sort_values("score")
    df_scan.to_csv(f"{outdir}/hyperparameter_scan_results.tsv", sep="\t", index=False)

    print("\n" + "=" * 60)
    print(f"[*] TUNING COMPLETE. Best Params: {best_params}")
    print("=" * 60 + "\n")

    return best_params
