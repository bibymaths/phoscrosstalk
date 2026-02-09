"""
multi_start.py
Implements multi-start optimization strategies and best-solution selection
using Fréchet distance.
"""
import numpy as np
from pymoo.algorithms.moo.unsga3 import UNSGA3
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.termination.default import DefaultMultiObjectiveTermination
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.core.result import Result

from phoscrosstalk.fretchet import frechet_distance
from phoscrosstalk.logger import get_logger

logger = get_logger()


def run_multi_start_optimization(problem, args, P_scaled):
    """
    Executes multiple optimization strategies (Ensemble Optimization) and selects
    the best solution based on the Discrete Fréchet Distance.

    Strategies implemented:
    1. **Balanced (UNSGA3)**: Standard reference direction density (Partitions=12).
    2. **High-Resolution (UNSGA3)**: Higher density reference directions (Partitions=16) for fine-tuning.
    3. **Diversity (NSGA2)**: Crowding-distance based sorting to explore Pareto extremes.

    Args:
        problem (ElementwiseProblem): The Pymoo problem instance (with parallel runner).
        args (Namespace): Command line arguments containing config (pop_size, gen, etc.).
        P_scaled (np.ndarray): The scaled target phosphosite data (used for scoring).

    Returns:
        tuple:
            - res (Result): A consolidated Pymoo Result object containing merged F and X.
            - best_idx (int): The index in res.X (and res.F) corresponding to the best solution.
            - frechet_scores (np.ndarray): Array of scores corresponding to res.X.
    """

    strategies = [
        {
            "name": "Balanced (UNSGA3, p=12)",
            "algo": "unsga3",
            "partitions": 12,
            "seed": 1
        },
        {
            "name": "High-Res (UNSGA3, p=15)",
            "algo": "unsga3",
            "partitions": 15,
            "seed": 42
        },
        {
            "name": "Super High-Res (UNSGA3, p=20)",
            "algo": "unsga3",
            "partitions": 20,
            "seed": 75
        },
        {
            "name": "Exploration (NSGA2)",
            "algo": "nsga2",
            "seed": 100
        }
    ]

    all_F = []
    all_X = []

    logger.header("[*] Starting Multi-Start Ensemble Optimization")

    # 1. Run Strategies
    for i, strat in enumerate(strategies):
        logger.info(f"--- Strategy {i + 1}/{len(strategies)}: {strat['name']} ---")

        # Configure Algorithm
        if strat["algo"] == "unsga3":
            # Dynamic Reference Directions based on strategy config
            # 'das-dennis' works well for 3 objectives
            ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=strat["partitions"])

            algorithm = UNSGA3(
                pop_size=args.pop_size,
                ref_dirs=ref_dirs
            )

        elif strat["algo"] == "nsga2":
            algorithm = NSGA2(
                pop_size=args.pop_size
            )

        # Configure Termination (Shared logic)
        termination = DefaultMultiObjectiveTermination(
            xtol=1e-8,
            cvtol=1e-6,
            ftol=0.0025,
            period=30,
            n_max_gen=args.gen,
            n_max_evals=1000000
        )

        # Execute
        res = minimize(
            problem,
            algorithm,
            termination,
            seed=strat["seed"],
            verbose=True
        )

        if res.X is not None and len(res.X) > 0:
            all_X.append(res.X)
            all_F.append(res.F)
            logger.info(f"    -> Finished. Found {len(res.X)} solutions.")
        else:
            logger.warning(f"    -> Strategy {strat['name']} returned no solutions.")

    # 2. Merge Results
    if not all_X:
        logger.critical("All strategies failed to produce solutions.")
        raise RuntimeError("Optimization failed.")

    X_combined = np.concatenate(all_X, axis=0)
    F_combined = np.concatenate(all_F, axis=0)

    logger.info(f"[*] Combined Population: {len(X_combined)} unique solutions.")

    # 3. Evaluate Best Solution (Fréchet Distance)
    logger.info("[*] Calculating Fréchet Distances for Model Selection...")

    frechet_scores = np.full(len(X_combined), np.inf, dtype=float)

    # Pre-convert true coords for Numba
    true_coords = np.ascontiguousarray(P_scaled, dtype=np.float64)

    # We iterate sequentially here; simulation is fast enough for the pareto set
    for i in range(len(X_combined)):
        # Simulate trajectory for this parameter set
        P_pred = problem.simulate(X_combined[i])

        # Ensure contiguous arrays for Numba
        pred_coords = np.ascontiguousarray(P_pred, dtype=np.float64)

        try:
            score = frechet_distance(true_coords, pred_coords)
            frechet_scores[i] = score
        except Exception as e:
            logger.warning(f"    Error calculating score for idx {i}: {e}")

    # Find Best
    best_idx = int(np.argmin(frechet_scores))
    best_score = frechet_scores[best_idx]

    logger.success(f"[*] Best Solution Found via Multi-Start: Fréchet Distance = {best_score:.6f}")

    # Construct a merged Result object to return to main.py
    merged_res = Result()
    merged_res.X = X_combined
    merged_res.F = F_combined
    # Note: We don't merge history or other meta-data to save memory

    return merged_res, best_idx, frechet_scores