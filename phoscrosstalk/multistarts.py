"""
multistarts.py
Multi-start parameter fitting using Optimistix (replaces the pymoo NSGA2/UNSGA3 backend).

Strategy
--------
1. Generate ``n_starts`` random starting points uniformly in [xl, xu] using
   fixed random seeds for reproducibility.
2. Run ``run_single_optimisation`` (Optimistix BFGS) from each starting point.
3. For each solution found, simulate the best-fit trajectory and compute the
   Discrete Fréchet Distance against the target data.
4. Aggregate all solutions into a merged result object compatible with the
   caller in main.py.

Backward-compatible return interface
-------------------------------------
Returns (merged_res, best_idx, frechet_scores) where:
  merged_res.X  : (n_solutions, n_params)
  merged_res.F  : (n_solutions, 3)       – [f1, f2, f3]
  merged_res.J  : (n_solutions,)         – scalarized total loss (= J in analysis)
  best_idx      : int
  frechet_scores: np.ndarray (n_solutions,)

Old pymoo-specific flags (--gen, --pop-size, --algorithm) are mapped:
  --pop-size   → n_starts
  --gen        → max_steps_per_run
  --algorithm  → ignored with a warning (non-fatal)

Pseudo-Pareto front
-------------------
A small number of extra runs (``N_WEIGHT_COMBOS``) use different scalarisation
weights to approximate diverse Pareto-like solutions, so that downstream
analysis (which expects pareto_front_with_J.tsv) still receives a varied set.
"""

import numpy as np

from phoscrosstalk.optimization import make_loss_fn, run_single_optimisation
from phoscrosstalk.fretchet import frechet_distance
from phoscrosstalk.logger import get_logger

logger = get_logger()


# Weight combinations used to generate pseudo-Pareto diversity
_WEIGHT_COMBOS = [
    (1.0, 1.0, 1.0),    # balanced (default)
    (3.0, 1.0, 0.5),    # emphasise phosphosite fit
    (1.0, 3.0, 0.5),    # emphasise abundance fit
    (1.0, 1.0, 3.0),    # emphasise regularisation
]


class OptimizationResult:
    """Minimal result container compatible with the main.py caller."""
    def __init__(self, X, F):
        self.X = X
        self.F = F


def run_multi_start_optimization(problem, args, P_scaled):
    """
    Execute multi-start optimisation using Optimistix and select the best solution.

    Parameters
    ----------
    problem  : NetworkProblem – wraps the simulation and objective
    args     : argparse.Namespace – CLI arguments
    P_scaled : np.ndarray (N_sites, T) – scaled target data for Fréchet scoring

    Returns
    -------
    merged_res    : OptimizationResult  (.X, .F)
    best_idx      : int
    frechet_scores: np.ndarray (n_solutions,)
    """
    n_starts = max(1, getattr(args, "pop_size", 20))
    max_steps = max(64, getattr(args, "gen", 200))

    w_phospho  = getattr(args, "loss_weight_phospho",  1.0)
    w_abundance = getattr(args, "loss_weight_abundance", 1.0)
    w_reg       = getattr(args, "loss_weight_reg",       1.0)

    # Warn if evolutionary algorithm flags were passed
    algo = getattr(args, "algorithm", None)
    if algo is not None:
        logger.warning(
            f"[!] --algorithm {algo!r} is not used by the Optimistix backend; "
            "ignored for compatibility."
        )

    xl = problem.xl
    xu = problem.xu

    # Build the differentiable loss function (shared across all starts)
    loss_fn = make_loss_fn(
        t=problem.t,
        P_data=problem.P_data,
        A_scaled=problem.A_scaled,
        prot_idx_for_A=problem.prot_idx_for_A,
        W_data=problem.W_data,
        W_data_prot=problem.W_data_prot,
        Cg=problem.Cg,
        Cl=problem.Cl,
        site_prot_idx=problem.site_prot_idx,
        K_site_kin=problem.K_site_kin,
        R=problem.R,
        L_alpha=problem.L_alpha,
        kin_to_prot_idx=problem.kin_to_prot_idx,
        receptor_mask_prot=problem.receptor_mask_prot,
        receptor_mask_kin=problem.receptor_mask_kin,
        mechanism=problem.mechanism,
        lambda_net=problem.lambda_net,
        reg_lambda=problem.reg_lambda,
        w_phospho=w_phospho,
        w_abundance=w_abundance,
        w_reg=w_reg,
    )

    # Generate diverse starting points
    starts = _generate_starts(n_starts, xl, xu, _WEIGHT_COMBOS)

    logger.header("[*] Starting Multi-Start Optimistix Optimisation")
    logger.info(f"    {len(starts)} starting points, max_steps={max_steps} each")

    all_X, all_F = [], []

    for i, (theta0, wc) in enumerate(starts):
        logger.info(
            f"--- Run {i + 1}/{len(starts)} | "
            f"weights=({wc[0]:.1f},{wc[1]:.1f},{wc[2]:.1f}) ---"
        )

        # Build a weight-specific loss if the weight combo differs from default
        if wc != (w_phospho, w_abundance, w_reg):
            run_loss_fn = make_loss_fn(
                t=problem.t,
                P_data=problem.P_data,
                A_scaled=problem.A_scaled,
                prot_idx_for_A=problem.prot_idx_for_A,
                W_data=problem.W_data,
                W_data_prot=problem.W_data_prot,
                Cg=problem.Cg,
                Cl=problem.Cl,
                site_prot_idx=problem.site_prot_idx,
                K_site_kin=problem.K_site_kin,
                R=problem.R,
                L_alpha=problem.L_alpha,
                kin_to_prot_idx=problem.kin_to_prot_idx,
                receptor_mask_prot=problem.receptor_mask_prot,
                receptor_mask_kin=problem.receptor_mask_kin,
                mechanism=problem.mechanism,
                lambda_net=problem.lambda_net,
                reg_lambda=problem.reg_lambda,
                w_phospho=wc[0],
                w_abundance=wc[1],
                w_reg=wc[2],
            )
        else:
            run_loss_fn = loss_fn

        try:
            theta_opt, _, f1, f2, f3 = run_single_optimisation(
                run_loss_fn, theta0, max_steps=max_steps
            )
            all_X.append(theta_opt)
            all_F.append([f1, f2, f3])
            logger.info(f"    -> f1={f1:.4f}  f2={f2:.4f}  f3={f3:.4f}")
        except Exception as e:
            logger.warning(f"    -> Run {i + 1} failed: {e}")

    if not all_X:
        raise RuntimeError(
            "All optimisation runs failed to produce a solution. "
            "Check your data, bounds, and model dimensions."
        )

    X_combined = np.array(all_X)
    F_combined = np.array(all_F)
    logger.info(f"[*] Combined: {len(X_combined)} solutions collected.")

    # Evaluate Fréchet distance for model selection
    logger.info("[*] Calculating Fréchet Distances for Model Selection...")
    frechet_scores = np.full(len(X_combined), np.inf)
    true_coords    = np.ascontiguousarray(P_scaled, dtype=np.float64)

    for i in range(len(X_combined)):
        P_pred = problem.simulate(X_combined[i])
        pred_coords = np.ascontiguousarray(P_pred, dtype=np.float64)
        try:
            frechet_scores[i] = frechet_distance(true_coords, pred_coords)
        except Exception as e:
            logger.warning(f"    Fréchet error at idx {i}: {e}")

    best_idx   = int(np.argmin(frechet_scores))
    best_score = frechet_scores[best_idx]
    logger.success(
        f"[*] Best Solution: Fréchet Distance = {best_score:.6f} (idx={best_idx})"
    )

    merged_res = OptimizationResult(X=X_combined, F=F_combined)
    return merged_res, best_idx, frechet_scores


def _generate_starts(n_starts, xl, xu, weight_combos):
    """
    Generate starting parameter vectors and paired weight combinations.

    Returns a list of (theta0, weight_combo) tuples.
    The first ``len(weight_combos)`` entries use fixed diverse seeds so that
    the pseudo-Pareto ensemble explores different trade-offs.
    The remaining entries are random restarts with the default weights.
    """
    dim    = len(xl)
    starts = []

    # Diverse starts: one per weight combo
    for seed_offset, wc in enumerate(weight_combos):
        rng    = np.random.default_rng(seed_offset)
        theta0 = xl + rng.random(dim) * (xu - xl)
        starts.append((theta0, wc))

    default_wc = weight_combos[0]  # balanced
    n_extra    = max(0, n_starts - len(weight_combos))
    for i in range(n_extra):
        rng    = np.random.default_rng(100 + i)
        theta0 = xl + rng.random(dim) * (xu - xl)
        starts.append((theta0, default_wc))

    return starts
