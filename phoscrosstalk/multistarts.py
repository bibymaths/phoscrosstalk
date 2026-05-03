"""
multistarts.py
Multi-start parameter fitting using slsqp-jax / Optimistix (single-objective framework).

Strategy
--------
1. Generate ``n_starts`` random starting points uniformly in [xl, xu] using
   fixed random seeds for reproducibility.
2. Run ``run_single_optimisation`` (slsqp-jax SLSQP) from each starting point
   using the same configured loss weights for all runs.
3. Select the best solution by minimum **total loss** (the scalar objective
   that was minimised).
4. Compute the Discrete Fréchet Distance for each solution as a diagnostic
   metric only (logged but not used for selection).
5. Aggregate all solutions into a result object compatible with the
   caller in main.py.

Backward-compatible return interface
-------------------------------------
Returns (merged_res, best_idx, total_losses) where:
  merged_res.X  : (n_solutions, n_params)
  merged_res.F  : (n_solutions, 3)       – [f1, f2, f3] loss components
  merged_res.J  : (n_solutions,)         – total loss per run (used for selection)
  best_idx      : int                    – index of lowest-total-loss solution
  total_losses  : np.ndarray             – total loss for each solution

Old pymoo-specific flags (--gen, --pop-size, --algorithm) are mapped:
  --pop-size   → n_starts
  --gen        → max_steps_per_run
  --algorithm  → ignored with a warning (non-fatal)
"""

import os
import numpy as np
import pandas as pd

from phoscrosstalk.optimization import make_loss_fn, run_single_optimisation
from phoscrosstalk.fretchet import frechet_distance
from phoscrosstalk.logger import get_logger

logger = get_logger()


class OptimizationResult:
    """Minimal result container compatible with the main.py caller."""

    def __init__(self, X, F, J):
        self.X = X
        self.F = F
        self.J = J  # total loss per run


def run_multi_start_optimization(problem, args, P_scaled):
    """
    Execute multi-start optimisation using slsqp-jax and select the best solution.

    Parameters
    ----------
    problem  : NetworkProblem – wraps the simulation and objective
    args     : argparse.Namespace – CLI arguments
    P_scaled : np.ndarray (N_sites, T) – scaled target data for diagnostic scoring

    Returns
    -------
    merged_res    : OptimizationResult  (.X, .F, .J)
    best_idx      : int                 – index of solution with lowest total loss
    total_losses  : np.ndarray (n_solutions,) – total loss per solution
    """
    # Derive the number of starts and max steps from the parsed CLI args.
    # Prefer the new attributes when present; fall back to deprecated ones.
    if hasattr(args, "n_starts") and args.n_starts is not None:
        n_starts = int(args.n_starts)
    else:
        n_starts = int(getattr(args, "pop_size", 20))
    if hasattr(args, "max_steps") and args.max_steps is not None:
        max_steps = int(args.max_steps)
    else:
        max_steps = int(getattr(args, "gen", 200))

    # Sanity defaults
    n_starts = max(1, n_starts)
    max_steps = max(64, max_steps)

    # Tolerance for the SLSQP solver
    rtol = float(getattr(args, "rtol", 1e-6))
    atol = float(getattr(args, "atol", 1e-6))

    # Warn if deprecated fields were used instead of the preferred ones
    if (
        hasattr(args, "pop_size")
        and args.pop_size is not None
        and (not hasattr(args, "n_starts") or args.n_starts is None)
    ):
        logger.warning(
            f"[!] --pop-size ({args.pop_size}) is deprecated; use --n-starts instead."
        )
    if (
        hasattr(args, "gen")
        and args.gen is not None
        and (not hasattr(args, "max_steps") or args.max_steps is None)
    ):
        logger.warning(
            f"[!] --gen ({args.gen}) is deprecated; use --max-steps instead."
        )

    w_phospho = getattr(args, "loss_weight_phospho", 1.0)
    w_abundance = getattr(args, "loss_weight_abundance", 1.0)
    w_reg = getattr(args, "loss_weight_reg", 1.0)

    # Warn if evolutionary algorithm flags were passed
    algo = getattr(args, "algorithm", None)
    if algo is not None:
        logger.warning(
            f"[!] --algorithm {algo!r} is not used by the slsqp-jax backend; "
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

    starts = _generate_starts(n_starts, xl, xu)

    logger.header("[*] Starting Multi-Start SLSQP Optimisation")
    logger.info(
        f"    {len(starts)} starting points, max_steps={max_steps} each"
    )
    logger.info(
        f"    weights: phospho={w_phospho}, abundance={w_abundance}, reg={w_reg}"
    )

    all_X, all_F, all_total, all_diag_rows = [], [], [], []

    for i, theta0 in enumerate(starts):
        logger.info(f"--- Run {i + 1}/{len(starts)} ---")

        try:
            theta_opt, total_loss, f1, f2, f3, result_status, diag = run_single_optimisation(
                loss_fn, theta0, xl, xu, max_steps=max_steps, rtol=rtol, atol=atol
            )
            finite_sol = bool(np.isfinite(theta_opt).all() and np.isfinite(total_loss))
            all_X.append(theta_opt)
            all_F.append([f1, f2, f3])
            all_total.append(total_loss)
            all_diag_rows.append({
                "start_id": i,
                "result_status": result_status,
                "total_loss": total_loss,
                "phosphosite_loss": f1,
                "abundance_loss": f2,
                "regularization_loss": f3,
                "finite_solution": finite_sol,
                "selected_best": False,
                **diag,
            })
            logger.info(
                f"    -> total={total_loss:.4f}  f1={f1:.4f}  f2={f2:.4f}  f3={f3:.4f}"
                f"  status={result_status}"
            )
        except Exception as e:
            logger.warning(f"    -> Run {i + 1} failed: {e}")
            all_diag_rows.append({
                "start_id": i,
                "result_status": f"EXCEPTION: {e}",
                "total_loss": float("inf"),
                "phosphosite_loss": float("nan"),
                "abundance_loss": float("nan"),
                "regularization_loss": float("nan"),
                "finite_solution": False,
                "selected_best": False,
            })

    if not all_X:
        raise RuntimeError(
            "All optimisation runs failed to produce a solution. "
            "Check your data, bounds, and model dimensions."
        )

    X_combined = np.array(all_X)
    F_combined = np.array(all_F)
    total_losses = np.array(all_total)

    logger.info(f"[*] Combined: {len(X_combined)} solutions collected.")

    # Select best solution by minimum total loss (single-objective criterion)
    best_idx = int(np.argmin(total_losses))
    best_loss = total_losses[best_idx]
    logger.success(
        f"[*] Best Solution: total_loss = {best_loss:.6f} (idx={best_idx})"
    )

    # Mark the selected best in diagnostics
    for row in all_diag_rows:
        if row["start_id"] == best_idx:
            row["selected_best"] = True

    # Compute Fréchet Distance as a diagnostic metric (not used for selection)
    logger.info("[*] Computing Fréchet Distances (diagnostic only)...")
    frechet_scores = np.full(len(X_combined), np.inf)
    true_coords = np.ascontiguousarray(P_scaled, dtype=np.float64)
    for i in range(len(X_combined)):
        P_pred = problem.simulate(X_combined[i])
        pred_coords = np.ascontiguousarray(P_pred, dtype=np.float64)
        try:
            frechet_scores[i] = frechet_distance(true_coords, pred_coords)
        except Exception as e:
            logger.warning(f"    Fréchet error at idx {i}: {e}")
    logger.info(
        f"    -> Fréchet at best (idx={best_idx}): {frechet_scores[best_idx]:.6f}"
    )

    # Add frechet scores to diagnostics rows (only for successful runs)
    freq_iter = iter(frechet_scores)
    for row in all_diag_rows:
        if row["finite_solution"]:
            try:
                row["frechet_distance"] = next(freq_iter)
            except StopIteration:
                row["frechet_distance"] = float("inf")
        else:
            row["frechet_distance"] = float("inf")

    # Save diagnostics to disk if outdir is available
    outdir = getattr(args, "outdir", None)
    if outdir is not None:
        _save_optimization_diagnostics(outdir, all_diag_rows, best_idx, X_combined, F_combined)

    merged_res = OptimizationResult(X=X_combined, F=F_combined, J=total_losses)
    return merged_res, best_idx, total_losses


def _save_optimization_diagnostics(outdir, diag_rows, best_idx, X_combined, F_combined):
    """Save optimization_diagnostics.tsv and loss_components.tsv to outdir."""
    os.makedirs(outdir, exist_ok=True)

    df_diag = pd.DataFrame(diag_rows)
    diag_path = os.path.join(outdir, "optimization_diagnostics.tsv")
    df_diag.to_csv(diag_path, sep="\t", index=False)
    logger.info(f"[*] Saved {diag_path}")

    # loss_components.tsv: one row per start
    n_runs = len(X_combined)
    obj_names = ["phosphosite_loss", "abundance_loss", "regularization_loss"]
    df_loss = pd.DataFrame(F_combined, columns=obj_names)
    df_loss.insert(0, "start_id", range(n_runs))
    df_loss["selected_best"] = [i == best_idx for i in range(n_runs)]
    loss_path = os.path.join(outdir, "loss_components.tsv")
    df_loss.to_csv(loss_path, sep="\t", index=False)
    logger.info(f"[*] Saved {loss_path}")


def _generate_starts(n_starts, xl, xu):
    """
    Generate ``n_starts`` random starting parameter vectors in [xl, xu].

    Each start uses a fixed seed (0, 1, 2, …) for reproducibility.

    Returns
    -------
    list of np.ndarray, each of shape (dim,)
    """
    dim = len(xl)
    starts = []
    for i in range(n_starts):
        rng = np.random.default_rng(i)
        theta0 = xl + rng.random(dim) * (xu - xl)
        starts.append(theta0)
    return starts
