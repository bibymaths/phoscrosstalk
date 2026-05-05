"""
multistarts.py
Multi-start parameter fitting using Optimistix (single-objective framework).

Strategy
--------
1. Generate ``n_starts`` random starting points uniformly in [xl, xu] using
   fixed random seeds for reproducibility.
2. Run ``run_single_optimisation`` (Optimistix LevenbergMarquardt) from each
   starting point, optionally in parallel via ``ProcessPoolExecutor``.
   Falls back to serial execution automatically when the residuals closure is
   not picklable (e.g. JAX closures on some platforms).
3. Select the best solution by minimum **total loss** (the scalar objective
   that was minimised).
4. Compute the Discrete Fréchet Distance for each solution as a diagnostic
   metric only (logged but not used for selection).  Optionally parallelised.
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

import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from phoscrosstalk.fretchet import frechet_distance
from phoscrosstalk.logger import get_logger
from phoscrosstalk.optimization import (
    make_residuals_fn,
    run_single_optimisation,
)

logger = get_logger(__name__)


class OptimizationResult:
    """Minimal result container compatible with the main.py caller."""

    def __init__(self, X, F, J):
        self.X = X
        self.F = F
        self.J = J  # total loss per run


# ---------------------------------------------------------------------------
# Module-level workers for ProcessPoolExecutor
# ---------------------------------------------------------------------------


def _run_single_start_worker(task):
    """
    Worker function for parallel multi-start execution.

    Must be defined at module level so it is picklable by ProcessPoolExecutor.
    Applies per-worker CPU thread caps before any JAX usage.

    Parameters
    ----------
    task : tuple
        (i, theta0, residuals_fn, max_steps, opt_rtol, opt_atol, opt_verbose,
         optx_adjoint, ls_solver, jac_mode, threads_per_run)

    Returns
    -------
    tuple
        (i, success, theta_opt, total_loss, f1, f2, f3, f4, error_message)
    """
    (
        i,
        theta0,
        residuals_fn,
        max_steps,
        opt_rtol,
        opt_atol,
        opt_verbose,
        optx_adjoint,
        ls_solver,
        jac_mode,
        threads_per_run,
    ) = task

    # Cap BLAS/OMP threads per worker before any computation.
    # The import is deferred intentionally: in spawn-based subprocesses the
    # env vars must be set *before* any JAX/XLA initialisation happens in
    # this process; importing runtime_env here (which is pure stdlib) is safe
    # and ensures apply_cpu_env() is called before any downstream JAX import.
    from phoscrosstalk import runtime_env  # noqa: PLC0415

    runtime_env.apply_cpu_env(threads_per_run, overwrite=True)

    try:
        theta_opt, total_loss, f1, f2, f3, f4 = run_single_optimisation(
            residuals_fn,
            theta0,
            max_steps=max_steps,
            rtol=opt_rtol,
            atol=opt_atol,
            verbose=opt_verbose,
            optx_adjoint=optx_adjoint,
            ls_solver=ls_solver,
            jac_mode=jac_mode,
        )
        return (i, True, theta_opt, total_loss, f1, f2, f3, f4, None)
    except Exception as exc:
        return (i, False, None, None, None, None, None, None, str(exc))


def _frechet_worker(task):
    """
    Worker function for parallel Fréchet distance computation.

    Parameters
    ----------
    task : tuple
        (i, theta, problem, true_coords)

    Returns
    -------
    tuple
        (i, score)  where score is np.inf on failure
    """
    i, theta, problem, true_coords = task
    try:
        P_pred = problem.simulate(theta)
        if not np.all(np.isfinite(P_pred)):
            return (i, np.inf)
        pred_coords = np.ascontiguousarray(P_pred.T, dtype=np.float64)
        score = frechet_distance(true_coords, pred_coords)
        return (i, score)
    except Exception:
        return (i, np.inf)


def _can_pickle(obj) -> bool:
    """Return True if *obj* can be serialised with pickle."""
    try:
        pickle.dumps(obj)
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_multi_start_optimization(problem, args, P_scaled):
    """
    Execute multi-start optimisation using Optimistix and select the best solution.

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
    w_mrna = getattr(problem, "loss_weight_rna", getattr(args, "loss_weight_mrna", 1.0))

    # Optimistix solver settings (optimisation-level, not ODE solver)
    opt_rtol = getattr(args, "opt_rtol", getattr(args, "rtol", 1e-8))
    opt_atol = getattr(args, "opt_atol", getattr(args, "atol", 1e-8))
    opt_verbose = getattr(args, "opt_verbose", False)

    # Warn if evolutionary algorithm flags were passed
    algo = getattr(args, "algorithm", None)
    if algo is not None:
        logger.warning(
            f"[!] --algorithm {algo!r} is not used by the Optimistix backend; "
            "ignored for compatibility."
        )

    xl = problem.xl
    xu = problem.xu

    # Build the residual-vector function for LevenbergMarquardt + optx.least_squares.
    # This is the primary fitting path following the canonical Diffrax+Optimistix approach.  # noqa: E501
    # We use make_residuals_fn (not make_loss_fn) so the optimizer sees a residual vector.  # noqa: E501
    residuals_fn = make_residuals_fn(
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
        w_mrna=w_mrna,
        rtol=getattr(args, "rtol", 1e-6),
        atol=getattr(args, "atol", 1e-9),
        max_steps=getattr(args, "solver_max_steps", 16384),
        k_act_fn=getattr(problem, "k_act_fn", None),
        s_prod_fn=getattr(problem, "s_prod_fn", None),
        t_mrna=getattr(problem, "t_rna", None),
        rna_data_scaled=getattr(problem, "rna_obs_matched", None),
        rna_model_prot_idx=getattr(problem, "rna_model_prot_idx", None),
        rna_obs_idx=getattr(problem, "rna_obs_idx", None),
        rna_fit_genes=getattr(problem, "rna_fit_genes", None),
        R_data0=getattr(problem, "R_data0", None),
        W_data_mrna=getattr(problem, "W_data_mrna", None),
        rna_relax=getattr(problem, "rna_relax", 0.1),
        ode_solver_kind=getattr(args, "ode_solver", "tsit5"),
        ode_adjoint_kind=getattr(args, "ode_adjoint", "forward"),
        dt0=getattr(args, "ode_dt0", 0.01),
        root_find_max_steps=getattr(args, "ode_root_find_max_steps", 10),
        xl=problem.xl,
        xu=problem.xu,
    )

    starts = _generate_starts(n_starts, xl, xu)

    # ------------------------------------------------------------------
    # CPU parallelism plan
    # ------------------------------------------------------------------
    from phoscrosstalk.runtime_env import plan_cpu_runtime  # noqa: PLC0415

    cpu_plan = plan_cpu_runtime(
        cpu_threads=getattr(args, "cpu_threads", "auto"),
        n_starts=n_starts,
        parallel_starts=getattr(args, "parallel_starts", "auto"),
        threads_per_start=getattr(args, "threads_per_start", "auto"),
        reserve_cores=getattr(args, "reserve_cores", 0),
        use_physical_cores=getattr(args, "use_physical_cores", True),
    )
    topo = cpu_plan.topo

    logger.header("[*] Starting Multi-Start Optimistix Optimisation")
    logger.info(f"    Optimizer: Optimistix LevenbergMarquardt(verbose={opt_verbose})")
    logger.info(
        "    ODE solver: "
        f"Diffrax {getattr(args, 'ode_solver', 'tsit5')} "
        f"+ PIDController + {getattr(args, 'ode_adjoint', 'forward')} adjoint"
    )
    logger.info(
        "    Optimistix: "
        f"{getattr(args, 'ls_solver', 'lm')} "
        f"+ jac={getattr(args, 'jac_mode', 'fwd')} "
        f"+ adjoint={getattr(args, 'optx_adjoint', 'implicit')}"
    )
    logger.info(f"    {len(starts)} starting points, max_steps={max_steps} each")
    logger.info(
        f"    weights: phospho={w_phospho}, abundance={w_abundance}, reg={w_reg}, mrna={w_mrna}"  # noqa: E501
    )
    logger.info(f"    Optimistix rtol={opt_rtol}, atol={opt_atol}")
    logger.info(
        f"    ODE rtol={getattr(args, 'rtol', 1e-6)}, atol={getattr(args, 'atol', 1e-9)}, max_steps={getattr(args, 'solver_max_steps', 16384)}"  # noqa: E501
    )
    logger.info(
        "[runtime] CPU plan:"
        f"\n    SLURM active         = {topo.slurm_cpus is not None}"
        f"\n    logical CPUs         = {topo.logical_cpus}"
        f"\n    physical cores       = {topo.physical_cores if topo.physical_cores is not None else 'unknown'}"  # noqa: E501
        f"\n    affinity CPUs        = {topo.affinity_cpus if topo.affinity_cpus is not None else 'n/a'}"  # noqa: E501
        f"\n    usable CPUs (budget) = {cpu_plan.total_available_cpus}"
        f"\n    CPU source           = {topo.source}"
        f"\n    n_starts             = {n_starts}"
        f"\n    parallel starts      = {cpu_plan.n_parallel_runs}"
        f"\n    threads per start    = {cpu_plan.threads_per_run}"
        f"\n    XLA intra-op threads = {cpu_plan.xla_threads}"
        f"\n    BLAS/OpenMP threads  = {cpu_plan.blas_threads}"
    )

    # ------------------------------------------------------------------
    # Decide whether to use ProcessPoolExecutor
    # ------------------------------------------------------------------
    use_parallel = cpu_plan.n_parallel_runs > 1

    if use_parallel and not _can_pickle(residuals_fn):
        logger.warning(
            "[runtime] residuals_fn is not picklable (likely a JAX closure); "
            "falling back to serial execution.  Each run will use "
            f"{cpu_plan.threads_per_run} threads as configured by XLA."
        )
        use_parallel = False

    all_X, all_F, all_total = [], [], []

    if use_parallel:
        logger.info(
            f"[runtime] Running {n_starts} starts across "
            f"{cpu_plan.n_parallel_runs} parallel workers "
            f"({cpu_plan.threads_per_run} threads/worker)."
        )
        tasks = [
            (
                i,
                theta0,
                residuals_fn,
                max_steps,
                opt_rtol,
                opt_atol,
                opt_verbose,
                getattr(args, "optx_adjoint", "implicit"),
                getattr(args, "ls_solver", "lm"),
                getattr(args, "jac_mode", "fwd"),
                cpu_plan.threads_per_run,
            )
            for i, theta0 in enumerate(starts)
        ]

        results_map = {}
        try:
            with ProcessPoolExecutor(max_workers=cpu_plan.n_parallel_runs) as pool:
                future_to_i = {
                    pool.submit(_run_single_start_worker, task): task[0]
                    for task in tasks
                }
                for future in as_completed(future_to_i):
                    orig_i = future_to_i[future]
                    try:
                        result = future.result()
                        results_map[result[0]] = result
                    except Exception as exc:
                        logger.warning(
                            f"    -> Run {orig_i + 1} raised exception: {exc}"
                        )
                        results_map[orig_i] = (
                            orig_i,
                            False,
                            None,
                            None,
                            None,
                            None,
                            None,
                            None,
                            str(exc),
                        )
        except Exception as exc:
            logger.warning(
                f"[runtime] ProcessPoolExecutor failed ({exc}); "
                "falling back to serial execution."
            )
            # Fall back: run any missing starts serially
            for i, theta0 in enumerate(starts):
                if i in results_map:
                    continue
                logger.info(f"--- Run {i + 1}/{len(starts)} (serial fallback) ---")
                try:
                    theta_opt, total_loss, f1, f2, f3, f4 = run_single_optimisation(
                        residuals_fn,
                        theta0,
                        max_steps=max_steps,
                        rtol=opt_rtol,
                        atol=opt_atol,
                        verbose=opt_verbose,
                        optx_adjoint=getattr(args, "optx_adjoint", "implicit"),
                        ls_solver=getattr(args, "ls_solver", "lm"),
                        jac_mode=getattr(args, "jac_mode", "fwd"),
                    )
                    results_map[i] = (
                        i,
                        True,
                        theta_opt,
                        total_loss,
                        f1,
                        f2,
                        f3,
                        f4,
                        None,
                    )
                except Exception as inner_exc:
                    logger.warning(f"    -> Run {i + 1} failed: {inner_exc}")
                    results_map[i] = (
                        i,
                        False,
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        str(inner_exc),
                    )

        # Collect results in original start order
        for i in sorted(results_map):
            _, success, theta_opt, total_loss, f1, f2, f3, f4, err = results_map[i]
            if success:
                all_X.append(theta_opt)
                all_F.append([f1, f2, f3, f4])
                all_total.append(total_loss)
                logger.info(
                    f"--- Run {i + 1}/{len(starts)} -> "
                    f"total={total_loss:.4f}  f1={f1:.4f}  f2={f2:.4f}  f3={f3:.4f}  f4={f4:.4f}"  # noqa: E501
                )
            else:
                logger.warning(f"    -> Run {i + 1} failed: {err}")

    else:
        # Serial execution
        for i, theta0 in enumerate(starts):
            logger.info(f"--- Run {i + 1}/{len(starts)} ---")
            try:
                theta_opt, total_loss, f1, f2, f3, f4 = run_single_optimisation(
                    residuals_fn,
                    theta0,
                    max_steps=max_steps,
                    rtol=opt_rtol,
                    atol=opt_atol,
                    verbose=opt_verbose,
                    optx_adjoint=getattr(args, "optx_adjoint", "implicit"),
                    ls_solver=getattr(args, "ls_solver", "lm"),
                    jac_mode=getattr(args, "jac_mode", "fwd"),
                )
                all_X.append(theta_opt)
                all_F.append([f1, f2, f3, f4])
                all_total.append(total_loss)
                logger.info(
                    f"    -> total={total_loss:.4f}  f1={f1:.4f}  f2={f2:.4f}  f3={f3:.4f}  f4={f4:.4f}"  # noqa: E501
                )
            except Exception as exc:
                logger.warning(f"    -> Run {i + 1} failed: {exc}")

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
    logger.success(f"[*] Best Solution: total_loss = {best_loss:.6f} (idx={best_idx})")

    # ------------------------------------------------------------------
    # Fréchet Distance diagnostics (optional, not used for selection)
    # ------------------------------------------------------------------
    logger.info("[*] Computing Fréchet Distances (diagnostic only)...")
    frechet_scores = np.full(len(X_combined), np.inf)
    # P_scaled shape is (N_sites, T).  Transpose to (T, N_sites) so that each
    # row is a time-point in N_sites-dimensional feature space, which is the
    # standard "curve of observations" orientation expected by frechet_distance.
    true_coords = np.ascontiguousarray(P_scaled.T, dtype=np.float64)

    parallel_frechet_cfg = getattr(args, "parallel_frechet", "auto")
    _pf_str = str(parallel_frechet_cfg).lower()
    if _pf_str == "auto":
        use_frechet_parallel = len(X_combined) > 2 and cpu_plan.n_parallel_runs > 1
    elif _pf_str in ("true", "1", "yes"):
        use_frechet_parallel = True
    else:
        use_frechet_parallel = False

    if use_frechet_parallel and not _can_pickle(problem):
        logger.info(
            "[runtime] problem is not picklable; using serial Fréchet computation."
        )
        use_frechet_parallel = False

    if use_frechet_parallel:
        logger.info(
            f"[runtime] Computing Fréchet scores in parallel "
            f"(max_workers={cpu_plan.n_parallel_runs})."
        )
        frechet_tasks = [
            (i, X_combined[i], problem, true_coords) for i in range(len(X_combined))
        ]
        try:
            with ProcessPoolExecutor(max_workers=cpu_plan.n_parallel_runs) as pool:
                for score_i, score_val in pool.map(_frechet_worker, frechet_tasks):
                    frechet_scores[score_i] = score_val
        except Exception as exc:
            logger.warning(
                f"[runtime] Parallel Fréchet failed ({exc}); falling back to serial."
            )
            use_frechet_parallel = False  # re-run serially below

    if not use_frechet_parallel:
        for i in range(len(X_combined)):
            P_pred = problem.simulate(X_combined[i])
            if not np.all(np.isfinite(P_pred)):
                logger.warning(
                    f"    Fréchet idx {i}: P_pred contains non-finite values; skipping."
                )
                continue
            pred_coords = np.ascontiguousarray(P_pred.T, dtype=np.float64)
            try:
                frechet_scores[i] = frechet_distance(true_coords, pred_coords)
            except Exception as exc:
                logger.warning(f"    Fréchet error at idx {i}: {exc}")

    logger.info(
        f"    -> Fréchet at best (idx={best_idx}): {frechet_scores[best_idx]:.6f}"
    )

    merged_res = OptimizationResult(X=X_combined, F=F_combined, J=total_losses)
    return merged_res, best_idx, total_losses


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
