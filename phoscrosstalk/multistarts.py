"""
multistarts.py
Multi-start parameter fitting using Optimistix (single-objective framework).

Strategy
--------
1. Generate ``n_starts`` random starting points uniformly in [xl, xu] using
   fixed random seeds for reproducibility.
2. Run ``run_single_optimisation`` (Optimistix LevenbergMarquardt) from each
   starting point, optionally in parallel via ``ProcessPoolExecutor``.
   Falls back to serial execution automatically when ``residual_kwargs`` is
   not picklable (e.g. contains non-picklable callables or objects).
3. Select the best solution by minimum **total loss** (the scalar objective
   that was minimised).
4. Compute the Discrete Fréchet Distance for each solution as a diagnostic
   metric only (logged but not used for selection).  Optionally parallelised.
5. Aggregate all solutions into a result object compatible with the
   caller in main.py.

Parallel worker contract
------------------------
Workers receive only picklable data (``residual_kwargs``), never a pre-built
JAX closure.  The worker rebuilds ``k_act_fn`` / ``s_prod_fn`` and then
calls ``make_residuals_fn`` locally, *after* applying per-worker CPU/XLA
environment settings.  This avoids pickling JAX closures while still
enabling process-level parallelism.

Backward-compatible return interface
-------------------------------------
Returns (merged_res, best_idx, total_losses) where:
  merged_res.X  : (n_solutions, n_params)
  merged_res.F  : (n_solutions, 4)       – [f1, f2, f3, f4] loss components
  merged_res.J  : (n_solutions,)         – total loss per run (used for selection)
  best_idx      : int                    – index of lowest-total-loss solution
  total_losses  : np.ndarray             – total loss for each solution

Old pymoo-specific flags (--gen, --pop-size, --algorithm) are mapped:
  --pop-size   → n_starts
  --gen        → max_steps_per_run
  --algorithm  → ignored with a warning (non-fatal)
"""

import multiprocessing as mp
import pickle
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from phoscrosstalk.fretchet import frechet_distance
from phoscrosstalk.logger import get_logger

# NOTE: make_residuals_fn and run_single_optimisation are imported lazily
# (inside functions) to avoid triggering JAX initialisation at module import
# time.  This is critical for spawn-based worker processes: the worker imports
# multistarts.py before apply_cpu_env() is called, so any top-level JAX import
# here would initialise XLA with the wrong thread count.

logger = get_logger()


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
    Applies per-worker CPU thread caps before any JAX usage, then delegates to
    :func:`_run_one_start` which rebuilds the correct callable (residuals_fn for
    ``"optimistix"`` or loss_fn for gradient-based backends) from picklable
    ``residual_kwargs``.

    Parameters
    ----------
    task : tuple
        (i, theta0, residual_kwargs, max_steps, opt_rtol, opt_atol,
         opt_verbose, optx_adjoint, ls_solver, jac_mode, threads_per_run,
         backend, backend_kwargs)

    Returns
    -------
    tuple
        (i, success, theta_opt, total_loss, f1, f2, f3, f4, error_message)
    """
    (
        i,
        theta0,
        residual_kwargs,
        max_steps,
        opt_rtol,
        opt_atol,
        opt_verbose,
        optx_adjoint,
        ls_solver,
        jac_mode,
        threads_per_run,
        backend,
        backend_kwargs,
    ) = task

    # Cap BLAS/OMP/XLA threads per worker BEFORE any JAX import.
    # The import is deferred intentionally: in spawn-based subprocesses the
    # env vars must be set *before* any JAX/XLA initialisation; importing
    # runtime_env here (which is pure stdlib) is safe.
    from phoscrosstalk import runtime_env  # noqa: PLC0415

    runtime_env.apply_cpu_env(threads_per_run, overwrite=True)

    # JAX-heavy work happens inside _run_one_start (deferred imports), so
    # XLA picks up the correct intra-op thread count on first import.
    try:
        theta_opt, total_loss, f1, f2, f3, f4 = _run_one_start(
            backend=backend,
            theta0=theta0,
            residual_kwargs=residual_kwargs,
            max_steps=max_steps,
            opt_rtol=opt_rtol,
            opt_atol=opt_atol,
            opt_verbose=opt_verbose,
            optx_adjoint=optx_adjoint,
            ls_solver=ls_solver,
            jac_mode=jac_mode,
            backend_kwargs=backend_kwargs,
        )
        return (i, True, theta_opt, total_loss, f1, f2, f3, f4, None)
    except Exception as exc:
        return (
            i,
            False,
            None,
            None,
            None,
            None,
            None,
            None,
            f"{type(exc).__name__}: {exc}",
        )


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


def _find_non_picklable_items(mapping: dict) -> list:
    """
    Return a list of keys in *mapping* whose values cannot be pickled.

    Useful for diagnosing which fields in ``residual_kwargs`` prevent parallel
    execution.

    Parameters
    ----------
    mapping : dict
        A flat dictionary of keyword arguments (e.g. ``residual_kwargs``).

    Returns
    -------
    list[str]
        Keys whose values raise an exception when passed to ``pickle.dumps``.
    """
    bad_keys = []
    for key, value in mapping.items():
        try:
            pickle.dumps(value)
        except Exception:
            bad_keys.append(key)
    return bad_keys


def _build_residual_kwargs(
    problem, args, w_phospho, w_abundance, w_reg, w_mrna
) -> dict:
    """
    Collect all inputs needed by ``make_residuals_fn`` into a picklable dict.

    The returned dict contains only plain numpy arrays, scalars, and strings –
    no JAX arrays, no JAX closures.  ``k_act_fn`` and ``s_prod_fn`` are
    replaced by the raw data needed to rebuild them inside a worker process
    (stored under the private keys ``_k_act_rebuild_kwargs`` and
    ``_s_prod_rebuild_kwargs``).  If the rebuild kwargs are not available on
    ``problem``, the callable is stored directly under ``k_act_fn`` /
    ``s_prod_fn`` and may cause the pickle check to fail, triggering serial
    fallback.

    Parameters
    ----------
    problem : NetworkProblem
    args    : argparse.Namespace / SimpleNamespace
    w_phospho, w_abundance, w_reg, w_mrna : float

    Returns
    -------
    dict
        Keyword arguments suitable for ``make_residuals_fn(**residual_kwargs)``
        after removing the ``_k_act_rebuild_kwargs`` / ``_s_prod_rebuild_kwargs``
        special keys and replacing them with the built callables.
    """
    kwargs: dict = {
        "dims": problem.dims,
        "t": problem.t,
        "P_data": problem.P_data,
        "A_scaled": problem.A_scaled,
        "prot_idx_for_A": problem.prot_idx_for_A,
        "W_data": problem.W_data,
        "W_data_prot": problem.W_data_prot,
        "Cg": problem.Cg,
        "Cl": problem.Cl,
        "site_prot_idx": problem.site_prot_idx,
        "K_site_kin": problem.K_site_kin,
        "R": problem.R,
        "L_alpha": problem.L_alpha,
        "kin_to_prot_idx": problem.kin_to_prot_idx,
        "receptor_mask_prot": problem.receptor_mask_prot,
        "receptor_mask_kin": problem.receptor_mask_kin,
        "mechanism": problem.mechanism,
        "lambda_net": problem.lambda_net,
        "reg_lambda": problem.reg_lambda,
        "w_phospho": w_phospho,
        "w_abundance": w_abundance,
        "w_reg": w_reg,
        "w_mrna": w_mrna,
        "rtol": getattr(args, "rtol", 1e-6),
        "atol": getattr(args, "atol", 1e-9),
        "max_steps": getattr(args, "solver_max_steps", 16384),
        "t_mrna": getattr(problem, "t_rna", None),
        "rna_data_scaled": getattr(problem, "rna_obs_matched", None),
        "rna_model_prot_idx": getattr(problem, "rna_model_prot_idx", None),
        "rna_obs_idx": getattr(problem, "rna_obs_idx", None),
        "rna_fit_genes": getattr(problem, "rna_fit_genes", None),
        "R_data0": getattr(problem, "R_data0", None),
        "W_data_mrna": getattr(problem, "W_data_mrna", None),
        "rna_relax": getattr(problem, "rna_relax", 0.1),
        "ode_solver_kind": getattr(args, "ode_solver", "tsit5"),
        "ode_adjoint_kind": getattr(args, "ode_adjoint", "forward"),
        "dt0": getattr(args, "ode_dt0", 0.01),
        "root_find_max_steps": getattr(args, "ode_root_find_max_steps", 10),
        "xl": problem.xl,
        "xu": problem.xu,
    }

    # k_act_fn: prefer picklable rebuild kwargs stored on problem; fall back
    # to the callable (which will fail the pickle check → serial fallback).
    k_act_rebuild = getattr(problem, "_k_act_rebuild_kwargs", None)
    if k_act_rebuild is not None:
        kwargs["_k_act_rebuild_kwargs"] = k_act_rebuild
        # k_act_fn will be built inside the worker from these raw kwargs
    else:
        kwargs["k_act_fn"] = getattr(problem, "k_act_fn", None)

    # s_prod_fn: same pattern
    s_prod_rebuild = getattr(problem, "_s_prod_rebuild_kwargs", None)
    if s_prod_rebuild is not None:
        kwargs["_s_prod_rebuild_kwargs"] = s_prod_rebuild
        # s_prod_fn will be built inside the worker from these raw kwargs
    else:
        kwargs["s_prod_fn"] = getattr(problem, "s_prod_fn", None)

    return kwargs


def _residuals_fn_from_kwargs(residual_kwargs: dict):
    """
    Build ``residuals_fn`` locally from ``residual_kwargs``.

    Handles the ``_k_act_rebuild_kwargs`` / ``_s_prod_rebuild_kwargs`` special
    keys by rebuilding the JAX closures before calling ``make_residuals_fn``.
    Used for the serial execution path.
    """
    from phoscrosstalk.derived_rates import (  # noqa: PLC0415
        make_k_act_fn,
        make_s_prod_fn,
    )
    from phoscrosstalk.optimization import make_residuals_fn  # noqa: PLC0415

    mkwargs = dict(residual_kwargs)

    if "_k_act_rebuild_kwargs" in mkwargs:
        k_act_rebuild = mkwargs.pop("_k_act_rebuild_kwargs")
        mkwargs["k_act_fn"] = make_k_act_fn(**k_act_rebuild)

    if "_s_prod_rebuild_kwargs" in mkwargs:
        s_prod_rebuild = mkwargs.pop("_s_prod_rebuild_kwargs")
        mkwargs["s_prod_fn"] = make_s_prod_fn(**s_prod_rebuild)

    return make_residuals_fn(**mkwargs)


def _residuals_fn_to_loss_fn(residuals_fn):
    """
    Adapt a residuals_fn to the scalar loss_fn interface.

    Gradient-based backends (jaxopt, optax, scipy_jax, mpax) expect::

        loss_fn(theta, args) -> (scalar_loss, (f1, f2, f3, f4))

    The canonical ``"optimistix"`` backend expects::

        residuals_fn(theta, args) -> (residuals_1d, (f1, f2, f3, f4))

    This adapter wraps the residuals_fn to produce a scalar by computing
    ``jnp.sum(residuals ** 2)``, which equals ``f1+f2+f3+f4`` because each
    residual block is the sqrt-weighted loss term.  The (f1, f2, f3, f4)
    auxiliary values are passed through unchanged.
    """
    def loss_fn(theta, args):
        import jax.numpy as jnp  # noqa: PLC0415
        residuals, aux = residuals_fn(theta, args)
        return jnp.sum(residuals ** 2), aux

    return loss_fn


def _run_one_start(
    backend: str,
    theta0,
    residual_kwargs: dict,
    max_steps: int,
    opt_rtol: float,
    opt_atol: float,
    opt_verbose: bool,
    optx_adjoint: str,
    ls_solver: str,
    jac_mode: str,
    backend_kwargs: dict,
    jaxpr_out_dir=None,
):
    """
    Central single-start dispatch: builds the correct callable for the
    selected backend, then calls ``dispatch_optimisation``.

    Callable contracts
    ------------------
    ``"optimistix"``
        Expects a *residuals_fn*: ``(theta, args) -> (residuals_1d, aux)``.
        Passes Optimistix-specific kwargs (ls_solver, optx_adjoint, jac_mode).

    All other backends
        Expect a *loss_fn*: ``(theta, args) -> (scalar_loss, aux)``.
        The residuals_fn is adapted via :func:`_residuals_fn_to_loss_fn`.
        ``max_steps`` is forwarded; for ``"mpax"`` it is mapped to
        ``max_sqp_steps`` unless overridden in ``backend_kwargs``.

    ``backend_kwargs`` (from ``args.optimizer_backend_kwargs``) can override
    any of the auto-forwarded kwargs for fine-grained per-backend tuning.

    Parameters
    ----------
    backend : str
        Backend key from AVAILABLE_BACKENDS.
    theta0 : np.ndarray
        Starting parameter vector.
    residual_kwargs : dict
        Picklable kwargs for ``make_residuals_fn``; must contain ``"xl"`` and
        ``"xu"`` keys for bounds.
    max_steps : int
        Iteration cap forwarded to the solver.
    opt_rtol, opt_atol : float
        Optimistix convergence tolerances (ignored for non-optimistix backends).
    opt_verbose : bool
        Enable solver verbosity.
    optx_adjoint, ls_solver, jac_mode : str
        Optimistix-specific settings (ignored for non-optimistix backends).
    backend_kwargs : dict
        Extra kwargs forwarded to the solver; override auto-injected values.
    jaxpr_out_dir : str or None
        Directory for jaxpr reports (optimistix backend only).

    Returns
    -------
    tuple
        (theta_opt, total_loss, f1, f2, f3, f4)
    """
    from phoscrosstalk.optimizers.dispatch import dispatch_optimisation  # noqa: PLC0415

    residuals_fn = _residuals_fn_from_kwargs(residual_kwargs)
    xl = residual_kwargs.get("xl")
    xu = residual_kwargs.get("xu")

    if backend == "optimistix":
        # Canonical residual-based path: pass residuals_fn directly.
        # dispatch_optimisation ignores xl/xu for this backend.
        kw = dict(
            max_steps=max_steps,
            rtol=opt_rtol,
            atol=opt_atol,
            verbose=opt_verbose,
            ls_solver=ls_solver,
            optx_adjoint=optx_adjoint,
            jac_mode=jac_mode,
            jaxpr_out_dir=jaxpr_out_dir,
        )
        kw.update(backend_kwargs)
        return dispatch_optimisation("optimistix", residuals_fn, theta0, xl, xu, **kw)
    else:
        # Gradient-based backends: adapt residuals_fn to scalar loss_fn.
        loss_fn = _residuals_fn_to_loss_fn(residuals_fn)
        if backend == "mpax":
            # MPAX uses max_sqp_steps instead of max_steps.
            kw = {"verbose": opt_verbose}
            if "max_sqp_steps" not in backend_kwargs:
                kw["max_sqp_steps"] = max_steps
        else:
            kw = {"max_steps": max_steps, "verbose": opt_verbose}
        kw.update(backend_kwargs)
        return dispatch_optimisation(backend, loss_fn, theta0, xl, xu, **kw)


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

    # Read backend selection from args (resolved once in main.py).
    # Default to "optimistix" for full backward-compatibility.
    backend = getattr(args, "optimizer_backend", "optimistix")
    backend_kwargs = dict(getattr(args, "optimizer_backend_kwargs", None) or {})

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

    # Build picklable residual kwargs dict (one dict shared by serial and parallel paths).
    residual_kwargs = _build_residual_kwargs(
        problem, args, w_phospho, w_abundance, w_reg, w_mrna
    )  # noqa: E501

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

    logger.header("[*] Starting Multi-Start Optimisation")
    logger.info(f"    Backend: {backend}")
    if backend == "optimistix":
        logger.info(f"    Optimizer: Optimistix LevenbergMarquardt(verbose={opt_verbose})")
        logger.info(
            "    Optimistix: "
            f"{getattr(args, 'ls_solver', 'lm')} "
            f"+ jac={getattr(args, 'jac_mode', 'fwd')} "
            f"+ adjoint={getattr(args, 'optx_adjoint', 'implicit')}"
        )
    logger.info(
        "    ODE solver: "
        f"Diffrax {getattr(args, 'ode_solver', 'tsit5')} "
        f"+ PIDController + {getattr(args, 'ode_adjoint', 'forward')} adjoint"
    )
    logger.info(f"    {len(starts)} starting points, max_steps={max_steps} each")
    logger.info(
        f"    weights: phospho={w_phospho}, abundance={w_abundance}, reg={w_reg}, mrna={w_mrna}"  # noqa: E501
    )
    if backend == "optimistix":
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
    t_multistart_begin = time.perf_counter()

    # ------------------------------------------------------------------
    # Decide whether to use ProcessPoolExecutor
    # ------------------------------------------------------------------
    use_parallel = cpu_plan.n_parallel_runs > 1

    if use_parallel and not _can_pickle(residual_kwargs):
        bad_keys = _find_non_picklable_items(residual_kwargs)
        logger.warning(
            "[runtime] residual_kwargs is not picklable; falling back to serial execution.\n"
            "    This usually means the residual kwargs contain non-picklable callables\n"
            "    (e.g. JAX closures stored under k_act_fn / s_prod_fn).\n"
            f"    Non-picklable residual kwargs: {', '.join(bad_keys) if bad_keys else '(unknown)'}"  # noqa: E501
        )
        use_parallel = False

    if use_parallel:
        logger.info(
            "[runtime] Multiprocessing strategy: workers rebuild callable from residual_kwargs."  # noqa: E501
        )

    all_X, all_F, all_total = [], [], []

    if use_parallel:
        logger.info(
            f"Running {n_starts} starts across "
            f"{cpu_plan.n_parallel_runs} workers with "
            f"{cpu_plan.threads_per_run} threads/worker."
        )
        tasks = [
            (
                i,
                theta0,
                residual_kwargs,
                max_steps,
                opt_rtol,
                opt_atol,
                opt_verbose,
                getattr(args, "optx_adjoint", "implicit"),
                getattr(args, "ls_solver", "lm"),
                getattr(args, "jac_mode", "fwd"),
                cpu_plan.threads_per_run,
                backend,
                backend_kwargs,
            )
            for i, theta0 in enumerate(starts)
        ]

        results_map = {}
        try:
            ctx = mp.get_context("spawn")
            with ProcessPoolExecutor(
                max_workers=cpu_plan.n_parallel_runs, mp_context=ctx
            ) as pool:
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
                            f"{type(exc).__name__}: {exc}",
                        )
        except Exception as exc:
            logger.warning(
                f"[runtime] ProcessPoolExecutor failed ({exc}); "
                "falling back to serial execution."
            )

            # Fall back: run any missing starts serially using _run_one_start.
            for i, theta0 in enumerate(starts):
                if i in results_map:
                    continue
                logger.info(
                    "[fit]  multistart  start=%02d/%02d  initializing  (serial fallback)",
                    i + 1, len(starts),
                )
                t_start = time.perf_counter()
                try:
                    theta_opt, total_loss, f1, f2, f3, f4 = _run_one_start(
                        backend=backend,
                        theta0=theta0,
                        residual_kwargs=residual_kwargs,
                        max_steps=max_steps,
                        opt_rtol=opt_rtol,
                        opt_atol=opt_atol,
                        opt_verbose=opt_verbose,
                        optx_adjoint=getattr(args, "optx_adjoint", "implicit"),
                        ls_solver=getattr(args, "ls_solver", "lm"),
                        jac_mode=getattr(args, "jac_mode", "fwd"),
                        backend_kwargs=backend_kwargs,
                    )
                    elapsed = time.perf_counter() - t_start
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
                    logger.info(
                        "[fit]  multistart  start=%02d/%02d  loss=%.4e"
                        "  f1=%.4e  f2=%.4e  f3=%.4e  f4=%.4e  t=%.2fs",
                        i + 1, len(starts),
                        float(total_loss), float(f1), float(f2), float(f3), float(f4),
                        elapsed,
                    )
                except Exception as inner_exc:
                    elapsed = time.perf_counter() - t_start
                    logger.warning(
                        "[fit]  multistart  start=%02d/%02d  FAILED  t=%.2fs  error=%s",
                        i + 1, len(starts), elapsed, inner_exc,
                    )
                    results_map[i] = (
                        i,
                        False,
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        f"{type(inner_exc).__name__}: {inner_exc}",
                    )

        # Collect results in original start order
        _best_loss_parallel = float("inf")
        for i in sorted(results_map):
            _, success, theta_opt, total_loss, f1, f2, f3, f4, err = results_map[i]
            if success:
                all_X.append(theta_opt)
                all_F.append([f1, f2, f3, f4])
                all_total.append(total_loss)
                logger.info(
                    "[fit]  multistart  start=%02d/%02d  loss=%.4e"
                    "  f1=%.4e  f2=%.4e  f3=%.4e  f4=%.4e",
                    i + 1, len(starts),
                    float(total_loss), float(f1), float(f2), float(f3), float(f4),
                )
                if float(total_loss) < _best_loss_parallel:
                    _best_loss_parallel = float(total_loss)
                    logger.info(
                        "[fit]  multistart  new best  start=%02d  loss=%.4e",
                        i + 1, _best_loss_parallel,
                    )
            else:
                logger.warning(f"    -> Run {i + 1} failed: {err}")

    else:
        # Serial execution: use _run_one_start which builds the correct callable.
        # _jaxpr_out_dir is stored on args._jaxpr_out_dir by main.py when
        # cfg.debug.save_jaxpr_reports is True; None otherwise.
        _jaxpr_out_dir = getattr(args, "_jaxpr_out_dir", None)

        _best_loss_serial = float("inf")
        for i, theta0 in enumerate(starts):
            logger.info(
                "[fit]  multistart  start=%02d/%02d  initializing",
                i + 1, n_starts,
            )
            t_start = time.perf_counter()
            try:
                theta_opt, total_loss, f1, f2, f3, f4 = _run_one_start(
                    backend=backend,
                    theta0=theta0,
                    residual_kwargs=residual_kwargs,
                    max_steps=max_steps,
                    opt_rtol=opt_rtol,
                    opt_atol=opt_atol,
                    opt_verbose=opt_verbose,
                    optx_adjoint=getattr(args, "optx_adjoint", "implicit"),
                    ls_solver=getattr(args, "ls_solver", "lm"),
                    jac_mode=getattr(args, "jac_mode", "fwd"),
                    backend_kwargs=backend_kwargs,
                    jaxpr_out_dir=_jaxpr_out_dir if i == 0 else None,
                )
                elapsed = time.perf_counter() - t_start
                all_X.append(theta_opt)
                all_F.append([f1, f2, f3, f4])
                all_total.append(total_loss)
                logger.info(
                    "[fit]  multistart  start=%02d/%02d  loss=%.4e"
                    "  f1=%.4e  f2=%.4e  f3=%.4e  f4=%.4e  t=%.2fs",
                    i + 1, n_starts,
                    float(total_loss), float(f1), float(f2), float(f3), float(f4),
                    elapsed,
                )
                if float(total_loss) < _best_loss_serial:
                    _best_loss_serial = float(total_loss)
                    logger.info(
                        "[fit]  multistart  new best  start=%02d  loss=%.4e",
                        i + 1, _best_loss_serial,
                    )
            except Exception as exc:
                elapsed = time.perf_counter() - t_start
                logger.warning(
                    "[fit]  multistart  start=%02d/%02d  FAILED  t=%.2fs  error=%s",
                    i + 1, n_starts, elapsed, exc,
                )

    if not all_X:
        raise RuntimeError(
            f"All {n_starts} optimisation runs failed to produce a solution. "
            "Check your data, bounds, and model dimensions."
        )

    X_combined = np.array(all_X)
    F_combined = np.array(all_F)
    total_losses = np.array(all_total)
    t_multistart_total = time.perf_counter() - t_multistart_begin

    logger.info(
        "[fit]  multistart  complete  n_solutions=%d  total_t=%.1fs",
        len(X_combined), t_multistart_total,
    )

    # Select best solution by minimum total loss (single-objective criterion)
    best_idx = int(np.argmin(total_losses))
    best_loss = total_losses[best_idx]
    logger.success(f"[*] Best Solution: total_loss = {best_loss:.6f} (idx={best_idx})")

    # ------------------------------------------------------------------
    # Fréchet Distance diagnostics (optional, not used for selection)
    # ------------------------------------------------------------------
    logger.info("[*] Computing Fréchet Distances...")
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
            ctx = mp.get_context("spawn")
            with ProcessPoolExecutor(
                max_workers=cpu_plan.n_parallel_runs, mp_context=ctx
            ) as pool:
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
    if xl is None or xu is None:
        raise ValueError(
            f"_generate_starts received xl={xl}, xu={xu}. "
            "Bounds must be finite numpy arrays — check that create_bounds() "
            "returned valid values and ModelDims were set before calling it."
        )
    dim = len(xl)
    starts = []
    for i in range(n_starts):
        rng = np.random.default_rng(i)
        theta0 = xl + rng.random(dim) * (xu - xl)
        starts.append(theta0)
    return starts
