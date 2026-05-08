# SPDX-License-Identifier: MIT
"""
optimization_jaxopt.py
JAXopt-based bounded optimisation backend for the phospho-network.

Solver options
--------------
``"lbfgsb"``
    ``jaxopt.ScipyBoundedMinimize`` with L-BFGS-B.  Hard box constraints via
    SciPy; quasi-Newton convergence.  Not JIT-able (calls NumPy internally).
    This is the recommended solver in this module.

``"projected_gradient"``
    ``jaxopt.ProjectedGradient`` with ``projection_box``.  Pure-JAX projected
    gradient descent.  JIT-able; fully differentiable through the solution via
    implicit-function theorem.  Use when you need the result to be
    differentiable w.r.t. hyper-parameters (e.g. meta-learning bounds).

⚠️  JAXopt is no longer actively developed upstream (as of 2025).  Use this
    module for legacy compatibility or when ProjectedGradient differentiability
    is strictly required.  Prefer ``optimization_optax.py`` for new work.

Interface (mirrors optimization.py)
------------------------------------
    run_single_optimisation_jaxopt(loss_fn, theta0, xl, xu, ...)
        -> (theta_opt, total_loss, f1, f2, f3, f4)

``loss_fn`` must have the signature produced by ``make_loss_fn`` in
``optimization.py``::

    loss_fn(theta, args) -> (scalar_loss, (f1, f2, f3, f4))
"""

import numpy as np
import jax
import jax.numpy as jnp

from phoscrosstalk.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _strip_aux(loss_fn):
    """Return a scalar-only wrapper; JAXopt solvers do not support has_aux."""
    def _fn(theta, *args, **kwargs):
        val, _ = loss_fn(theta, None)
        return val
    return _fn


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_single_optimisation_jaxopt(
    loss_fn,
    theta0,
    xl,
    xu,
    *,
    max_steps: int = 500,
    tol: float = 1e-6,
    verbose: bool = False,
    solver_kind: str = "lbfgsb",
    # ProjectedGradient-specific
    stepsize: float = 1e-3,
    acceleration: bool = True,
):
    """
    Run a single JAXopt bounded optimisation.

    Parameters
    ----------
    loss_fn : callable
        ``(theta, args) -> (scalar_loss, (f1, f2, f3, f4))``
        as returned by ``make_loss_fn()`` in ``optimization.py``.
    theta0 : np.ndarray
        Initial parameter vector.
    xl, xu : np.ndarray
        Lower/upper bounds vectors (same shape as *theta0*).
        These come from ``create_bounds()`` / ``problem.xl`` / ``problem.xu``.
    max_steps : int
        Maximum iterations.
    tol : float
        Solver convergence tolerance (gradient norm for L-BFGS-B; step norm for
        ProjectedGradient).
    solver_kind : {"lbfgsb", "projected_gradient"}
        ``"lbfgsb"``          – SciPy L-BFGS-B via JAXopt wrapper (recommended).
        ``"projected_gradient"`` – pure-JAX projected GD.
    stepsize : float
        Step size for ``"projected_gradient"`` only; ignored for ``"lbfgsb"``.
    acceleration : bool
        Enable Nesterov acceleration for ``"projected_gradient"`` (FISTA-style).

    Returns
    -------
    theta_opt : np.ndarray
        Best-fit parameter vector (float64, hard-clipped to [xl, xu]).
    total_loss : float
        f1 + f2 + f3 + f4.
    f1, f2, f3, f4 : float
        Diagnostic loss components (phosphosite / abundance / reg / mRNA).
    """
    # Lazy import so the rest of the codebase does not require JAXopt
    try:
        from jaxopt import ProjectedGradient, ScipyBoundedMinimize
        from jaxopt.projection import projection_box
    except ImportError as exc:
        raise ImportError(
            "JAXopt is required for this solver.  "
            "Install with: pip install jaxopt"
        ) from exc

    scalar_loss = _strip_aux(loss_fn)

    theta0_j = jnp.asarray(theta0, dtype=jnp.float64)
    xl_j = jnp.asarray(xl, dtype=jnp.float64)
    xu_j = jnp.asarray(xu, dtype=jnp.float64)
    bounds = (xl_j, xu_j)  # JAXopt convention: tuple (lower, upper)

    if solver_kind == "lbfgsb":
        # ------------------------------------------------------------------ #
        # jaxopt.ScipyBoundedMinimize
        #
        # API (v0.8):
        #   ScipyBoundedMinimize(fun, method, tol, options, maxiter, has_aux)
        #   .run(init_params, bounds=(lower, upper), *args) -> OptStep
        #   OptStep.params  – solution
        #   OptStep.state   – solver state (nit, fun_val, …)
        #
        # Note: NOT JIT-able (calls scipy internally).
        # Note: fun must accept (params, *args); args forwarded from .run().
        # ------------------------------------------------------------------ #
        solver = ScipyBoundedMinimize(
            fun=scalar_loss,
            method="l-bfgs-b",
            tol=tol,
            maxiter=max_steps,
            options={"disp": verbose},
        )
        sol = solver.run(theta0_j, bounds=bounds)
        theta_opt_j = sol.params

        if verbose:
            logger.info(
                "[jaxopt/lbfgsb] nit=%d  fun_val=%.4e",
                int(sol.state.iter_num),
                float(sol.state.fun_val),
            )

    elif solver_kind == "projected_gradient":
        # ------------------------------------------------------------------ #
        # jaxopt.ProjectedGradient
        #
        # API (v0.8):
        #   ProjectedGradient(fun, projection, stepsize, maxiter, tol,
        #                     acceleration, verbose)
        #   .run(init_params, hyperparams_proj=bounds, *args) -> OptStep
        #   OptStep.params  – solution
        #   OptStep.state   – (iter_num, error, …)
        #
        # hyperparams_proj is forwarded verbatim to projection_box, which
        # expects a (lower, upper) tuple.
        # ------------------------------------------------------------------ #
        solver = ProjectedGradient(
            fun=scalar_loss,
            projection=projection_box,
            stepsize=stepsize,
            maxiter=max_steps,
            tol=tol,
            acceleration=acceleration,
            verbose=verbose,
        )
        sol = solver.run(
            theta0_j,
            hyperparams_proj=bounds,
        )
        theta_opt_j = sol.params

        if verbose:
            logger.info(
                "[jaxopt/projected_gradient] nit=%d  error=%.4e",
                int(sol.state.iter_num),
                float(sol.state.error),
            )

    else:
        raise ValueError(
            f"Unknown solver_kind {solver_kind!r}. "
            "Choose 'lbfgsb' or 'projected_gradient'."
        )

    # Hard-clip to bounds for safety (L-BFGS-B may return values at machine eps
    # outside bounds; ProjectedGradient should already be feasible).
    theta_opt_j = jnp.clip(theta_opt_j, xl_j, xu_j)
    theta_opt = np.asarray(theta_opt_j, dtype=np.float64)

    # Recompute diagnostics at the solution point.
    _, (f1, f2, f3, f4) = loss_fn(theta_opt_j, None)
    f1, f2, f3, f4 = float(f1), float(f2), float(f3), float(f4)
    total_loss = f1 + f2 + f3 + f4

    logger.info(
        "[jaxopt/%s] final loss=%.4e  f1=%.3e f2=%.3e f3=%.3e f4=%.3e",
        solver_kind, total_loss, f1, f2, f3, f4,
    )
    return theta_opt, total_loss, f1, f2, f3, f4
