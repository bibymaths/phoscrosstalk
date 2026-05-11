# SPDX-License-Identifier: MIT
"""
JAXopt-based bounded optimisation backend for the phospho-network.

Solver options
--------------
"lbfgsb"
    jaxopt.ScipyBoundedMinimize with L-BFGS-B.  Hard box constraints via
    SciPy; quasi-Newton convergence.  Not JIT-able (calls NumPy internally).
    This is the recommended solver in this module.

"projected_gradient"
    jaxopt.ProjectedGradient with projection_box.  Pure-JAX projected
    gradient descent.  JIT-able; fully differentiable through the solution via
    implicit-function theorem.  Use when you need the result to be
    differentiable w.r.t. hyper-parameters (e.g. meta-learning bounds).

"osqp"
    jaxopt.OSQP solving a local quadratic surrogate of the nonlinear ODE loss.
    Constructs Q, c, G, h from the Hessian and gradient of the scalar loss at
    theta0, then solves the resulting QP and clips/projects back to bounds.
    Experimental; use only as a local quadratic-approximation step.

"box_osqp"
    jaxopt.BoxOSQP with box constraints passed directly as params_ineq.
    Same local-surrogate construction as "osqp".

"eq_qp"
    jaxopt.EqualityConstrainedQP.  No inequality/box constraints are supported
    directly; bounds are enforced by post-solve clipping.
    Same local-surrogate construction as "osqp".
"""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
from typing import Callable

from phoscrosstalk.logger import get_logger

logger = get_logger()


# ---------------------------------------------------------------------------
# Internal helpers – existing solvers
# ---------------------------------------------------------------------------


def _strip_aux(loss_fn):
    """Return a scalar-only wrapper; JAXopt solvers do not support has_aux."""

    def _fn(theta, *args, **kwargs):
        val, _ = loss_fn(theta, None)
        return val

    return _fn


# ---------------------------------------------------------------------------
# QP data helpers
# ---------------------------------------------------------------------------


def _regularize_psd_matrix(H: jnp.ndarray, ridge: float = 1e-6) -> jnp.ndarray:
    """
    Project H to the positive semi-definite cone, then add a ridge.

    Steps:
    1. Symmetrize: ``H_sym = 0.5 * (H + H.T)``.
    2. Eigendecompose and clamp negative eigenvalues to zero
       (projection onto the PSD cone).
    3. Add ``ridge * I`` to ensure strict positive definiteness.

    This guarantees that the returned matrix is PSD regardless of whether the
    input Hessian has negative eigenvalues (indefinite or negative definite).

    Args:
        H: Square matrix (Hessian or Gauss-Newton approximation).
        ridge: Diagonal regularisation strength added after PSD projection
            (default 1e-6).

    Returns:
        PSD matrix ``Q = V @ diag(max(λ, 0)) @ V.T + ridge * I``.
    """
    H_sym = 0.5 * (H + H.T)
    eigenvalues, eigenvectors = jnp.linalg.eigh(H_sym)
    eigenvalues = jnp.maximum(eigenvalues, 0.0)
    H_psd = eigenvectors @ jnp.diag(eigenvalues) @ eigenvectors.T
    return H_psd + ridge * jnp.eye(H_psd.shape[0], dtype=H_psd.dtype)


def _extract_qp_data_from_kwargs(kwargs: dict) -> dict | None:
    """
    Return explicit QP data if present in *kwargs*, otherwise ``None``.

    Recognised keys: ``params_obj``, ``params_eq``, ``params_ineq``,
    ``qp_mode``.  Returns the sub-dict only when ``qp_mode == "explicit"``
    and ``params_obj`` is present.

    Args:
        kwargs: optimizer_backend_kwargs dict from config / caller.

    Returns:
        Dict with ``params_obj``, ``params_eq``, ``params_ineq`` or ``None``.
    """
    if kwargs.get("qp_mode") != "explicit":
        return None
    params_obj = kwargs.get("params_obj")
    if params_obj is None:
        return None
    # Validate that params_obj looks like a (Q, c) tuple
    if not (isinstance(params_obj, (tuple, list)) and len(params_obj) == 2):
        raise ValueError(
            "optimizer_backend_kwargs['params_obj'] must be a (Q, c) tuple "
            f"when qp_mode='explicit'; got {type(params_obj)!r}."
        )
    return {
        "params_obj": params_obj,
        "params_eq": kwargs.get("params_eq"),
        "params_ineq": kwargs.get("params_ineq"),
    }


def _build_local_quadratic_qp(
        scalar_loss: Callable,
        theta0: jnp.ndarray,
        ridge: float = 1e-6,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Build a local quadratic approximation of *scalar_loss* around *theta0*.

    The surrogate is:

        loss(theta0 + d) ≈ loss(theta0) + g^T d + 0.5 d^T H d

    Re-expressed in theta (not d):

        0.5 theta^T Q theta + c^T theta + const

    where:

        Q = H_regularized
        c = g - H_regularized @ theta0

    Args:
        scalar_loss: Scalar callable ``f(theta) -> scalar``.
        theta0: Expansion point (float64 JAX array).
        ridge: Diagonal regularisation added to Hessian before use.

    Returns:
        Tuple ``(Q, c)`` where Q is the (n, n) PSD cost matrix and c is the
        (n,) linear cost vector.

    Raises:
        RuntimeError: If Hessian computation fails.
    """
    try:
        g = jax.grad(scalar_loss)(theta0)
        H = jax.hessian(scalar_loss)(theta0)
    except Exception as exc:
        raise RuntimeError(
            "QP surrogate construction failed: could not compute Hessian of "
            "the scalar loss at theta0.  Check that the loss function is "
            "twice-differentiable with respect to all parameters.\n"
            f"Original error: {exc}"
        ) from exc

    Q = _regularize_psd_matrix(H, ridge=ridge)
    c = g - Q @ theta0
    return Q, c


def _bounds_to_osqp_ineq(
        xl: jnp.ndarray,
        xu: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Convert box bounds [xl, xu] to a pair (G, h) for OSQP inequality G @ x <= h.

    Encodes:
        theta <= xu   →  I @ theta <= xu
        -theta <= -xl →  -I @ theta <= -xl

    so G = [I; -I], h = [xu; -xl].

    Args:
        xl: Lower bound vector, shape (n,).
        xu: Upper bound vector, shape (n,).

    Returns:
        (G, h) where G has shape (2n, n) and h has shape (2n,).
    """
    n = xl.shape[0]
    I = jnp.eye(n, dtype=xl.dtype)
    G = jnp.concatenate([I, -I], axis=0)
    h = jnp.concatenate([xu, -xl], axis=0)
    return G, h


def _bounds_to_box_osqp_ineq(
        xl: jnp.ndarray,
        xu: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Return ``(A, l, u)`` for BoxOSQP with ``l <= A @ x <= u``.

    BoxOSQP expects:
      - ``params_eq``   = A
      - ``params_ineq`` = (l, u)
    with constraints ``l <= A @ x <= u``.
    For pure box constraints on theta itself, ``A = I``, ``l = xl``, ``u = xu``.

    Args:
        xl: Lower bound vector, shape (n,).
        xu: Upper bound vector, shape (n,).

    Returns:
        ``(A, xl, xu)`` where ``A`` is the (n, n) identity matrix.
    """
    n = xl.shape[0]
    A = jnp.eye(n, dtype=xl.dtype)
    return A, xl, xu


def _run_osqp_qp(
        Q: jnp.ndarray,
        c: jnp.ndarray,
        G: jnp.ndarray,
        h: jnp.ndarray,
        theta0: jnp.ndarray,
        **solver_kwargs,
) -> jnp.ndarray:
    """
    Solve a QP with jaxopt.OSQP.

    Problem:

        min  0.5 x^T Q x + c^T x
        s.t. G x <= h

    The primal solution is extracted from ``sol.params.primal[0]``.

    Args:
        Q: Cost matrix (n, n), PSD.
        c: Linear cost vector (n,).
        G: Inequality constraint matrix (m, n).
        h: Inequality right-hand side (m,).
        theta0: Initial guess (used as warm-start where supported).
        **solver_kwargs: Forwarded to ``jaxopt.OSQP()``.

    Returns:
        Primal solution vector (n,).
    """
    try:
        from jaxopt import OSQP
    except ImportError as exc:
        raise ImportError(
            "JAXopt is required for the OSQP solver.  "
            "Install with: pip install jaxopt"
        ) from exc

    n = Q.shape[0]
    # OSQP params_obj = (Q, c)
    # OSQP params_ineq = (G, h) where G @ x <= h
    solver = OSQP(**solver_kwargs)
    sol = solver.run(
        params_obj=(Q, c),
        params_ineq=(G, h),
    )
    return sol.params.primal[0]


def _run_box_osqp_qp(
        Q: jnp.ndarray,
        c: jnp.ndarray,
        xl: jnp.ndarray,
        xu: jnp.ndarray,
        theta0: jnp.ndarray,
        params_ineq: tuple | None = None,
        **solver_kwargs,
) -> jnp.ndarray:
    """
    Solve a box-constrained QP with jaxopt.BoxOSQP.

    Problem:

        min  0.5 x^T Q x + c^T x
        s.t. xl <= x <= xu

    BoxOSQP solves ``l <= C @ x <= u`` and expects
    ``params_ineq = (C, l, u)``.  For box constraints on theta, ``C = I``.

    When *params_ineq* is not None it is passed directly to BoxOSQP (allowing
    custom constraint matrices for ``qp_mode="explicit"``); otherwise the box
    bounds are converted automatically via :func:`_bounds_to_box_osqp_ineq`.

    Args:
        Q: Cost matrix (n, n), PSD.
        c: Linear cost vector (n,).
        xl: Lower bound vector (n,).
        xu: Upper bound vector (n,).
        theta0: Initial guess.
        params_ineq: Optional explicit ``(C, l, u)`` triple.  If provided,
            *xl* and *xu* are ignored for constraint construction.
        **solver_kwargs: Forwarded to ``jaxopt.BoxOSQP()``.

    Returns:
        Primal solution vector (n,).
    """
    try:
        from jaxopt import BoxOSQP
    except ImportError as exc:
        raise ImportError(
            "JAXopt is required for the BoxOSQP solver.  "
            "Install with: pip install jaxopt"
        ) from exc

    if params_ineq is None:
        A_eq, l_ineq, u_ineq = _bounds_to_box_osqp_ineq(xl, xu)
    else:
        if not isinstance(params_ineq, (tuple, list)):
            raise ValueError(
                "params_ineq for box_osqp must be a tuple/list of length 2 or 3."
            )
        if len(params_ineq) == 3:
            A_eq, l_ineq, u_ineq = params_ineq
        elif len(params_ineq) == 2:
            # Backward-compatible path: user provides only (l, u); use identity A.
            A_eq = jnp.eye(xl.shape[0], dtype=xl.dtype)
            l_ineq, u_ineq = params_ineq
        else:
            raise ValueError(
                "params_ineq for box_osqp must be (l, u) or (A, l, u)."
            )

    solver = BoxOSQP(**solver_kwargs)
    sol = solver.run(
        params_obj=(Q, c),
        params_eq=A_eq,
        params_ineq=(l_ineq, u_ineq),
    )
    return sol.params.primal[0]


def _run_eq_qp(
        Q: jnp.ndarray,
        c: jnp.ndarray,
        A_eq: jnp.ndarray | None,
        b_eq: jnp.ndarray | None,
        theta0: jnp.ndarray,
        **solver_kwargs,
) -> jnp.ndarray:
    """
    Solve an equality-constrained QP with jaxopt.EqualityConstrainedQP.

    Problem:

        min  0.5 x^T Q x + c^T x
        s.t. A_eq x = b_eq   (if A_eq is not None)

    If no equality constraints are provided, a dummy zero constraint is used
    so that the solver effectively solves the unconstrained QP.

    Bounds are NOT handled here; apply post-solve clipping after calling.

    Args:
        Q: Cost matrix (n, n), PSD.
        c: Linear cost vector (n,).
        A_eq: Equality constraint matrix (p, n) or None.
        b_eq: Equality right-hand side (p,) or None.
        theta0: Initial guess.
        **solver_kwargs: Forwarded to ``jaxopt.EqualityConstrainedQP()``.

    Returns:
        Primal solution vector (n,).
    """
    try:
        from jaxopt import EqualityConstrainedQP
    except ImportError as exc:
        raise ImportError(
            "JAXopt is required for the EqualityConstrainedQP solver.  "
            "Install with: pip install jaxopt"
        ) from exc

    n = Q.shape[0]
    if A_eq is None or b_eq is None:
        # Dummy equality constraint: 0^T x = 0 (effectively unconstrained)
        A_eq = jnp.zeros((1, n), dtype=Q.dtype)
        b_eq = jnp.zeros((1,), dtype=Q.dtype)

    solver = EqualityConstrainedQP(**solver_kwargs)
    sol = solver.run(
        params_obj=(Q, c),
        params_eq=(A_eq, b_eq),
    )
    return sol.params.primal[0]


def _line_search(
        scalar_loss: Callable,
        theta0: jnp.ndarray,
        theta_qp: jnp.ndarray,
        n_steps: int = 8,
) -> jnp.ndarray:
    """
    Backtracking line search between *theta0* and *theta_qp* on the original
    nonlinear scalar loss.

    Evaluates alpha values (1.0, 0.5, 0.25, …) plus the fallback theta0
    (alpha=0), stacks all candidate losses into a JAX array, finds the argmin
    in JAX to avoid scalar device→host transfers inside the loop, and returns
    the best theta.

    Args:
        scalar_loss: Original nonlinear scalar loss ``f(theta) -> scalar``.
        theta0: Starting point (alpha = 0 fallback).
        theta_qp: QP solution (alpha = 1 full step).
        n_steps: Number of halving steps to evaluate (default 8, covering
            alpha in {1.0, 0.5, …, ~0.004}).

    Returns:
        Best theta_trial (float64 JAX array).
    """
    direction = theta_qp - theta0
    # Build candidate thetas: theta0 (fallback) + n_steps trial steps
    alphas = [0.0] + [1.0 * 0.5 ** i for i in range(n_steps)]
    candidates = [theta0 + a * direction for a in alphas]

    # Evaluate all losses and find the best using JAX operations only
    # (no float() or int() conversions to avoid device→host transfers)
    losses = jnp.array([scalar_loss(t) for t in candidates])
    candidates_stacked = jnp.stack(candidates)  # shape (n_steps+1, n)
    best_idx = jnp.argmin(losses)
    return candidates_stacked[best_idx]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_single_optimisation_jaxopt(
        loss_fn: Callable,
        theta0: np.ndarray,
        xl: np.ndarray,
        xu: np.ndarray,
        *,
        max_steps: int = 500,
        tol: float = 1e-6,
        verbose: bool = False,
        solver_kind: str = "lbfgsb",
        # ProjectedGradient-specific
        stepsize: float = 1e-3,
        acceleration: bool = True,
        # QP-specific
        qp_mode: str = "local_quadratic",
        hessian_mode: str = "hessian",
        ridge: float = 1e-6,
        line_search: bool = True,
        line_search_steps: int = 8,
        params_obj=None,
        params_eq=None,
        params_ineq=None,
) -> tuple[np.ndarray, float, float, float, float, float]:
    """
    Run a single JAXopt bounded optimisation.

    Args:
        loss_fn: ``(theta, args) -> (scalar_loss, (f1, f2, f3, f4))``
            as returned by make_loss_fn() in optimization.py.
        theta0: Initial parameter vector.
        xl: Lower bounds vector (same shape as theta0).
            These come from create_bounds() / problem.xl / problem.xu.
        xu: Upper bounds vector (same shape as theta0).
        max_steps: Maximum iterations (L-BFGS-B and ProjectedGradient only).
        tol: Solver convergence tolerance (gradient norm for L-BFGS-B; step
            norm for ProjectedGradient).
        verbose: If True, emit per-iteration diagnostics via the logger.
        solver_kind: One of:
            - "lbfgsb" – SciPy L-BFGS-B via JAXopt wrapper (recommended).
            - "projected_gradient" – pure-JAX projected GD.
            - "osqp" – local quadratic surrogate via JAXopt OSQP.
            - "box_osqp" – local quadratic surrogate via JAXopt BoxOSQP.
            - "eq_qp" – local quadratic surrogate via JAXopt EqualityConstrainedQP.
        stepsize: Step size for "projected_gradient" only.
        acceleration: Nesterov acceleration for "projected_gradient".
        qp_mode: For QP solvers, either "local_quadratic" (default, build a
            Hessian-based surrogate) or "explicit" (use caller-supplied
            params_obj/params_eq/params_ineq).
        hessian_mode: "hessian" (default, full ``jax.hessian``).  Only
            "hessian" is implemented; reserved for future "gauss_newton".
        ridge: Diagonal regularisation for local quadratic Hessian (default 1e-6).
        line_search: If True (default for QP solvers), run a backtracking line
            search between theta0 and the QP solution on the original nonlinear loss.
        line_search_steps: Number of halving steps in the line search (default 8).
        params_obj: Explicit QP objective tuple ``(Q, c)`` for "explicit" mode.
        params_eq: Explicit equality constraint tuple ``(A, b)`` or None.
        params_ineq: Explicit inequality constraints (format depends on solver).

    Returns:
        Tuple of (theta_opt, total_loss, f1, f2, f3, f4):

        - theta_opt: Best-fit parameter vector (float64, hard-clipped to [xl, xu]).
        - total_loss: f1 + f2 + f3 + f4.
        - f1, f2, f3, f4: Diagnostic loss components (phosphosite / abundance /
            reg / mRNA).

    Raises:
        ImportError: If jaxopt is not installed.
        ValueError: If solver_kind or hessian_mode is not recognised.
        RuntimeError: If the Hessian-based QP surrogate cannot be constructed.
    """
    _VALID_HESSIAN_MODES = {"hessian"}
    if hessian_mode not in _VALID_HESSIAN_MODES:
        raise ValueError(
            f"Unsupported hessian_mode={hessian_mode!r}. "
            f"Only 'hessian' is currently implemented."
        )

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
        #   OptStep.state   – solver state (iter_num, fun_val, …)
        #
        # Note: NOT JIT-able (calls scipy internally).
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

    elif solver_kind in ("osqp", "box_osqp", "eq_qp"):
        # ------------------------------------------------------------------ #
        # QP solvers – local quadratic surrogate of the nonlinear ODE loss.
        #
        # These solvers do NOT globally optimise the PhosCrosstalk ODE model.
        # They solve a Hessian-based local quadratic approximation around
        # theta0, then optionally line-search back on the original loss.
        # ------------------------------------------------------------------ #
        explicit_data = _extract_qp_data_from_kwargs({
            "qp_mode": qp_mode,
            "params_obj": params_obj,
            "params_eq": params_eq,
            "params_ineq": params_ineq,
        })

        if explicit_data is not None:
            Q, c = explicit_data["params_obj"]
            Q = jnp.asarray(Q, dtype=jnp.float64)
            c = jnp.asarray(c, dtype=jnp.float64)
        else:
            # Build local quadratic surrogate
            Q, c = _build_local_quadratic_qp(scalar_loss, theta0_j, ridge=ridge)

        if solver_kind == "osqp":
            if explicit_data is not None and explicit_data.get("params_ineq") is not None:
                G, h = explicit_data["params_ineq"]
                G = jnp.asarray(G, dtype=jnp.float64)
                h = jnp.asarray(h, dtype=jnp.float64)
            else:
                G, h = _bounds_to_osqp_ineq(xl_j, xu_j)
            theta_opt_j = _run_osqp_qp(Q, c, G, h, theta0_j)

        elif solver_kind == "box_osqp":
            # Honor explicit (C, l, u) params_ineq when provided
            explicit_ineq = None
            if explicit_data is not None and explicit_data.get("params_ineq") is not None:
                raw = explicit_data["params_ineq"]
                if not (isinstance(raw, (tuple, list)) and len(raw) == 3):
                    raise ValueError(
                        "For box_osqp qp_mode='explicit', params_ineq must be a "
                        f"(C, l, u) triple; got {type(raw)!r} of length "
                        f"{len(raw) if hasattr(raw, '__len__') else '?'}."
                    )
                C_ineq = jnp.asarray(raw[0], dtype=jnp.float64)
                l_ineq = jnp.asarray(raw[1], dtype=jnp.float64)
                u_ineq = jnp.asarray(raw[2], dtype=jnp.float64)
                explicit_ineq = (C_ineq, l_ineq, u_ineq)
            theta_opt_j = _run_box_osqp_qp(Q, c, xl_j, xu_j, theta0_j,
                                            params_ineq=explicit_ineq)

        else:  # eq_qp
            logger.warning(
                "[jaxopt/eq_qp] EqualityConstrainedQP does not support box "
                "bounds directly.  Bounds will be enforced by post-solve "
                "clipping.  Results may be outside [xl, xu] before clipping."
            )
            if explicit_data is not None and explicit_data.get("params_eq") is not None:
                A_eq, b_eq = explicit_data["params_eq"]
                A_eq = jnp.asarray(A_eq, dtype=jnp.float64)
                b_eq = jnp.asarray(b_eq, dtype=jnp.float64)
            else:
                A_eq, b_eq = None, None
            theta_opt_j = _run_eq_qp(Q, c, A_eq, b_eq, theta0_j)

        # Optionally line-search on original nonlinear loss
        if line_search:
            theta_opt_j = _line_search(
                scalar_loss, theta0_j, theta_opt_j, n_steps=line_search_steps
            )

        if verbose:
            logger.info(
                "[jaxopt/%s] QP solve complete  loss_at_solution=%.4e",
                solver_kind,
                float(scalar_loss(theta_opt_j)),
            )

    else:
        raise ValueError(
            f"Unknown solver_kind {solver_kind!r}. "
            "Choose 'lbfgsb', 'projected_gradient', 'osqp', 'box_osqp', or 'eq_qp'."
        )

    # Hard-clip to bounds for safety (L-BFGS-B may return values at machine eps
    # outside bounds; ProjectedGradient should already be feasible; QP solvers
    # may drift slightly outside for eq_qp or due to line search).
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
