# SPDX-License-Identifier: MIT
"""
MPAX (Mathematical Programming in JAX) QP-subproblem backend.

Strategy — Sequential Quadratic Programming (SQP)
--------------------------------------------------
MPAX solves LP and QP problems of the form::

    min_{l <= x <= u}  0.5 x^T Q x + c^T x
    s.t.               Ax = b,  Gx >= h

This module uses MPAX as a QP subproblem solver inside a Sequential
Quadratic Programming (SQP) outer loop::

    For k = 0, 1, ..., max_sqp_steps:
        1. Compute gradient g = grad(loss)(theta_k) and
                  Hessian H = hessian(loss)(theta_k)  (via jax.hessian).
        2. Regularise: H_reg = H + delta*I  to ensure positive-definiteness.
        3. Solve the local QP:
               min_{d_theta}  0.5 d_theta^T H_reg d_theta + g^T d_theta
               s.t.      max(xl - theta_k, -r) <= d_theta <= min(xu - theta_k, r)
           where r = trust_radius (trust-region constraint on step size).
        4. Update: theta_{k+1} = clip(theta_k + d_theta, xl, xu).

After the QP warm-start, an optional fine-tuning step with
jax.scipy.optimize.minimize (BFGS) can be enabled via finetune=True.

MPAX API (verified, PyPI >= 0.1)::

    from mpax import create_qp, raPDHG

    qp = create_qp(
        Q,    # (n, n) quadratic cost matrix
        c,    # (n,)   linear cost vector
        A,    # (m, n) equality constraint matrix   (0 rows if no eq. constraints)
        b,    # (m,)   equality RHS
        G,    # (p, n) inequality constraint matrix (0 rows if no ineq. constraints)
        h,    # (p,)   inequality RHS  (Gx >= h)
        l,    # (n,)   variable lower bounds
        u,    # (n,)   variable upper bounds
        use_sparse_matrix=False,   # True for scipy sparse input
    )

    solver = raPDHG(eps_abs=1e-4, eps_rel=1e-4, verbose=False)
    result = solver.optimize(qp)
    # result.primal_solution  – (n,) solution vector

Interface (mirrors optimization.py)::

    run_single_optimisation_mpax(loss_fn, theta0, xl, xu, ...)
        -> (theta_opt, total_loss, f1, f2, f3, f4)

loss_fn must have the signature produced by make_loss_fn::

    loss_fn(theta, args) -> (scalar_loss, (f1, f2, f3, f4))

Warning:
    The full Hessian jax.hessian is O(n^2) memory and O(n^3) compute.
    For the default parameter dimension n = 2K + 2 + 3M + N + 4 this is
    tractable for small-to-medium networks.  For large networks, pass
    diagonal_hessian=True to use only the diagonal of the Hessian as a
    cheap positive-definite approximation (essentially scaled gradient descent
    with per-parameter curvature).

Warning:
    This backend requires JAX x64 mode (JAX_ENABLE_X64=true).  Call
    enable_x64() from phoscrosstalk.runtime_env before importing JAX.
"""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
from typing import Callable

from phoscrosstalk.logger import get_logger

logger = get_logger()

# Hessian regularisation added to the diagonal for positive-definiteness.
_HESS_REG: float = 1e-5


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_single_optimisation_mpax(
        loss_fn: Callable,
        theta0: np.ndarray,
        xl: np.ndarray,
        xu: np.ndarray,
        *,
        max_sqp_steps: int = 30,
        eps_abs: float = 1e-4,
        eps_rel: float = 1e-4,
        verbose: bool = False,
        trust_radius: float = 1.0,
        hess_reg: float = _HESS_REG,
        diagonal_hessian: bool = False,
        finetune: bool = False,
        finetune_max_steps: int = 200,
        finetune_gtol: float = 1e-5,
) -> tuple[np.ndarray, float, float, float, float, float]:
    """
    Run SQP optimisation using MPAX as the QP subproblem solver.

    Args:
        loss_fn: ``(theta, args) -> (scalar_loss, (f1, f2, f3, f4))``.
        theta0: Initial parameter vector.
        xl: Lower bounds.  Sourced from problem.xl / problem.xu
            (output of create_bounds()).
        xu: Upper bounds.
        max_sqp_steps: Number of outer SQP iterations.
        eps_abs: MPAX raPDHG absolute tolerance for the inner QP solve.
        eps_rel: MPAX raPDHG relative tolerance for the inner QP solve.
        verbose: If True, emit per-step diagnostics via the logger.
        trust_radius: Maximum step magnitude per SQP iteration (d_theta clipped
            to +/-trust_radius).
        hess_reg: Diagonal regularisation added to the Hessian; ensures PD.
        diagonal_hessian: If True, use only the diagonal of the Hessian
            (memory-efficient for large networks).
        finetune: Run a BFGS fine-tuning step (jax.scipy.optimize.minimize)
            after the SQP loop, with the reparameterised bounds strategy.
        finetune_max_steps: Max BFGS iterations for fine-tuning.
        finetune_gtol: Gradient tolerance for fine-tuning BFGS.

    Returns:
        Tuple of (theta_opt, total_loss, f1, f2, f3, f4):

        - theta_opt: float64 numpy array.
        - total_loss: float.
        - f1, f2, f3, f4: float loss components.

    Raises:
        RuntimeError: If JAX x64 mode is not enabled.
        ImportError: If mpax is not installed.
    """
    if not jax.config.jax_enable_x64:
        raise RuntimeError(
            "MPAX backend requires JAX x64 mode. "
            "Call enable_x64() from phoscrosstalk.runtime_env before importing JAX."
        )

    try:
        from mpax import create_qp, raPDHG
    except ImportError as exc:
        raise ImportError(
            "MPAX is required for this solver.  "
            "Install with: pip install mpax"
        ) from exc

    xl_j = jnp.asarray(xl, dtype=jnp.float64)
    xu_j = jnp.asarray(xu, dtype=jnp.float64)
    theta = jnp.clip(jnp.asarray(theta0, dtype=jnp.float64), xl_j, xu_j)
    n = theta.shape[0]

    # Scalar loss (no aux).
    def scalar_loss(t):
        val, _ = loss_fn(t, None)
        return val

    grad_fn = jax.grad(scalar_loss)
    hess_fn = jax.hessian(scalar_loss) if not diagonal_hessian else None

    # Empty equality/inequality constraint matrices.
    A_eq = jnp.zeros((0, n), dtype=jnp.float64)
    b_eq = jnp.zeros(0, dtype=jnp.float64)
    G_ineq = jnp.zeros((0, n), dtype=jnp.float64)
    h_ineq = jnp.zeros(0, dtype=jnp.float64)

    # ------------------------------------------------------------------ #
    # MPAX raPDHG solver:
    #   raPDHG(eps_abs, eps_rel, verbose) — instantiated once, reused.
    # ------------------------------------------------------------------ #
    solver = raPDHG(eps_abs=eps_abs, eps_rel=eps_rel, verbose=False)

    best_theta = theta
    best_loss = jnp.asarray(jnp.inf, dtype=jnp.float64)

    for step in range(max_sqp_steps):
        g = grad_fn(theta)  # (n,)

        if diagonal_hessian:
            # Cheap diagonal approximation: scale gradient by per-element
            # second-order finite-difference estimate.
            eps_fd = 1e-4
            g_plus = grad_fn(theta + eps_fd)
            diag_H = jnp.clip((g_plus - g) / eps_fd, 0.0, None)
            H_reg = jnp.diag(diag_H + hess_reg)
        else:
            H = hess_fn(theta)  # (n, n)
            H_reg = H + hess_reg * jnp.eye(n, dtype=jnp.float64)

        # Trust-region box in the step space.
        l_qp = jnp.maximum(xl_j - theta, -trust_radius)
        u_qp = jnp.minimum(xu_j - theta, trust_radius)

        # ---------------------------------------------------------------- #
        # create_qp API:
        #   create_qp(Q, c, A, b, G, h, l, u, use_sparse_matrix=False)
        # ---------------------------------------------------------------- #
        qp = create_qp(
            H_reg, g,
            A_eq, b_eq,
            G_ineq, h_ineq,
            l_qp, u_qp,
            use_sparse_matrix=False,
        )
        result = solver.optimize(qp)
        dtheta = result.primal_solution  # (n,)

        theta = jnp.clip(theta + dtheta, xl_j, xu_j)

        loss_val, (f1, f2, f3, f4) = loss_fn(theta, None)
        total = float(loss_val)

        if verbose:
            logger.info(
                "[mpax/SQP] step=%d  loss=%.4e  f1=%.3e f2=%.3e f3=%.3e f4=%.3e",
                step, total, float(f1), float(f2), float(f3), float(f4),
            )

        if total < float(best_loss):
            best_loss = loss_val
            best_theta = theta

    # Optional BFGS fine-tuning with reparameterisation.
    if finetune:
        logger.info("[mpax/SQP] running BFGS fine-tuning ...")
        from phoscrosstalk.optimizers.scipy_jax_backend import run_single_optimisation_scipy_jax
        best_theta_np = np.asarray(best_theta, dtype=np.float64)
        ft_theta, ft_loss, ft_f1, ft_f2, ft_f3, ft_f4 = run_single_optimisation_scipy_jax(
            loss_fn,
            best_theta_np,
            xl, xu,
            max_steps=finetune_max_steps,
            gtol=finetune_gtol,
            verbose=verbose,
            bounds_strategy="reparameterize",
        )
        if ft_loss < float(best_loss):
            logger.info(
                "[mpax/SQP] fine-tuning improved loss %.4e -> %.4e",
                float(best_loss), ft_loss,
            )
            return ft_theta, ft_loss, ft_f1, ft_f2, ft_f3, ft_f4

    theta_opt = np.asarray(best_theta, dtype=np.float64)
    _, (f1, f2, f3, f4) = loss_fn(best_theta, None)
    f1, f2, f3, f4 = float(f1), float(f2), float(f3), float(f4)
    total_loss = f1 + f2 + f3 + f4

    logger.info(
        "[mpax/SQP] final loss=%.4e  f1=%.3e f2=%.3e f3=%.3e f4=%.3e",
        total_loss, f1, f2, f3, f4,
    )
    return theta_opt, total_loss, f1, f2, f3, f4
