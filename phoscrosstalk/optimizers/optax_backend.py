# SPDX-License-Identifier: MIT
"""
optax_backend.py
Optax projected gradient / L-BFGS backend for the phospho-network.

Two solver kinds are provided, both enforcing hard box constraints via
optax.projections.projection_box:

"adam" / "sgd"
    First-order gradient descent with Optax optimiser.  After each gradient
    step, parameters are projected back into [xl, xu] using
    optax.projections.projection_box.

    Optax update loop::

        updates, state = optimizer.update(grads, state)
        theta = projection_box(apply_updates(theta, updates), xl, xu)

"lbfgs"
    optax.lbfgs() (quasi-Newton).  L-BFGS requires the value and grad
    at the current point to compute the update, supplied via
    optax.value_and_grad_from_state.  Bounds are applied via
    projection_box after each step exactly as for Adam/SGD.

    L-BFGS update loop (verified API, Optax >= 0.2)::

        value_and_grad = optax.value_and_grad_from_state(scalar_loss)
        value, grad = value_and_grad(theta, state=opt_state)
        updates, opt_state = optimizer.update(
            grad, opt_state, theta,
            value=value, grad=grad, value_fn=scalar_loss,
        )
        theta = projection_box(apply_updates(theta, updates), xl, xu)

Interface (mirrors optimization.py)::

    run_single_optimisation_optax(loss_fn, theta0, xl, xu, ...)
        -> (theta_opt, total_loss, f1, f2, f3, f4)

loss_fn must have the signature produced by make_loss_fn::

    loss_fn(theta, args) -> (scalar_loss, (f1, f2, f3, f4))
"""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
import optax
from optax.projections import projection_box
from typing import Callable

from phoscrosstalk.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_single_optimisation_optax(
    loss_fn: Callable,
    theta0: np.ndarray,
    xl: np.ndarray,
    xu: np.ndarray,
    *,
    max_steps: int = 1000,
    learning_rate: float = 1e-3,
    optimizer_kind: str = "adam",
    verbose: bool = False,
    log_every: int = 100,
    convergence_tol: float = 1e-7,
) -> tuple[np.ndarray, float, float, float, float, float]:
    """
    Run projected gradient optimisation via Optax with hard box constraints.

    Args:
        loss_fn: ``(theta, args) -> (scalar_loss, (f1, f2, f3, f4))``.
        theta0: Initial parameter vector.
        xl: Lower bounds.  Sourced from problem.xl / problem.xu
            (output of create_bounds()).
        xu: Upper bounds.
        max_steps: Maximum gradient steps.
        learning_rate: Step size for "adam" / "sgd"; ignored for "lbfgs"
            (L-BFGS uses a line-search-controlled step).
        optimizer_kind: Optax optimiser to use; one of "adam", "sgd", "lbfgs".
        verbose: If True, emit per-step diagnostics via the logger.
        log_every: Log diagnostics every log_every steps (only when verbose=True).
        convergence_tol: Stop early if the absolute change in loss falls below
            this threshold.

    Returns:
        Tuple of (theta_opt, total_loss, f1, f2, f3, f4):

        - theta_opt: float64 numpy array.
        - total_loss: float.
        - f1, f2, f3, f4: float loss components.

    Raises:
        ValueError: If optimizer_kind is not recognised.
    """
    xl_j = jnp.asarray(xl, dtype=jnp.float64)
    xu_j = jnp.asarray(xu, dtype=jnp.float64)
    theta = jnp.asarray(theta0, dtype=jnp.float64)
    # Guarantee feasible start.
    theta = projection_box(theta, xl_j, xu_j)

    # ----------------------------------------------------------------------- #
    # Build the Optax optimiser.
    #
    # optax.lbfgs() API (>= 0.2):
    #   optimizer.update(grad, state, params, value=..., grad=..., value_fn=...)
    #   Requires value and grad at the current point; use
    #   optax.value_and_grad_from_state to avoid recomputing the function.
    # ----------------------------------------------------------------------- #
    if optimizer_kind == "adam":
        optimizer = optax.adam(learning_rate)
    elif optimizer_kind == "sgd":
        optimizer = optax.sgd(learning_rate, momentum=0.9)
    elif optimizer_kind == "lbfgs":
        optimizer = optax.lbfgs()
    else:
        raise ValueError(
            f"Unknown optimizer_kind {optimizer_kind!r}. "
            "Choose 'adam', 'sgd', or 'lbfgs'."
        )

    opt_state = optimizer.init(theta)

    # Scalar loss (no aux) — used for L-BFGS value_fn.
    def scalar_loss(t):
        val, _ = loss_fn(t, None)
        return val

    # Scalar loss with aux — used for Adam/SGD to get diagnostics cheaply.
    def scalar_loss_with_aux(t):
        return loss_fn(t, None)  # returns (scalar, (f1,f2,f3,f4))

    # For L-BFGS we use optax.value_and_grad_from_state to read the cached
    # value from the optimiser state rather than recomputing it each step.
    if optimizer_kind == "lbfgs":
        value_and_grad_fn = optax.value_and_grad_from_state(scalar_loss)

    prev_loss = jnp.asarray(jnp.inf, dtype=jnp.float64)
    best_theta = theta
    best_loss = jnp.asarray(jnp.inf, dtype=jnp.float64)

    for step in range(max_steps):
        if optimizer_kind == "lbfgs":
            # --------------------------------------------------------------- #
            # L-BFGS update loop (verified Optax >= 0.2 API):
            #
            #   value_and_grad_from_state(scalar_loss)(params, state=opt_state)
            #     -> (value, grad)   [reads cached value from state if available]
            #
            #   optimizer.update(grad, state, params,
            #                    value=value, grad=grad, value_fn=scalar_loss)
            #     -> (updates, new_state)
            # --------------------------------------------------------------- #
            value, grad = value_and_grad_fn(theta, state=opt_state)
            updates, opt_state = optimizer.update(
                grad, opt_state, theta,
                value=value, grad=grad, value_fn=scalar_loss,
            )
            loss_val = value
            # Diagnostics from a cheap aux call at current theta.
            _, (f1, f2, f3, f4) = scalar_loss_with_aux(theta)
        else:
            # --------------------------------------------------------------- #
            # Adam / SGD update loop:
            #
            #   jax.value_and_grad(scalar_loss_with_aux, has_aux=True)(theta)
            #     -> ((scalar, aux), grad)
            #
            #   optimizer.update(grad, state)
            #     -> (updates, new_state)
            # --------------------------------------------------------------- #
            (loss_val, (f1, f2, f3, f4)), grad = jax.value_and_grad(
                scalar_loss_with_aux, has_aux=True
            )(theta)
            updates, opt_state = optimizer.update(grad, opt_state)

        # Apply gradient update then project onto the box.
        theta = projection_box(optax.apply_updates(theta, updates), xl_j, xu_j)

        total = float(loss_val)
        if total < float(best_loss):
            best_loss = loss_val
            best_theta = theta

        if verbose and step % log_every == 0:
            logger.info(
                "[optax/%s] step=%d  loss=%.4e  f1=%.3e f2=%.3e f3=%.3e f4=%.3e",
                optimizer_kind, step, total,
                float(f1), float(f2), float(f3), float(f4),
            )

        if abs(float(prev_loss) - total) < convergence_tol:
            logger.info("[optax/%s] converged at step %d", optimizer_kind, step)
            break
        prev_loss = loss_val

    theta_opt = np.asarray(best_theta, dtype=np.float64)

    # Final diagnostics at the best point found.
    _, (f1, f2, f3, f4) = loss_fn(best_theta, None)
    f1, f2, f3, f4 = float(f1), float(f2), float(f3), float(f4)
    total_loss = f1 + f2 + f3 + f4

    logger.info(
        "[optax/%s] final loss=%.4e  f1=%.3e f2=%.3e f3=%.3e f4=%.3e",
        optimizer_kind, total_loss, f1, f2, f3, f4,
    )
    return theta_opt, total_loss, f1, f2, f3, f4
