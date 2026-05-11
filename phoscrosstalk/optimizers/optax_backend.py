# SPDX-License-Identifier: MIT
"""
Optax projected-gradient / L-BFGS backend for the phospho-network.

Supported optimiser kinds
-------------------------
"adam"
    Adam with hard box projection after every update.

"sgd"
    Momentum SGD with hard box projection after every update.

"lbfgs"
    Optax L-BFGS with optional zoom line search and hard box projection after
    every update.

Important note on bounds
------------------------
For all Optax modes, bounds are enforced by post-update projection:

    theta = projection_box(theta + update, xl, xu)

This guarantees feasibility, but for L-BFGS it is not mathematically identical
to native L-BFGS-B. Projection can distort the quasi-Newton curvature history
when many parameters hit bounds. For strict bounded quasi-Newton optimisation,
jaxopt_lbfgsb remains the cleaner backend.

This backend is still useful because it is fully JAX/Optax based and supports
modern adaptive optimisers and Optax L-BFGS line-search machinery.
"""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
import optax
from optax.projections import projection_box

from phoscrosstalk.logger import get_logger

logger = get_logger()


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _as_optional_float(value):
    """
    Convert config values to float or None.

    This keeps compatibility with TOML/config loaders that may pass "none",
    "null", "", or None.
    """
    if value is None:
        return None

    if isinstance(value, str):
        if value.strip().lower() in {"", "none", "null"}:
            return None

    return float(value)


def _make_lbfgs_linesearch(kind: str | None, max_linesearch_steps: int):
    if kind is None:
        return None

    kind_norm = str(kind).strip().lower()

    if kind_norm in {"none", "null", "false", "off", ""}:
        return None

    if kind_norm in {"zoom", "default", "auto"}:
        return optax.scale_by_zoom_linesearch(
            max_linesearch_steps=int(max_linesearch_steps)
        )

    raise ValueError(
        f"Unknown lbfgs_linesearch={kind!r}. "
        "Use 'zoom' or 'none'."
    )

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
        learning_rate: float | None = 1e-3,
        optimizer_kind: str = "adam",
        verbose: bool = False,
        log_every: int = 100,
        convergence_tol: float = 1e-7,
        lbfgs_memory_size: int = 20,
        lbfgs_scale_init_precond: bool = True,
        lbfgs_linesearch: str | None = "zoom",
        lbfgs_max_linesearch_steps: int = 20,
) -> tuple[np.ndarray, float, float, float, float, float]:
    """
    Run projected Optax optimisation with hard box constraints.

    Args:
        loss_fn:
            Callable with signature ``loss_fn(theta, args) ->
            (scalar_loss, (f1, f2, f3, f4))``.

        theta0:
            Initial parameter vector.

        xl:
            Lower bounds, same shape as theta0.

        xu:
            Upper bounds, same shape as theta0.

        max_steps:
            Maximum optimisation steps.

        learning_rate:
            Step size for Adam/SGD. For L-BFGS, this is an optional global
            scaling factor. When using line search, 1.0 is usually appropriate.
            None is allowed for L-BFGS but not for Adam/SGD.

        optimizer_kind:
            One of "adam", "sgd", or "lbfgs".

        verbose:
            Emit progress logs.

        log_every:
            Log every N steps when verbose=True.

        convergence_tol:
            Stop when absolute loss improvement between consecutive accepted
            iterates is below this value.

        lbfgs_memory_size:
            Number of past parameter/gradient differences retained by L-BFGS.

        lbfgs_scale_init_precond:
            Whether to scale the initial L-BFGS inverse-Hessian preconditioner.

        lbfgs_linesearch:
            "zoom" to use Optax zoom line search, or "none" to disable line
            search.

        lbfgs_max_linesearch_steps:
            Maximum number of line search steps to take before giving up.

    Returns:
        Tuple:
            theta_opt, total_loss, f1, f2, f3, f4

    Raises:
        ValueError:
            On invalid optimizer_kind or incompatible options.
    """
    optimizer_kind = str(optimizer_kind).strip().lower()

    if optimizer_kind not in {"adam", "sgd", "lbfgs"}:
        raise ValueError(
            f"Unknown optimizer_kind {optimizer_kind!r}. "
            "Choose 'adam', 'sgd', or 'lbfgs'."
        )

    if max_steps <= 0:
        raise ValueError(f"max_steps must be > 0. Got {max_steps}.")

    if log_every <= 0:
        raise ValueError(f"log_every must be > 0. Got {log_every}.")

    if convergence_tol < 0:
        raise ValueError(
            f"convergence_tol must be >= 0. Got {convergence_tol}."
        )

    xl_j = jnp.asarray(xl, dtype=jnp.float64)
    xu_j = jnp.asarray(xu, dtype=jnp.float64)
    theta = jnp.asarray(theta0, dtype=jnp.float64)

    if xl_j.shape != theta.shape or xu_j.shape != theta.shape:
        raise ValueError(
            "theta0, xl, and xu must have identical shapes. "
            f"Got theta0={theta.shape}, xl={xl_j.shape}, xu={xu_j.shape}."
        )

    if bool(jnp.any(xu_j <= xl_j)):
        raise ValueError("All upper bounds must be greater than lower bounds.")

    # Guarantee feasible start.
    theta = projection_box(theta, xl_j, xu_j)

    lr = _as_optional_float(learning_rate)

    # ------------------------------------------------------------------ #
    # Scalar loss wrappers                                                #
    # ------------------------------------------------------------------ #
    def scalar_loss(params):
        val, _aux = loss_fn(params, None)
        return val

    def scalar_loss_with_aux(params):
        return loss_fn(params, None)

    # ------------------------------------------------------------------ #
    # Build optimiser                                                     #
    # ------------------------------------------------------------------ #
    if optimizer_kind == "adam":
        if lr is None:
            raise ValueError("learning_rate must be provided for optax.adam.")
        optimizer = optax.adam(lr)

    elif optimizer_kind == "sgd":
        if lr is None:
            raise ValueError("learning_rate must be provided for optax.sgd.")
        optimizer = optax.sgd(lr, momentum=0.9)

    else:  # optimizer_kind == "lbfgs"
        if lbfgs_memory_size <= 0:
            raise ValueError(
                f"lbfgs_memory_size must be > 0. Got {lbfgs_memory_size}."
            )

        linesearch = _make_lbfgs_linesearch(
            lbfgs_linesearch,
            lbfgs_max_linesearch_steps,
        )

        optimizer = optax.lbfgs(
            learning_rate=lr,
            memory_size=int(lbfgs_memory_size),
            scale_init_precond=bool(lbfgs_scale_init_precond),
            linesearch=linesearch,
        )

    opt_state = optimizer.init(theta)

    # ------------------------------------------------------------------ #
    # JIT derivative/value functions                                      #
    # ------------------------------------------------------------------ #
    # Adam/SGD use value_and_grad with aux.
    value_and_grad_with_aux_fn = jax.jit(
        jax.value_and_grad(scalar_loss_with_aux, has_aux=True)
    )

    # L-BFGS uses Optax's state-aware value/grad helper. Do not wrap this
    # helper blindly with jax.jit because it reads optimiser-state caches.
    if optimizer_kind == "lbfgs":
        lbfgs_value_and_grad_fn = optax.value_and_grad_from_state(scalar_loss)

    # Aux diagnostics after the projected update.
    scalar_loss_with_aux_jit = jax.jit(scalar_loss_with_aux)

    # ------------------------------------------------------------------ #
    # Optimisation loop                                                   #
    # ------------------------------------------------------------------ #
    prev_loss = jnp.asarray(jnp.inf, dtype=jnp.float64)
    best_theta = theta
    best_loss = jnp.asarray(jnp.inf, dtype=jnp.float64)

    for step in range(int(max_steps)):
        if optimizer_kind == "lbfgs":
            value, grad = lbfgs_value_and_grad_fn(theta, state=opt_state)

            updates, opt_state = optimizer.update(
                grad,
                opt_state,
                theta,
                value=value,
                grad=grad,
                value_fn=scalar_loss,
            )

        else:
            (value, _aux), grad = value_and_grad_with_aux_fn(theta)
            updates, opt_state = optimizer.update(grad, opt_state)

        # Hard projection after every update.
        theta_new = projection_box(
            optax.apply_updates(theta, updates),
            xl_j,
            xu_j,
        )

        # Evaluate accepted/projected iterate. This is important for L-BFGS
        # because the line search sees the unprojected candidate, while the
        # actual iterate is projected.
        loss_new, (f1, f2, f3, f4) = scalar_loss_with_aux_jit(theta_new)

        if not bool(jnp.isfinite(loss_new)):
            logger.warning(
                "[optax/%s] step=%d produced non-finite loss after projection; "
                "stopping and returning best feasible theta.",
                optimizer_kind,
                step,
            )
            break

        theta = theta_new
        total = float(loss_new)

        if total < float(best_loss):
            best_loss = loss_new
            best_theta = theta

        if verbose and step % int(log_every) == 0:
            logger.info(
                "[optax/%s] step=%d  loss=%.4e  f1=%.3e f2=%.3e f3=%.3e f4=%.3e",
                optimizer_kind,
                step,
                total,
                float(f1),
                float(f2),
                float(f3),
                float(f4),
            )

        if abs(float(prev_loss) - total) < float(convergence_tol):
            logger.info(
                "[optax/%s] converged at step %d; |Δloss|=%.3e < %.3e",
                optimizer_kind,
                step,
                abs(float(prev_loss) - total),
                float(convergence_tol),
            )
            break

        prev_loss = loss_new

    # ------------------------------------------------------------------ #
    # Final diagnostics at best point                                     #
    # ------------------------------------------------------------------ #
    theta_opt = np.asarray(best_theta, dtype=np.float64)

    _, (f1, f2, f3, f4) = loss_fn(best_theta, None)
    f1 = float(f1)
    f2 = float(f2)
    f3 = float(f3)
    f4 = float(f4)
    total_loss = f1 + f2 + f3 + f4

    logger.info(
        "[optax/%s] final loss=%.4e  f1=%.3e f2=%.3e f3=%.3e f4=%.3e",
        optimizer_kind,
        total_loss,
        f1,
        f2,
        f3,
        f4,
    )

    return theta_opt, total_loss, f1, f2, f3, f4