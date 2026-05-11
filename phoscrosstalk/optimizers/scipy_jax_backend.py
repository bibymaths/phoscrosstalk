# SPDX-License-Identifier: MIT
"""
jax.scipy.optimize.minimize backend for the phospho-network.

jax.scipy.optimize.minimize only supports unconstrained BFGS (the only
method currently implemented in JAX).

"reparameterize" (default, recommended)
    theta = xl + (xu - xl) * sigmoid(phi).
    Optimise over the unconstrained phi instead of theta directly.
    The mapping is bijective and smooth; the Jacobian is well-conditioned away
    from the corners.  phi0 is initialised from theta0 via the inverse logit.
    This strategy preserves full end-to-end differentiability of the solution
    w.r.t. xl / xu.

Note on jax.scipy.optimize.minimize:
    API (as of JAX 0.4.x)::

        jax.scipy.optimize.minimize(
            fun,          # f(x) -> scalar
            x0,           # initial point, 1-D array
            args=(),      # extra positional args passed to fun
            method=None,  # only "BFGS" supported
            tol=None,     # gradient norm tolerance
            options=dict, # {"maxiter": int, "gtol": float, "norm": float}
        ) -> OptimizeResults
            .x            – solution
            .success      – bool
            .status       – int (0 = success)
            .fun          – final function value
            .nit          – number of iterations

    The function must be JIT-able (no Python side-effects inside fun).
"""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.optimize import minimize as jax_minimize
from typing import Callable

from phoscrosstalk.logger import get_logger

logger = get_logger()

# Coefficient for the quadratic out-of-bounds penalty ("penalty" strategy).
_PENALTY_COEFF: float = 1e4


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_single_optimisation_scipy_jax(
        loss_fn: Callable,
        theta0: np.ndarray,
        xl: np.ndarray,
        xu: np.ndarray,
        *,
        max_steps: int = 500,
        gtol: float = 1e-5,
        verbose: bool = False,
        bounds_strategy: str = "reparameterize",
) -> tuple[np.ndarray, float, float, float, float, float]:
    """
    Run BFGS via jax.scipy.optimize.minimize with bounds enforced.

    Args:
        loss_fn: ``(theta, args) -> (scalar_loss, (f1, f2, f3, f4))``.
        theta0: Initial parameter vector.
        xl: Lower bounds.  Sourced from problem.xl / problem.xu
            (output of create_bounds()).
        xu: Upper bounds.
        max_steps: Maximum BFGS iterations (options["maxiter"]).
        gtol: Gradient norm convergence tolerance (options["gtol"]).
        verbose: If True, log result metadata after optimisation.
        bounds_strategy: One of "reparameterize" (sigmoid mapping, recommended)
            or "penalty" (quadratic penalty; see module docstring).

    Returns:
        Tuple of (theta_opt, total_loss, f1, f2, f3, f4):

        - theta_opt: float64 numpy array.
        - total_loss: float.
        - f1, f2, f3, f4: float loss components.

    Raises:
        ValueError: If bounds_strategy is not recognised.
    """
    xl_j = jnp.asarray(xl, dtype=jnp.float64)
    xu_j = jnp.asarray(xu, dtype=jnp.float64)
    theta0_j = jnp.asarray(theta0, dtype=jnp.float64)

    options = {"maxiter": max_steps, "gtol": gtol}

    if bounds_strategy == "reparameterize":
        # ------------------------------------------------------------------ #
        # theta = xl + (xu - xl) * sigmoid(phi)
        # phi   = logit((theta - xl) / (xu - xl))   [inverse map]
        #
        # Clip ratio away from {0, 1} to avoid logit = +/-inf at the boundary.
        # ------------------------------------------------------------------ #
        def phi_to_theta(phi):
            return xl_j + (xu_j - xl_j) * jax.nn.sigmoid(phi)

        ratio = jnp.clip(
            (theta0_j - xl_j) / (xu_j - xl_j + 1e-12),
            1e-6, 1.0 - 1e-6,
        )
        phi0 = jnp.log(ratio / (1.0 - ratio))

        def unconstrained_loss(phi):
            val, _ = loss_fn(phi_to_theta(phi), None)
            return val

        result = jax_minimize(
            unconstrained_loss,
            phi0,
            method="BFGS",
            options=options,
        )
        theta_opt_j = phi_to_theta(result.x)
    else:
        raise ValueError(
            f"Unknown bounds_strategy {bounds_strategy!r}. "
        )

    if verbose:
        logger.info(
            "[scipy_jax/BFGS/%s] success=%s  nit=%d  fun=%.4e",
            bounds_strategy,
            bool(result.success),
            int(result.nit),
            float(result.fun),
        )

    theta_opt = np.asarray(theta_opt_j, dtype=np.float64)

    # Recompute diagnostics at the solution.
    _, (f1, f2, f3, f4) = loss_fn(theta_opt_j, None)
    f1, f2, f3, f4 = float(f1), float(f2), float(f3), float(f4)
    total_loss = f1 + f2 + f3 + f4

    logger.info(
        "[scipy_jax/BFGS/%s] final loss=%.4e  f1=%.3e f2=%.3e f3=%.3e f4=%.3e",
        bounds_strategy, total_loss, f1, f2, f3, f4,
    )
    return theta_opt, total_loss, f1, f2, f3, f4
