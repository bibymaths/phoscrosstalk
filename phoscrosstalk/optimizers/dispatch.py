# SPDX-License-Identifier: MIT
"""
dispatch.py
Unified dispatcher for all solver backends.

Provides a single dispatch_optimisation entry point that selects among the
available backends by name.  This is the recommended way to switch solvers
in multistart runners or experiment scripts without changing call sites.

Solver key         Module                              Notes
-----------------  ----------------------------------  --------------------------------
"optimistix"       optimization.py                     Canonical LM / Gauss-Newton
"jaxopt_lbfgsb"    optimizers/jaxopt_backend.py        SciPy L-BFGS-B via JAXopt
"jaxopt_pgd"       optimizers/jaxopt_backend.py        Projected GD via JAXopt
"scipy_jax"        optimizers/scipy_jax_backend.py     BFGS + reparameterisation
"scipy_jax_pen"    optimizers/scipy_jax_backend.py     BFGS + quadratic penalty
"optax_adam"       optimizers/optax_backend.py         Adam + projection_box
"optax_sgd"        optimizers/optax_backend.py         SGD  + projection_box
"optax_lbfgs"      optimizers/optax_backend.py         L-BFGS + projection_box
"mpax"             optimizers/mpax_backend.py          SQP with MPAX QP subproblem

All backends share the same return signature::

    (theta_opt: np.ndarray, total_loss: float, f1, f2, f3, f4)

loss_fn must be the scalar-loss closure from make_loss_fn::

    loss_fn(theta, args) -> (scalar_loss, (f1, f2, f3, f4))
"""

from __future__ import annotations

import numpy as np
from typing import Callable

from phoscrosstalk.logger import get_logger

logger = get_logger()


# ---------------------------------------------------------------------------
# Lazy imports (avoid loading heavy deps at module import time)
# ---------------------------------------------------------------------------


def _import_optimistix_runner():
    from phoscrosstalk.optimization import run_single_optimisation
    return run_single_optimisation


def _import_jaxopt_runner():
    from phoscrosstalk.optimizers.jaxopt_backend import run_single_optimisation_jaxopt
    return run_single_optimisation_jaxopt


def _import_scipy_jax_runner():
    from phoscrosstalk.optimizers.scipy_jax_backend import run_single_optimisation_scipy_jax
    return run_single_optimisation_scipy_jax


def _import_optax_runner():
    from phoscrosstalk.optimizers.optax_backend import run_single_optimisation_optax
    return run_single_optimisation_optax


def _import_mpax_runner():
    from phoscrosstalk.optimizers.mpax_backend import run_single_optimisation_mpax
    return run_single_optimisation_mpax


# ---------------------------------------------------------------------------
# Dispatch table
# ---------------------------------------------------------------------------

# Each entry: (import_fn, extra_kwargs_to_inject)
_DISPATCH: dict[str, tuple] = {
    # Existing Optimistix backend — uses residuals_fn, not loss_fn.
    # Pass residuals_fn as loss_fn; the optimistix runner handles has_aux.
    "optimistix": (_import_optimistix_runner, {}),

    # JAXopt backends
    "jaxopt_lbfgsb": (_import_jaxopt_runner, {"solver_kind": "lbfgsb"}),
    "jaxopt_pgd": (_import_jaxopt_runner, {"solver_kind": "projected_gradient"}),

    # jax.scipy.optimize backends
    "scipy_jax": (_import_scipy_jax_runner, {"bounds_strategy": "reparameterize"}),
    "scipy_jax_pen": (_import_scipy_jax_runner, {"bounds_strategy": "penalty"}),

    # Optax backends
    "optax_adam": (_import_optax_runner, {"optimizer_kind": "adam"}),
    "optax_sgd": (_import_optax_runner, {"optimizer_kind": "sgd"}),
    "optax_lbfgs": (_import_optax_runner, {"optimizer_kind": "lbfgs"}),

    # MPAX SQP backend
    "mpax": (_import_mpax_runner, {}),
}

AVAILABLE_BACKENDS: list[str] = list(_DISPATCH.keys())


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def dispatch_optimisation(
        backend: str,
        loss_fn: Callable,
        theta0: np.ndarray,
        xl: np.ndarray,
        xu: np.ndarray,
        **kwargs: object,
) -> tuple[np.ndarray, float, float, float, float, float]:
    """
    Dispatch a single optimisation to the requested solver backend.

    Args:
        backend: One of AVAILABLE_BACKENDS.
        loss_fn: ``(theta, args) -> (scalar_loss, (f1, f2, f3, f4))`` closure
            from make_loss_fn in optimization.py.
        theta0: Starting parameter vector.
        xl: Lower bounds vector (same shape as theta0).
        xu: Upper bounds vector (same shape as theta0).
        **kwargs: Forwarded to the underlying solver.  Backend-specific keys
            (e.g. solver_kind, optimizer_kind) are injected automatically from
            the dispatch table and do not need to be repeated.

    Returns:
        Tuple of (theta_opt, total_loss, f1, f2, f3, f4).

    Raises:
        ValueError: If backend is not in AVAILABLE_BACKENDS.
    """
    if backend not in _DISPATCH:
        raise ValueError(
            f"Unknown backend {backend!r}. "
            f"Available: {AVAILABLE_BACKENDS}"
        )

    import_fn, injected_kwargs = _DISPATCH[backend]
    runner = import_fn()

    # Merge injected kwargs (low priority) with caller-supplied kwargs (high).
    merged = {**injected_kwargs, **kwargs}

    logger.info("dispatch_optimisation: backend=%s  extra_kwargs=%s", backend, merged)

    # The "optimistix" backend has a different signature (no xl/xu positional).
    if backend == "optimistix":
        return runner(loss_fn, theta0, **merged)
    else:
        return runner(loss_fn, theta0, xl, xu, **merged)
