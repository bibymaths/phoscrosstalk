# SPDX-License-Identifier: MIT
"""
optimizers
Optimizer backends for the phospho-network ODE model.

This subpackage provides a unified dispatch interface and individual backend
modules for running bounded parameter optimisation.

Attributes:
    AVAILABLE_BACKENDS: List of registered backend keys.

Example:
    from phoscrosstalk.optimizers import dispatch_optimisation, AVAILABLE_BACKENDS

    theta_opt, loss, f1, f2, f3, f4 = dispatch_optimisation(
        "optax_adam", loss_fn, theta0, xl, xu, max_steps=500
    )
"""

from phoscrosstalk.optimizers.dispatch import dispatch_optimisation, AVAILABLE_BACKENDS
from phoscrosstalk.optimizers.jaxopt_backend import run_single_optimisation_jaxopt
from phoscrosstalk.optimizers.optax_backend import run_single_optimisation_optax
from phoscrosstalk.optimizers.scipy_jax_backend import run_single_optimisation_scipy_jax
from phoscrosstalk.optimizers.mpax_backend import run_single_optimisation_mpax

__all__ = [
    "dispatch_optimisation",
    "AVAILABLE_BACKENDS",
    "run_single_optimisation_jaxopt",
    "run_single_optimisation_optax",
    "run_single_optimisation_scipy_jax",
    "run_single_optimisation_mpax",
]
