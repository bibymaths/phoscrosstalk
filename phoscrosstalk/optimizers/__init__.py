# SPDX-License-Identifier: MIT
"""
Optimizer backends for the phospho-network ODE model.

This subpackage provides a unified dispatch interface and individual backend
modules for running bounded parameter optimisation.

Attributes:
    AVAILABLE_BACKENDS: List of registered backend keys.
"""

from phoscrosstalk.optimizers.dispatch import dispatch_optimisation, AVAILABLE_BACKENDS
from phoscrosstalk.optimizers.jaxopt_backend import run_single_optimisation_jaxopt
from phoscrosstalk.optimizers.scipy_jax_backend import run_single_optimisation_scipy_jax

__all__ = [
    "dispatch_optimisation",
    "AVAILABLE_BACKENDS",
    "run_single_optimisation_jaxopt",
    "run_single_optimisation_scipy_jax",
]
