# SPDX-License-Identifier: MIT
# Backward-compatibility re-export.  New code should import from
# phoscrosstalk.optimizers.scipy_jax_backend directly.
from phoscrosstalk.optimizers.scipy_jax_backend import run_single_optimisation_scipy_jax

__all__ = ["run_single_optimisation_scipy_jax"]
