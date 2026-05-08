# SPDX-License-Identifier: MIT
# Backward-compatibility re-export.  New code should import from
# phoscrosstalk.optimizers.optax_backend directly.
from phoscrosstalk.optimizers.optax_backend import run_single_optimisation_optax

__all__ = ["run_single_optimisation_optax"]
