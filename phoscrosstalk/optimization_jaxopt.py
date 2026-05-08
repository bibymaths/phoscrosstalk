# SPDX-License-Identifier: MIT
# Backward-compatibility re-export.  New code should import from
# phoscrosstalk.optimizers.jaxopt_backend directly.
from phoscrosstalk.optimizers.jaxopt_backend import run_single_optimisation_jaxopt

__all__ = ["run_single_optimisation_jaxopt"]
