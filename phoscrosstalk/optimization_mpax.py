# SPDX-License-Identifier: MIT
# Backward-compatibility re-export.  New code should import from
# phoscrosstalk.optimizers.mpax_backend directly.
from phoscrosstalk.optimizers.mpax_backend import run_single_optimisation_mpax

__all__ = ["run_single_optimisation_mpax"]
