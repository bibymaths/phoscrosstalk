# SPDX-License-Identifier: MIT
# Backward-compatibility re-export.  New code should import from
# phoscrosstalk.optimizers.dispatch directly.
from phoscrosstalk.optimizers.dispatch import dispatch_optimisation, AVAILABLE_BACKENDS

__all__ = ["dispatch_optimisation", "AVAILABLE_BACKENDS"]
