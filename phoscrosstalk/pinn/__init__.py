# SPDX-License-Identifier: MIT
"""
phoscrosstalk.pinn
==================
PINN / Universal ODE subpackage for PhosCrosstalk.

Implements an additive neural augmentation to the mechanistic ODE RHS::

    dx/dt = f_mechanistic(x, t; θ) + f_pinn(x, t; φ)

The mechanistic model remains the primary model.  The neural term is a residual
correction for missing regulatory edges, unmeasured regulators, and topology
limitations.  It is regularised so it cannot dominate the mechanistic RHS.

Public API
----------
run_pinn_pipeline(...)
    Main PINN execution entry point.  Called by main.py when pinn.enabled = true.

PINNAugmentation
    Equinox module implementing the neural correction term.

make_combined_rhs(...)
    Construct the augmented ODE RHS (mechanistic + PINN).
"""

from phoscrosstalk.pinn.model import PINNAugmentation
from phoscrosstalk.pinn.outputs import load_pinn_model_bundle, save_pinn_model_bundle
from phoscrosstalk.pinn.rhs import make_combined_rhs
from phoscrosstalk.pinn.runner import run_pinn_pipeline

__all__ = [
    "PINNAugmentation",
    "make_combined_rhs",
    "run_pinn_pipeline",
    "save_pinn_model_bundle",
    "load_pinn_model_bundle",
]
