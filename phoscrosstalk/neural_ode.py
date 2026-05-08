"""
neural_ode.py
PEP-8 alias for phoscrosstalk.neuralODE – re-exports the public API unchanged.
"""

from phoscrosstalk.neuralODE import (  # noqa: F401
    LatentRateMLP,
    NeuralRateGenerator,
    JointNeuralMechanisticModel,
    run_neural_latent_rate_refinement,
    save_neural_ode_plots,
)

__all__ = [
    "LatentRateMLP",
    "NeuralRateGenerator",
    "JointNeuralMechanisticModel",
    "run_neural_latent_rate_refinement",
    "save_neural_ode_plots",
]
