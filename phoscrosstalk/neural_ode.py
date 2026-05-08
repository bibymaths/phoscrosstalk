"""
phoscrosstalk/neural_ode.py
---------------------------
Backward-compatibility alias for ``phoscrosstalk.neuralODE``.

The canonical module name used by the test suite is ``phoscrosstalk.neural_ode``
(PEP-8 style).  This thin re-export keeps the original ``neuralODE.py``
unchanged while exposing the same public symbols under the expected dotted name.

Importing from either name is equivalent:

    from phoscrosstalk.neural_ode import run_neural_latent_rate_refinement
    from phoscrosstalk.neuralODE import run_neural_latent_rate_refinement
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
