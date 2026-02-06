"""
config.py
Global configuration and state management for the Phospho-Network Model.
"""
import numpy as np


# Global dimensions container
class ModelDims:
    """
    Global container for storing model dimensions (Proteins, Kinases, Sites).

    Acts as a static state holder to avoid passing dimensions recursively
    through every function in the simulation pipeline.
    """
    K: int = None  # Number of Proteins
    M: int = None  # Number of Kinases
    N: int = None  # Number of Phosphosites

    @classmethod
    def set_dims(cls, k, m, n):
        """
        Set the global dimensions for the current model context.

        Args:
            k (int): Number of unique proteins (K).
            m (int): Number of kinases (M).
            n (int): Number of phosphorylation sites (N).

        Returns:
            None
        """
        cls.K = k
        cls.M = m
        cls.N = n


DEFAULT_TIMEPOINTS = np.array(
    [0.0, 0.5, 0.75, 1.0, 2.0, 4.0, 8.0, 16.0,
     30.0, 60.0, 120.0, 240.0, 480.0, 960.0]
)

EPS = 1e-8
