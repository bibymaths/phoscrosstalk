"""
config.py
Global configuration and state management for the Phospho-Network Model.
"""
import numpy as np


# Global dimensions container
class ModelDims:
    K: int = None  # Number of Proteins
    M: int = None  # Number of Kinases
    N: int = None  # Number of Phosphosites

    @classmethod
    def set_dims(cls, k, m, n):
        cls.K = k
        cls.M = m
        cls.N = n


DEFAULT_TIMEPOINTS = np.array(
    [0.0, 0.5, 0.75, 1.0, 2.0, 4.0, 8.0, 16.0,
     30.0, 60.0, 120.0, 240.0, 480.0, 960.0]
)

EPS = 1e-8
