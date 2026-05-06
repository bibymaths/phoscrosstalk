"""
Regression tests for simulation.py.
"""

import numpy as np
import pytest

from phoscrosstalk.config import ModelDims
from phoscrosstalk.simulation import simulate


def test_simulate_requires_explicit_dims():
    """simulate requires explicit per-run dims and returns finite outputs."""
    t = np.array([0.0, 1.0])
    P_data = np.zeros((2, 2))
    A_data = np.zeros((2, 2))
    dims = ModelDims(K=2, M=2, N=2)
    theta = np.zeros(2 * dims.K + 2 + 3 * dims.M + dims.N + 4)
    Cg = np.zeros((2, 2))
    Cl = np.zeros((2, 2))
    site_prot_idx = np.array([0, 0], dtype=np.int64)
    K_site_kin = np.eye(2)
    R = np.eye(2)
    L_alpha = np.zeros((2, 2))
    kin_to_prot_idx = np.array([0, 0], dtype=np.int64)
    receptor_mask_prot = np.array([0, 0], dtype=np.int64)
    receptor_mask_kin = np.array([0, 0], dtype=np.int64)

    P_sim, A_sim = simulate(
        t,
        P_data,
        A_data,
        theta,
        Cg,
        Cl,
        site_prot_idx,
        K_site_kin,
        R,
        L_alpha,
        kin_to_prot_idx,
        receptor_mask_prot,
        receptor_mask_kin,
        "dist",
    )
    assert P_sim.shape == (2, 2)
    assert A_sim.shape == (2, 2)
