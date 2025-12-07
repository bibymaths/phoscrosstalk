"""
simulation.py
Wrapper for scipy.integrate.odeint to simulate the network.
"""
import numpy as np
from scipy.integrate import odeint
from phoscrosstalk.core_mechanisms import network_rhs
from phoscrosstalk.config import ModelDims


def simulate_p_scipy(t_arr, P_data0, A_data0, theta,
                     Cg, Cl, site_prot_idx,
                     K_site_kin, R,
                     L_alpha, kin_to_prot_idx,
                     receptor_mask_prot, receptor_mask_kin,
                     mechanism: str):
    K, M, N = ModelDims.K, ModelDims.M, ModelDims.N
    state_dim = 2 * K + M + N

    x0 = np.zeros((state_dim,))
    # Initial Conditions
    # A = A_data0[:, 0] or default
    a0 = A_data0[:, 0].astype(np.float64)
    a0 = np.nan_to_num(a0, nan=1.0, posinf=5.0, neginf=0.0)
    a0 = np.clip(a0, 0.0, 5.0)
    x0[K:2 * K] = a0

    # P = P_data0[:, 0] or default
    p0 = P_data0[:, 0].astype(np.float64)
    p0 = np.nan_to_num(p0, nan=0.0, posinf=1.0, neginf=0.0)
    p0 = np.clip(p0, 0.0, 1.0)
    x0[2 * K + M:] = p0

    if not np.all(np.isfinite(x0)):
        N_sites = P_data0.shape[0]
        T = len(t_arr)
        return np.full((N_sites, T), np.nan), np.full((K, T), np.nan)

    # Simulation
    xs = odeint(
        network_rhs,
        x0,
        t_arr,
        args=(theta, Cg, Cl, site_prot_idx,
              K_site_kin, R, L_alpha, kin_to_prot_idx,
              receptor_mask_prot, receptor_mask_kin,
              mechanism),
    )

    np.clip(xs, 1e-6, None, out=xs)
    A_sim = xs[:, K:2 * K].T
    P_sim = xs[:, 2 * K + M:].T

    return P_sim, A_sim


def build_full_A0(K, T, A_scaled, prot_idx_for_A):
    """
    Build a (K, T) abundance matrix from subset A_scaled (K_obs, T)
    using mapping prot_idx_for_A (len = K_obs).

    Unobserved proteins get zeros.
    """
    A0_full = np.zeros((K, T), dtype=float)

    if A_scaled.size > 0:
        for k, p_idx in enumerate(prot_idx_for_A):
            # copy the whole time-course; simulate_p_scipy only uses [:, 0]
            A0_full[p_idx, :] = A_scaled[k, :]

    return A0_full
