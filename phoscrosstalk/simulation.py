"""
simulation.py
Wrapper for scipy.integrate.odeint to simulate the network.
"""
import warnings

import numpy as np
from numba import njit
from scipy.integrate import odeint, ODEintWarning

from phoscrosstalk.core_mechanisms import network_rhs, rhs_nb_dispatch
from phoscrosstalk.config import ModelDims

warnings.filterwarnings("ignore", category=ODEintWarning)
warnings.filterwarnings("ignore", message="Excess work done on this call")
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

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
        Dfun=fd_jacobian,
        col_deriv=False,
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

def fd_jacobian(
    x,
    t,
    theta,
    Cg, Cl,
    site_prot_idx,
    K_site_kin, R,
    L_alpha,
    kin_to_prot_idx,
    receptor_mask_prot,
    receptor_mask_kin,
    mechanism: str,
):
    """
    Python wrapper for fd_jacobian_nb_core so SciPy can call it.

    This keeps the SciPy signature compatible with `network_rhs`.
    """

    # Map mechanism string → small int for numba
    if mechanism == "dist":
        mech_code = 0
    elif mechanism == "seq":
        mech_code = 1
    elif mechanism == "rand":
        mech_code = 2
    else:  # fallback
        mech_code = 0

    # Ensure x is a contiguous float64 array
    x_arr = np.asarray(x, dtype=np.float64)

    return fd_jacobian_nb_core(
        x_arr,
        t,
        theta,
        Cg, Cl,
        site_prot_idx,
        K_site_kin, R,
        L_alpha,
        kin_to_prot_idx,
        receptor_mask_prot,
        receptor_mask_kin,
        mech_code,
        ModelDims.K,
        ModelDims.M,
        ModelDims.N,
    )

@njit(cache=True, fastmath=True)
def fd_jacobian_nb_core(
    x,
    t,
    theta,
    Cg, Cl,
    site_prot_idx,
    K_site_kin, R,
    L_alpha,
    kin_to_prot_idx,
    receptor_mask_prot,
    receptor_mask_kin,
    mech_code,
    K, M, N,
    eps=1e-8
):
    """
    Finite-difference Jacobian of the numba RHS w.r.t. x.

    J[i, j] = d f_i / d x_j  (forward difference)
    """

    n = x.size
    J = np.empty((n, n), dtype=np.float64)

    # f(x)
    f0 = rhs_nb_dispatch(
        x, t, theta,
        Cg, Cl, site_prot_idx,
        K_site_kin, R, L_alpha,
        kin_to_prot_idx,
        receptor_mask_prot,
        receptor_mask_kin,
        K, M, N,
        mech_code,
    )

    # forward difference on each column j
    for j in range(n):
        x_pert = x.copy()
        # scale step by magnitude of x_j
        h = eps * max(1.0, abs(x[j]))
        x_pert[j] += h

        fj = rhs_nb_dispatch(
            x_pert, t, theta,
            Cg, Cl, site_prot_idx,
            K_site_kin, R, L_alpha,
            kin_to_prot_idx,
            receptor_mask_prot,
            receptor_mask_kin,
            K, M, N,
            mech_code,
        )

        # (f(x + h e_j) - f(x)) / h  → column j
        for i in range(n):
            J[i, j] = (fj[i] - f0[i]) / h

    return J