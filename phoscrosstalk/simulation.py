"""
simulation.py
Wrapper for scipy.integrate.odeint to simulate the network.
"""
import math
import warnings
from typing import cast

import numpy as np
from numba import njit
from scipy.integrate import odeint, ODEintWarning

from phoscrosstalk.core_mechanisms import network_rhs, rhs_nb_dispatch_dense
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
    xs = cast(np.ndarray, cast(object, odeint(
        network_rhs,
        x0,
        t_arr,
        args=(theta, Cg, Cl, site_prot_idx,
              K_site_kin, R, L_alpha, kin_to_prot_idx,
              receptor_mask_prot, receptor_mask_kin,
              mechanism),
        # Dfun=fd_jacobian,
        col_deriv=False,
        rtol=1e-6,
        atol=1e-9,
        mxstep=5000,
        # full_output=True,
    )))

    # if solver failed, return NaNs early
    # if infodict.get("message", "").lower().find("successful") == -1:
    #     N_sites = P_data0.shape[0]
    #     T = len(t_arr)
    #     return np.full((N_sites, T), np.nan), np.full((K, T), np.nan)

    # Fail if any non-finite
    if not np.all(np.isfinite(xs)):
        N_sites = P_data0.shape[0]
        T = len(t_arr)
        return np.full((N_sites, T), np.nan), np.full((K, T), np.nan)

    # Slice
    S_sim = xs[:, 0:K]
    A_sim = xs[:, K:2 * K]
    Kdyn_sim = xs[:, 2 * K:2 * K + M]
    P_sim = xs[:, 2 * K + M:2 * K + M + N]

    # Clip only bounded states
    np.clip(S_sim, 0.0, 1.0, out=S_sim)
    np.clip(Kdyn_sim, 0.0, 1.0, out=Kdyn_sim)
    np.clip(P_sim, 0.0, 1.0, out=P_sim)

    # A: keep your biological cap consistent with x0
    np.clip(A_sim, 0.0, 5.0, out=A_sim)

    return P_sim.T, A_sim.T


def build_full_A0(K, T, A_scaled, prot_idx_for_A):
    """
    Build a (K, T) abundance matrix from subset A_scaled (K_obs, T)
    using mapping prot_idx_for_A (len = K_obs).

    Unobserved proteins get zeros.
    """
    A0_full = np.zeros((K, T), dtype=float)

    if A_scaled.size > 0:
        for k, p_idx in enumerate(prot_idx_for_A):
            # copy the whole time-course simulate_p_scipy only uses [:, 0]
            A0_full[p_idx, :] = A_scaled[k, :]

    return A0_full

def fd_jacobian(
    x, t, theta,
    Cg, Cl,
    site_prot_idx,
    K_site_kin, R,
    L_alpha,
    kin_to_prot_idx,
    receptor_mask_prot,
    receptor_mask_kin,
    mechanism: str,
):
    if mechanism == "dist":
        mech_code = 0
    elif mechanism == "seq":
        mech_code = 1
    elif mechanism == "rand":
        mech_code = 2
    else:
        mech_code = 0

    x_arr = np.ascontiguousarray(x, dtype=np.float64)
    theta_arr = np.ascontiguousarray(theta, dtype=np.float64)

    return fd_jacobian_nb_core(
        x_arr, t, theta_arr,
        np.ascontiguousarray(Cg, dtype=np.float64),
        np.ascontiguousarray(Cl, dtype=np.float64),
        np.ascontiguousarray(site_prot_idx, dtype=np.int64),
        np.ascontiguousarray(K_site_kin, dtype=np.float64),
        np.ascontiguousarray(R, dtype=np.float64),
        np.ascontiguousarray(L_alpha, dtype=np.float64),
        np.ascontiguousarray(kin_to_prot_idx, dtype=np.int64),
        np.ascontiguousarray(receptor_mask_prot, dtype=np.int64),
        np.ascontiguousarray(receptor_mask_kin, dtype=np.int64),
        mech_code,
        ModelDims.K, ModelDims.M, ModelDims.N,
    )

@njit(cache=True, fastmath=True)
def fd_jacobian_nb_core(
    x, t, theta,
    Cg, Cl,
    site_prot_idx,
    K_site_kin, R,
    L_alpha,
    kin_to_prot_idx,
    receptor_mask_prot,
    receptor_mask_kin,
    mech_code,
    K, M, N,
    eps=1e-6,      # larger than 1e-8 for stiff-ish, clipped systems
    h_min=1e-8
):
    n = x.size
    J = np.empty((n, n), dtype=np.float64)

    # baseline
    f0 = rhs_nb_dispatch_dense(
        x, t, theta,
        Cg, Cl, site_prot_idx,
        K_site_kin, R, L_alpha,
        kin_to_prot_idx,
        receptor_mask_prot,
        receptor_mask_kin,
        K, M, N,
        mech_code,
    )

    x_pert = x.copy()

    for j in range(n):
        xj0 = x[j]

        # step size
        h = eps * (1.0 + math.fabs(xj0))
        if h < h_min:
            h = h_min

        # +h
        x_pert[j] = xj0 + h
        f_plus = rhs_nb_dispatch_dense(
            x_pert, t, theta,
            Cg, Cl, site_prot_idx,
            K_site_kin, R, L_alpha,
            kin_to_prot_idx,
            receptor_mask_prot,
            receptor_mask_kin,
            K, M, N,
            mech_code,
        )

        # -h
        x_pert[j] = xj0 - h
        f_minus = rhs_nb_dispatch_dense(
            x_pert, t, theta,
            Cg, Cl, site_prot_idx,
            K_site_kin, R, L_alpha,
            kin_to_prot_idx,
            receptor_mask_prot,
            receptor_mask_kin,
            K, M, N,
            mech_code,
        )

        # restore
        x_pert[j] = xj0

        inv2h = 0.5 / h
        for i in range(n):
            J[i, j] = (f_plus[i] - f_minus[i]) * inv2h

    return J