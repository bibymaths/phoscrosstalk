"""
simulation.py
Diffrax-based ODE solver wrapper for the phospho-network model.

Public interface (unchanged from the SciPy version):
    simulate_ode(...)   – primary function
    simulate_p_scipy    – backward-compatible alias for simulate_ode
    build_full_A0(...)  – helper to build the full protein abundance matrix

The SciPy / Numba backend has been replaced by a Diffrax + JAX pipeline:
    - RHS is provided by jax_mechanisms.make_rhs
    - Integration uses diffrax.Tsit5 with PIDController adaptive stepping
    - Results are converted back to NumPy arrays at the module boundary
"""

import numpy as np
import jax
import jax.numpy as jnp
import diffrax

from phoscrosstalk.config import ModelDims
from phoscrosstalk.jax_mechanisms import make_rhs, compute_prev_site_idx


def simulate_ode(
    t_arr,
    P_data0,
    A_data0,
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
    mechanism,
    full_output=False,
    rtol=1e-6,
    atol=1e-9,
    max_steps=16384,
    dt0=0.01,
):
    """
    Simulate the phosphoproteomic network dynamics using Diffrax (JAX backend).

    Builds the initial state vector x = [S, A, K_dyn, p], integrates the
    ODE system over t_arr with the Tsit5 adaptive solver, clips bounded
    states, and returns the result as NumPy arrays.

    Args:
        t_arr (np.ndarray): Time points for the simulation.
        P_data0 (np.ndarray): Initial phosphosite data (used for t=0 state).
        A_data0 (np.ndarray): Initial protein abundance data (used for t=0 state).
        theta (np.ndarray): Flattened parameter vector.
        Cg, Cl (np.ndarray): Global and Local coupling matrices.
        site_prot_idx (np.ndarray): Mapping indices for sites to proteins.
        K_site_kin (np.ndarray): Kinase-site interaction matrix.
        R (np.ndarray): Receptor input matrix.
        L_alpha (np.ndarray): Kinase network Laplacian.
        kin_to_prot_idx (np.ndarray): Mapping indices for kinases to proteins.
        receptor_mask_prot, receptor_mask_kin (np.ndarray): Input masks.
        mechanism (str): Kinetic mechanism ('dist', 'seq', 'rand').
        full_output (bool): If True, also return S_sim and Kdyn_sim.
        rtol, atol (float): Solver tolerances.
        max_steps (int): Maximum solver steps.
        dt0 (float): Initial step size.

    Returns:
        tuple:
            - P_sim (np.ndarray): Simulated phosphosite trajectories (N_sites x T).
            - A_sim (np.ndarray): Simulated protein abundance trajectories (K_proteins x T).
            Returns arrays of NaNs if integration fails.
    """
    K, M, N = ModelDims.K, ModelDims.M, ModelDims.N
    if K is None or M is None or N is None:
        raise RuntimeError(
            "ModelDims have not been set. Call ModelDims.set_dims(K, M, N) before "
            "running a simulation."
        )

    T       = len(t_arr)
    N_sites = P_data0.shape[0]

    # Build initial conditions
    x0 = np.zeros(2 * K + M + N, dtype=np.float64)

    a0 = np.nan_to_num(A_data0[:, 0].astype(np.float64), nan=1.0, posinf=5.0, neginf=0.0)
    a0 = np.clip(a0, 0.0, 5.0)
    x0[K : 2 * K] = a0

    p0 = np.nan_to_num(P_data0[:, 0].astype(np.float64), nan=0.0, posinf=1.0, neginf=0.0)
    p0 = np.clip(p0, 0.0, 1.0)
    x0[2 * K + M :] = p0

    nan_result = _nan_result(N_sites, K, M, T, full_output)
    if not np.all(np.isfinite(x0)):
        return nan_result

    # Precompute static topology for sequential mechanism
    prev_site_idx = compute_prev_site_idx(site_prot_idx.astype(np.int32), N)

    # Convert all topology arrays to JAX float32
    args = (
        jnp.asarray(theta,               dtype=jnp.float32),
        jnp.asarray(Cg,                  dtype=jnp.float32),
        jnp.asarray(Cl,                  dtype=jnp.float32),
        jnp.asarray(site_prot_idx,       dtype=jnp.int32),
        jnp.asarray(K_site_kin,          dtype=jnp.float32),
        jnp.asarray(R,                   dtype=jnp.float32),
        jnp.asarray(L_alpha,             dtype=jnp.float32),
        jnp.asarray(kin_to_prot_idx,     dtype=jnp.int32),
        jnp.asarray(receptor_mask_prot,  dtype=jnp.float32),
        jnp.asarray(receptor_mask_kin,   dtype=jnp.float32),
        jnp.asarray(prev_site_idx,       dtype=jnp.int32),
    )

    rhs_fn        = make_rhs(K, M, N, mechanism)
    term          = diffrax.ODETerm(rhs_fn)
    t_eval        = jnp.asarray(t_arr, dtype=jnp.float32)
    y0_jax        = jnp.asarray(x0,   dtype=jnp.float32)
    saveat        = diffrax.SaveAt(ts=t_eval)
    stepsize_ctrl = diffrax.PIDController(rtol=rtol, atol=atol)
    solver        = diffrax.Tsit5()

    try:
        sol = diffrax.diffeqsolve(
            term,
            solver,
            t0=float(t_arr[0]),
            t1=float(t_arr[-1]),
            dt0=dt0,
            y0=y0_jax,
            args=args,
            saveat=saveat,
            stepsize_controller=stepsize_ctrl,
            max_steps=max_steps,
            throw=False,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Diffrax solver failed. mechanism={mechanism!r}, "
            f"theta.shape={theta.shape}, t0={t_arr[0]}, t1={t_arr[-1]}. "
            f"Original error: {exc}"
        ) from exc

    # sol.ys shape: (T, state_dim)
    xs = np.asarray(sol.ys, dtype=np.float64)

    if not np.all(np.isfinite(xs)):
        return nan_result

    # Slice and clip bounded states
    S_sim    = xs[:, :K]
    A_sim    = xs[:, K : 2 * K]
    Kdyn_sim = xs[:, 2 * K : 2 * K + M]
    P_sim    = xs[:, 2 * K + M : 2 * K + M + N]

    np.clip(S_sim,    0.0, 1.0, out=S_sim)
    np.clip(Kdyn_sim, 0.0, 1.0, out=Kdyn_sim)
    np.clip(P_sim,    0.0, 1.0, out=P_sim)
    np.clip(A_sim,    0.0, 5.0, out=A_sim)

    if full_output:
        return P_sim.T, A_sim.T, S_sim.T, Kdyn_sim.T

    return P_sim.T, A_sim.T


def _nan_result(N_sites, K, M, T, full_output):
    """Return NaN sentinel arrays matching expected output shape."""
    if full_output:
        return (
            np.full((N_sites, T), np.nan),
            np.full((K,       T), np.nan),
            np.full((K,       T), np.nan),
            np.full((M,       T), np.nan),
        )
    return np.full((N_sites, T), np.nan), np.full((K, T), np.nan)


# Backward-compatible alias used throughout analysis, app, sensitivity, etc.
simulate_p_scipy = simulate_ode


def build_full_A0(K, T, A_scaled, prot_idx_for_A):
    """
    Constructs the full-dimension protein abundance matrix from partial observations.

    Maps the observed protein data (which may only cover a subset of proteins) into the
    full model state space K x T. Unobserved proteins are initialized to zero.

    Args:
        K (int): Total number of proteins in the model.
        T (int): Number of time points.
        A_scaled (np.ndarray): Observed protein data (K_obs x T).
        prot_idx_for_A (np.ndarray): Indices mapping observations to the full protein list.

    Returns:
        np.ndarray: Full abundance matrix (K x T).
    """
    A0_full = np.zeros((K, T), dtype=float)
    if A_scaled.size > 0:
        for k, p_idx in enumerate(prot_idx_for_A):
            A0_full[p_idx, :] = A_scaled[k, :]
    return A0_full
