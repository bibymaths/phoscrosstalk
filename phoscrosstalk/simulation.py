"""
simulation.py
Diffrax-based ODE solver wrapper for the phospho-network model.

Public interface:
    simulate_ode(...)        – primary function
    simulate_p_scipy         – backward-compatible alias for simulate_ode
    build_full_A0(...)       – helper to build the full protein abundance matrix

State layout (new):
    y = [R_rna, S, A, Kdyn, p]    (dim = 3*K + M + N)

    R_rna : mRNA levels for model proteins        shape (K,)
    S     : protein signalling/activation state   shape (K,)
    A     : protein abundance state               shape (K,)
    Kdyn  : kinase activity state                 shape (M,)
    p     : relative phosphosite signal            shape (N,)

The SciPy / Numba backend has been replaced by a Diffrax + JAX pipeline:
    - RHS is provided by jax_mechanisms.make_rhs
    - Integration uses diffrax.Tsit5 with PIDController adaptive stepping
    - Results are converted back to NumPy arrays at the module boundary
"""

import diffrax
import jax.numpy as jnp
import numpy as np

from phoscrosstalk.config import ModelDims
from phoscrosstalk.jax_mechanisms import compute_prev_site_idx, make_rhs
from phoscrosstalk.solver_config import (
    make_diffrax_adjoint,
    make_diffrax_solver,
    make_stepsize_controller,
)


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
    return_full=False,
    rtol=1e-6,
    atol=1e-9,
    max_steps=16384,
    dt0=0.01,
    ode_solver_kind="tsit5",
    root_find_max_steps=10,
    k_act_fn=None,
    s_prod_fn=None,
    t_extra=None,
    t_rna=None,
    R_data0=None,
    rna_relax=0.1,
    ode_adjoint_kind="recursive",
):
    """
    Simulate the phosphoproteomic network dynamics using Diffrax (JAX backend).

    State layout: y = [R_rna, S, A, Kdyn, p]  (dim = 3*K + M + N)

    Builds the initial state vector, integrates the ODE system over a unified
    time grid (``t_arr ∪ t_rna ∪ t_extra`` if provided), clips bounded states,
    and returns results sampled at ``t_arr`` (and optionally at ``t_rna``).

    Args:
        t_arr (np.ndarray): Primary time points for protein/phosphosite output.
        P_data0 (np.ndarray): Initial phosphosite data (N_sites x T) for t=0 IC.
        A_data0 (np.ndarray): Initial protein abundance data (K x T) for t=0 IC.
        theta (np.ndarray): Flattened parameter vector (length 2*K+2+3*M+N+4).
        Cg, Cl (np.ndarray): Global and Local coupling matrices.
        site_prot_idx (np.ndarray): Mapping indices for sites to proteins.
        K_site_kin (np.ndarray): Kinase-site interaction matrix.
        R (np.ndarray): Receptor/kinase input matrix (NOT the mRNA state).
        L_alpha (np.ndarray): Kinase network Laplacian.
        kin_to_prot_idx (np.ndarray): Mapping indices for kinases to proteins.
        receptor_mask_prot, receptor_mask_kin (np.ndarray): Input masks.
        mechanism (str): Kinetic mechanism ('dist', 'seq', 'rand').
        full_output (bool): If True, return (P_sim, A_sim, S_sim, Kdyn_sim).
        return_full (bool): If True, return a dict with all state components
            plus R_sim (at t_arr) and R_sim_rna (at t_rna).
        rtol, atol (float): Solver tolerances.
        max_steps (int): Maximum solver steps.
        dt0 (float): Initial step size.
        k_act_fn (callable | None): JAX closure for derived k_act(t) -> (K,).
        s_prod_fn (callable | None): JAX closure for derived s_prod(t) -> (K,).
        t_extra (np.ndarray | None): Additional time points to include in solver.
        t_rna (np.ndarray | None): mRNA-specific time points for RNA output.
        R_data0 (np.ndarray | None): RNA initial condition matrix (K x T_rna or K,);
            R_rna state is initialized from first column (or vector). Defaults 1.0.

    Returns:
        If return_full=True: dict with keys
            P_sim, A_sim, S_sim, Kdyn_sim, R_sim, R_sim_rna, t, t_rna, solver_times
        elif full_output=True: (P_sim, A_sim, S_sim, Kdyn_sim)  – (N|K|K|M x T)
        else: (P_sim, A_sim)  – (N_sites x T, K x T)
        Returns NaN arrays if integration fails.
    """
    K, M, N = ModelDims.K, ModelDims.M, ModelDims.N
    if K is None or M is None or N is None:
        raise RuntimeError(
            "ModelDims have not been set. Call ModelDims.set_dims(K, M, N) before "
            "running a simulation."
        )

    T = len(t_arr)
    N_sites = P_data0.shape[0]

    # Resolve RNA time grid:
    # t_rna is the preferred argument; t_extra is a legacy alias kept for
    # backward compatibility.  When both are None there is no separate RNA grid.
    if t_rna is None and t_extra is not None:
        t_rna = t_extra
    t_rna_arr = (
        np.asarray(t_rna, dtype=np.float64)
        if t_rna is not None and len(t_rna) > 0
        else None
    )

    # Build unified solver time grid: t_arr ∪ t_rna ∪ t_extra
    parts = [t_arr]
    if t_rna_arr is not None:
        parts.append(t_rna_arr)
    if t_extra is not None and len(t_extra) > 0:
        parts.append(np.asarray(t_extra, dtype=np.float64))
    solver_times = np.sort(np.unique(np.concatenate(parts)))

    # Index maps: where in solver_times each primary/RNA time falls
    prot_time_idx = np.searchsorted(solver_times, t_arr)
    if t_rna_arr is not None:
        rna_time_idx = np.searchsorted(solver_times, t_rna_arr)
    else:
        rna_time_idx = None

    # Build initial conditions: x0 = [R_rna, S, A, Kdyn, p]  (3*K + M + N)
    x0 = np.zeros(3 * K + M + N, dtype=np.float64)

    # R_rna initial condition (mRNA): from R_data0 or default 1.0
    if R_data0 is not None:
        r_data = np.asarray(R_data0, dtype=np.float64)
        if r_data.ndim == 1:
            r0 = r_data.copy()
        else:
            r0 = r_data[:, 0].copy()
        r0 = np.nan_to_num(r0, nan=1.0, posinf=5.0, neginf=0.0)
        r0 = np.clip(r0, 0.0, 10.0)
    else:
        r0 = np.ones(K, dtype=np.float64)
    x0[:K] = r0

    # A initial condition (protein abundance)
    a0 = np.nan_to_num(
        A_data0[:, 0].astype(np.float64), nan=1.0, posinf=5.0, neginf=0.0
    )
    a0 = np.clip(a0, 0.0, 5.0)
    x0[2 * K : 3 * K] = a0

    # p initial condition (relative phosphosite signal, clipped to [0, +inf))
    p0 = np.nan_to_num(
        P_data0[:, 0].astype(np.float64), nan=0.0, posinf=10.0, neginf=0.0
    )
    p0 = np.clip(p0, 0.0, None)
    x0[3 * K + M :] = p0

    nan_result = _nan_result(N_sites, K, M, T, full_output, return_full, t_rna_arr)
    if not np.all(np.isfinite(x0)):
        return nan_result

    # Precompute static topology for sequential mechanism
    prev_site_idx = compute_prev_site_idx(site_prot_idx.astype(np.int32), N)

    # Convert all topology arrays to JAX float32
    args = (
        jnp.asarray(theta, dtype=jnp.float32),
        jnp.asarray(Cg, dtype=jnp.float32),
        jnp.asarray(Cl, dtype=jnp.float32),
        jnp.asarray(site_prot_idx, dtype=jnp.int32),
        jnp.asarray(K_site_kin, dtype=jnp.float32),
        jnp.asarray(R, dtype=jnp.float32),
        jnp.asarray(L_alpha, dtype=jnp.float32),
        jnp.asarray(kin_to_prot_idx, dtype=jnp.int32),
        jnp.asarray(receptor_mask_prot, dtype=jnp.float32),
        jnp.asarray(receptor_mask_kin, dtype=jnp.float32),
        jnp.asarray(prev_site_idx, dtype=jnp.int32),
    )

    rhs_fn = make_rhs(
        K, M, N, mechanism, k_act_fn=k_act_fn, s_prod_fn=s_prod_fn, rna_relax=rna_relax
    )
    term = diffrax.ODETerm(rhs_fn)
    t_eval = jnp.asarray(solver_times, dtype=jnp.float32)
    y0_jax = jnp.asarray(x0, dtype=jnp.float32)
    saveat = diffrax.SaveAt(ts=t_eval)
    stepsize_ctrl = make_stepsize_controller(rtol=rtol, atol=atol)
    solver = make_diffrax_solver(
        ode_solver_kind,
        root_find_max_steps=root_find_max_steps,
    )
    adjoint = make_diffrax_adjoint(ode_adjoint_kind)

    try:
        sol = diffrax.diffeqsolve(
            term,
            solver,
            t0=float(solver_times[0]),
            t1=float(solver_times[-1]),
            dt0=dt0,
            y0=y0_jax,
            args=args,
            saveat=saveat,
            stepsize_controller=stepsize_ctrl,
            max_steps=max_steps,
            throw=False,
            adjoint=adjoint,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Diffrax solver failed. mechanism={mechanism!r}, "
            f"theta.shape={theta.shape}, t0={solver_times[0]}, t1={solver_times[-1]}. "
            f"Original error: {exc}"
        ) from exc

    # sol.ys shape: (T_unified, 3*K + M + N)
    xs_all = np.asarray(sol.ys, dtype=np.float64)

    if not np.all(np.isfinite(xs_all)) or sol.result != diffrax.RESULTS.successful:
        return nan_result

    # Sample at protein/phosphosite time indices
    xs = xs_all[prot_time_idx, :]

    # Slice and clip bounded states (new layout: [R_rna, S, A, Kdyn, p])
    R_rna_sim = xs[:, :K]
    S_sim = xs[:, K : 2 * K]
    A_sim = xs[:, 2 * K : 3 * K]
    Kdyn_sim = xs[:, 3 * K : 3 * K + M]
    P_sim = xs[:, 3 * K + M : 3 * K + M + N]

    np.clip(R_rna_sim, 0.0, None, out=R_rna_sim)
    np.clip(S_sim, 0.0, 1.0, out=S_sim)
    np.clip(Kdyn_sim, 0.0, 1.0, out=Kdyn_sim)

    # p is now a non-negative relative phosphosite signal, not a bounded occupancy.
    # Keep only the biological/numerical lower bound.
    np.clip(P_sim, 0.0, None, out=P_sim)

    np.clip(A_sim, 0.0, 5.0, out=A_sim)

    # return_full: rich dict with all outputs including RNA at RNA time points
    if return_full:
        if rna_time_idx is not None:
            xs_rna = xs_all[rna_time_idx, :]
            R_rna_at_rna = np.clip(xs_rna[:, :K], 0.0, None)
        else:
            R_rna_at_rna = R_rna_sim.copy()  # fallback: same grid
        return {
            "P_sim": P_sim.T,  # (N, T)
            "A_sim": A_sim.T,  # (K, T)
            "S_sim": S_sim.T,  # (K, T)
            "Kdyn_sim": Kdyn_sim.T,  # (M, T)
            "R_sim": R_rna_sim.T,  # (K, T)  sampled at t_arr
            "R_sim_rna": R_rna_at_rna.T,  # (K, T_rna) sampled at t_rna
            "t": t_arr,
            "t_rna": t_rna_arr if t_rna_arr is not None else t_arr,
            "solver_times": solver_times,
        }

    if full_output:
        return P_sim.T, A_sim.T, S_sim.T, Kdyn_sim.T

    return P_sim.T, A_sim.T


def _nan_result(N_sites, K, M, T, full_output, return_full=False, t_rna_arr=None):
    """Return NaN sentinel arrays matching expected output shape."""
    if return_full:
        T_rna = len(t_rna_arr) if t_rna_arr is not None else T
        return {
            "P_sim": np.full((N_sites, T), np.nan),
            "A_sim": np.full((K, T), np.nan),
            "S_sim": np.full((K, T), np.nan),
            "Kdyn_sim": np.full((M, T), np.nan),
            "R_sim": np.full((K, T), np.nan),
            "R_sim_rna": np.full((K, T_rna), np.nan),
            "t": None,
            "t_rna": t_rna_arr,
            "solver_times": None,
        }
    if full_output:
        return (
            np.full((N_sites, T), np.nan),
            np.full((K, T), np.nan),
            np.full((K, T), np.nan),
            np.full((M, T), np.nan),
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
    """  # noqa: E501
    A0_full = np.zeros((K, T), dtype=float)
    if A_scaled.size > 0:
        for k, p_idx in enumerate(prot_idx_for_A):
            A0_full[p_idx, :] = A_scaled[k, :]
    return A0_full
