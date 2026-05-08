# SPDX-License-Identifier: MIT
"""
pinn/loss.py
PINN objective function builder.

Reuses the existing residual / loss infrastructure from optimization.py and
adds f_pinn_reg, the neural regularisation term.

Total objective:
    total = f1 + f2 + f3 + f4 + f_pinn_reg

where f1–f4 are the standard mechanistic loss components (phosphosite,
abundance, regularisation, mRNA) and f_pinn_reg regularises the neural term.

Two regularisation strategies are supported (pinn_cfg.regularize):

  "residual_l2"  – mean L2 norm of the PINN correction over the ODE trajectory.
                   This ties the regularisation to the actual neural contribution.
  "param_l2"     – L2 norm of the raw PINN network parameters.
                   Cheaper (no extra ODE solve needed) but coarser.
"""

from __future__ import annotations

import logging

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from phoscrosstalk.config import ModelDims
from phoscrosstalk.mechanisms import compute_prev_site_idx
from phoscrosstalk.simulation import build_full_A0
from phoscrosstalk.solver_config import (
    make_diffrax_adjoint,
    make_diffrax_solver,
    make_stepsize_controller,
)
from phoscrosstalk.pinn.rhs import make_combined_rhs

_logger = logging.getLogger(__name__)

# Per-element penalty returned when the ODE solve fails.
_FAILED_SOLVE_PENALTY: float = 1e3

# Upper bound for clipping R_rna (fold-change scale).
_RNA_CLIP_UPPER: float = 20.0


def make_pinn_loss_fn(
    dims: ModelDims,
    t,
    P_data,
    A_scaled,
    prot_idx_for_A,
    W_data,
    W_data_prot,
    Cg,
    Cl,
    site_prot_idx,
    K_site_kin,
    R,
    L_alpha,
    kin_to_prot_idx,
    receptor_mask_prot,
    receptor_mask_kin,
    mechanism: str,
    lambda_net: float,
    reg_lambda: float,
    lambda_pinn: float,
    regularize: str = "residual_l2",
    w_phospho: float = 1.0,
    w_abundance: float = 1.0,
    w_reg: float = 1.0,
    w_mrna: float = 1.0,
    rtol: float = 1e-6,
    atol: float = 1e-9,
    max_steps: int = 16384,
    dt0: float = 0.01,
    k_act_fn=None,
    s_prod_fn=None,
    t_mrna=None,
    rna_data_scaled=None,
    rna_model_prot_idx=None,
    R_data0=None,
    W_data_mrna=None,
    rna_relax: float = 0.1,
    abundance_max: float = 5.0,
    ode_solver_kind: str = "tsit5",
    ode_adjoint_kind: str = "recursive",
    root_find_max_steps: int = 10,
    xl=None,
    xu=None,
):
    """
    Build a scalar loss function for PINN joint optimisation.

    The returned function has signature::

        loss_fn((theta, pinn_model), args) -> (total_loss, (f1, f2, f3, f4, f_pinn_reg))

    where the trainable is a two-element tuple ``(theta, pinn_model)``.

    The PINN model is the last element in the ODE args tuple so diffrax
    correctly traces gradients through the neural correction.

    Parameters mirror those of ``make_residuals_fn`` in optimization.py, with
    the following additions:

    lambda_pinn : float
        Weight for the PINN regularisation term.
    regularize : str
        Strategy: ``"residual_l2"`` or ``"param_l2"``.
    """
    K, M, N = dims.K, dims.M, dims.N

    state_dim = 3 * K + M + N
    n_p   = max(1, P_data.size)
    n_A   = max(1, A_scaled.size)
    n_var = 2 * K + 2 + 3 * M + N + 4

    prev_site_idx = compute_prev_site_idx(site_prot_idx.astype(np.int32), N)

    # Build unified time grid
    if t_mrna is not None and len(t_mrna) > 0:
        all_times = np.union1d(t, t_mrna)
    else:
        all_times = np.unique(t)
    all_times = np.sort(all_times).astype(np.float64)

    prot_time_idx = np.searchsorted(all_times, t)
    if t_mrna is not None and len(t_mrna) > 0:
        mrna_time_idx = np.searchsorted(all_times, t_mrna)
    else:
        mrna_time_idx = None

    # Initial state
    T_prot = P_data.shape[1]
    A0_full = build_full_A0(K, T_prot, A_scaled, prot_idx_for_A)

    x0 = np.zeros(state_dim, dtype=np.float64)
    if R_data0 is not None:
        r_data = np.asarray(R_data0, dtype=np.float64)
        r0 = r_data[:, 0].copy() if r_data.ndim > 1 else r_data.copy()
        r0 = np.nan_to_num(r0, nan=1.0, posinf=5.0, neginf=0.0)
        x0[:K] = np.clip(r0, 0.0, 10.0)
    else:
        x0[:K] = 1.0
    a0 = np.nan_to_num(A0_full[:, 0], nan=1.0, posinf=5.0, neginf=0.0)
    x0[2 * K : 3 * K] = np.clip(a0, 0.0, 5.0)
    p0 = np.nan_to_num(P_data[:, 0], nan=0.0, posinf=10.0, neginf=0.0)
    x0[3 * K + M :] = np.clip(p0, 0.0, None)

    # Convert to JAX
    Cg_j   = jnp.asarray(Cg,              dtype=jnp.float64)
    Cl_j   = jnp.asarray(Cl,              dtype=jnp.float64)
    K_sk_j = jnp.asarray(K_site_kin,      dtype=jnp.float64)
    R_j    = jnp.asarray(R,               dtype=jnp.float64)
    La_j   = jnp.asarray(L_alpha,         dtype=jnp.float64)
    spi_j  = jnp.asarray(site_prot_idx,   dtype=jnp.int32)
    k2p_j  = jnp.asarray(kin_to_prot_idx, dtype=jnp.int32)
    rmp_j  = jnp.asarray(receptor_mask_prot, dtype=jnp.float64)
    rmk_j  = jnp.asarray(receptor_mask_kin,  dtype=jnp.float64)
    psi_j  = jnp.asarray(prev_site_idx,   dtype=jnp.int32)

    y0_j      = jnp.asarray(x0,       dtype=jnp.float64)
    t_eval    = jnp.asarray(all_times, dtype=jnp.float64)
    P_data_j  = jnp.asarray(P_data,   dtype=jnp.float64)
    A_scaled_j = jnp.asarray(A_scaled, dtype=jnp.float64)
    prot_idx_j = jnp.asarray(prot_idx_for_A, dtype=jnp.int32)
    prot_idx_solver = jnp.asarray(prot_time_idx, dtype=jnp.int32)

    W_data_j  = jnp.asarray(W_data,      dtype=jnp.float64)
    W_prot_j  = jnp.asarray(W_data_prot, dtype=jnp.float64)

    has_abundance = A_scaled.size > 0

    has_mrna = (
        t_mrna is not None
        and rna_data_scaled is not None
        and len(t_mrna) > 0
        and mrna_time_idx is not None
        and rna_model_prot_idx is not None
        and len(rna_model_prot_idx) > 0
    )

    if has_mrna:
        rna_j           = jnp.asarray(rna_data_scaled, dtype=jnp.float64)
        mrna_idx_j      = jnp.asarray(mrna_time_idx,   dtype=jnp.int32)
        rna_prot_idx_j  = jnp.asarray(rna_model_prot_idx, dtype=jnp.int32)
        n_matched = len(rna_model_prot_idx)
        T_rna     = len(t_mrna)
        W_rna_base = (
            np.asarray(W_data_mrna, dtype=np.float64)
            if W_data_mrna is not None
            else np.ones((n_matched, T_rna), dtype=np.float64)
        )
        W_rna_j  = jnp.asarray(W_rna_base, dtype=jnp.float64)
        n_rna    = max(1, rna_data_scaled.size)
    else:
        rna_j = mrna_idx_j = rna_prot_idx_j = W_rna_j = None
        n_rna = 1

    has_bounds = xl is not None and xu is not None
    if has_bounds:
        xl_j = jnp.asarray(xl, dtype=jnp.float64)
        xu_j = jnp.asarray(xu, dtype=jnp.float64)

    has_net_reg = lambda_net > 0.0

    # Build combined RHS factory (doesn't hold pinn_model yet)
    combined_rhs_fn = make_combined_rhs(
        K, M, N, mechanism,
        k_act_fn=k_act_fn,
        s_prod_fn=s_prod_fn,
        rna_relax=rna_relax,
        abundance_max=abundance_max,
    )

    term = diffrax.ODETerm(combined_rhs_fn)
    ode_solver = make_diffrax_solver(ode_solver_kind, root_find_max_steps=root_find_max_steps)
    sctrl   = make_stepsize_controller(rtol=rtol, atol=atol)
    saveat  = diffrax.SaveAt(ts=t_eval)
    adjoint = make_diffrax_adjoint(ode_adjoint_kind)

    t0_val = float(all_times[0])
    t1_val = float(all_times[-1])

    PENALTY = jnp.asarray(_FAILED_SOLVE_PENALTY, dtype=jnp.float64)

    def loss_fn(trainable, _args):
        """
        Compute scalar PINN loss and auxiliary diagnostics.

        Parameters
        ----------
        trainable : (theta, pinn_model)
        _args : None (not used; kept for dispatch compatibility)

        Returns
        -------
        (total_loss, (f1, f2, f3, f4, f_pinn_reg))
        """
        theta_raw, pinn_model = trainable
        theta_j = jnp.asarray(theta_raw, dtype=jnp.float64)
        if has_bounds:
            theta_j = jnp.clip(theta_j, xl_j, xu_j)

        ode_args = (
            theta_j,
            Cg_j, Cl_j, spi_j, K_sk_j, R_j, La_j,
            k2p_j, rmp_j, rmk_j, psi_j,
            pinn_model,   # last element – consumed by combined_rhs
        )

        sol = diffrax.diffeqsolve(
            term,
            ode_solver,
            t0=t0_val,
            t1=t1_val,
            dt0=dt0,
            y0=y0_j,
            args=ode_args,
            saveat=saveat,
            stepsize_controller=sctrl,
            max_steps=max_steps,
            adjoint=adjoint,
            throw=False,
        )

        xs = sol.ys  # (T_unified, state_dim)
        solve_ok = jnp.all(jnp.isfinite(xs)) & (sol.result == diffrax.RESULTS.successful)

        xs_prot = xs[prot_idx_solver, :]
        P_sim = jnp.clip(xs_prot[:, 3 * K + M :], 0.0, None).T  # (N, T_prot)
        A_sim = jnp.clip(xs_prot[:, 2 * K : 3 * K], 0.0, 5.0).T  # (K, T_prot)

        # --- f1: phosphosite loss ---
        diff_p = P_sim - P_data_j
        f1 = jnp.sum(W_data_j * diff_p * diff_p) / n_p

        # --- f2: abundance loss ---
        if has_abundance:
            A_sim_obs = A_sim[prot_idx_j, :]
            diff_A = A_sim_obs - A_scaled_j
            f2 = jnp.sum(W_prot_j * diff_A * diff_A) / n_A
        else:
            f2 = jnp.asarray(0.0, dtype=jnp.float64)

        # --- f3: mechanistic regularisation (L2 + Laplacian) ---
        f3_l2 = jnp.asarray(reg_lambda, dtype=jnp.float64) * jnp.dot(theta_j, theta_j)
        if has_net_reg:
            alpha_raw = theta_j[2 * K + 2 : 2 * K + 2 + M]
            alpha = jnp.exp(jnp.clip(alpha_raw, -20.0, 10.0))
            f3_net = jnp.asarray(lambda_net, dtype=jnp.float64) * jnp.dot(alpha, La_j @ alpha)
        else:
            f3_net = jnp.asarray(0.0, dtype=jnp.float64)
        f3 = (f3_l2 + f3_net) / jnp.asarray(max(n_var, 1), dtype=jnp.float64)

        # --- f4: mRNA / R_rna loss ---
        if has_mrna:
            xs_rna = xs[mrna_idx_j, :]
            R_sim_rna = jnp.clip(xs_rna[:, :K], 0.0, _RNA_CLIP_UPPER).T
            R_sim_matched = R_sim_rna[rna_prot_idx_j, :]
            diff_R = R_sim_matched - rna_j
            f4 = jnp.sum(W_rna_j * diff_R * diff_R) / n_rna
        else:
            f4 = jnp.asarray(0.0, dtype=jnp.float64)

        # --- f_pinn_reg: neural regularisation ---
        if regularize == "residual_l2":
            # Mean L2 of PINN correction over trajectory.
            # We pass t=0 to the PINN model for all trajectory points because
            # (a) the regularisation only needs an estimate of the correction
            #     magnitude, not a time-accurate evaluation, and
            # (b) passing varying traced time values would require vmapping over
            #     t values from the ODE output, adding overhead.
            # This simplification is conservative: if the PINN output is large
            # at t=0 it is likely large elsewhere too.  For "residual_l2" the
            # time argument is only used as one feature alongside the full ODE
            # state, so fixing it to a constant does not meaningfully bias the
            # regularisation.
            pinn_corrections = jax.vmap(
                lambda y_row: pinn_model(y_row, jnp.asarray(0.0, dtype=jnp.float64))
            )(xs)  # (T, state_dim)
            f_pinn_reg = jnp.mean(jnp.sum(pinn_corrections ** 2, axis=-1))
        else:  # "param_l2"
            leaves, _ = jax.tree_util.tree_flatten(
                eqx.filter(pinn_model, eqx.is_array)
            )
            sq_sum = sum(jnp.sum(l ** 2) for l in leaves if l is not None)
            n_params = max(1, sum(l.size for l in leaves if l is not None))
            f_pinn_reg = sq_sum / jnp.asarray(n_params, dtype=jnp.float64)

        f_pinn_reg_weighted = jnp.asarray(lambda_pinn, dtype=jnp.float64) * f_pinn_reg

        total = (
            jnp.asarray(w_phospho,  dtype=jnp.float64) * f1
            + jnp.asarray(w_abundance, dtype=jnp.float64) * f2
            + jnp.asarray(w_reg,  dtype=jnp.float64) * f3
            + jnp.asarray(w_mrna, dtype=jnp.float64) * f4
            + f_pinn_reg_weighted
        )

        # Guard against non-finite results
        f1 = jnp.where(jnp.isfinite(f1), f1, PENALTY)
        f2 = jnp.where(jnp.isfinite(f2), f2, PENALTY)
        f3 = jnp.where(jnp.isfinite(f3), f3, PENALTY)
        f4 = jnp.where(jnp.isfinite(f4), f4, PENALTY)
        f_pinn_reg = jnp.where(jnp.isfinite(f_pinn_reg), f_pinn_reg, PENALTY)
        total = jnp.where(jnp.isfinite(total), total, PENALTY)
        total = jnp.where(solve_ok, total, PENALTY)

        return total, (f1, f2, f3, f4, f_pinn_reg)

    return loss_fn
