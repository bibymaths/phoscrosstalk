# SPDX-License-Identifier: MIT
"""
optimization.py
Optimistix-based objective functions and parameter fitting for the phospho-network.

Primary fitting path (canonical Diffrax + Optimistix approach):
  - ODE solve inside the optimization objective using diffrax.Tsit5.
  - diffrax.SaveAt(ts=...) with explicit saved timepoints.
  - diffrax.DirectAdjoint() for differentiating through the ODE solve (required
    because Optimistix least-squares solvers use forward-mode autodiff).
  - Optimistix LevenbergMarquardt with optx.least_squares for the main fitting path.
  - Residual vector (not scalar) as the optimization objective.

Loss components (diagnostics, computed from residuals):
  f1 : phosphosite relative-signal loss – mean(W_data  * (P_sim - P_data)^2)
  f2 : protein abundance loss       – mean(W_prot  * (A_sim - A_data)^2)
  f3 : regularisation               – L2 + Laplacian network term
  f4 : mRNA / R_rna state loss      – mean(W_mrna  * (R_sim - R_obs)^2)
                                      zero when no RNA data provided

Residual vector structure (for optx.least_squares):
  [sqrt(w_phospho*W_data) * (P_sim - P_data),   shape: (N*T_prot,)
   sqrt(w_abundance*W_prot) * (A_sim - A_data),  shape: (K_obs*T_prot,)
   sqrt(w_mrna*W_mrna) * (R_sim - R_obs),        shape: (n_rna,)
   sqrt(reg_lambda) * theta,                      shape: (n_var,)
   sqrt(lambda_net) * alpha_net_reg]              shape: (M,)  [when lambda_net>0]

State layout: y = [R_rna, S, A, Kdyn, p]  (dim = 3*K + M + N)
"""

import os
import pathlib
from collections.abc import Callable, Sequence

import diffrax
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optimistix as optx
from numba import njit

from phoscrosstalk.logger import get_logger
from phoscrosstalk.config import ModelDims
from phoscrosstalk.core_mechanisms import decode_theta
from phoscrosstalk.jax_mechanisms import (
    compute_objectives_jax,
    compute_prev_site_idx,
    make_rhs,
)
from phoscrosstalk.simulation import build_full_A0, simulate_ode
from phoscrosstalk.solver_config import (
    make_diffrax_adjoint,
    make_diffrax_solver,
    make_ls_solver,
    make_optx_adjoint,
    make_stepsize_controller,
)

logger = get_logger()

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------


# Upper bound for clipping R_rna (fold-change scale).
# RNA fold-change values >20 are biologically implausible and risk float32 overflow.
# This bound is a soft cap that still allows the optimizer to distinguish signals.
_RNA_CLIP_UPPER: float = 20.0

# Per-element penalty value returned when the ODE solve fails or produces non-finite
# states.  This is large enough to guide the optimizer away from bad regions
# but small enough to remain representable in float32 (~3.4e38 max).
_FAILED_SOLVE_PENALTY: float = 1e3


@njit(cache=True)
def bio_score_nb(theta, K, M, N):
    """
    Numba-compiled kernel to calculate the Biological Plausibility Score.

    Derives the half-lives (t_half = ln(2)/k) for kinases and proteins from the parameter
    vector theta and penalizes deviations from expected biological time scales.

    Args:
        theta (np.ndarray): Parameter vector (length 2*K+2+3*M+N+4).
        K, M, N (int): Model dimensions.

    Returns:
        float: The calculated biological score (lower is better/more plausible).
    """  # noqa: E501
    (k_deact, d_deg, _, _, _, kK_act, kK_deact, _, _, _, _, _) = decode_theta(
        theta, K, M, N
    )
    t_half_kinase = np.log(2.0) / kK_deact
    t_half_protein = np.log(2.0) / d_deg

    median_t_kinase = np.sort(t_half_kinase)[len(t_half_kinase) // 2]
    median_t_protein = np.sort(t_half_protein)[len(t_half_protein) // 2]

    return (np.log10(median_t_kinase) - np.log10(10.0)) ** 2 + (
        np.log10(median_t_protein) - np.log10(600.0)
    ) ** 2


def bio_score(theta):
    """
    Wrapper function to calculate the biological plausibility score for a parameter set.

    Args:
        theta (np.ndarray): Parameter vector.

    Returns:
        float: Biological score.
    """
    return float(bio_score_nb(theta, ModelDims.K, ModelDims.M, ModelDims.N))


def create_bounds(K, M, N):
    """
    Generates the lower (xl) and upper (xu) bound vectors for the optimization search space.

    ``k_act`` and ``s_prod`` are no longer optimisation variables.
    The dimension is now ``2*K + 2 + 3*M + N + 4``.

    Args:
        K, M, N (int): Model dimensions.

    Returns:
        tuple: (xl, xu, dim)
    """  # noqa: E501
    dim = 2 * K + 2 + 3 * M + N + 4
    xl, xu = np.zeros(dim), np.zeros(dim)
    idx = 0
    # Protein: k_deact, d_deg (k_act and s_prod removed – derived from data)
    # k_deact
    xl[idx : idx + K] = np.log(1e-5)
    xu[idx : idx + K] = np.log(10.0)
    idx += K
    # d_deg (restricted upper bound for biological plausibility)
    xl[idx : idx + K] = np.log(1e-5)
    xu[idx : idx + K] = np.log(0.5)
    idx += K
    # Coupling
    xl[idx] = np.log(1e-5)
    xu[idx] = np.log(10.0)
    idx += 1
    xl[idx] = np.log(1e-5)
    xu[idx] = np.log(10.0)
    idx += 1
    # Kinase: alpha, kK_act, kK_deact
    xl[idx : idx + M] = np.log(1e-5)
    xu[idx : idx + M] = np.log(10.0)
    idx += M
    xl[idx : idx + M] = np.log(1e-5)
    xu[idx : idx + M] = np.log(3.0)
    idx += M
    xl[idx : idx + M] = np.log(1e-5)
    xu[idx : idx + M] = np.log(3.0)
    idx += M
    # Site: k_off
    xl[idx : idx + N] = np.log(1e-5)
    xu[idx : idx + N] = np.log(5.0)
    idx += N
    # Gammas (tanh raw)
    xl[idx : idx + 4] = -3.0
    xu[idx : idx + 4] = 3.0
    idx += 4
    return xl, xu, dim


def build_parameter_labels(K: int, M: int, N: int) -> list[str]:
    """
    Return human-readable labels for every element of the flattened theta vector.

    The theta vector layout (length ``2*K + 2 + 3*M + N + 4``) is:

    ============  =====================  ============================
    Slice         Length                 Content
    ============  =====================  ============================
    [0 : K)       K                      log_k_deact[0..K-1]
    [K : 2K)      K                      log_d_deg[0..K-1]
    [2K : 2K+1)   1                      log_beta_g
    [2K+1 : 2K+2) 1                      log_beta_l
    [2K+2 : ...)  M                      log_alpha[0..M-1]
    [... : ...)   M                      log_kK_act[0..M-1]
    [... : ...)   M                      log_kK_deact[0..M-1]
    [... : ...)   N                      log_k_off[0..N-1]
    [... : end)   4                      gamma_raw[0..3]
    ============  =====================  ============================

    Parameters
    ----------
    K : int
        Number of model proteins.
    M : int
        Number of model kinases.
    N : int
        Number of phosphosites.

    Returns
    -------
    list[str]
        Parameter labels in the exact order they appear in theta.
        Length is ``2*K + 2 + 3*M + N + 4``.

    Notes
    -----
    These labels are intended for use with :func:`compute_second_order_sensitivities`
    to annotate Hessian rows/columns.
    """
    labels: list[str] = []
    # log-rate protein kinetics
    for k in range(K):
        labels.append(f"log_k_deact[{k}]")
    for k in range(K):
        labels.append(f"log_d_deg[{k}]")
    # coupling
    labels.append("log_beta_g")
    labels.append("log_beta_l")
    # kinase kinetics
    for m in range(M):
        labels.append(f"log_alpha[{m}]")
    for m in range(M):
        labels.append(f"log_kK_act[{m}]")
    for m in range(M):
        labels.append(f"log_kK_deact[{m}]")
    # site off-rates
    for n in range(N):
        labels.append(f"log_k_off[{n}]")
    # gamma raw coefficients
    for i in range(4):
        labels.append(f"gamma_raw[{i}]")
    return labels


# ---------------------------------------------------------------------------
# Scalarized JAX loss for Optimistix
# ---------------------------------------------------------------------------


def make_loss_fn(
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
    mechanism,
    lambda_net,
    reg_lambda,
    w_phospho=1.0,
    w_abundance=1.0,
    w_reg=1.0,
    rtol=1e-6,
    atol=1e-9,
    max_steps=16384,
    k_act_fn=None,
    s_prod_fn=None,
    t_mrna=None,
    rna_data_scaled=None,
    w_mrna=1.0,
    rna_model_prot_idx=None,
    R_data0=None,
    W_data_rna=None,
    rna_relax=0.1,
    ode_solver_kind="tsit5",
    ode_adjoint_kind="recursive",
    dt0=0.01,
    root_find_max_steps=10,
    scan_kind=None,
):
    """
    Build a JAX-differentiable scalarized loss function for Optimistix.

    The returned function ``loss_fn(theta, args)`` is compatible with
    ``optimistix.minimise``. It:
      1. Runs diffrax.diffeqsolve inside the loss (over a unified time grid).
      2. Computes f1 (phosphosite), f2 (abundance), f3 (regularisation).
      3. Optionally computes f4 (mRNA/R_rna) when *t_mrna* and *rna_data_scaled*
         are provided.
      4. Returns the weighted total loss plus (f1, f2, f3, f4) as auxiliary output.

    Parameters are frozen at creation time; only ``theta`` varies.

    State layout: y = [R_rna, S, A, Kdyn, p]  (dim = 3*K + M + N)
    """
    K, M, N = ModelDims.K, ModelDims.M, ModelDims.N

    n_p = max(1, P_data.size)
    n_A = max(1, A_scaled.size)
    n_var = 2 * K + 2 + 3 * M + N + 4

    prev_site_idx = compute_prev_site_idx(site_prot_idx.astype(np.int32), N)

    # Build unified time grid (protein ∪ mRNA)
    if t_mrna is not None and len(t_mrna) > 0:
        all_times = np.union1d(t, t_mrna)
    else:
        all_times = np.unique(t)
    all_times = np.sort(all_times).astype(np.float64)

    # Index maps: where in solver output do the protein / mRNA times land?
    prot_time_idx = np.searchsorted(all_times, t)
    if t_mrna is not None and len(t_mrna) > 0:
        mrna_time_idx = np.searchsorted(all_times, t_mrna)
    else:
        mrna_time_idx = None

    # Build initial state from data (new layout: [R_rna, S, A, Kdyn, p])
    T_prot = P_data.shape[1]
    A0_full = build_full_A0(K, T_prot, A_scaled, prot_idx_for_A)

    x0 = np.zeros(3 * K + M + N, dtype=np.float64)
    # R_rna initial condition
    if R_data0 is not None:
        r_data = np.asarray(R_data0, dtype=np.float64)
        if r_data.ndim == 1:
            r0 = r_data.copy()
        else:
            r0 = r_data[:, 0].copy()
        r0 = np.nan_to_num(r0, nan=1.0, posinf=5.0, neginf=0.0)
        x0[:K] = np.clip(r0, 0.0, 10.0)
    else:
        x0[:K] = 1.0  # default fold-change = 1.0
    # A initial condition
    a0 = np.nan_to_num(A0_full[:, 0], nan=1.0, posinf=5.0, neginf=0.0)
    x0[2 * K : 3 * K] = np.clip(a0, 0.0, 5.0)
    # p initial condition
    p0 = np.nan_to_num(P_data[:, 0], nan=0.0, posinf=10.0, neginf=0.0)
    x0[3 * K + M :] = np.clip(p0, 0.0, None)

    # JAX static arrays
    Cg_j = jnp.asarray(Cg, dtype=jnp.float32)
    Cl_j = jnp.asarray(Cl, dtype=jnp.float32)
    K_sk_j = jnp.asarray(K_site_kin, dtype=jnp.float32)
    R_j = jnp.asarray(R, dtype=jnp.float32)
    La_j = jnp.asarray(L_alpha, dtype=jnp.float32)
    spi_j = jnp.asarray(site_prot_idx, dtype=jnp.int32)
    k2p_j = jnp.asarray(kin_to_prot_idx, dtype=jnp.int32)
    rmp_j = jnp.asarray(receptor_mask_prot, dtype=jnp.float32)
    rmk_j = jnp.asarray(receptor_mask_kin, dtype=jnp.float32)
    psi_j = jnp.asarray(prev_site_idx, dtype=jnp.int32)

    y0_j = jnp.asarray(x0, dtype=jnp.float32)
    t_eval = jnp.asarray(all_times, dtype=jnp.float32)

    P_data_j = jnp.asarray(P_data, dtype=jnp.float32)
    A_scaled_j = jnp.asarray(A_scaled, dtype=jnp.float32)
    W_data_j = jnp.asarray(W_data, dtype=jnp.float32)
    W_prot_j = jnp.asarray(W_data_prot, dtype=jnp.float32)
    prot_idx_j = jnp.asarray(prot_idx_for_A, dtype=jnp.int32)
    La_loss_j = jnp.asarray(L_alpha, dtype=jnp.float32)

    prot_idx_solver = jnp.asarray(prot_time_idx, dtype=jnp.int32)

    # mRNA arrays (if available)
    # Only t_mrna, rna_data_scaled, and rna_model_prot_idx are strictly required.
    # W_data_rna, rna_obs_idx, and rna_fit_genes are optional metadata; when
    # W_data_rna is absent a uniform (all-ones) weight matrix is used.
    has_mrna = (
        t_mrna is not None
        and rna_data_scaled is not None
        and len(t_mrna) > 0
        and mrna_time_idx is not None
        and rna_model_prot_idx is not None
        and len(rna_model_prot_idx) > 0
    )

    if has_mrna:
        # All of t_mrna, rna_data_scaled, rna_model_prot_idx are guaranteed non-None
        # and non-empty here by the has_mrna gate above.
        assert t_mrna is not None  # type checker hint
        rna_j = jnp.asarray(rna_data_scaled, dtype=jnp.float32)
        mrna_idx_j = jnp.asarray(mrna_time_idx, dtype=jnp.int32)
        rna_prot_idx_j = jnp.asarray(rna_model_prot_idx, dtype=jnp.int32)
        n_matched = len(rna_model_prot_idx)
        T_rna = len(t_mrna)
        if W_data_rna is not None:
            W_rna_j = jnp.asarray(W_data_rna, dtype=jnp.float32)
        else:
            W_rna_j = jnp.ones((n_matched, T_rna), dtype=jnp.float32)
        n_rna = max(1, rna_data_scaled.size)
    else:
        rna_j = None
        mrna_idx_j = None
        rna_prot_idx_j = None
        W_rna_j = None
        n_rna = 1

    rhs_fn = make_rhs(
        K, M, N, mechanism, k_act_fn=k_act_fn, s_prod_fn=s_prod_fn, rna_relax=rna_relax
    )
    term = diffrax.ODETerm(rhs_fn)
    solver = make_diffrax_solver(
        ode_solver_kind,
        root_find_max_steps=root_find_max_steps,
        scan_kind=scan_kind
    )
    sctrl = make_stepsize_controller(rtol=rtol, atol=atol)
    saveat = diffrax.SaveAt(ts=t_eval)
    adjoint = make_diffrax_adjoint(ode_adjoint_kind)

    logger.info(f"Using Diffrax adjoint: {adjoint}")
    logger.info(f"Using Diffrax solver: {solver}")

    t0_val = float(all_times[0])
    t1_val = float(all_times[-1])

    FAILED_SOLVE_PENALTY = jnp.float32(1e6)

    def loss_fn(theta, _args):
        theta_j = jnp.asarray(theta, dtype=jnp.float32)

        ode_args = (
            theta_j,
            Cg_j,
            Cl_j,
            spi_j,
            K_sk_j,
            R_j,
            La_j,
            k2p_j,
            rmp_j,
            rmk_j,
            psi_j,
        )

        sol = diffrax.diffeqsolve(
            term,
            solver,
            t0=t0_val,
            t1=t1_val,
            dt0=dt0,
            y0=y0_j,
            args=ode_args,
            saveat=saveat,
            stepsize_controller=sctrl,
            max_steps=max_steps,
            throw=False,
            adjoint=adjoint
        )

        xs = sol.ys  # (T_unified, 3K+M+N) – new state layout

        # Sample at protein time indices
        xs_prot = xs[prot_idx_solver, :]
        # New slicing: [R_rna, S, A, Kdyn, p]
        P_sim = jnp.clip(xs_prot[:, 3 * K + M :], 0.0, None).T  # (N, T_prot)
        A_sim = jnp.clip(xs_prot[:, 2 * K : 3 * K], 0.0, 5.0).T  # (K, T_prot)

        f1, f2, f3 = compute_objectives_jax(
            theta_j,
            P_data_j,
            P_sim,
            A_scaled_j,
            A_sim,
            W_data_j,
            W_prot_j,
            prot_idx_j,
            La_loss_j,
            lambda_net,
            reg_lambda,
            n_p,
            n_A,
            n_var,
            K,
            M,
            N,
        )

        # f4: mRNA / R_rna loss (only when RNA data is available)
        # Use raw MSE (not log1p) to avoid overflow to inf when R_sim is far from obs.
        if has_mrna:
            xs_rna = xs[mrna_idx_j, :]
            R_sim_rna = jnp.clip(
                xs_rna[:, :K], 0.0, _RNA_CLIP_UPPER
            ).T  # (K, T_rna); clip to prevent float32 overflow
            R_sim_matched = R_sim_rna[rna_prot_idx_j, :]  # (n_match, T_rna)
            diff_R = rna_j - R_sim_matched
            f4 = jnp.sum(W_rna_j * diff_R * diff_R) / n_rna
        else:
            f4 = jnp.float32(0.0)

        total = (
            jnp.float32(w_phospho) * f1
            + jnp.float32(w_abundance) * f2
            + jnp.float32(w_reg) * f3
            + jnp.float32(w_mrna) * f4
        )

        # Penalise non-finite results without crashing
        total = jnp.where(jnp.isfinite(total), total, FAILED_SOLVE_PENALTY)
        return total, (f1, f2, f3, f4)

    return loss_fn


def make_residuals_fn(
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
    mechanism,
    lambda_net,
    reg_lambda,
    w_phospho=1.0,
    w_abundance=1.0,
    w_reg=1.0,
    rtol=1e-6,
    atol=1e-9,
    max_steps=16384,
    k_act_fn=None,
    s_prod_fn=None,
    t_mrna=None,
    rna_data_scaled=None,
    w_mrna=1.0,
    rna_model_prot_idx=None,
    rna_obs_idx=None,
    rna_fit_genes=None,
    R_data0=None,
    W_data_mrna=None,
    rna_relax=0.1,
    ode_solver_kind="tsit5",
    ode_adjoint_kind="forward",
    dt0=0.01,
    root_find_max_steps=10,
    xl=None,
    xu=None,
):
    """
    Build a JAX-differentiable residual-vector function for Optimistix least_squares.

    Follows the canonical Diffrax + Optimistix approach:
      - ODE solve inside the objective.
      - diffrax.Tsit5() solver.
      - diffrax.SaveAt(ts=t_eval) with explicit saved timepoints.
      - diffrax.DirectAdjoint() for differentiating through the ODE solve.
        (required because Optimistix LM uses forward-mode autodiff)

    The returned function ``residuals_fn(theta, args)`` is compatible with
    ``optimistix.least_squares``. It returns:
      - 1D finite residual vector (concatenation of all modality residuals)
      - auxiliary tuple (f1, f2, f3, f4) as diagnostics  [via has_aux=True]

    Residual blocks:
      sqrt(w_phospho * W_data)  * (P_sim - P_data)   phosphosite block
      sqrt(w_abundance * W_prot) * (A_sim - A_data)   abundance block
      sqrt(w_mrna * W_mrna)     * (R_sim - R_obs)     mRNA block
      sqrt(reg_lambda)          * theta                L2 regularisation
      sqrt(lambda_net)          * L_alpha @ alpha      Laplacian regularisation

    Failed ODE solves return a large finite penalty vector (not inf/nan).

    Parameters are frozen at creation time; only ``theta`` varies.

    State layout: y = [R_rna, S, A, Kdyn, p]  (dim = 3*K + M + N)
    """
    K, M, N = ModelDims.K, ModelDims.M, ModelDims.N

    n_p = max(1, P_data.size)
    n_A = max(1, A_scaled.size)
    n_var = 2 * K + 2 + 3 * M + N + 4

    prev_site_idx = compute_prev_site_idx(site_prot_idx.astype(np.int32), N)

    # Build unified time grid (protein ∪ mRNA)
    if t_mrna is not None and len(t_mrna) > 0:
        all_times = np.union1d(t, t_mrna)
    else:
        all_times = np.unique(t)
    all_times = np.sort(all_times).astype(np.float64)

    # Index maps
    prot_time_idx = np.searchsorted(all_times, t)
    if t_mrna is not None and len(t_mrna) > 0:
        mrna_time_idx = np.searchsorted(all_times, t_mrna)
    else:
        mrna_time_idx = None

    # Build initial state from data (layout: [R_rna, S, A, Kdyn, p])
    T_prot = P_data.shape[1]
    A0_full = build_full_A0(K, T_prot, A_scaled, prot_idx_for_A)

    x0 = np.zeros(3 * K + M + N, dtype=np.float64)
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

    # JAX static arrays
    Cg_j = jnp.asarray(Cg, dtype=jnp.float32)
    Cl_j = jnp.asarray(Cl, dtype=jnp.float32)
    K_sk_j = jnp.asarray(K_site_kin, dtype=jnp.float32)
    R_j = jnp.asarray(R, dtype=jnp.float32)
    La_j = jnp.asarray(L_alpha, dtype=jnp.float32)
    spi_j = jnp.asarray(site_prot_idx, dtype=jnp.int32)
    k2p_j = jnp.asarray(kin_to_prot_idx, dtype=jnp.int32)
    rmp_j = jnp.asarray(receptor_mask_prot, dtype=jnp.float32)
    rmk_j = jnp.asarray(receptor_mask_kin, dtype=jnp.float32)
    psi_j = jnp.asarray(prev_site_idx, dtype=jnp.int32)

    y0_j = jnp.asarray(x0, dtype=jnp.float32)
    t_eval = jnp.asarray(all_times, dtype=jnp.float32)

    P_data_j = jnp.asarray(P_data, dtype=jnp.float32)
    A_scaled_j = jnp.asarray(A_scaled, dtype=jnp.float32)
    prot_idx_j = jnp.asarray(prot_idx_for_A, dtype=jnp.int32)
    prot_idx_solver = jnp.asarray(prot_time_idx, dtype=jnp.int32)

    # Weight arrays with modality loss weights baked in as sqrt factors
    # so that ||sqrt(w*W)*(sim-obs)||^2 == w * sum(W * (sim-obs)^2)
    sqrt_wp = jnp.sqrt(jnp.float32(w_phospho)) * jnp.sqrt(
        jnp.asarray(W_data, dtype=jnp.float32)
    )
    has_abundance = A_scaled.size > 0
    if has_abundance:
        sqrt_wa = jnp.sqrt(jnp.float32(w_abundance)) * jnp.sqrt(
            jnp.asarray(W_data_prot, dtype=jnp.float32)
        )
    else:
        sqrt_wa = None

    # Convert weight arrays to JAX for use inside the JIT-traced residuals_fn
    W_data_j_diag = jnp.asarray(W_data, dtype=jnp.float32)
    W_prot_j_diag = (
        jnp.asarray(W_data_prot, dtype=jnp.float32) if has_abundance else None
    )

    # mRNA arrays
    has_mrna = (
        t_mrna is not None
        and rna_data_scaled is not None
        and len(t_mrna) > 0
        and mrna_time_idx is not None
        and rna_model_prot_idx is not None
        and len(rna_model_prot_idx) > 0
    )

    if has_mrna:
        assert t_mrna is not None
        rna_j = jnp.asarray(rna_data_scaled, dtype=jnp.float32)
        mrna_idx_j = jnp.asarray(mrna_time_idx, dtype=jnp.int32)
        rna_prot_idx_j = jnp.asarray(rna_model_prot_idx, dtype=jnp.int32)
        n_matched = len(rna_model_prot_idx)
        T_rna = len(t_mrna)
        W_rna_base = (
            np.asarray(W_data_mrna, dtype=np.float32)
            if W_data_mrna is not None
            else np.ones((n_matched, T_rna), dtype=np.float32)
        )
        W_rna_j_diag = jnp.asarray(W_rna_base, dtype=jnp.float32)
        sqrt_wr = jnp.sqrt(jnp.float32(w_mrna)) * jnp.sqrt(W_rna_j_diag)
        n_rna = max(1, rna_data_scaled.size)
    else:
        rna_j = mrna_idx_j = rna_prot_idx_j = sqrt_wr = W_rna_j_diag = None
        n_rna = 1

    # Regularisation residuals – fixed structure, no ODE needed
    sqrt_reg = jnp.float32(np.sqrt(float(reg_lambda)))
    has_net_reg = lambda_net > 0.0
    if has_net_reg:
        sqrt_lnet = jnp.float32(np.sqrt(float(lambda_net)))

    # Bounds for hard-clipping theta inside residuals_fn (prevents LM from escaping
    # the biological parameter space and producing stiff/divergent ODEs).
    has_bounds = xl is not None and xu is not None
    if has_bounds:
        xl_j = jnp.asarray(xl, dtype=jnp.float32)
        xu_j = jnp.asarray(xu, dtype=jnp.float32)

    rhs_fn = make_rhs(
        K, M, N, mechanism, k_act_fn=k_act_fn, s_prod_fn=s_prod_fn, rna_relax=rna_relax
    )

    term = diffrax.ODETerm(rhs_fn)
    ode_solver = make_diffrax_solver(
        ode_solver_kind,
        root_find_max_steps=root_find_max_steps
    )
    sctrl = make_stepsize_controller(rtol=rtol, atol=atol)
    saveat = diffrax.SaveAt(ts=t_eval)
    adjoint = make_diffrax_adjoint(ode_adjoint_kind)

    logger.info(f"Using Diffrax adjoint: {adjoint}")
    logger.info(f"Using Diffrax solver: {ode_solver}")

    t0_val = float(all_times[0])
    t1_val = float(all_times[-1])

    PENALTY = jnp.float32(_FAILED_SOLVE_PENALTY)

    def residuals_fn(theta, _args):
        """
        Compute residual vector and diagnostic loss components.

        Compatible with optx.least_squares(..., has_aux=True):
          returns (residuals_1d, (f1, f2, f3, f4))

        Uses diffrax.DirectAdjoint so that Optimistix LM can compute JVPs
        through the ODE solve without storing all intermediate states.
        """
        theta_j = jnp.asarray(theta, dtype=jnp.float32)

        # Clip theta to parameter bounds.  Optimistix LM is unconstrained by
        # default; without this guard the solver can drift to biologically
        # implausible values (e.g. alpha > 1000) that cause stiff/divergent ODEs.
        if has_bounds:
            theta_j = jnp.clip(theta_j, xl_j, xu_j)

        ode_args = (
            theta_j,
            Cg_j,
            Cl_j,
            spi_j,
            K_sk_j,
            R_j,
            La_j,
            k2p_j,
            rmp_j,
            rmk_j,
            psi_j,
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
            adjoint=adjoint,  # DirectAdjoint for forward-mode AD compatibility
            throw=False,
        )

        xs = sol.ys  # (T_unified, 3K+M+N)
        # A successful solve requires both finite states AND that the solver
        # did not stop early (e.g. hit max_steps with throw=False).
        solve_ok = jnp.all(jnp.isfinite(xs)) & (
            sol.result == diffrax.RESULTS.successful
        )

        # Sample at protein time indices (new layout: [R_rna, S, A, Kdyn, p])
        xs_prot = xs[prot_idx_solver, :]
        P_sim = jnp.clip(xs_prot[:, 3 * K + M :], 0.0, None).T  # (N, T_prot)
        A_sim = jnp.clip(xs_prot[:, 2 * K : 3 * K], 0.0, 5.0).T  # (K, T_prot)

        # --- Phosphosite residuals ---
        diff_p = P_sim - P_data_j  # (N, T_prot)
        r_phospho = (sqrt_wp * diff_p).ravel()  # (N*T_prot,)

        # --- f1 diagnostic (mean unweighted-by-modality MSE) ---
        f1 = jnp.sum(W_data_j_diag * diff_p * diff_p) / n_p

        # --- Abundance residuals ---
        if has_abundance:
            A_sim_obs = A_sim[prot_idx_j, :]  # (K_obs, T_prot)
            diff_A = A_sim_obs - A_scaled_j  # (K_obs, T_prot)
            r_abund = (sqrt_wa * diff_A).ravel()
            f2 = jnp.sum(W_prot_j_diag * diff_A * diff_A) / n_A
        else:
            r_abund = jnp.zeros(0, dtype=jnp.float32)
            f2 = jnp.float32(0.0)

        # --- mRNA residuals ---
        if has_mrna:
            xs_rna = xs[mrna_idx_j, :]
            # Clip to prevent float32 overflow; R_rna is on fold-change scale (~0-20)
            R_sim_rna = jnp.clip(xs_rna[:, :K], 0.0, _RNA_CLIP_UPPER).T  # (K, T_rna)
            R_sim_matched = R_sim_rna[rna_prot_idx_j, :]  # (n_match, T_rna)
            diff_R = R_sim_matched - rna_j  # (n_match, T_rna)
            r_rna = (sqrt_wr * diff_R).ravel()
            f4 = jnp.sum(W_rna_j_diag * diff_R * diff_R) / n_rna
        else:
            r_rna = jnp.zeros(0, dtype=jnp.float32)
            f4 = jnp.float32(0.0)

        # --- Regularisation residuals (no ODE needed) ---
        # L2 on theta
        r_reg_l2 = sqrt_reg * theta_j  # (n_var,)
        # Laplacian network regularisation on alpha (decoded from theta)
        if has_net_reg:
            alpha_raw = theta_j[2 * K + 2 : 2 * K + 2 + M]
            alpha = jnp.exp(jnp.clip(alpha_raw, -20.0, 10.0))
            r_reg_net = sqrt_lnet * (La_j @ alpha)  # (M,)
            r_reg = jnp.concatenate([r_reg_l2, r_reg_net])
        else:
            r_reg = r_reg_l2

        # f3 diagnostic
        f3_l2 = jnp.float32(reg_lambda) * jnp.dot(theta_j, theta_j)
        if has_net_reg:
            alpha_raw = theta_j[2 * K + 2 : 2 * K + 2 + M]
            alpha = jnp.exp(jnp.clip(alpha_raw, -20.0, 10.0))
            f3_net = jnp.float32(lambda_net) * jnp.dot(alpha, La_j @ alpha)
        else:
            f3_net = jnp.float32(0.0)
        f3 = (f3_l2 + f3_net) / jnp.float32(max(n_var, 1))

        # --- Concatenate residual vector ---
        residuals = jnp.concatenate([r_phospho, r_abund, r_rna, r_reg])

        # --- Replace non-finite residuals with finite penalty ---
        # This handles ODE solve failures gracefully without crashing the optimizer.
        finite_residuals = jnp.where(jnp.isfinite(residuals), residuals, PENALTY)
        finite_residuals = jnp.where(
            solve_ok, finite_residuals, jnp.full_like(finite_residuals, PENALTY)
        )

        # Recalculate finite diagnostics for aux output
        f1 = jnp.where(jnp.isfinite(f1), f1, jnp.float32(1e6))
        f2 = jnp.where(jnp.isfinite(f2), f2, jnp.float32(1e6))
        f3 = jnp.where(jnp.isfinite(f3), f3, jnp.float32(1e6))
        f4 = jnp.where(jnp.isfinite(f4), f4, jnp.float32(1e6))

        return finite_residuals, (f1, f2, f3, f4)

    return residuals_fn

def run_single_optimisation(
    residuals_fn,
    theta0,
    max_steps: int = 500,
    rtol: float = 1e-8,
    atol: float = 1e-8,
    verbose: bool = False,
    *,
    ls_solver: str = "lm",
    optx_adjoint: str = "implicit",
    jac_mode: str = "fwd",
):
    """
    Run a single Optimistix least-squares optimisation of the parameter vector.

    Canonical setup:
      - Residual-vector objective via optimistix.least_squares
      - Gauss-Newton-type solver (Levenberg-Marquardt by default)
      - Optional alternative solvers and adjoints, switchable via strings

    The ``residuals_fn`` must have signature::

        residuals_fn(theta, args) -> (residuals_1d, (f1, f2, f3, f4))

    as returned by ``make_residuals_fn``.

    Parameters
    ----------
    residuals_fn : callable
        (theta, args) -> (1D residuals, (f1, f2, f3, f4)).
    theta0 : np.ndarray
        Starting parameter vector.
    max_steps : int
        Optimistix iteration cap (not ODE steps).
    rtol, atol : float
        Optimistix convergence tolerances.
    verbose : bool
        Enable per-step progress logging in the solver.
    ls_solver : {"lm", "indirect_lm", "dogleg", "gauss_newton"}
        Least-squares solver type. Default "lm" reproduces previous behaviour.
    optx_adjoint : {"implicit", "checkpoint"}
        Optimistix adjoint used to differentiate through the fixed-point solve.
        Default "implicit" is Optimistix's recommended choice.
    jac_mode : {"fwd", "rev"}
        Jacobian mode for Optimistix. Default "fwd" is recommended for most cases.

    Returns
    -------
    theta_opt : np.ndarray
        Best-fit parameter vector (float64).
    total_loss : float
        f1 + f2 + f3 + f4 (diagnostic sum; per-modality weights already included
        in the residuals).
    f1, f2, f3, f4 : float
        Diagnostic loss components.
    """
    # Construct solver and adjoint from simple string flags.
    solver = make_ls_solver(ls_solver, rtol=rtol, atol=atol, verbose=verbose)
    adjoint = make_optx_adjoint(optx_adjoint)

    if verbose:
        logger.info(f"Using Optimistix adjoint: {optx_adjoint}")
        logger.info(f"Using Optimistix solver: {ls_solver}")
        logger.info(f"Using Optimistix Jacobian mode: {jac_mode}")

    sol = optx.least_squares(
        residuals_fn,
        solver,
        jnp.asarray(theta0, dtype=jnp.float32),
        args=None,
        options={"jac": jac_mode},
        has_aux=True,
        max_steps=max_steps,
        adjoint=adjoint,
        throw=False,
    )
    theta_opt = np.asarray(sol.value, dtype=np.float64)

    # Recompute diagnostics at the optimal point (aux from the last solver step)
    _, (f1, f2, f3, f4) = residuals_fn(sol.value, None)
    f1, f2, f3, f4 = float(f1), float(f2), float(f3), float(f4)
    total_loss = f1 + f2 + f3 + f4  # diagnostic sum; modality weights are in residuals

    return theta_opt, total_loss, f1, f2, f3, f4


# ---------------------------------------------------------------------------
# Problem shape validation
# ---------------------------------------------------------------------------


def validate_problem_shapes(problem):
    """
    Validate all array shapes in a NetworkProblem before starting optimisation.

    Raises
    ------
    ValueError  if any shape invariant is violated or required arrays are None.

    Parameters
    ----------
    problem : NetworkProblem
    """
    K, M, N = ModelDims.K, ModelDims.M, ModelDims.N
    problem.P_data.shape[1]

    errors = []

    def _chk(cond, msg):
        if not cond:
            errors.append(msg)

    # Core shape checks
    _chk(
        problem.P_data.shape == problem.W_data.shape,
        f"P_data.shape {problem.P_data.shape} != W_data.shape {problem.W_data.shape}",
    )
    if problem.A_scaled.size > 0:
        _chk(
            problem.A_scaled.shape == problem.W_data_prot.shape,
            f"A_scaled.shape {problem.A_scaled.shape} != W_data_prot.shape {problem.W_data_prot.shape}",  # noqa: E501
        )
    _chk(
        problem.K_site_kin.shape == (N, M),
        f"K_site_kin.shape {problem.K_site_kin.shape} != (N={N}, M={M})",
    )
    _chk(
        problem.R.shape == (M, N),
        f"R.shape {problem.R.shape} != (M={M}, N={N})",
    )
    _chk(
        problem.Cg.shape == (N, N),
        f"Cg.shape {problem.Cg.shape} != (N={N}, N={N})",
    )
    _chk(
        problem.Cl.shape == (N, N),
        f"Cl.shape {problem.Cl.shape} != (N={N}, N={N})",
    )
    _chk(
        problem.L_alpha.shape == (M, M),
        f"L_alpha.shape {problem.L_alpha.shape} != (M={M}, M={M})",
    )
    _chk(
        len(problem.site_prot_idx) == N,
        f"len(site_prot_idx)={len(problem.site_prot_idx)} != N={N}",
    )
    if len(problem.site_prot_idx) > 0:
        _chk(
            int(problem.site_prot_idx.max()) < K,
            f"max(site_prot_idx)={problem.site_prot_idx.max()} >= K={K}",
        )
    _chk(
        len(problem.kin_to_prot_idx) == M,
        f"len(kin_to_prot_idx)={len(problem.kin_to_prot_idx)} != M={M}",
    )

    # RNA shape checks
    has_rna = (
        problem.t_rna is not None
        and problem.rna_obs_matched is not None
        and problem.rna_model_prot_idx is not None
        and getattr(problem, "rna_fit_genes", None) is not None
        and len(getattr(problem, "rna_fit_genes", [])) > 0
    )
    if has_rna:
        T_rna = len(problem.t_rna)
        obs_shape = problem.rna_obs_matched.shape
        _chk(
            obs_shape[1] == T_rna,
            f"rna_obs_matched.shape[1]={obs_shape[1]} != len(t_rna)={T_rna}",
        )
        if problem.W_data_mrna is not None:
            _chk(
                obs_shape == problem.W_data_mrna.shape,
                f"rna_obs_matched.shape {obs_shape} != W_data_mrna.shape {problem.W_data_mrna.shape}",  # noqa: E501
            )
        n_matched = obs_shape[0]
        _chk(
            len(problem.rna_model_prot_idx) == n_matched,
            f"len(rna_model_prot_idx)={len(problem.rna_model_prot_idx)} != rna_obs_matched.shape[0]={n_matched}",  # noqa: E501
        )
        if len(problem.rna_model_prot_idx) > 0:
            _chk(
                int(np.asarray(problem.rna_model_prot_idx).max()) < K,
                f"max(rna_model_prot_idx)={np.asarray(problem.rna_model_prot_idx).max()} >= K={K}",  # noqa: E501
            )

    # Finiteness checks on core arrays
    for name, arr in [
        ("P_data", problem.P_data),
        ("W_data", problem.W_data),
    ]:
        n_bad = int((~np.isfinite(arr)).sum())
        if n_bad > 0:
            errors.append(f"{name} has {n_bad} non-finite value(s) before optimization")

    if errors:
        raise ValueError(
            "validate_problem_shapes found errors:\n  " + "\n  ".join(errors)
        )


# ---------------------------------------------------------------------------
# Biological input validation
# ---------------------------------------------------------------------------


def validate_biological_inputs(
    P_data=None,
    A_scaled=None,
    rna_data_scaled=None,
    W_data=None,
    W_data_prot=None,
    theta=None,
    K=None,
    M=None,
    N=None,
):
    """
    Defensive validation of biological inputs before fitting.

    Checks
    ------
    * All observed data matrices contain only finite values.
    * Observed relative phosphosite signal (P_data) is non-negative;
      negative values are invalid for this model.
    * Observed protein abundance (A_scaled) values are non-negative
      (fold-change data must be ≥ 0).
    * Observed mRNA data (rna_data_scaled) values are non-negative;
      negative fold-change values are invalid for this model.
    * All loss weight matrices are finite and non-negative.
    * When a decoded ``theta`` vector is provided together with K, M, N,
      the positive-definite rate parameters are finite and > 0, and the
      gamma parameters are finite.

    Parameters
    ----------
    P_data          : np.ndarray | None – (N, T) relative phosphosite signal data.
    A_scaled        : np.ndarray | None – (K_obs, T) protein abundance data.
    rna_data_scaled : np.ndarray | None – (n_genes, T_rna) mRNA fold-change.
    W_data          : np.ndarray | None – per-element phosphosite weights.
    W_data_prot     : np.ndarray | None – per-element abundance weights.
    theta           : np.ndarray | None – flat parameter vector.
    K, M, N         : int | None        – model dimensions for theta decoding.

    Raises
    ------
    ValueError  on the first detected violation (with a descriptive message).
    """
    errors = []

    def _chk_finite(name, arr):
        arr = np.asarray(arr, dtype=float)
        n_bad = int((~np.isfinite(arr)).sum())
        if n_bad > 0:
            errors.append(
                f"{name} contains {n_bad} non-finite value(s) (NaN or Inf). "
                "Replace or impute these before fitting."
            )

    def _chk_nonneg(name, arr):
        arr = np.asarray(arr, dtype=float)
        finite = arr[np.isfinite(arr)]
        if finite.size > 0 and float(finite.min()) < 0.0:
            n_neg = int((finite < 0.0).sum())
            raise ValueError(
                f"{name} contains {n_neg} negative value(s). "
                "Negative fold-change values are invalid for this model. "
                "Transform or correct the data before fitting."
            )

    # --- Observed data arrays ---
    if P_data is not None:
        _chk_finite("P_data", P_data)
        _chk_nonneg("P_data (relative phosphosite signal)", P_data)
    if A_scaled is not None and np.asarray(A_scaled).size > 0:
        _chk_finite("A_scaled", A_scaled)
        _chk_nonneg("A_scaled (protein abundance)", A_scaled)
    if rna_data_scaled is not None and np.asarray(rna_data_scaled).size > 0:
        _chk_finite("rna_data_scaled", rna_data_scaled)
        _chk_nonneg("rna_data_scaled (mRNA fold-change)", rna_data_scaled)

    # --- Loss weights ---
    for wname, warr in [("W_data", W_data), ("W_data_prot", W_data_prot)]:
        if warr is not None and np.asarray(warr).size > 0:
            warr_np = np.asarray(warr, dtype=float)
            _chk_finite(wname, warr_np)
            finite_w = warr_np[np.isfinite(warr_np)]
            if finite_w.size > 0 and float(finite_w.min()) < 0.0:
                errors.append(
                    f"{wname} contains negative weight(s). "
                    "Loss weights must be non-negative."
                )

    # --- Decoded parameter check ---
    if theta is not None and K is not None and M is not None and N is not None:
        from phoscrosstalk.core_mechanisms import decode_theta

        (
            k_deact,
            d_deg,
            beta_g,
            beta_l,
            alpha,
            kK_act,
            kK_deact,
            k_off,
            gamma_S_p,
            gamma_A_S,
            gamma_A_p,
            gamma_K_net,
        ) = decode_theta(np.asarray(theta, dtype=np.float64), K, M, N)

        for pname, parr in [
            ("k_deact", k_deact),
            ("d_deg", d_deg),
            ("alpha", alpha),
            ("kK_act", kK_act),
            ("kK_deact", kK_deact),
            ("k_off", k_off),
        ]:
            parr_np = np.asarray(parr, dtype=float)
            if not np.all(np.isfinite(parr_np)):
                errors.append(
                    f"Decoded parameter '{pname}' contains non-finite values."
                )
            elif float(parr_np.min()) < 1e-15:
                errors.append(
                    f"Decoded parameter '{pname}' has non-positive value(s) "
                    f"(min={float(parr_np.min()):.3g}). Rate parameters must be > 0."
                )

        for sname, sval in [
            ("beta_g", beta_g),
            ("beta_l", beta_l),
            ("gamma_S_p", gamma_S_p),
            ("gamma_A_S", gamma_A_S),
            ("gamma_A_p", gamma_A_p),
            ("gamma_K_net", gamma_K_net),
        ]:
            if not np.isfinite(float(sval)):
                errors.append(f"Decoded parameter '{sname}' is non-finite.")

    if errors:
        raise ValueError(
            "validate_biological_inputs found issues:\n  " + "\n  ".join(errors)
        )


class NetworkProblem:
    """
    Minimal problem wrapper that preserves the .simulate() interface used by
    steadystate, knockouts, sensitivity, and app modules.

    Does NOT inherit from pymoo.  The _evaluate / optimisation logic has moved
    to make_loss_fn + run_single_optimisation.

    RNA-related attributes (optional):
        t_rna             : np.ndarray | None – RNA time points
        rna_obs_matched   : np.ndarray | None – observed RNA for matched proteins (n_match, T_rna)
        rna_model_prot_idx: np.ndarray | None – protein indices for matched RNA rows
        rna_obs_idx       : np.ndarray | None – gene indices in gene_ids for matched rows
        rna_fit_genes     : list | None       – matched gene/protein names
        loss_weight_rna   : float             – RNA loss weight
        R_data0           : np.ndarray | None – RNA initial condition (K, T_rna or K,)
    """  # noqa: E501

    def __init__(
        self,
        t,
        P_data,
        Cg,
        Cl,
        site_prot_idx,
        K_site_kin,
        R,
        A_scaled,
        prot_idx_for_A,
        W_data,
        W_data_prot,
        L_alpha,
        kin_to_prot_idx,
        lambda_net,
        reg_lambda,
        receptor_mask_prot,
        receptor_mask_kin,
        mechanism,
        xl,
        xu,
        k_act_fn=None,
        s_prod_fn=None,
        t_rna=None,
        rna_obs_matched=None,
        rna_model_prot_idx=None,
        rna_obs_idx=None,
        rna_fit_genes=None,
        loss_weight_rna=1.0,
        R_data0=None,
        rna_relax=0.1,
        W_data_mrna=None,
        ode_solver_kind="tsit5",
        ode_dt0=0.01,
        ode_root_find_max_steps=10,
        ode_adjoint_kind="forward",
        rtol=1e-6,
        atol=1e-9,
        max_steps=16384,
        **kwargs,  # absorb legacy keyword args (elementwise_runner, etc.)
    ):
        self.t = t
        self.P_data = P_data
        self.Cg = Cg
        self.Cl = Cl
        self.site_prot_idx = site_prot_idx
        self.K_site_kin = K_site_kin
        self.R = R
        self.A_scaled = A_scaled
        self.prot_idx_for_A = prot_idx_for_A
        self.W_data = W_data
        self.W_data_prot = W_data_prot
        self.W_data_mrna = W_data_mrna
        self.L_alpha = L_alpha
        self.kin_to_prot_idx = kin_to_prot_idx
        self.lambda_net = lambda_net
        self.reg_lambda = reg_lambda
        self.receptor_mask_prot = receptor_mask_prot
        self.receptor_mask_kin = receptor_mask_kin
        self.mechanism = mechanism
        self.xl = xl
        self.xu = xu
        self.k_act_fn = k_act_fn
        self.s_prod_fn = s_prod_fn
        self.rna_relax = rna_relax
        # RNA-specific
        self.t_rna = t_rna
        self.rna_obs_matched = rna_obs_matched
        self.rna_model_prot_idx = rna_model_prot_idx
        self.rna_obs_idx = rna_obs_idx
        self.rna_fit_genes = rna_fit_genes
        self.loss_weight_rna = loss_weight_rna
        self.R_data0 = R_data0
        self.ode_solver_kind = ode_solver_kind
        self.ode_dt0 = ode_dt0
        self.ode_root_find_max_steps = ode_root_find_max_steps
        self.ode_adjoint_kind = ode_adjoint_kind
        self.rtol = rtol
        self.atol = atol
        self.max_steps = max_steps

    def simulate(self, x):
        """
        Run a simulation for parameter vector x and return phosphosite trajectories.

        Args:
            x (np.ndarray): Parameter vector.

        Returns:
            np.ndarray: P_sim (N_sites x T).
        """
        theta = np.asarray(x, dtype=np.float64)
        K, T = ModelDims.K, self.P_data.shape[1]
        A0 = build_full_A0(K, T, self.A_scaled, self.prot_idx_for_A)

        P_sim, _A_sim = simulate_ode(
            self.t,
            self.P_data,
            A0,
            theta,
            self.Cg,
            self.Cl,
            self.site_prot_idx,
            self.K_site_kin,
            self.R,
            self.L_alpha,
            self.kin_to_prot_idx,
            self.receptor_mask_prot,
            self.receptor_mask_kin,
            self.mechanism,
            k_act_fn=self.k_act_fn,
            s_prod_fn=self.s_prod_fn,
            t_rna=self.t_rna,
            R_data0=self.R_data0,
            rna_relax=self.rna_relax,
            ode_adjoint_kind=self.ode_adjoint_kind,
            root_find_max_steps=self.ode_root_find_max_steps,
            ode_solver_kind=self.ode_solver_kind,
            dt0=self.ode_dt0,
            rtol=self.rtol,
            atol=self.atol,
            max_steps=self.max_steps,
        )
        return P_sim

    def simulate_full(self, x):
        """
        Run a full simulation returning all state components including R_sim_rna.

        Args:
            x (np.ndarray): Parameter vector.

        Returns:
            dict: Keys: P_sim, A_sim, S_sim, Kdyn_sim, R_sim, R_sim_rna, t, t_rna,
                solver_times.
        """
        theta = np.asarray(x, dtype=np.float64)
        K, T = ModelDims.K, self.P_data.shape[1]
        A0 = build_full_A0(K, T, self.A_scaled, self.prot_idx_for_A)

        return simulate_ode(
            self.t,
            self.P_data,
            A0,
            theta,
            self.Cg,
            self.Cl,
            self.site_prot_idx,
            self.K_site_kin,
            self.R,
            self.L_alpha,
            self.kin_to_prot_idx,
            self.receptor_mask_prot,
            self.receptor_mask_kin,
            self.mechanism,
            return_full=True,
            k_act_fn=self.k_act_fn,
            s_prod_fn=self.s_prod_fn,
            t_rna=self.t_rna,
            R_data0=self.R_data0,
            rna_relax=self.rna_relax,
            ode_solver_kind=self.ode_solver_kind,
            dt0=self.ode_dt0,
            root_find_max_steps=self.ode_root_find_max_steps,
            ode_adjoint_kind=self.ode_adjoint_kind,
            rtol=self.rtol,
            atol=self.atol,
            max_steps=self.max_steps,
        )


# Legacy alias so that any remaining code that imports NetworkOptimizationProblem
# still works without crashing.
NetworkOptimizationProblem = NetworkProblem


# ---------------------------------------------------------------------------
# Second-order (Hessian) sensitivity analysis
# ---------------------------------------------------------------------------


def compute_second_order_sensitivities(
    theta: np.ndarray,
    loss_fn: Callable[[jnp.ndarray, object], tuple],
    param_labels: Sequence[str],
    out_dir: str | os.PathLike,
    prefix: str = "loss_hessian",
    jit: bool = True,
) -> np.ndarray:
    """
    Compute and save the Hessian of the scalarised loss w.r.t. the parameter vector.

    Uses ``jax.hessian`` applied to the scalar output of *loss_fn*.  The loss
    function must have been built with :func:`make_loss_fn` (which uses
    ``Tsit5(scan_kind="bounded")`` to support higher-order autodiff through
    Diffrax).

    Parameters
    ----------
    theta : np.ndarray
        Flattened parameter vector at which to evaluate the Hessian
        (e.g. the optimised point).  Shape ``(n,)``.
    loss_fn : Callable[[jnp.ndarray, Any], tuple]
        Scalar loss function with signature ``loss_fn(theta, args) ->
        (total_loss, aux)``, as returned by :func:`make_loss_fn`.
    param_labels : Sequence[str]
        Human-readable labels for each element of *theta*, e.g. as
        returned by :func:`build_parameter_labels`.  Must have the same
        length as *theta*.
    out_dir : str or os.PathLike
        Directory where output files are written.  Created if absent.
    prefix : str, optional
        Base name (without extension) for all output files.
        Default ``"loss_hessian"``.
    jit : bool, optional
        When ``True`` (default) the Hessian function is wrapped with
        ``jax.jit`` before evaluation, which is faster for repeated calls.

    Returns
    -------
    np.ndarray
        Hessian matrix as a float64 NumPy array of shape
        ``(len(param_labels), len(param_labels))``.

    Outputs
    -------
    ``<out_dir>/<prefix>.npy``
        NumPy binary containing the Hessian matrix.
    ``<out_dir>/<prefix>.tsv``
        Tab-separated matrix with parameter-label row/column headers and
        values formatted as ``{val:.6e}``.
    ``<out_dir>/<prefix>_heatmap.png``
        Heatmap visualisation saved at 200 dpi.

    Notes
    -----
    * Higher-order autodiff through Diffrax requires the solver to use
      ``scan_kind="bounded"`` (i.e. ``Tsit5(scan_kind="bounded")``).
      :func:`make_loss_fn` already sets this; other solver paths (residuals,
      LM) are **not** modified.
    * The Hessian is computed in float32 (matching JAX's default) and
      immediately up-cast to float64 for numerical consistency.
    * For ``n > 40`` parameter labels the heatmap omits dense tick labels
      to remain readable.
    """
    theta_j = jnp.asarray(theta, dtype=jnp.float32)
    n = len(param_labels)

    def _loss_only(th):
        total, _aux = loss_fn(th, None)
        return total

    hess_fn = jax.hessian(_loss_only)
    if jit:
        hess_fn = jax.jit(hess_fn)

    H_raw = hess_fn(theta_j)
    H = np.asarray(H_raw, dtype=np.float64)

    if H.shape != (n, n):
        raise ValueError(
            f"Hessian shape {H.shape} does not match "
            f"len(param_labels)={n}.  Ensure theta and param_labels "
            "have the same length."
        )

    out_path = pathlib.Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # --- .npy ---
    np.save(out_path / f"{prefix}.npy", H)

    # --- .tsv ---
    tsv_path = out_path / f"{prefix}.tsv"
    with tsv_path.open("w") as fh:
        # Header row: tab + tab-separated column labels
        fh.write("\t" + "\t".join(param_labels) + "\n")
        for i, row_label in enumerate(param_labels):
            row_vals = "\t".join(f"{v:.6e}" for v in H[i])
            fh.write(f"{row_label}\t{row_vals}\n")

    # --- heatmap ---
    fig, ax = plt.subplots(figsize=(max(6, n // 4), max(5, n // 4)))
    im = ax.imshow(H, aspect="auto")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("∂² loss / ∂θ_i ∂θ_j")
    ax.set_title("Hessian of scalarised loss wrt parameters")
    ax.set_xlabel("parameters")
    ax.set_ylabel("parameters")
    if n <= 40:
        ax.set_xticks(range(n))
        ax.set_xticklabels(param_labels, rotation=90, fontsize=6)
        ax.set_yticks(range(n))
        ax.set_yticklabels(param_labels, fontsize=6)
    else:
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
    fig.savefig(out_path / f"{prefix}_heatmap.png", dpi=200)
    plt.close(fig)

    return H
