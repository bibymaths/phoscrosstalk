# SPDX-License-Identifier: MIT
"""
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
from functools import partial
from collections.abc import Callable, Sequence

import diffrax
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optimistix as optx

from phoscrosstalk.config import ModelDims
from phoscrosstalk.mechanisms import (
    compute_prev_site_idx,
    make_rhs, decode_theta,
)
from phoscrosstalk.logger import get_logger
from phoscrosstalk.simulation import build_full_A0, simulate
from phoscrosstalk.solver_config import (
    make_diffrax_adjoint,
    make_diffrax_solver,
    make_ls_solver,
    make_optx_adjoint,
    make_stepsize_controller,
)

logger = get_logger()

# Module-level standard logger used by jax.debug.callback (must be a plain
# logging.Logger — jax.debug.callback executes the callback outside the trace).
_debug_logger = get_logger().logger


def _log_residuals_step(total_loss):
    """Plain Python callback for jax.debug.callback — logs the scalar total loss."""
    _debug_logger.info("[fit]  step  loss=%.4e", float(total_loss))


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------


# Upper bound for clipping R_rna (fold-change scale).
# RNA fold-change values >20 are biologically implausible.
# This bound is a soft cap that still allows the optimizer to distinguish signals.
_RNA_CLIP_UPPER: float = 20.0

# Per-element penalty value returned when the ODE solve fails or produces non-finite
# states.  Large enough to guide the optimizer away from bad regions.
_FAILED_SOLVE_PENALTY: float = 1e3


@partial(jax.jit, static_argnames=("K", "M", "N"))
def bio_score_jax(theta, K: int, M: int, N: int):
    """
    JAX Biological Plausibility Score.

    K, M, and N must be static because decode_theta slices theta using
    these dimensions.
    """
    (
        k_deact,
        d_deg,
        _,
        _,
        _,
        kK_act,
        kK_deact,
        _,
        _,
        _,
        _,
        _,
    ) = decode_theta(theta, K, M, N)

    eps = jnp.asarray(1e-12, dtype=theta.dtype)

    t_half_kinase = jnp.log(jnp.asarray(2.0, dtype=theta.dtype)) / jnp.maximum(
        kK_deact, eps
    )
    t_half_protein = jnp.log(jnp.asarray(2.0, dtype=theta.dtype)) / jnp.maximum(
        d_deg, eps
    )

    median_t_kinase = jnp.median(t_half_kinase)
    median_t_protein = jnp.median(t_half_protein)

    score_kinase = (
                           jnp.log10(jnp.maximum(median_t_kinase, eps))
                           - jnp.log10(jnp.asarray(10.0, dtype=theta.dtype))
                   ) ** 2

    score_protein = (
                            jnp.log10(jnp.maximum(median_t_protein, eps))
                            - jnp.log10(jnp.asarray(600.0, dtype=theta.dtype))
                    ) ** 2

    return score_kinase + score_protein


def bio_score(theta, dims: ModelDims | None = None):
    """
    NumPy/Python wrapper used by analysis code.

    Keep this non-jitted wrapper so callers can pass NumPy arrays and receive
    a normal Python float.
    """
    if dims is None:
        dims = ModelDims.current()
        if dims is None:
            return float("nan")
    theta = jnp.asarray(theta)
    return float(
        bio_score_jax(
            theta,
            K=int(dims.K),
            M=int(dims.M),
            N=int(dims.N),
        )
    )


def create_bounds(K, M, N, bounds=None):
    """
    Create lower and upper bound vectors for the optimization parameter space.

    The optimized parameter vector is stored in transformed coordinates. Positive
    rate parameters are represented on a log scale, while the four gamma
    parameters are represented as raw signed values used by the parameter
    decoder.

    ``k_act`` and ``s_prod`` are not optimization variables. They are derived
    from input data through the rate-construction functions. Therefore, the
    parameter dimension is:

    ``2 * K + 2 + 3 * M + N + 4``

    The parameter blocks are ordered as:

    - ``k_deact``: protein signalling deactivation rates, length ``K``.
    - ``d_deg``: protein degradation rates, length ``K``.
    - ``beta_g``: global crosstalk coupling strength, scalar.
    - ``beta_l``: local crosstalk coupling strength, scalar.
    - ``alpha``: kinase-site activation strengths, length ``M``.
    - ``kK_act``: kinase activation rates, length ``M``.
    - ``kK_deact``: kinase deactivation rates, length ``M``.
    - ``k_off``: phosphosite dephosphorylation rates, length ``N``.
    - ``gamma`` values: signed regulatory coupling parameters, length ``4``.

    Args:
        K (int):
            Number of model proteins.

        M (int):
            Number of kinases.

        N (int):
            Number of phosphosites.

        bounds (types.SimpleNamespace | None, optional):
            Optional configuration namespace corresponding to the ``[bounds]``
            section of ``config.toml``. When provided, the following attributes
            override the built-in defaults:

            - ``rate_min``: lower bound for ALL positive rate parameters.
            - ``rate_max``: generic upper bound; only used where no dedicated
              ceiling exists.
            - ``protein_degradation_max``: upper bound for ``d_deg``.
            - ``k_deact_max``: upper bound for ``k_deact``; separate from
              ``rate_max`` so deactivation rates can be constrained tightly.
            - ``kinase_rate_max``: upper bound for ``kK_act`` and ``kK_deact``.
            - ``phosphatase_rate_max``: upper bound for ``k_off``.
            - ``beta_coupling_max``: upper bound for ``beta_g`` and ``beta_l``;
              separate from ``rate_max`` to prevent large crosstalk coupling.
            - ``alpha_min``: lower bound for ``alpha``; prevents collapse near
              zero which would silence kinase contributions entirely.
            - ``gamma_abs_max``: absolute bound for the raw gamma parameters
              (tanh-encoded; NOT log-space).

            Attributes not consumed here (no effect on ``create_bounds``):

            - ``rna_max``: hard clip for ``R_rna`` in ``make_rhs()`` / simulation
              code; NOT a parameter bound and NOT used by ``create_bounds()``.
            - ``abundance_max``: hard clip for protein abundance ``A`` in
              ``make_rhs()`` / simulation code; NOT a parameter bound and NOT
              used by ``create_bounds()``.

            If ``bounds`` is ``None`` or an attribute is missing, the function
            falls back to hard-coded defaults.

    Returns:
        (tuple): ``(xl, xu, dim)`` where ``xl`` is the lower-bound vector of shape
            ``(dim,)``, ``xu`` is the upper-bound vector of shape ``(dim,)``, and
            ``dim`` is the total number of optimization variables.
            Bounds for positive biological rates are in log space; bounds for gamma
            parameters are in raw parameter space.
    """
    # Resolve bound values from config or fall back to hard-coded defaults.
    if bounds is not None:
        _rate_min = float(getattr(bounds, "rate_min", 1e-5))
        _rate_max = float(getattr(bounds, "rate_max", 10.0))
        _ddeg_max = float(getattr(bounds, "protein_degradation_max", 0.5))
        _kdeact_max = float(getattr(bounds, "k_deact_max", 2.0))
        _kin_max = float(getattr(bounds, "kinase_rate_max", 3.0))
        _phos_max = float(getattr(bounds, "phosphatase_rate_max", 5.0))
        _beta_max = float(getattr(bounds, "beta_coupling_max", 3.0))
        _alpha_min = float(getattr(bounds, "alpha_min", 0.01))
        _gamma_max = float(getattr(bounds, "gamma_abs_max", 3.0))
        # rna_max is consumed by make_rhs() as a clipping value, not by create_bounds().
        # abundance_max is consumed by make_rhs() / simulation code as a clipping value,
        # not by create_bounds().
    else:
        _rate_min = 1e-5
        _rate_max = 10.0
        _ddeg_max = 0.5
        _kdeact_max = 2.0
        _kin_max = 3.0
        _phos_max = 5.0
        _beta_max = 3.0
        _alpha_min = 0.01
        _gamma_max = 3.0

    dim = 2 * K + 2 + 3 * M + N + 4
    xl, xu = np.zeros(dim), np.zeros(dim)
    idx = 0
    # Protein: k_deact, d_deg (k_act and s_prod removed – derived from data)
    # k_deact — dedicated ceiling separate from generic rate_max
    xl[idx: idx + K] = np.log(_rate_min)
    xu[idx: idx + K] = np.log(_kdeact_max)
    idx += K  # k_deact
    # d_deg (restricted upper bound for biological plausibility)
    xl[idx: idx + K] = np.log(_rate_min)
    xu[idx: idx + K] = np.log(_ddeg_max)
    idx += K
    # Coupling — dedicated beta_coupling_max ceiling, not generic rate_max
    xl[idx] = np.log(_rate_min)
    xu[idx] = np.log(_beta_max)
    idx += 1  # beta_g
    xl[idx] = np.log(_rate_min)
    xu[idx] = np.log(_beta_max)
    idx += 1  # beta_l
    # Kinase: alpha, kK_act, kK_deact
    # alpha — dedicated lower bound to prevent collapse near zero
    xl[idx: idx + M] = np.log(_alpha_min)
    xu[idx: idx + M] = np.log(_rate_max)
    idx += M  # alpha
    xl[idx: idx + M] = np.log(_rate_min)
    xu[idx: idx + M] = np.log(_kin_max)
    idx += M
    xl[idx: idx + M] = np.log(_rate_min)
    xu[idx: idx + M] = np.log(_kin_max)
    idx += M
    # Site: k_off
    xl[idx: idx + N] = np.log(_rate_min)
    xu[idx: idx + N] = np.log(_phos_max)
    idx += N
    # Gammas (tanh raw)
    xl[idx: idx + 4] = -_gamma_max
    xu[idx: idx + 4] = _gamma_max
    idx += 4
    return xl, xu, dim


def bounds_to_original_scale(xl, xu, K, M, N):
    """
    Convert transformed optimizer bounds back to original biological scale.

    Positive rate parameters are log-transformed, so use exp().
    Gamma parameters are raw signed values, so keep them unchanged.
    """

    xl = np.asarray(xl, dtype=float)
    xu = np.asarray(xu, dtype=float)

    xl_orig = xl.copy()
    xu_orig = xu.copy()

    idx = 0

    # k_deact, length K
    xl_orig[idx: idx + K] = np.exp(xl[idx: idx + K])
    xu_orig[idx: idx + K] = np.exp(xu[idx: idx + K])
    idx += K

    # d_deg, length K
    xl_orig[idx: idx + K] = np.exp(xl[idx: idx + K])
    xu_orig[idx: idx + K] = np.exp(xu[idx: idx + K])
    idx += K

    # beta_g, scalar
    xl_orig[idx] = np.exp(xl[idx])
    xu_orig[idx] = np.exp(xu[idx])
    idx += 1

    # beta_l, scalar
    xl_orig[idx] = np.exp(xl[idx])
    xu_orig[idx] = np.exp(xu[idx])
    idx += 1

    # alpha, length M
    xl_orig[idx: idx + M] = np.exp(xl[idx: idx + M])
    xu_orig[idx: idx + M] = np.exp(xu[idx: idx + M])
    idx += M

    # kK_act, length M
    xl_orig[idx: idx + M] = np.exp(xl[idx: idx + M])
    xu_orig[idx: idx + M] = np.exp(xu[idx: idx + M])
    idx += M

    # kK_deact, length M
    xl_orig[idx: idx + M] = np.exp(xl[idx: idx + M])
    xu_orig[idx: idx + M] = np.exp(xu[idx: idx + M])
    idx += M

    # k_off, length N
    xl_orig[idx: idx + N] = np.exp(xl[idx: idx + N])
    xu_orig[idx: idx + N] = np.exp(xu[idx: idx + N])
    idx += N

    # gamma values, length 4
    # Already raw signed values. Do not exponentiate.
    idx += 4

    assert idx == len(xl), f"Parameter dimension mismatch: idx={idx}, len(xl)={len(xl)}"

    return xl_orig, xu_orig


def build_parameter_labels(K: int, M: int, N: int) -> list[str]:
    """
    Return human-readable labels for every element of the flattened theta vector.

    The theta vector layout (length ``2*K + 2 + 3*M + N + 4``) is:

    ============  =====================  ============================
    Slice         Length                 Content
    ============  =====================  ============================
    ``[0:K)``     K                      log_k_deact[0..K-1]
    ``[K:2K)``    K                      log_d_deg[0..K-1]
    ``[2K:2K+1)`` 1                      log_beta_g
    ``[2K+1:2K+2)`` 1                    log_beta_l
    ``[2K+2:...)``  M                    log_alpha[0..M-1]
    ``[...:...)``   M                    log_kK_act[0..M-1]
    ``[...:...)``   M                    log_kK_deact[0..M-1]
    ``[...:...)``   N                    log_k_off[0..N-1]
    ``[...:end)``   4                    gamma_raw[0..3]
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
# JAX objective computation (loss components)
# ---------------------------------------------------------------------------


def compute_objectives_jax(
        theta,
        P_data,
        P_sim,
        A_scaled,
        A_sim,
        W_data,
        W_data_prot,
        prot_idx_for_A,
        L_alpha,
        lambda_net: float,
        reg_lambda: float,
        n_p: int,
        n_A: int,
        n_var: int,
        K: int,
        M: int,
        N: int,
):
    """
    Compute the three objective components of the loss in JAX.

    JAX-native equivalent of optimization.compute_objectives_nb.

    Parameters
    ----------
    theta                   : jax.Array, flat parameter vector (2K+2+3M+N+4)
    P_data                  : jax.Array (N_sites, T)
    P_sim                   : jax.Array (N_sites, T)  – simulation output
    A_scaled                : jax.Array (K_obs, T_A) or shape (0,)
    A_sim                   : jax.Array (K, T)        – full-dim simulation
    W_data, W_data_prot     : jax.Array – per-element loss weights
    prot_idx_for_A          : jax.Array (K_obs,) int  – protein indices
    L_alpha                 : jax.Array (M, M)
    lambda_net, reg_lambda  : float
    n_p, n_A, n_var         : int  – normalisation counts
    K, M, N                 : int

    Returns
    -------
    (f1, f2, f3) : tuple of JAX scalars
    """
    # --- Decode for regularisation ---
    _, _, _, _, alpha, _, _, _, _, _, _, _ = decode_theta(theta, K, M, N)

    # 1. Phosphosite loss: weighted MSLE
    diff_p = P_data - P_sim
    f1 = jnp.sum(jnp.log1p(W_data * diff_p * diff_p)) / max(n_p, 1)

    # 2. Protein abundance loss
    if A_scaled.size > 0:
        A_sim_obs = A_sim[prot_idx_for_A, :]  # (K_obs, T_A)
        diff_A = A_scaled - A_sim_obs
        f2 = jnp.sum(jnp.log1p(W_data_prot * diff_A * diff_A)) / max(n_A, 1)
    else:
        f2 = jnp.array(0.0)

    # 3. Regularisation: L2 + Laplacian network term
    reg = reg_lambda * jnp.dot(theta, theta)
    reg_net = lambda_net * jnp.dot(alpha, L_alpha @ alpha)
    f3 = (reg + reg_net) / max(n_var, 1)

    return f1, f2, f3


# ---------------------------------------------------------------------------
# Scalarized JAX loss for Optimistix
# ---------------------------------------------------------------------------
def make_loss_fn(
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
        ode_adjoint_kind="forward",
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
    K, M, N = dims.K, dims.M, dims.N

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
    x0[2 * K: 3 * K] = np.clip(a0, 0.0, 5.0)
    # p initial condition
    p0 = np.nan_to_num(P_data[:, 0], nan=0.0, posinf=10.0, neginf=0.0)
    x0[3 * K + M:] = np.clip(p0, 0.0, None)

    # JAX static arrays
    Cg_j = jnp.asarray(Cg, dtype=jnp.float64)
    Cl_j = jnp.asarray(Cl, dtype=jnp.float64)
    K_sk_j = jnp.asarray(K_site_kin, dtype=jnp.float64)
    R_j = jnp.asarray(R, dtype=jnp.float64)
    La_j = jnp.asarray(L_alpha, dtype=jnp.float64)
    spi_j = jnp.asarray(site_prot_idx, dtype=jnp.int32)
    k2p_j = jnp.asarray(kin_to_prot_idx, dtype=jnp.int32)
    rmp_j = jnp.asarray(receptor_mask_prot, dtype=jnp.float64)
    rmk_j = jnp.asarray(receptor_mask_kin, dtype=jnp.float64)
    psi_j = jnp.asarray(prev_site_idx, dtype=jnp.int32)

    y0_j = jnp.asarray(x0, dtype=jnp.float64)
    t_eval = jnp.asarray(all_times, dtype=jnp.float64)

    P_data_j = jnp.asarray(P_data, dtype=jnp.float64)
    A_scaled_j = jnp.asarray(A_scaled, dtype=jnp.float64)
    W_data_j = jnp.asarray(W_data, dtype=jnp.float64)
    W_prot_j = jnp.asarray(W_data_prot, dtype=jnp.float64)
    prot_idx_j = jnp.asarray(prot_idx_for_A, dtype=jnp.int32)
    La_loss_j = jnp.asarray(L_alpha, dtype=jnp.float64)

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
        rna_j = jnp.asarray(rna_data_scaled, dtype=jnp.float64)
        mrna_idx_j = jnp.asarray(mrna_time_idx, dtype=jnp.int32)
        rna_prot_idx_j = jnp.asarray(rna_model_prot_idx, dtype=jnp.int32)
        n_matched = len(rna_model_prot_idx)
        T_rna = len(t_mrna)
        if W_data_rna is not None:
            W_rna_j = jnp.asarray(W_data_rna, dtype=jnp.float64)
        else:
            W_rna_j = jnp.ones((n_matched, T_rna), dtype=jnp.float64)
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
        ode_solver_kind, root_find_max_steps=root_find_max_steps, scan_kind=scan_kind
    )
    sctrl = make_stepsize_controller(rtol=rtol, atol=atol)
    saveat = diffrax.SaveAt(ts=t_eval)
    adjoint = make_diffrax_adjoint(ode_adjoint_kind)

    logger.info(f"Using Diffrax adjoint: {adjoint}")
    logger.info(f"Using Diffrax solver: {solver}")

    t0_val = float(all_times[0])
    t1_val = float(all_times[-1])

    FAILED_SOLVE_PENALTY = jnp.asarray(1e6, dtype=jnp.float64)

    def loss_fn(theta, _args):
        theta_j = jnp.asarray(theta, dtype=jnp.float64)

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
            adjoint=adjoint,
        )

        xs = sol.ys  # (T_unified, 3K+M+N) – new state layout

        # Sample at protein time indices
        xs_prot = xs[prot_idx_solver, :]
        # New slicing: [R_rna, S, A, Kdyn, p]
        P_sim = jnp.clip(xs_prot[:, 3 * K + M:], 0.0, None).T  # (N, T_prot)
        A_sim = jnp.clip(xs_prot[:, 2 * K: 3 * K], 0.0, 5.0).T  # (K, T_prot)

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
            ).T  # (K, T_rna); clip to prevent overflow
            R_sim_matched = R_sim_rna[rna_prot_idx_j, :]  # (n_match, T_rna)
            diff_R = rna_j - R_sim_matched
            f4 = jnp.sum(W_rna_j * diff_R * diff_R) / n_rna
        else:
            f4 = jnp.asarray(0.0, dtype=jnp.float64)

        total = (
                jnp.asarray(w_phospho, dtype=jnp.float64) * f1
                + jnp.asarray(w_abundance, dtype=jnp.float64) * f2
                + jnp.asarray(w_reg, dtype=jnp.float64) * f3
                + jnp.asarray(w_mrna, dtype=jnp.float64) * f4
        )

        # Penalise non-finite results without crashing
        total = jnp.where(jnp.isfinite(total), total, FAILED_SOLVE_PENALTY)
        return total, (f1, f2, f3, f4)

    return loss_fn


def make_residuals_fn(
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
    K, M, N = dims.K, dims.M, dims.N

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
    x0[2 * K: 3 * K] = np.clip(a0, 0.0, 5.0)
    p0 = np.nan_to_num(P_data[:, 0], nan=0.0, posinf=10.0, neginf=0.0)
    x0[3 * K + M:] = np.clip(p0, 0.0, None)

    # JAX static arrays
    Cg_j = jnp.asarray(Cg, dtype=jnp.float64)
    Cl_j = jnp.asarray(Cl, dtype=jnp.float64)
    K_sk_j = jnp.asarray(K_site_kin, dtype=jnp.float64)
    R_j = jnp.asarray(R, dtype=jnp.float64)
    La_j = jnp.asarray(L_alpha, dtype=jnp.float64)
    spi_j = jnp.asarray(site_prot_idx, dtype=jnp.int32)
    k2p_j = jnp.asarray(kin_to_prot_idx, dtype=jnp.int32)
    rmp_j = jnp.asarray(receptor_mask_prot, dtype=jnp.float64)
    rmk_j = jnp.asarray(receptor_mask_kin, dtype=jnp.float64)
    psi_j = jnp.asarray(prev_site_idx, dtype=jnp.int32)

    y0_j = jnp.asarray(x0, dtype=jnp.float64)
    t_eval = jnp.asarray(all_times, dtype=jnp.float64)

    P_data_j = jnp.asarray(P_data, dtype=jnp.float64)
    A_scaled_j = jnp.asarray(A_scaled, dtype=jnp.float64)
    prot_idx_j = jnp.asarray(prot_idx_for_A, dtype=jnp.int32)
    prot_idx_solver = jnp.asarray(prot_time_idx, dtype=jnp.int32)

    # Weight arrays with modality loss weights baked in as sqrt factors
    # so that ||sqrt(w*W)*(sim-obs)||^2 == w * sum(W * (sim-obs)^2)
    sqrt_wp = jnp.sqrt(jnp.asarray(w_phospho, dtype=jnp.float64)) * jnp.sqrt(
        jnp.asarray(W_data, dtype=jnp.float64)
    )
    has_abundance = A_scaled.size > 0
    if has_abundance:
        sqrt_wa = jnp.sqrt(jnp.asarray(w_abundance, dtype=jnp.float64)) * jnp.sqrt(
            jnp.asarray(W_data_prot, dtype=jnp.float64)
        )
    else:
        sqrt_wa = None

    # Convert weight arrays to JAX for use inside the JIT-traced residuals_fn
    W_data_j_diag = jnp.asarray(W_data, dtype=jnp.float64)
    W_prot_j_diag = (
        jnp.asarray(W_data_prot, dtype=jnp.float64) if has_abundance else None
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
        rna_j = jnp.asarray(rna_data_scaled, dtype=jnp.float64)
        mrna_idx_j = jnp.asarray(mrna_time_idx, dtype=jnp.int32)
        rna_prot_idx_j = jnp.asarray(rna_model_prot_idx, dtype=jnp.int32)
        n_matched = len(rna_model_prot_idx)
        T_rna = len(t_mrna)
        W_rna_base = (
            np.asarray(W_data_mrna, dtype=np.float64)
            if W_data_mrna is not None
            else np.ones((n_matched, T_rna), dtype=np.float64)
        )
        W_rna_j_diag = jnp.asarray(W_rna_base, dtype=jnp.float64)
        sqrt_wr = jnp.sqrt(jnp.asarray(w_mrna, dtype=jnp.float64)) * jnp.sqrt(W_rna_j_diag)
        n_rna = max(1, rna_data_scaled.size)
    else:
        rna_j = mrna_idx_j = rna_prot_idx_j = sqrt_wr = W_rna_j_diag = None
        n_rna = 1

    # Regularisation residuals – fixed structure, no ODE needed
    sqrt_reg = jnp.asarray(np.sqrt(float(reg_lambda)), dtype=jnp.float64)
    has_net_reg = lambda_net > 0.0
    if has_net_reg:
        sqrt_lnet = jnp.asarray(np.sqrt(float(lambda_net)), dtype=jnp.float64)

    # Bounds for hard-clipping theta inside residuals_fn (prevents LM from escaping
    # the biological parameter space and producing stiff/divergent ODEs).
    has_bounds = xl is not None and xu is not None
    if has_bounds:
        xl_j = jnp.asarray(xl, dtype=jnp.float64)
        xu_j = jnp.asarray(xu, dtype=jnp.float64)

    rhs_fn = make_rhs(
        K, M, N, mechanism, k_act_fn=k_act_fn, s_prod_fn=s_prod_fn, rna_relax=rna_relax
    )

    term = diffrax.ODETerm(rhs_fn)
    ode_solver = make_diffrax_solver(
        ode_solver_kind, root_find_max_steps=root_find_max_steps
    )
    sctrl = make_stepsize_controller(rtol=rtol, atol=atol)
    saveat = diffrax.SaveAt(ts=t_eval)
    adjoint = make_diffrax_adjoint(ode_adjoint_kind)

    logger.info(f"Using Diffrax adjoint: {adjoint}")
    logger.info(f"Using Diffrax solver: {ode_solver}")

    t0_val = float(all_times[0])
    t1_val = float(all_times[-1])

    PENALTY = jnp.asarray(_FAILED_SOLVE_PENALTY, dtype=jnp.float64)

    def residuals_fn(theta, _args):
        """
        Compute residual vector and diagnostic loss components.

        Compatible with optx.least_squares(..., has_aux=True):
          returns (residuals_1d, (f1, f2, f3, f4))

        Uses diffrax.DirectAdjoint so that Optimistix LM can compute JVPs
        through the ODE solve without storing all intermediate states.
        """
        theta_j = jnp.asarray(theta, dtype=jnp.float64)
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
        P_sim = jnp.clip(xs_prot[:, 3 * K + M:], 0.0, None).T  # (N, T_prot)
        A_sim = jnp.clip(xs_prot[:, 2 * K: 3 * K], 0.0, 5.0).T  # (K, T_prot)

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
            r_abund = jnp.zeros(0, dtype=jnp.float64)
            f2 = jnp.asarray(0.0, dtype=jnp.float64)

        # --- mRNA residuals ---
        if has_mrna:
            xs_rna = xs[mrna_idx_j, :]
            # Clip to prevent overflow; R_rna is on fold-change scale (~0-20)
            R_sim_rna = jnp.clip(xs_rna[:, :K], 0.0, _RNA_CLIP_UPPER).T  # (K, T_rna)
            R_sim_matched = R_sim_rna[rna_prot_idx_j, :]  # (n_match, T_rna)
            diff_R = R_sim_matched - rna_j  # (n_match, T_rna)
            r_rna = (sqrt_wr * diff_R).ravel()
            f4 = jnp.sum(W_rna_j_diag * diff_R * diff_R) / n_rna
        else:
            r_rna = jnp.zeros(0, dtype=jnp.float64)
            f4 = jnp.asarray(0.0, dtype=jnp.float64)

        # --- Regularisation residuals (no ODE needed) ---
        # L2 on theta
        r_reg_l2 = sqrt_reg * theta_j  # (n_var,)
        # Laplacian network regularisation on alpha (decoded from theta)
        if has_net_reg:
            alpha_raw = theta_j[2 * K + 2: 2 * K + 2 + M]
            alpha = jnp.exp(jnp.clip(alpha_raw, -20.0, 10.0))
            r_reg_net = sqrt_lnet * (La_j @ alpha)  # (M,)
            r_reg = jnp.concatenate([r_reg_l2, r_reg_net])
        else:
            r_reg = r_reg_l2

        # f3 diagnostic
        f3_l2 = jnp.asarray(reg_lambda, dtype=jnp.float64) * jnp.dot(theta_j, theta_j)
        if has_net_reg:
            alpha_raw = theta_j[2 * K + 2: 2 * K + 2 + M]
            alpha = jnp.exp(jnp.clip(alpha_raw, -20.0, 10.0))
            f3_net = jnp.asarray(lambda_net, dtype=jnp.float64) * jnp.dot(alpha, La_j @ alpha)
        else:
            f3_net = jnp.asarray(0.0, dtype=jnp.float64)
        f3 = (f3_l2 + f3_net) / jnp.asarray(max(n_var, 1), dtype=jnp.float64)

        # --- Concatenate residual vector ---
        residuals = jnp.concatenate([r_phospho, r_abund, r_rna, r_reg])

        # --- Replace non-finite residuals with finite penalty ---
        # This handles ODE solve failures gracefully without crashing the optimizer.
        finite_residuals = jnp.where(jnp.isfinite(residuals), residuals, PENALTY)
        finite_residuals = jnp.where(
            solve_ok, finite_residuals, jnp.full_like(finite_residuals, PENALTY)
        )

        # Recalculate finite diagnostics for aux output
        f1 = jnp.where(jnp.isfinite(f1), f1, jnp.asarray(1e6, dtype=jnp.float64))
        f2 = jnp.where(jnp.isfinite(f2), f2, jnp.asarray(1e6, dtype=jnp.float64))
        f3 = jnp.where(jnp.isfinite(f3), f3, jnp.asarray(1e6, dtype=jnp.float64))
        f4 = jnp.where(jnp.isfinite(f4), f4, jnp.asarray(1e6, dtype=jnp.float64))

        jax.debug.callback(_log_residuals_step, f1 + f2 + f3 + f4)

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
        jaxpr_out_dir=None,
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
        f1 + f2 + f3 + f4 (selection metric; all four loss components
        contribute so that the best run is chosen consistently whether
        or not RNA data are present).
    f1, f2, f3, f4 : float
        Diagnostic loss components.
    """
    # Construct solver (default: optx.LevenbergMarquardt) and adjoint from string flags.
    solver = make_ls_solver(ls_solver, rtol=rtol, atol=atol, verbose=verbose)
    adjoint = make_optx_adjoint(optx_adjoint)

    if verbose:
        logger.info(f"Using Optimistix adjoint: {optx_adjoint}")
        logger.info(f"Using Optimistix solver: {ls_solver}")
        logger.info(f"Using Optimistix Jacobian mode: {jac_mode}")

    # Optional: capture jaxpr for the residuals function before running the solver.
    if jaxpr_out_dir is not None:
        from pathlib import Path as _Path
        from phoscrosstalk.jaxpr_reporter import capture_and_save_jaxpr
        capture_and_save_jaxpr(
            fn=residuals_fn,
            example_args=(jnp.asarray(theta0, dtype=jnp.float64), None),
            step_label="multistart_residuals_fn",
            module_label="phoscrosstalk.optimization",
            out_dir=_Path(jaxpr_out_dir),
        )

    sol = optx.least_squares(
        residuals_fn,
        solver,
        jnp.asarray(theta0, dtype=jnp.float64),
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
    # Include all four loss terms in the selection metric so that best-run
    # selection is consistent when RNA data are present (f4 > 0).
    total_loss = f1 + f2 + f3 + f4

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
    if hasattr(problem, "dims") and problem.dims is not None:
        K, M, N = problem.dims.K, problem.dims.M, problem.dims.N
    else:
        N = int(problem.P_data.shape[0])
        M = int(problem.R.shape[0])
        if getattr(problem, "A_scaled", None) is not None and np.asarray(problem.A_scaled).size > 0:
            K = int(np.asarray(problem.A_scaled).shape[0])
        else:
            K = N
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
            f"len(rna_model_prot_idx)={len(problem.rna_model_prot_idx)} != rna_obs_matched.shape[0]={n_matched}",
            # noqa: E501
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
        from phoscrosstalk.mechanisms import decode_theta

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
            dims: ModelDims,
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
            pinn_model=None,
            **kwargs,  # absorb legacy keyword args (elementwise_runner, etc.)
    ):
        # Note: ode_adjoint_kind was previously defaulted to the non-existent value
        # "adjoint".  The corrected default is "forward", which is the right choice
        # for LM + jac_mode="fwd" (Optimistix forward-mode AD through Diffrax).
        # If you were relying on a different adjoint, set ode_adjoint_kind explicitly
        # in your config under [solver] ode_adjoint.
        self.t = t
        self.dims = dims
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
        # Optional picklable rebuild kwargs for parallel multi-start workers.
        # Set these after construction (e.g. in main.py) with the raw numpy
        # arrays / config strings used to build k_act_fn / s_prod_fn so that
        # spawned worker processes can reconstruct the JAX closures without
        # receiving a non-picklable callable across process boundaries.
        #
        # _k_act_rebuild_kwargs expected keys (matching make_k_act_fn signature):
        #   t_rna, rna_data, tf_prot_weights, K, interp_mode,
        #   protein_self_rna_idx
        #
        # _s_prod_rebuild_kwargs expected keys (matching make_s_prod_fn signature):
        #   t_protein, Y_data, R_kin_site, kin_to_prot_idx, K, M,
        #   s_prod_fn_type, interp_mode
        self._k_act_rebuild_kwargs: dict | None = None
        self._s_prod_rebuild_kwargs: dict | None = None
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
        self.pinn_model = pinn_model

    def simulate(self, x):
        """
        Run a simulation for parameter vector x and return phosphosite trajectories.

        Args:
            x (np.ndarray): Parameter vector.

        Returns:
            (np.ndarray): P_sim (N_sites x T).
        """
        theta = np.asarray(x, dtype=np.float64)
        K, T = self.dims.K, self.P_data.shape[1]
        A0 = build_full_A0(K, T, self.A_scaled, self.prot_idx_for_A)

        P_sim, _A_sim = simulate(
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
            dims=self.dims,
        )
        return P_sim

    def simulate_full(self, x):
        """
        Run a full simulation returning all state components including R_sim_rna.

        Args:
            x (np.ndarray): Parameter vector.

        Returns:
            (dict): Keys: P_sim, A_sim, S_sim, Kdyn_sim, R_sim, R_sim_rna, t, t_rna,
                solver_times.
        """
        theta = np.asarray(x, dtype=np.float64)
        K, T = self.dims.K, self.P_data.shape[1]
        A0 = build_full_A0(K, T, self.A_scaled, self.prot_idx_for_A)

        return simulate(
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
            dims=self.dims,
            max_steps=self.max_steps,
        )

    def _simulate_pinn(self, theta):
        """
        Run PINN-augmented simulation using the fitted PINN model.

        This mirrors the normal simulate(..., return_full=True) interface, but
        uses the combined mechanistic + PINN RHS.
        """
        from phoscrosstalk.pinn.rhs import make_combined_rhs

        K, M, N = self.dims.K, self.dims.M, self.dims.N
        T = self.P_data.shape[1]

        A0 = build_full_A0(K, T, self.A_scaled, self.prot_idx_for_A)

        x0 = np.zeros(3 * K + M + N, dtype=np.float64)

        if self.R_data0 is not None:
            r0 = np.asarray(self.R_data0, dtype=np.float64)
            r0 = r0[:, 0] if r0.ndim > 1 else r0
            x0[:K] = np.clip(np.nan_to_num(r0, nan=1.0), 0.0, 10.0)
        else:
            x0[:K] = 1.0

        x0[2 * K: 3 * K] = np.clip(
            np.nan_to_num(A0[:, 0], nan=1.0),
            0.0,
            5.0,
        )
        x0[3 * K + M:] = np.clip(
            np.nan_to_num(self.P_data[:, 0], nan=0.0),
            0.0,
            None,
        )

        prev_idx = compute_prev_site_idx(
            np.asarray(self.site_prot_idx, dtype=np.int32),
            N,
        )

        combined_rhs = make_combined_rhs(
            K,
            M,
            N,
            self.mechanism,
            k_act_fn=self.k_act_fn,
            s_prod_fn=self.s_prod_fn,
            rna_relax=self.rna_relax,
            abundance_max=5.0,
        )

        theta_j = jnp.asarray(theta, dtype=jnp.float64)
        y0_j = jnp.asarray(x0, dtype=jnp.float64)

        # Use the same unified output grid as pinn/loss.py:
        # protein/phosphosite outputs are read at self.t,
        # RNA outputs are read at self.t_rna.
        if self.t_rna is not None and len(self.t_rna) > 0:
            all_times = np.union1d(
                np.asarray(self.t, dtype=np.float64),
                np.asarray(self.t_rna, dtype=np.float64),
            )
        else:
            all_times = np.unique(np.asarray(self.t, dtype=np.float64))

        all_times = np.sort(all_times).astype(np.float64)

        self.prot_time_idx = np.searchsorted(all_times, np.asarray(self.t, dtype=np.float64))

        if self.t_rna is not None and len(self.t_rna) > 0:
            self.mrna_time_idx = np.searchsorted(
                all_times,
                np.asarray(self.t_rna, dtype=np.float64),
            )
        else:
            self.mrna_time_idx = None

        t_eval = jnp.asarray(all_times, dtype=jnp.float64)

        ode_args = (
            theta_j,
            jnp.asarray(self.Cg, dtype=jnp.float64),
            jnp.asarray(self.Cl, dtype=jnp.float64),
            jnp.asarray(self.site_prot_idx, dtype=jnp.int32),
            jnp.asarray(self.K_site_kin, dtype=jnp.float64),
            jnp.asarray(self.R, dtype=jnp.float64),
            jnp.asarray(self.L_alpha, dtype=jnp.float64),
            jnp.asarray(self.kin_to_prot_idx, dtype=jnp.int32),
            jnp.asarray(self.receptor_mask_prot, dtype=jnp.float64),
            jnp.asarray(self.receptor_mask_kin, dtype=jnp.float64),
            jnp.asarray(prev_idx, dtype=jnp.int32),
            self.pinn_model,
        )

        sol = diffrax.diffeqsolve(
            diffrax.ODETerm(combined_rhs),
            make_diffrax_solver(
                self.ode_solver_kind,
                root_find_max_steps=self.ode_root_find_max_steps,
            ),
            t0=float(all_times[0]),
            t1=float(all_times[-1]),
            dt0=self.ode_dt0,
            y0=y0_j,
            args=ode_args,
            saveat=diffrax.SaveAt(ts=t_eval),
            stepsize_controller=make_stepsize_controller(
                rtol=self.rtol,
                atol=self.atol,
            ),
            max_steps=self.max_steps,
            throw=False,
        )

        ys = np.asarray(sol.ys, dtype=np.float64)

        # Diffrax returns (T_unified, state_dim).
        if ys.shape[0] != len(all_times):
            raise RuntimeError(
                f"PINN simulation returned unexpected shape {ys.shape}; "
                f"expected first dimension len(all_times)={len(all_times)}."
            )

        xs_prot = ys[self.prot_time_idx, :]  # (T_prot, state_dim)

        R_sim = xs_prot[:, 0:K].T
        S_sim = xs_prot[:, K: 2 * K].T
        A_sim = xs_prot[:, 2 * K: 3 * K].T
        Kdyn_sim = xs_prot[:, 3 * K: 3 * K + M].T
        P_sim = xs_prot[:, 3 * K + M: 3 * K + M + N].T

        if self.mrna_time_idx is not None:
            xs_rna = ys[self.mrna_time_idx, :]  # (T_rna, state_dim)
            self.R_sim_rna = xs_rna[:, 0:K].T  # (K, T_rna)
        else:
            self.R_sim_rna = None

        return {
            "P_sim": P_sim,
            "A_sim": A_sim,
            "S_sim": S_sim,
            "Kdyn_sim": Kdyn_sim,
            "R_sim": R_sim,
            "R_sim_rna": self.R_sim_rna,
            "t": np.asarray(self.t, dtype=np.float64),
            "t_rna": (
                np.asarray(self.t_rna, dtype=np.float64)
                if self.t_rna is not None
                else None
            ),
            "solver_times": all_times,
        }


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
    * The Hessian is computed in float64.
    * For ``n > 40`` parameter labels the heatmap omits dense tick labels
      to remain readable.
    """
    theta_j = jnp.asarray(theta, dtype=jnp.float64)
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
