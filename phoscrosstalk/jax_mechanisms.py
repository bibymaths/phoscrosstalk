#!/usr/bin/env python3
"""
jax_mechanisms.py

JAX-compatible ODE right-hand side (RHS) functions for the phospho-network model.

Mirrors the biological logic of core_mechanisms.py but uses jax.numpy so that
the functions can be:
  - traced by JAX JIT,
  - differentiated via jax.grad / jax.value_and_grad,
  - composed with diffrax.ODETerm.

All three phosphorylation mechanisms are supported:
  0 / "dist"  – Distributive
  1 / "seq"   – Sequential (ordered gating using prev_site_idx)
  2 / "rand"  – Random/Cooperative (competitive crowding)

Public helpers
--------------
decode_theta_jax(theta, K, M, N)
    JAX version of core_mechanisms.decode_theta. Returns the same tuple of
    decoded biological parameters.

make_rhs(K, M, N, mechanism)
    Factory that returns a mechanism-specific JAX RHS callable compatible with
    diffrax.ODETerm(rhs).

compute_prev_site_idx(site_prot_idx, N)
    Pure-NumPy helper (called once at setup time) that precomputes the static
    "predecessor site" index array needed by the sequential mechanism.

compute_objectives_jax(theta, P_data, P_sim, A_scaled, A_sim, W_data,
                       W_data_prot, prot_idx_for_A, L_alpha,
                       lambda_net, reg_lambda, n_p, n_A, n_var, K, M, N)
    JAX version of optimization.compute_objectives_nb.  Returns (f1, f2, f3)
    as JAX scalars so the whole loss pipeline stays differentiable.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

# ---------------------------------------------------------------------------
# Parameter decoding
# ---------------------------------------------------------------------------


def decode_theta_jax(theta, K: int, M: int, N: int):
    """
    Decode the flat log-scale parameter vector into biological rate constants.

    ``k_act`` and ``s_prod`` are no longer optimisation variables; they are
    derived from experimental data (see :mod:`derived_rates`).  The parameter
    vector therefore has dimension ``2*K + 2 + 3*M + N + 4``.

    JAX-native equivalent of core_mechanisms.decode_theta (Numba).  All
    operations use jax.numpy, so the output is differentiable w.r.t. theta.

    Parameters
    ----------
    theta : jax.Array, shape (2*K + 2 + 3*M + N + 4,)
    K, M, N : int  –  proteins, kinases, phosphosites

    Returns
    -------
    tuple of 12 entries:
        k_deact, d_deg         (K,)
        beta_g, beta_l         scalar
        alpha, kK_act, kK_deact  (M,)
        k_off                  (N,)
        gamma_S_p, gamma_A_S, gamma_A_p, gamma_K_net  scalar
    """
    idx = 0
    log_k_deact = theta[idx : idx + K]
    idx += K
    log_d_deg = theta[idx : idx + K]
    idx += K

    log_beta_g = theta[idx]
    idx += 1
    log_beta_l = theta[idx]
    idx += 1

    log_alpha = theta[idx : idx + M]
    idx += M
    log_kK_act = theta[idx : idx + M]
    idx += M
    log_kK_deact = theta[idx : idx + M]
    idx += M

    log_k_off = theta[idx : idx + N]
    idx += N
    raw_gamma = theta[idx : idx + 4]

    def clip(v):
        return jnp.clip(v, -20.0, 10.0)

    k_deact = jnp.exp(clip(log_k_deact))
    d_deg = jnp.exp(clip(log_d_deg))
    alpha = jnp.exp(clip(log_alpha))
    kK_act = jnp.exp(clip(log_kK_act))
    kK_deact = jnp.exp(clip(log_kK_deact))
    k_off = jnp.exp(clip(log_k_off))

    beta_g = jnp.exp(jnp.clip(log_beta_g, -20.0, 10.0))
    beta_l = jnp.exp(jnp.clip(log_beta_l, -20.0, 10.0))

    gamma_S_p = 2.0 * jnp.tanh(raw_gamma[0])
    gamma_A_S = 2.0 * jnp.tanh(raw_gamma[1])
    gamma_A_p = 2.0 * jnp.tanh(raw_gamma[2])
    gamma_K_net = 2.0 * jnp.tanh(raw_gamma[3])

    return (
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
    )


# ---------------------------------------------------------------------------
# Sequential-mechanism topology helper (pure NumPy, called once at setup)
# ---------------------------------------------------------------------------


def compute_prev_site_idx(site_prot_idx: np.ndarray, N: int) -> np.ndarray:
    """
    Build the predecessor-site index array for the sequential mechanism.

    For each site ``i``, stores the index ``j < i`` of the most recently seen
    site on the same protein in the flat site ordering, or ``-1`` if ``i`` is
    the first site of that protein.

    Parameters
    ----------
    site_prot_idx : np.ndarray, shape (N,)  – integer protein index per site
    N             : int                     – number of sites

    Returns
    -------
    prev_site_idx : np.ndarray, shape (N,), dtype int32
    """
    prev_site_idx = np.full(N, -1, dtype=np.int32)
    last_seen: dict[int, int] = {}
    for i in range(N):
        prot = int(site_prot_idx[i])
        if prot in last_seen:
            prev_site_idx[i] = last_seen[prot]
        last_seen[prot] = i
    return prev_site_idx


# ---------------------------------------------------------------------------
# JAX RHS factory
# ---------------------------------------------------------------------------


def make_rhs(
    K: int,
    M: int,
    N: int,
    mechanism: str,
    k_act_fn=None,
    s_prod_fn=None,
    rna_relax: float = 0.1,
):
    """
    Return a JAX-compatible RHS function for ``diffrax.ODETerm``.

    The returned function has the signature::

        rhs(t, y, args) -> dy

    where ``args`` is a tuple::

        (theta, Cg, Cl, site_prot_idx, K_site_kin, R, L_alpha,
         kin_to_prot_idx, receptor_mask_prot, receptor_mask_kin,
         prev_site_idx)

    ``prev_site_idx`` is only used by the sequential mechanism; it can be
    ``None`` (or all-``-1``) for distributive/rand.

    ``k_act_fn`` and ``s_prod_fn`` are optional JAX callables closed over at
    construction time::

        k_act_fn(t) -> jnp.array(shape=(K,))   # protein activation rate
        s_prod_fn(t) -> jnp.array(shape=(K,))  # protein synthesis rate

    When *None*, constant defaults are used: ``k_act = 1.0``, ``s_prod = 0.1``.

    Parameters
    ----------
    K, M, N      : int   – proteins, kinases, phosphosites
    mechanism    : str   – ``"dist"`` | ``"seq"`` | ``"rand"``
    k_act_fn     : callable | None
    s_prod_fn    : callable | None

    Returns
    -------
    callable  – JAX-traceable RHS
    """
    if mechanism not in {"dist", "seq", "rand"}:
        raise ValueError(
            f"Unknown mechanism '{mechanism}'. Use 'dist', 'seq', or 'rand'."
        )

    # Build constant fallbacks so the RHS never branches on None
    _k_act_const = jnp.ones(K, dtype=jnp.float32)
    _s_prod_const = jnp.full(K, 0.1, dtype=jnp.float32)

    if k_act_fn is None:

        def _k_act_fn(t):
            return _k_act_const
    else:
        _k_act_fn = k_act_fn

    if s_prod_fn is None:

        def _s_prod_fn(t):
            return _s_prod_const
    else:
        _s_prod_fn = s_prod_fn

    def rhs(t, y, args):
        (
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
            prev_site_idx,
        ) = args

        # --- Decode parameters --------------------------------------------------
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
        ) = decode_theta_jax(theta, K, M, N)

        # Derived rates (time-varying, not optimised)
        k_act = _k_act_fn(t)
        s_prod = _s_prod_fn(t)

        # --- Unpack + clip state ------------------------------------------------
        # New state layout: y = [R_rna, S, A, Kdyn, p]  (dim = 3*K + M + N)
        R_rna = jnp.clip(y[:K], 0.0, None)  # mRNA state
        S = y[K : 2 * K]  # protein signalling
        A = y[2 * K : 3 * K]  # protein abundance
        Kdyn = jnp.clip(y[3 * K : 3 * K + M], 0.0, 1.0)  # kinase activity
        p = jnp.clip(y[3 * K + M :], 0.0, 1.0)  # phosphosite occupancy

        # Smooth external stimulus: sigmoid ramp from 0→1
        u = 1.0 / (1.0 + jnp.exp(-t / 0.1))

        # --- Coupling -----------------------------------------------------------
        coup = jnp.tanh(beta_g * (Cg @ p) + beta_l * (Cl @ p))

        # --- Per-protein aggregate means ----------------------------------------
        num_p = jnp.zeros(K).at[site_prot_idx].add(p)
        den = jnp.zeros(K).at[site_prot_idx].add(1.0)
        num_c = jnp.zeros(K).at[site_prot_idx].add(coup)

        safe_den = jnp.where(den > 0.0, den, 1.0)
        mp = num_p / safe_den
        mc = num_c / safe_den

        # --- 0. mRNA state (R_rna) -----------------------------------------------
        # Relaxation ODE: dR/dt = rna_relax * (k_act(t) - R)
        #
        # k_act(t) is the TF-derived mRNA drive signal on the fold-change scale
        # (~1.0 for most genes).  Using a first-order relaxation instead of
        # "k_act - d_deg * R" prevents R from accumulating to k_act / d_deg >> 1
        # when d_deg is a small protein degradation rate.  At steady state
        # R_ss = k_act(t), which matches the observed fold-change scale.
        # rna_relax is configurable via [derived_rates] rna_relax in config.toml.
        _rna_relax = jnp.float32(rna_relax)
        dR_rna = _rna_relax * (k_act - R_rna)

        # --- 1. Protein signalling state (S) ------------------------------------
        D_S = 1.0 + gamma_S_p * mp + mc + receptor_mask_prot * u
        D_S = jnp.clip(D_S, 0.0, None)
        dS = k_act * D_S * (1.0 - S) - k_deact * S

        # --- 2. Protein abundance (A) -------------------------------------------
        # R_rna couples mRNA level to protein synthesis: s_eff ∝ R_rna
        s_eff = jnp.clip(s_prod * R_rna * (1.0 + gamma_A_S * S), 0.0, None)
        dA = s_eff - d_deg * A

        # --- 3. Kinase dynamics (Kdyn) ------------------------------------------
        u_sub = R @ p

        u_net = -(L_alpha @ Kdyn)

        U = u_sub + gamma_K_net * u_net

        valid_prot = kin_to_prot_idx >= 0
        safe_p_idx = jnp.where(valid_prot, kin_to_prot_idx, 0)
        prot_contrib = gamma_A_S * S[safe_p_idx] + gamma_A_p * A[safe_p_idx]
        U = U + jnp.where(valid_prot, prot_contrib, 0.0)
        U = U + receptor_mask_kin * u

        dKdyn = kK_act * jnp.tanh(U) * (1.0 - Kdyn) - kK_deact * Kdyn

        # --- 4. Phosphosite dynamics (p) ----------------------------------------
        k_on_eff = K_site_kin @ (alpha * Kdyn)
        coup_clamp = jnp.clip(coup, 0.0, None)

        if mechanism == "dist":
            gate = jnp.ones(N)

        elif mechanism == "seq":
            safe_prev = jnp.where(prev_site_idx >= 0, prev_site_idx, 0)
            gate = jnp.where(prev_site_idx >= 0, p[safe_prev], 1.0)

        else:
            CROWDING_BASELINE = 0.5
            CROWDING_WEIGHT = 0.5
            CROWDING_EPSILON = 1e-9
            vacant_frac = 1.0 - mp[site_prot_idx]
            gate = 1.0 / (
                CROWDING_BASELINE + CROWDING_WEIGHT * vacant_frac + CROWDING_EPSILON
            )

        v_raw = k_on_eff * (1.0 + coup_clamp) * gate * (1.0 - p)
        v_on = v_raw / (1.0 + jnp.abs(v_raw))

        v_off_r = k_off * p
        v_off = v_off_r / (1.0 + v_off_r)

        dp = v_on - v_off

        # New state order: [R_rna, S, A, Kdyn, p]  (3*K + M + N)
        return jnp.concatenate([dR_rna, dS, dA, dKdyn, dp])

    return rhs


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
    _, _, _, _, alpha, _, _, _, _, _, _, _ = decode_theta_jax(theta, K, M, N)

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
