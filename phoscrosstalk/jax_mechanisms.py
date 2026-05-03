#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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

import numpy as np
import jax
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Parameter decoding
# ---------------------------------------------------------------------------


def decode_theta_jax(theta, K: int, M: int, N: int):
    """
    Decode the flat log-scale parameter vector into biological rate constants.

    JAX-native equivalent of core_mechanisms.decode_theta (Numba).  All
    operations use jax.numpy, so the output is differentiable w.r.t. theta.

    Parameters
    ----------
    theta : jax.Array, shape (4*K + 2 + 3*M + N + 4,)
    K, M, N : int  –  proteins, kinases, phosphosites

    Returns
    -------
    tuple of 14 entries matching core_mechanisms.decode_theta:
        k_act, k_deact, s_prod, d_deg  (K,)
        beta_g, beta_l                 scalar
        alpha, kK_act, kK_deact        (M,)
        k_off                          (N,)
        gamma_S_p, gamma_A_S, gamma_A_p, gamma_K_net  scalar
    """
    idx = 0
    log_k_act   = theta[idx : idx + K]; idx += K
    log_k_deact = theta[idx : idx + K]; idx += K
    log_s_prod  = theta[idx : idx + K]; idx += K
    log_d_deg   = theta[idx : idx + K]; idx += K

    log_beta_g = theta[idx]; idx += 1
    log_beta_l = theta[idx]; idx += 1

    log_alpha   = theta[idx : idx + M]; idx += M
    log_kK_act  = theta[idx : idx + M]; idx += M
    log_kK_deact = theta[idx : idx + M]; idx += M

    log_k_off   = theta[idx : idx + N]; idx += N
    raw_gamma   = theta[idx : idx + 4]

    clip = lambda v: jnp.clip(v, -20.0, 10.0)

    k_act    = jnp.exp(clip(log_k_act))
    k_deact  = jnp.exp(clip(log_k_deact))
    s_prod   = jnp.exp(clip(log_s_prod))
    d_deg    = jnp.exp(clip(log_d_deg))
    alpha    = jnp.exp(clip(log_alpha))
    kK_act   = jnp.exp(clip(log_kK_act))
    kK_deact = jnp.exp(clip(log_kK_deact))
    k_off    = jnp.exp(clip(log_k_off))

    beta_g = jnp.exp(jnp.clip(log_beta_g, -20.0, 10.0))
    beta_l = jnp.exp(jnp.clip(log_beta_l, -20.0, 10.0))

    gamma_S_p   = 2.0 * jnp.tanh(raw_gamma[0])
    gamma_A_S   = 2.0 * jnp.tanh(raw_gamma[1])
    gamma_A_p   = 2.0 * jnp.tanh(raw_gamma[2])
    gamma_K_net = 2.0 * jnp.tanh(raw_gamma[3])

    return (
        k_act, k_deact, s_prod, d_deg,
        beta_g, beta_l,
        alpha, kK_act, kK_deact,
        k_off,
        gamma_S_p, gamma_A_S, gamma_A_p, gamma_K_net,
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


def make_rhs(K: int, M: int, N: int, mechanism: str):
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

    Parameters
    ----------
    K, M, N   : int   – proteins, kinases, phosphosites
    mechanism : str   – ``"dist"`` | ``"seq"`` | ``"rand"``

    Returns
    -------
    callable  – JAX-traceable RHS
    """
    if mechanism not in {"dist", "seq", "rand"}:
        raise ValueError(f"Unknown mechanism '{mechanism}'. Use 'dist', 'seq', or 'rand'.")

    def rhs(t, y, args):
        (
            theta,
            Cg, Cl,
            site_prot_idx,
            K_site_kin, R, L_alpha,
            kin_to_prot_idx,
            receptor_mask_prot, receptor_mask_kin,
            prev_site_idx,
        ) = args

        # --- Decode parameters --------------------------------------------------
        (
            k_act, k_deact, s_prod, d_deg,
            beta_g, beta_l,
            alpha, kK_act, kK_deact,
            k_off,
            gamma_S_p, gamma_A_S, gamma_A_p, gamma_K_net,
        ) = decode_theta_jax(theta, K, M, N)

        # --- Unpack + clip state ------------------------------------------------
        S    = y[:K]
        A    = y[K : 2 * K]
        Kdyn = jnp.clip(y[2 * K : 2 * K + M], 0.0, 1.0)
        p    = jnp.clip(y[2 * K + M :],        0.0, 1.0)

        # Smooth external stimulus: sigmoid ramp from 0→1
        u = 1.0 / (1.0 + jnp.exp(-t / 0.1))

        # --- Coupling -----------------------------------------------------------
        # coup = tanh(beta_g * Cg @ p  +  beta_l * Cl @ p)
        coup = jnp.tanh(beta_g * (Cg @ p) + beta_l * (Cl @ p))

        # --- Per-protein aggregate means ----------------------------------------
        # num_p[k] = sum of p[i] for all sites i on protein k
        # den[k]   = count of sites on protein k
        num_p = jnp.zeros(K).at[site_prot_idx].add(p)
        den   = jnp.zeros(K).at[site_prot_idx].add(1.0)
        num_c = jnp.zeros(K).at[site_prot_idx].add(coup)

        safe_den = jnp.where(den > 0.0, den, 1.0)
        mp = num_p / safe_den   # mean phospho per protein  (K,)
        mc = num_c / safe_den   # mean coupling per protein (K,)

        # --- 1. Protein signalling state (S) ------------------------------------
        D_S = 1.0 + gamma_S_p * mp + mc + receptor_mask_prot * u
        D_S = jnp.clip(D_S, 0.0, None)
        dS  = k_act * D_S * (1.0 - S) - k_deact * S

        # --- 2. Protein abundance (A) -------------------------------------------
        s_eff = jnp.clip(s_prod * (1.0 + gamma_A_S * S), 0.0, None)
        dA    = s_eff - d_deg * A

        # --- 3. Kinase dynamics (Kdyn) ------------------------------------------
        u_sub = R @ p                          # (M,)  substrate pressure

        # Network regularisation: L_alpha @ Kdyn (only if lambda_net non-zero)
        u_net = -(L_alpha @ Kdyn)              # (M,)  note: sign matches Numba code

        U = u_sub + gamma_K_net * u_net

        # Add protein-state contributions for kinases that map to a protein
        valid_prot  = kin_to_prot_idx >= 0
        safe_p_idx  = jnp.where(valid_prot, kin_to_prot_idx, 0)
        prot_contrib = (
            gamma_A_S * S[safe_p_idx] + gamma_A_p * A[safe_p_idx]
        )
        U = U + jnp.where(valid_prot, prot_contrib, 0.0)
        U = U + receptor_mask_kin * u

        dKdyn = kK_act * jnp.tanh(U) * (1.0 - Kdyn) - kK_deact * Kdyn

        # --- 4. Phosphosite dynamics (p) ----------------------------------------
        k_on_eff   = K_site_kin @ (alpha * Kdyn)      # (N,)
        coup_clamp = jnp.clip(coup, 0.0, None)        # (N,)

        if mechanism == "dist":
            # Distributive: all sites phosphorylated independently
            gate = jnp.ones(N)

        elif mechanism == "seq":
            # Sequential: gate = occupancy of the preceding site on the same
            # protein (1.0 for the first site, which has prev_site_idx == -1)
            safe_prev = jnp.where(prev_site_idx >= 0, prev_site_idx, 0)
            gate      = jnp.where(prev_site_idx >= 0, p[safe_prev], 1.0)

        else:
            # rand / competitive crowding: gate proportional to 1/(vacant fraction)
            # Biological rationale: kinase is a limited resource; all unphosphorylated
            # sites compete for it. More vacant sites → lower per-site rate.
            # Gate = 1 / (CROWDING_BASELINE + CROWDING_WEIGHT * vacant_fraction + CROWDING_EPSILON)
            # The 0.5 + 0.5 decomposition ensures gate → 1 when all sites are occupied
            # and gate → 2 when no sites are occupied (maximum competition).
            CROWDING_BASELINE = 0.5   # minimum denominator contribution
            CROWDING_WEIGHT   = 0.5   # scales the vacant-fraction contribution
            CROWDING_EPSILON  = 1e-9  # numerical stability guard against zero division
            vacant_frac = 1.0 - mp[site_prot_idx]      # (N,)
            gate = 1.0 / (CROWDING_BASELINE + CROWDING_WEIGHT * vacant_frac + CROWDING_EPSILON)

        v_raw   = k_on_eff * (1.0 + coup_clamp) * gate * (1.0 - p)
        v_on    = v_raw / (1.0 + jnp.abs(v_raw))

        v_off_r = k_off * p
        v_off   = v_off_r / (1.0 + v_off_r)

        dp = v_on - v_off

        return jnp.concatenate([dS, dA, dKdyn, dp])

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
    theta                   : jax.Array, flat parameter vector
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
    _, _, _, _, _, _, alpha, _, _, _, _, _, _, _ = decode_theta_jax(theta, K, M, N)

    # 1. Phosphosite loss: weighted MSLE
    diff_p = P_data - P_sim
    f1 = jnp.sum(jnp.log1p(W_data * diff_p * diff_p)) / max(n_p, 1)

    # 2. Protein abundance loss
    if A_scaled.size > 0:
        A_sim_obs = A_sim[prot_idx_for_A, :]      # (K_obs, T_A)
        diff_A    = A_scaled - A_sim_obs
        f2 = jnp.sum(jnp.log1p(W_data_prot * diff_A * diff_A)) / max(n_A, 1)
    else:
        f2 = jnp.array(0.0)

    # 3. Regularisation: L2 + Laplacian network term
    reg     = reg_lambda * jnp.dot(theta, theta)
    reg_net = lambda_net * jnp.dot(alpha, L_alpha @ alpha)
    f3 = (reg + reg_net) / max(n_var, 1)

    return f1, f2, f3
