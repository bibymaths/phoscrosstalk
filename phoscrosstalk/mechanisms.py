#!/usr/bin/env python3
"""
mechanisms.py

This module provides JAX‑compatible right‑hand sides for the phospho‑network
model.
"""

from __future__ import annotations

import os

# Ensure double precision throughout JAX
os.environ["JAX_ENABLE_X64"] = "true"

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np


# ---------------------------------------------------------------------------
# Parameter decoding
# ---------------------------------------------------------------------------

def decode_theta(theta, K: int, M: int, N: int):
    """
    Decode the flat log-scale parameter vector into biological rate constants.

    ``k_act`` and ``s_prod`` are no longer optimisation variables; they are
    derived from experimental data (see :mod:`derived_rates`).  The parameter
    vector therefore has dimension ``2*K + 2 + 3*M + N + 4``.

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
    theta = jnp.asarray(theta, dtype=jnp.float64)

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
        return jnp.clip(v, jnp.float64(-20.0), jnp.float64(10.0))

    k_deact = jnp.exp(clip(log_k_deact))
    d_deg = jnp.exp(clip(log_d_deg))
    alpha = jnp.exp(clip(log_alpha))
    kK_act = jnp.exp(clip(log_kK_act))
    kK_deact = jnp.exp(clip(log_kK_deact))
    k_off = jnp.exp(clip(log_k_off))

    beta_g = jnp.exp(jnp.clip(log_beta_g, jnp.float64(-20.0), jnp.float64(10.0)))
    beta_l = jnp.exp(jnp.clip(log_beta_l, jnp.float64(-20.0), jnp.float64(10.0)))

    two = jnp.float64(2.0)

    gamma_S_p = two * jnp.tanh(raw_gamma[0])
    gamma_A_S = two * jnp.tanh(raw_gamma[1])
    gamma_A_p = two * jnp.tanh(raw_gamma[2])
    gamma_K_net = two * jnp.tanh(raw_gamma[3])

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
# Sequential‑mechanism topology helper
# ---------------------------------------------------------------------------

def compute_prev_site_idx(site_prot_idx: np.ndarray, N: int) -> np.ndarray:
    """
    Precompute the predecessor index for the sequential phosphorylation
    mechanism.

    Parameters
    ----------
    site_prot_idx : np.ndarray, shape (N,)
        The protein index for each phosphosite.
    N : int
        Number of phosphosites.

    Returns
    -------
    np.ndarray
        Array of length ``N`` containing the index of the previous site on
        the same protein, or ``-1`` for the first site.
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
    abundance_max: float = 5.0,
):
    """
    Construct a JAX‑compatible right‑hand side (RHS) function for use with
    ``diffrax.ODETerm``.

    The state vector is interpreted as ``y = [R_rna, S, A, Kdyn, p]`` with
    shapes ``(K,)``, ``(K,)``, ``(K,)``, ``(M,)`` and ``(N,)`` respectively.

    Mechanisms "dist", "seq" and "rand" are supported.
    """
    if mechanism not in {"dist", "seq", "rand"}:
        raise ValueError(
            f"Unknown mechanism '{mechanism}'. Use 'dist', 'seq', or 'rand'."
        )

    # ------------------------------------------------------------------
    # Constant fallbacks so RHS never branches on None during tracing
    # ------------------------------------------------------------------
    _k_act_const = jnp.ones(K, dtype=jnp.float64)
    _s_prod_const = jnp.full(K, 0.1, dtype=jnp.float64)

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

        t = jnp.asarray(t, dtype=jnp.float64)
        y = jnp.asarray(y, dtype=jnp.float64)

        theta = jnp.asarray(theta, dtype=jnp.float64)
        Cg = jnp.asarray(Cg, dtype=jnp.float64)
        Cl = jnp.asarray(Cl, dtype=jnp.float64)
        K_site_kin = jnp.asarray(K_site_kin, dtype=jnp.float64)
        R = jnp.asarray(R, dtype=jnp.float64)
        L_alpha = jnp.asarray(L_alpha, dtype=jnp.float64)
        receptor_mask_prot = jnp.asarray(receptor_mask_prot, dtype=jnp.float64)
        receptor_mask_kin = jnp.asarray(receptor_mask_kin, dtype=jnp.float64)

        site_prot_idx = jnp.asarray(site_prot_idx, dtype=jnp.int32)
        kin_to_prot_idx = jnp.asarray(kin_to_prot_idx, dtype=jnp.int32)
        prev_site_idx = jnp.asarray(prev_site_idx, dtype=jnp.int32)

        # Small constant to avoid division by zero in smooth denominators
        eps = jnp.float64(1e-8)
        kinase_basal = jnp.float64(0.05)

        # ------------------------------------------------------------------
        # Decode parameters
        # ------------------------------------------------------------------
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
        ) = decode_theta(theta, K, M, N)

        # ------------------------------------------------------------------
        # Derived external rates
        # ------------------------------------------------------------------
        # Smoothly enforce non‑negativity of k_act and s_prod.
        def smooth_pos0(x, eps=jnp.float64(1e-6)):
            return 0.5 * (x + jnp.sqrt(x * x + eps * eps)) - 0.5 * eps

        k_act_raw = _k_act_fn(t)
        s_prod_raw = _s_prod_fn(t)
        k_act = smooth_pos0(k_act_raw)
        s_prod = smooth_pos0(s_prod_raw)

        # ------------------------------------------------------------------
        # Unpack state without clipping
        # ------------------------------------------------------------------
        R_rna = y[:K]
        S = y[K : 2 * K]
        A = y[2 * K : 3 * K]
        Kdyn = y[3 * K : 3 * K + M]
        p = y[3 * K + M :]

        # S transform of p to guarantee non‑negativity in downstream
        # computations.
        p_pos = smooth_pos0(p)
        q = p_pos / (1.0 + p_pos)

        # Smooth external receptor stimulus.
        u = 1.0 / (1.0 + jnp.exp(-t / 0.1))

        # ------------------------------------------------------------------
        # Degree‑normalized network fields
        # ------------------------------------------------------------------
        def row_norm(x):
            return jnp.sqrt((jnp.sum(jnp.abs(x), axis=1)) ** 2 + eps ** 2)

        Cg_row_scale = row_norm(Cg)
        Cl_row_scale = row_norm(Cl)
        R_row_scale = row_norm(R)
        Ksk_row_scale = row_norm(K_site_kin)
        L_row_scale = row_norm(L_alpha)

        Cg_q = (Cg @ q) / Cg_row_scale
        Cl_q = (Cl @ q) / Cl_row_scale

        # Signed crosstalk field.
        # beta_g and beta_l are positive decoded
        # parameters,
        # so sign comes from graph structure / phosphosite state.

        coup_field = beta_g * Cg_q + beta_l * Cl_q
        coup = jnp.tanh(coup_field)
        coup_factor = jnp.exp(coup)

        # ------------------------------------------------------------------
        # Per‑protein aggregate phosphosite summaries
        # ------------------------------------------------------------------
        num_q = jnp.zeros(K, dtype=jnp.float64).at[site_prot_idx].add(q)
        den = jnp.zeros(K, dtype=jnp.float64).at[site_prot_idx].add(1.0)
        num_c = jnp.zeros(K, dtype=jnp.float64).at[site_prot_idx].add(coup)

        # Smoothly avoid division by zero using root‑sum‑square
        safe_den = jnp.sqrt(den ** 2 + eps ** 2)
        mq = num_q / safe_den
        mc = num_c / safe_den

        # ------------------------------------------------------------------
        # 0. mRNA / transcriptional state
        # ------------------------------------------------------------------
        _rna_relax = jnp.asarray(rna_relax, dtype=jnp.float64)

        rna_field = gamma_S_p * mq + mc + receptor_mask_prot * u
        rna_reg = jnp.exp(jnp.float64(0.5) * jnp.tanh(rna_field))
        R_target = rna_reg
        dR_rna = _rna_relax * (R_target - R_rna)

        # ------------------------------------------------------------------
        # 1. Protein signalling state S
        # ------------------------------------------------------------------

        S_field = gamma_S_p * mq + mc + receptor_mask_prot * u
        S_drive = 1.0 / (1.0 + jnp.exp(-S_field))
        # Smoothly saturate k_act: compute a fraction in (0,1)
        k_act_sig = k_act / (1.0 + k_act)

        # When S < 0 the first term is
        # positive and the second term is negative of a negative number,
        # resulting in dS > 0.  When S > 1 the (1-S) term becomes negative,
        # pushing dS downward.  No derivative guards are required.

        dS = k_act_sig * S_drive * (1.0 - S) - k_deact * S

        # ------------------------------------------------------------------
        # 2. Protein abundance state A
        # ------------------------------------------------------------------

        # Normalize s_prod so that A* ≈ A_init at baseline.
        A_basal_target = s_prod
        A_signal_mod = jnp.float64(0.5) * jnp.tanh(
            gamma_A_S * (S - jnp.float64(0.5))
            + jnp.float64(0.25) * gamma_A_p * (mq - jnp.float64(0.5))
        )
        s_eff = A_basal_target * (1.0 + A_signal_mod)

        # Smooth positive
        s_eff = smooth_pos0(s_eff)
        dA = s_eff - d_deg * A

        # ------------------------------------------------------------------
        # 3. Kinase dynamics Kdyn
        # ------------------------------------------------------------------

        # Substrate feedback from bounded phosphosite proxy q to kinase activity.
        u_sub = (R @ q) / R_row_scale
        # Stabilizing network diffusion / consensus term.
        u_net = -(L_alpha @ Kdyn) / L_row_scale

        # Protein context for kinases that map to model proteins.
        valid_prot = kin_to_prot_idx >= 0
        safe_p_idx = jnp.where(valid_prot, kin_to_prot_idx, 0)
        S_for_kin = S[safe_p_idx]

        # Keeping only S terms improves dynamic range of Kdyn while
        # still allowing inhibition via kK_deact.

        prot_contrib = gamma_A_S * S_for_kin
        prot_contrib = jnp.where(valid_prot, prot_contrib, 0.0)

        U = u_sub + gamma_K_net * u_net + prot_contrib + receptor_mask_kin * u

        # Nonnegative, saturating kinase activation drive.  A softplus on the
        # exponent avoids discontinuity at zero.  The logistic form ensures
        # K_drive remains in (kinase_basal, 1).

        K_drive = kinase_basal + (1.0 - kinase_basal) / (1.0 + jnp.exp(-U))
        dKdyn = kK_act * K_drive * (1.0 - Kdyn) - kK_deact * Kdyn

        # ------------------------------------------------------------------
        # 4. Phosphosite dynamics p
        # ------------------------------------------------------------------

        # Degree‑normalized kinase‑to‑site activation.
        kinase_signal = alpha * Kdyn
        k_on_eff = (K_site_kin @ kinase_signal) / Ksk_row_scale

        # Smooth positive
        k_on_eff = smooth_pos0(k_on_eff)

        # Mechanism‑specific gates

        if mechanism == "dist":
            gate = jnp.ones(N, dtype=jnp.float64)

        elif mechanism == "seq":
            safe_prev = jnp.where(prev_site_idx >= 0, prev_site_idx, 0)
            has_prev = prev_site_idx >= 0
            prev_occ = q[safe_prev]
            seq_half = jnp.float64(0.10)
            pred_enable = prev_occ / (seq_half + prev_occ)
            site_available = 1.0 - q
            gate = jnp.where(has_prev, pred_enable * site_available, site_available)

        else:  # "rand"

            occupied_frac = mq[site_prot_idx]
            gate = 1.0 / (1.0 + occupied_frac)

        # On/off fluxes.  The raw production rate v_on_raw is non‑negative
        # thanks to the smooth_pos0 above and the gate ∈ [0,1].

        A_site = A[site_prot_idx] / jnp.asarray(abundance_max, dtype=jnp.float64)
        v_on_raw = k_on_eff * coup_factor * gate * (1.0 + jnp.float64(0.5) * A_site)

        # Smooth positive
        v_on_raw = smooth_pos0(v_on_raw)

        # Nornmalizing the flux
        v_on = v_on_raw / (1.0 + v_on_raw)

        # First‑order loss of relative phosphosite signal uses the positive
        # proxy of p.  Negative p values are therefore pulled towards zero
        # rather than growing unbounded in the negative direction.

        v_off = k_off * p_pos
        dp = v_on - v_off

        # Concatenate derivatives for integration.  Boundary guards are
        # intentionally omitted: the flux forms above are designed so that
        # negative states are driven upward and states above their biological
        # maxima are driven downward without explicit clipping.  This ensures
        # differentiability of the entire RHS.

        return jnp.concatenate([dR_rna, dS, dA, dKdyn, dp])

    return rhs