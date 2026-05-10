#!/usr/bin/env python3
"""
JAX-compatible right-hand sides for the phospho-network model.

State vector
------------
The ODE state is:

    y = [R_rna, S, A, Kdyn, p]

with blocks:

    R_rna : (K,)  latent mRNA / transcriptional state
    S     : (K,)  protein signalling state
    A     : (K,)  protein abundance state
    Kdyn  : (M,)  kinase activity state
    p     : (N,)  phosphosite state

Model design
------------
k_act(t) and s_prod(t) are no longer fitted parameters. They are derived
external rate functions, usually built in derived_rates.py and passed into
make_rhs().

This RHS assumes k_act(t) and s_prod(t) are already biologically bounded /
regularized by their factory functions. Therefore the RHS uses them directly
as positive rate magnitudes, with only a final smooth positivity guard.
"""

from __future__ import annotations

import os

# Must be set before JAX array creation. In main.py this is already handled
# earlier, but keeping it here makes this module safer when imported directly.
os.environ["JAX_ENABLE_X64"] = "true"

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# Parameter decoding
# ---------------------------------------------------------------------------

def decode_theta(theta, K: int, M: int, N: int):
    """
    Decode the flat parameter vector into biological rate constants.

    k_act and s_prod are not optimisation variables. They are derived from
    external data/network functions and passed through make_rhs().

    Current theta dimension
    -----------------------
        2*K + 2 + 3*M + N + 4

    Blocks
    ------
        log_k_deact     (K,)
        log_d_deg       (K,)
        log_beta_g      scalar
        log_beta_l      scalar
        log_alpha       (M,)
        log_kK_act      (M,)
        log_kK_deact    (M,)
        log_k_off       (N,)
        raw_gamma       (4,)

    Returns
    -------
    tuple
        k_deact, d_deg,
        beta_g, beta_l,
        alpha, kK_act, kK_deact,
        k_off,
        gamma_S_p, gamma_A_S, gamma_A_p, gamma_K_net

    Notes
    -----
    gamma_A_p is retained for backward compatibility with existing bounds,
    labels, reports, and saved parameter outputs. The reduced RHS below does
    not use gamma_A_p in the abundance equation because s_prod(t) already
    contains phosphosite/kinase-derived synthesis regulation.
    """
    theta = jnp.asarray(theta, dtype=jnp.float64)

    idx = 0

    log_k_deact = theta[idx: idx + K]
    idx += K

    log_d_deg = theta[idx: idx + K]
    idx += K

    log_beta_g = theta[idx]
    idx += 1

    log_beta_l = theta[idx]
    idx += 1

    log_alpha = theta[idx: idx + M]
    idx += M

    log_kK_act = theta[idx: idx + M]
    idx += M

    log_kK_deact = theta[idx: idx + M]
    idx += M

    log_k_off = theta[idx: idx + N]
    idx += N

    raw_gamma = theta[idx: idx + 4]

    def _clip_log(v):
        return jnp.clip(v, jnp.float64(-20.0), jnp.float64(10.0))

    k_deact = jnp.exp(_clip_log(log_k_deact))
    d_deg = jnp.exp(_clip_log(log_d_deg))

    beta_g = jnp.exp(_clip_log(log_beta_g))
    beta_l = jnp.exp(_clip_log(log_beta_l))

    alpha = jnp.exp(_clip_log(log_alpha))
    kK_act = jnp.exp(_clip_log(log_kK_act))
    kK_deact = jnp.exp(_clip_log(log_kK_deact))

    k_off = jnp.exp(_clip_log(log_k_off))

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
# Sequential-mechanism topology helper
# ---------------------------------------------------------------------------

def compute_prev_site_idx(site_prot_idx: np.ndarray, N: int) -> np.ndarray:
    """
    Precompute predecessor indices for sequential phosphorylation.

    For each phosphosite, this returns the previous site on the same protein,
    or -1 if the site is the first site observed for that protein.

    Args:
        site_prot_idx:
            Array of shape (N,) mapping each phosphosite to its protein index.
        N:
            Number of phosphosites.

    Returns:
        np.ndarray of shape (N,), dtype int32.
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
# Small smooth numerical helpers
# ---------------------------------------------------------------------------

def _smooth_pos0(x, eps=jnp.float64(1e-6)):
    """
    Smooth positive-part approximation.

    Returns approximately max(x, 0), but differentiably.

    This is used as a numerical safety guard. It should not be used as the main
    biological bounding transform for k_act/s_prod; that belongs in
    derived_rates.py.
    """
    return jnp.float64(0.5) * (x + jnp.sqrt(x * x + eps * eps)) - jnp.float64(0.5) * eps


def _row_l1_norm(x, eps):
    """
    Smooth row-wise L1 normalization denominator.

    Args:
        x:   Matrix.
        eps: Small positive scalar.

    Returns:
        Vector of row normalization scales.
    """
    return jnp.sqrt((jnp.sum(jnp.abs(x), axis=1)) ** 2 + eps ** 2)

def _derive_kinase_basal_hyperparams(
        K_site_kin,
        R,
        receptor_mask_kin,
        kK_act,
        kK_deact,
        *,
        eps=jnp.float64(1e-8),
):
    """
    Derive kinase basal hyperparameters from fitted kinetic parameters and
    existing kinase/network structure.

    Returns:
        basal_min:      scalar
        basal_max:      scalar
        receptor_bonus: scalar

    Interpretation:
        basal_min:
            Lower basal activation floor, derived from the lower tail of the
            fitted kinase activation/deactivation equilibrium.

        basal_max:
            Upper basal activation floor, derived from the upper tail of the
            same fitted equilibrium.

        receptor_bonus:
            Network-scale receptor bonus, derived from how strongly receptor
            kinases are represented in the existing kinase/support structure.
    """
    # ------------------------------------------------------------
    # 1. Fitted kinetic equilibrium proxy for each kinase
    # ------------------------------------------------------------
    # eq_m is in (0, 1): high when kK_act >> kK_deact.
    eq = kK_act / (kK_act + kK_deact + eps)

    # Robust tails. These are data/parameter-derived, not fixed constants.
    q_low = jnp.quantile(eq, jnp.float64(0.10))
    q_high = jnp.quantile(eq, jnp.float64(0.90))

    # Keep a small numerical floor only to avoid exact zero.
    basal_min = jnp.maximum(jnp.float64(1e-6), jnp.float64(0.25) * q_low)
    basal_max = jnp.maximum(basal_min + eps, jnp.float64(0.75) * q_high)

    # Do not allow basal_max to exceed the fitted equilibrium upper envelope.
    basal_max = jnp.minimum(basal_max, jnp.float64(0.95))

    # ------------------------------------------------------------
    # 2. Structural receptor bonus
    # ------------------------------------------------------------
    # K_site_kin: (N, M), kinase support by column.
    # R:          (M, N), kinase feedback support by row.
    site_support = jnp.sum(jnp.abs(K_site_kin), axis=0)      # (M,)
    feedback_support = jnp.sum(jnp.abs(R), axis=1)           # (M,)

    support = site_support + feedback_support
    receptor_mask = receptor_mask_kin > 0.0

    global_support = jnp.mean(support)

    receptor_support = jnp.where(
        jnp.any(receptor_mask),
        jnp.mean(jnp.where(receptor_mask, support, 0.0)),
        global_support,
    )

    # Bonus is relative receptor support above global support.
    receptor_enrichment = receptor_support / (global_support + eps)

    # Scale receptor bonus by the fitted kinase equilibrium dynamic range.
    receptor_bonus = (basal_max - basal_min) * receptor_enrichment

    # Avoid pathological dominance if receptor support is extreme.
    receptor_bonus = jnp.minimum(receptor_bonus, basal_max)

    return basal_min, basal_max, receptor_bonus

def _network_kinase_basal(
        K_site_kin,
        R,
        receptor_mask_kin,
        kK_act,
        kK_deact,
        *,
        eps=jnp.float64(1e-8),
):
    """
    Build a per-kinase basal activation floor from fitted kinase kinetics and
    existing network structure.

    Returns:
        kinase_basal: (M,) vector.
    """
    basal_min, basal_max, receptor_bonus = _derive_kinase_basal_hyperparams(
        K_site_kin=K_site_kin,
        R=R,
        receptor_mask_kin=receptor_mask_kin,
        kK_act=kK_act,
        kK_deact=kK_deact,
        eps=eps,
    )

    # Structural support per kinase.
    site_support = jnp.sum(jnp.abs(K_site_kin), axis=0)    # (M,)
    feedback_support = jnp.sum(jnp.abs(R), axis=1)         # (M,)

    raw_support = site_support + feedback_support

    # Receptor kinases get a derived bonus, not a fixed constant.
    raw_support = raw_support + receptor_bonus * receptor_mask_kin

    max_support = jnp.max(raw_support)
    support_norm = raw_support / (max_support + eps)

    kinase_basal = basal_min + (basal_max - basal_min) * support_norm

    return kinase_basal

def _derive_rna_relax_rate(
        k_deact,
        d_deg,
        *,
        eps=jnp.float64(1e-8),
):
    """
    Derive per-protein RNA relaxation rate from existing fitted kinetic rates.

    Uses the geometric mean of signalling deactivation and abundance degradation:

        rho_R[g] = sqrt(k_deact[g] * d_deg[g])

    This removes the external rna_relax hyperparameter while preserving a
    fitted, protein-specific RNA timescale.

    Returns:
        (K,) vector.
    """
    return jnp.sqrt(_smooth_pos0(k_deact) * _smooth_pos0(d_deg) + eps)

def _derive_rna_exp_scale(
        k_act,
        gamma_S_p,
        gamma_A_S,
        gamma_K_net,
        *,
        eps=jnp.float64(1e-8),
):
    """
    Derive the exponent scale for rna_reg from current derived rates and
    fitted coupling parameters.

    rna_reg uses:

        rna_reg = exp(scale * tanh(rna_field))

    Since exp(scale * tanh(.)) has an approximate symmetric fold range:

        [exp(-scale), exp(+scale)]

    the implied max/min fold span is exp(2 * scale). Therefore:

        scale = 0.5 * log(max(k_act) / min(k_act))

    The fallback scale is derived from fitted gamma magnitudes when k_act has
    little cross-protein variation.
    """
    k_pos = _smooth_pos0(k_act) + eps

    k_min = jnp.min(k_pos)
    k_max = jnp.max(k_pos)

    # Mathematically derived from symmetric exp range:
    # max/min = exp(2 * scale)
    k_span_scale = jnp.float64(0.5) * jnp.log((k_max + eps) / (k_min + eps))

    # Fallback from fitted coupling strength.
    gamma_rms = jnp.sqrt(
        (
            gamma_S_p ** 2
            + gamma_A_S ** 2
            + gamma_K_net ** 2
        )
        / jnp.float64(3.0)
        + eps
    )

    # Maps gamma_rms >= 0 to [0, 1), without adding a new hyperparameter.
    gamma_scale = gamma_rms / (jnp.float64(1.0) + gamma_rms)

    return jnp.maximum(k_span_scale, gamma_scale)
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
    Construct a JAX-compatible RHS function for diffrax.ODETerm.

    Supported mechanisms
    --------------------
        dist : distributive phosphorylation
        seq  : sequential phosphorylation
        rand : random phosphorylation

    Args:
        K:
            Number of proteins.
        M:
            Number of kinases.
        N:
            Number of phosphosites.
        mechanism:
            One of {"dist", "seq", "rand"}.
        k_act_fn:
            Optional callable k_act_fn(t) -> (K,). Should return a bounded,
            positive or near-positive protein activation-rate vector.
        s_prod_fn:
            Optional callable s_prod_fn(t) -> (K,). Should return a bounded,
            positive or near-positive synthesis-rate vector.
        rna_relax:
            Relaxation rate for latent R_rna state.
        abundance_max:
            Scaling constant for abundance contribution to phosphosite flux.

    Returns:
        rhs(t, y, args)
            JAX-compatible RHS callable.
    """
    if mechanism not in {"dist", "seq", "rand"}:
        raise ValueError(
            f"Unknown mechanism {mechanism!r}. Expected 'dist', 'seq', or 'rand'."
        )

    K = int(K)
    M = int(M)
    N = int(N)

    if K <= 0:
        raise ValueError(f"K must be positive. Got {K}.")

    if M < 0:
        raise ValueError(f"M must be non-negative. Got {M}.")

    if N <= 0:
        raise ValueError(f"N must be positive. Got {N}.")

    # Constant fallbacks. These are closed over so the traced RHS never branches
    # on None.
    _k_act_const = jnp.ones(K, dtype=jnp.float64)
    _s_prod_const = jnp.full(K, jnp.float64(0.1), dtype=jnp.float64)

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

    _rna_relax = jnp.asarray(rna_relax, dtype=jnp.float64)
    _abundance_max = jnp.asarray(abundance_max, dtype=jnp.float64)

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

        # ------------------------------------------------------------------
        # Convert dynamic/static inputs to JAX arrays
        # ------------------------------------------------------------------
        t = jnp.asarray(t, dtype=jnp.float64)
        y = jnp.asarray(y, dtype=jnp.float64)

        theta = jnp.asarray(theta, dtype=jnp.float64)

        Cg = jnp.asarray(Cg, dtype=jnp.float64)
        Cl = jnp.asarray(Cl, dtype=jnp.float64)
        K_site_kin = jnp.asarray(K_site_kin, dtype=jnp.float64)
        R = jnp.asarray(R, dtype=jnp.float64)
        L_alpha = jnp.asarray(L_alpha, dtype=jnp.float64)

        site_prot_idx = jnp.asarray(site_prot_idx, dtype=jnp.int32)
        kin_to_prot_idx = jnp.asarray(kin_to_prot_idx, dtype=jnp.int32)
        prev_site_idx = jnp.asarray(prev_site_idx, dtype=jnp.int32)

        receptor_mask_prot = jnp.asarray(receptor_mask_prot, dtype=jnp.float64)
        receptor_mask_kin = jnp.asarray(receptor_mask_kin, dtype=jnp.float64)

        eps = jnp.float64(1e-8)

        # ------------------------------------------------------------------
        # Decode fitted parameters
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
            _gamma_A_p_unused,
            gamma_K_net,
        ) = decode_theta(theta, K, M, N)

        kinase_basal = _network_kinase_basal(
            K_site_kin=K_site_kin,
            R=R,
            receptor_mask_kin=receptor_mask_kin,
            kK_act=kK_act,
            kK_deact=kK_deact,
            eps=eps,
        )

        # ------------------------------------------------------------------
        # External derived rates
        # ------------------------------------------------------------------
        # k_act_fn/s_prod_fn should already be bounded in derived_rates.py.
        # This smooth positive guard is only a final numerical safety layer.
        k_act = _smooth_pos0(_k_act_fn(t))
        s_prod = _smooth_pos0(_s_prod_fn(t))

        # ------------------------------------------------------------------
        # Unpack state
        # ------------------------------------------------------------------
        R_rna = y[:K]
        S = y[K: 2 * K]
        A = y[2 * K: 3 * K]
        Kdyn = y[3 * K: 3 * K + M]
        p = y[3 * K + M:]

        # Positive phosphosite proxy used in downstream bounded summaries.
        p_pos = _smooth_pos0(p)
        q = p_pos / (jnp.float64(1.0) + p_pos)

        # Smooth external receptor stimulus.
        # Equivalent to a near-step around t=0, but differentiable.
        u = jax.nn.sigmoid(t / jnp.float64(0.1))

        # ------------------------------------------------------------------
        # Degree-normalized network fields
        # ------------------------------------------------------------------
        Cg_row_scale = _row_l1_norm(Cg, eps)
        Cl_row_scale = _row_l1_norm(Cl, eps)
        R_row_scale = _row_l1_norm(R, eps)
        Ksk_row_scale = _row_l1_norm(K_site_kin, eps)
        L_row_scale = _row_l1_norm(L_alpha, eps)

        Cg_q = (Cg @ q) / Cg_row_scale
        Cl_q = (Cl @ q) / Cl_row_scale

        # Signed crosstalk field. beta_g and beta_l are positive; signs come
        # from Cg/Cl structure and q.
        coup_field = beta_g * Cg_q + beta_l * Cl_q
        coup = jnp.tanh(coup_field)

        # Multiplicative crosstalk factor in approximately [exp(-1), exp(1)].
        coup_factor = jnp.exp(coup)

        # ------------------------------------------------------------------
        # Per-protein phosphosite summaries
        # ------------------------------------------------------------------
        num_q = jnp.zeros(K, dtype=jnp.float64).at[site_prot_idx].add(q)
        den = jnp.zeros(K, dtype=jnp.float64).at[site_prot_idx].add(jnp.float64(1.0))
        num_c = jnp.zeros(K, dtype=jnp.float64).at[site_prot_idx].add(coup)

        safe_den = jnp.sqrt(den ** 2 + eps ** 2)

        mq = num_q / safe_den
        mc = num_c / safe_den

        # ------------------------------------------------------------------
        # 0. Latent mRNA / transcriptional state R_rna
        # ------------------------------------------------------------------
        rna_field = gamma_S_p * mq + mc + receptor_mask_prot * u

        rna_exp_scale = _derive_rna_exp_scale(
            k_act=k_act,
            gamma_S_p=gamma_S_p,
            gamma_A_S=gamma_A_S,
            gamma_K_net=gamma_K_net,
            eps=eps,
        )

        rna_relax_auto = _derive_rna_relax_rate(
            k_deact=k_deact,
            d_deg=d_deg,
            eps=eps,
        )

        # Positive bounded transcriptional target.
        rna_reg = jnp.exp(rna_exp_scale * jnp.tanh(rna_field))

        dR_rna = rna_relax_auto * (rna_reg - R_rna)

        # ------------------------------------------------------------------
        # 1. Protein signalling state S
        # ------------------------------------------------------------------
        S_field = gamma_S_p * mq + mc + receptor_mask_prot * u
        S_drive = jax.nn.sigmoid(S_field)

        # k_act is already a bounded external derived activation rate.
        # Do not re-compress it with k_act / (1 + k_act).
        dS = k_act * S_drive * (jnp.float64(1.0) - S) - k_deact * S

        # ------------------------------------------------------------------
        # 2. Protein abundance state A
        # ------------------------------------------------------------------
        # s_prod is already the bounded synthesis drive derived from
        # kinase/phosphosite information. Therefore, do not re-inject mq here.
        # S provides the internal signalling-state modulation.
        A_signal_mod = jnp.float64(0.5) * jnp.tanh(
            gamma_A_S * (S - jnp.float64(0.5))
        )

        s_eff = _smooth_pos0(s_prod * (jnp.float64(1.0) + A_signal_mod))
        dA = s_eff - d_deg * A

        # ------------------------------------------------------------------
        # 3. Kinase activity state Kdyn
        # ------------------------------------------------------------------
        # Substrate feedback from bounded phosphosite proxy q.
        u_sub = (R @ q) / R_row_scale

        # Stabilizing network diffusion / consensus term.
        u_net = -(L_alpha @ Kdyn) / L_row_scale

        # Protein context for kinases that map to model proteins.
        valid_prot = kin_to_prot_idx >= 0
        safe_p_idx = jnp.where(valid_prot, kin_to_prot_idx, 0)
        S_for_kin = S[safe_p_idx]

        prot_contrib = gamma_A_S * S_for_kin
        prot_contrib = jnp.where(valid_prot, prot_contrib, jnp.float64(0.0))

        U = (
            u_sub
            + gamma_K_net * u_net
            + prot_contrib
            + receptor_mask_kin * u
        )

        K_drive = kinase_basal + (
            jnp.float64(1.0) - kinase_basal
        ) * jax.nn.sigmoid(U)

        dKdyn = kK_act * K_drive * (jnp.float64(1.0) - Kdyn) - kK_deact * Kdyn

        # ------------------------------------------------------------------
        # 4. Phosphosite state p
        # ------------------------------------------------------------------
        kinase_signal = alpha * Kdyn
        k_on_eff = (K_site_kin @ kinase_signal) / Ksk_row_scale
        k_on_eff = _smooth_pos0(k_on_eff)

        if mechanism == "dist":
            gate = jnp.ones(N, dtype=jnp.float64)

        elif mechanism == "seq":
            safe_prev = jnp.where(prev_site_idx >= 0, prev_site_idx, 0)
            has_prev = prev_site_idx >= 0

            prev_occ = q[safe_prev]
            seq_half = jnp.float64(0.10)

            pred_enable = prev_occ / (seq_half + prev_occ + eps)
            site_available = jnp.float64(1.0) - q

            gate = jnp.where(
                has_prev,
                pred_enable * site_available,
                site_available,
            )

        else:
            # Random mechanism: more occupied proteins have lower availability.
            occupied_frac = mq[site_prot_idx]
            gate = jnp.float64(1.0) / (jnp.float64(1.0) + occupied_frac)

        # Abundance contribution to phosphorylation flux.
        A_site = A[site_prot_idx] / (_abundance_max + eps)
        A_site = _smooth_pos0(A_site)

        v_on_raw = (
            k_on_eff
            * coup_factor
            * gate
            * (jnp.float64(1.0) + jnp.float64(0.5) * A_site)
        )

        v_on_raw = _smooth_pos0(v_on_raw)

        # Saturated phosphorylation influx.
        v_on = v_on_raw / (jnp.float64(1.0) + v_on_raw)

        # First-order phosphosite loss.
        v_off = k_off * p_pos

        dp = v_on - v_off

        return jnp.concatenate([dR_rna, dS, dA, dKdyn, dp])

    return rhs