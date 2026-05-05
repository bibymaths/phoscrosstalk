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
    abundance_max: float = 5.0,
):
    """
    Return a JAX-compatible RHS function for ``diffrax.ODETerm``.

    State layout:
        y = [R_rna, S, A, Kdyn, p]

    where:
        R_rna : shape (K,)   mRNA / transcriptional drive state
        S     : shape (K,)   protein signalling state, bounded [0, 1]
        A     : shape (K,)   protein abundance state
        Kdyn  : shape (M,)   kinase activity state, bounded [0, 1]
        p     : shape (N,)   relative phosphosite signal, nonnegative

    Args tuple:
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
        )

    Mechanisms:
        "dist" : distributive, independent site phosphorylation
        "seq"  : sequential phosphorylation with leaky predecessor gating
        "rand" : random/cooperative/crowding-aware phosphorylation
    """
    if mechanism not in {"dist", "seq", "rand"}:
        raise ValueError(
            f"Unknown mechanism '{mechanism}'. Use 'dist', 'seq', or 'rand'."
        )

    # ------------------------------------------------------------------
    # Constant fallbacks so RHS never branches on None during tracing
    # ------------------------------------------------------------------
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

        # Small constants used only for fixed numerical structure.
        eps = jnp.float32(1e-8)
        seq_leak = jnp.float32(1e-3)
        kinase_basal = jnp.float32(0.05)

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
        ) = decode_theta_jax(theta, K, M, N)

        # ------------------------------------------------------------------
        # Derived external rates
        # ------------------------------------------------------------------
        # k_act: protein signalling / mRNA drive input
        # s_prod: protein synthesis drive
        k_act = jnp.clip(_k_act_fn(t), 0.0, None)
        s_prod = jnp.clip(_s_prod_fn(t), 0.0, None)

        # ------------------------------------------------------------------
        # Unpack state
        # ------------------------------------------------------------------
        # Keep clipping for compatibility and numerical safety. The fluxes below
        # are written so that the bounded states are also naturally self-limiting.
        R_rna = jnp.clip(y[:K], 0.0, None)
        S = jnp.clip(y[K : 2 * K], 0.0, 1.0)
        A = jnp.clip(y[2 * K : 3 * K], 0.0, abundance_max)
        Kdyn = jnp.clip(y[3 * K : 3 * K + M], 0.0, 1.0)
        p = jnp.clip(y[3 * K + M :], 0.0, None)

        # Bounded proxy used only for occupancy-like regulation.
        # p itself remains relative phosphosite signal and can exceed 1.
        q = p / (1.0 + p)

        # Smooth external receptor stimulus.
        u = 1.0 / (1.0 + jnp.exp(-t / 0.1))

        # ------------------------------------------------------------------
        # Degree-normalized network fields
        # ------------------------------------------------------------------
        # These normalizations make the RHS more transferable across small and
        # large networks by turning raw graph sums into mean upstream evidence.
        Cg_row_scale = jnp.sum(jnp.abs(Cg), axis=1) + eps
        Cl_row_scale = jnp.sum(jnp.abs(Cl), axis=1) + eps
        R_row_scale = jnp.sum(jnp.abs(R), axis=1) + eps
        Ksk_row_scale = jnp.sum(jnp.abs(K_site_kin), axis=1) + eps
        L_row_scale = jnp.sum(jnp.abs(L_alpha), axis=1) + eps

        Cg_q = (Cg @ q) / Cg_row_scale
        Cl_q = (Cl @ q) / Cl_row_scale

        # Signed crosstalk field.
        # beta_g and beta_l are positive decoded parameters, so sign comes from
        # graph structure / phosphosite state. tanh bounds the field.
        coup_field = beta_g * Cg_q + beta_l * Cl_q
        coup = jnp.tanh(coup_field)

        # Positive multiplicative crosstalk factor.
        # Unlike clipping negative coupling to zero, this allows suppression
        # and enhancement:
        #     coup_factor in approximately [exp(-1), exp(1)]
        coup_factor = jnp.exp(coup)

        # ------------------------------------------------------------------
        # Per-protein aggregate phosphosite summaries
        # ------------------------------------------------------------------
        num_q = jnp.zeros(K, dtype=jnp.float32).at[site_prot_idx].add(q)
        den = jnp.zeros(K, dtype=jnp.float32).at[site_prot_idx].add(1.0)
        num_c = jnp.zeros(K, dtype=jnp.float32).at[site_prot_idx].add(coup)

        safe_den = jnp.where(den > 0.0, den, 1.0)
        mq = num_q / safe_den
        mc = num_c / safe_den

        # ------------------------------------------------------------------
        # 0. mRNA / transcriptional state
        # ------------------------------------------------------------------
        # Relaxation to the external/derived transcriptional drive.
        _rna_relax = jnp.float32(rna_relax)
        dR_rna = _rna_relax * (k_act - R_rna)

        # ------------------------------------------------------------------
        # 1. Protein signalling state S
        # ------------------------------------------------------------------
        # Regulatory field for signalling activation.
        # Use sigmoid rather than hard clipping 1 + field, so negative evidence
        # suppresses activation smoothly without creating dead regions.
        S_field = gamma_S_p * mq + mc + receptor_mask_prot * u
        S_drive = 1.0 / (1.0 + jnp.exp(-S_field))

        dS = k_act * S_drive * (1.0 - S) - k_deact * S

        # ------------------------------------------------------------------
        # 2. Protein abundance state A
        # ------------------------------------------------------------------
        # Positive synthesis modulation. This avoids hard zeroing of
        # s_prod * R_rna * (1 + gamma_A_S * S) when the parenthesis is negative.
        A_drive = 1.0 / (1.0 + jnp.exp(-(gamma_A_S * S)))
        s_eff = s_prod * R_rna * A_drive

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
        A_for_kin = A[safe_p_idx] / jnp.float32(abundance_max)

        prot_contrib = gamma_A_S * S_for_kin + gamma_A_p * A_for_kin
        prot_contrib = jnp.where(valid_prot, prot_contrib, 0.0)

        # Latent kinase activation field.
        U = u_sub + gamma_K_net * u_net + prot_contrib + receptor_mask_kin * u

        # Nonnegative, saturating kinase activation drive.
        # This prevents the kinase state from being structurally pinned at zero.
        K_drive = kinase_basal + (1.0 - kinase_basal) / (1.0 + jnp.exp(-U))

        dKdyn = kK_act * K_drive * (1.0 - Kdyn) - kK_deact * Kdyn

        # ------------------------------------------------------------------
        # 4. Phosphosite dynamics p
        # ------------------------------------------------------------------
        # Degree-normalized kinase-to-site activation.
        # This makes a site with many annotated kinases comparable to a site
        # with few annotated kinases.
        kinase_signal = alpha * Kdyn
        k_on_eff = (K_site_kin @ kinase_signal) / Ksk_row_scale
        k_on_eff = jnp.clip(k_on_eff, 0.0, None)

        if mechanism == "dist":
            # Distributive mechanism: each site can be phosphorylated independently.
            gate = jnp.ones(N, dtype=jnp.float32)

        elif mechanism == "seq":
            # Sequential mechanism: downstream site depends on predecessor site.
            # A small leak avoids exact structural blocking when the predecessor
            # bounded proxy q is near zero.
            safe_prev = jnp.where(prev_site_idx >= 0, prev_site_idx, 0)
            prev_occ = q[safe_prev]
            gate = jnp.where(
                prev_site_idx >= 0,
                seq_leak + (1.0 - seq_leak) * prev_occ,
                1.0,
            )

        else:
            # Random / crowding-aware mechanism.
            # As the protein-level bounded proxy (mq) rises, the gate
            # decreases smoothly, modelling crowding of available substrate.
            occupied_frac = mq[site_prot_idx]
            gate = 1.0 / (1.0 + occupied_frac)

        gate = jnp.clip(gate, 0.0, 1.0)

        # On/off fluxes.
        # v_on is positive, saturating, and includes:
        #   - kinase drive
        #   - signed network crosstalk as positive multiplicative factor
        #   - mechanism-specific gate
        #   - remaining unphosphorylated fraction
        # Protein abundance provides available substrate scale for relative phosphosite signal.  # noqa: E501
        A_site = A[site_prot_idx] / jnp.float32(abundance_max)
        A_site = jnp.clip(A_site, 0.0, None)

        v_on_raw = k_on_eff * coup_factor * gate * (1.0 + A_site)
        v_on_raw = jnp.clip(v_on_raw, 0.0, None)

        # Saturating production prevents runaway while still allowing p > 1.
        v_on = v_on_raw / (1.0 + v_on_raw)

        # First-order loss of relative phosphosite signal.
        # Do not saturate this too strongly; otherwise high p cannot come down.
        v_off = k_off * p

        dp = v_on - v_off

        # ------------------------------------------------------------------
        # Derivative boundary guards
        # ------------------------------------------------------------------
        # These are retained for compatibility and numerical safety.
        # The flux structure above already makes S, Kdyn, and p self-bounding.
        dR_rna = jnp.where((R_rna <= 0.0) & (dR_rna < 0.0), 0.0, dR_rna)

        dS = jnp.where((S <= 0.0) & (dS < 0.0), 0.0, dS)
        dS = jnp.where((S >= 1.0) & (dS > 0.0), 0.0, dS)

        dA = jnp.where((A <= 0.0) & (dA < 0.0), 0.0, dA)

        dKdyn = jnp.where((Kdyn <= 0.0) & (dKdyn < 0.0), 0.0, dKdyn)
        dKdyn = jnp.where((Kdyn >= 1.0) & (dKdyn > 0.0), 0.0, dKdyn)

        dp = jnp.where((p <= 0.0) & (dp < 0.0), 0.0, dp)

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
