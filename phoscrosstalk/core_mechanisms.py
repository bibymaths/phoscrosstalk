"""
core_mechanisms.py
Numba-compiled Right-Hand Side (RHS) functions for ODE integration.
"""
import math
import numpy as np
from numba import njit
from phoscrosstalk.config import ModelDims


@njit(cache=True)
def clip_scalar(x, lo, hi):
    if x < lo:
        return lo
    elif x > hi:
        return hi
    else:
        return x


@njit(cache=True)
def decode_theta(theta, K, M, N):
    idx0 = 0
    # protein rates (arrays)
    log_k_act = theta[idx0:idx0 + K];
    idx0 += K
    log_k_deact = theta[idx0:idx0 + K];
    idx0 += K
    log_s_prod = theta[idx0:idx0 + K];
    idx0 += K
    log_d_deg = theta[idx0:idx0 + K];
    idx0 += K

    # coupling (scalars)
    log_beta_g = theta[idx0];
    idx0 += 1
    log_beta_l = theta[idx0];
    idx0 += 1

    # kinase params (arrays)
    log_alpha = theta[idx0:idx0 + M];
    idx0 += M
    log_kK_act = theta[idx0:idx0 + M];
    idx0 += M
    log_kK_deact = theta[idx0:idx0 + M];
    idx0 += M

    # site params (array)
    log_k_off = theta[idx0:idx0 + N];
    idx0 += N

    # raw gamma (array of 4 scalars)
    raw_gamma = theta[idx0:idx0 + 4]

    k_act = np.exp(np.clip(log_k_act, -20.0, 10.0))
    k_deact = np.exp(np.clip(log_k_deact, -20.0, 10.0))
    s_prod = np.exp(np.clip(log_s_prod, -20.0, 10.0))
    d_deg = np.exp(np.clip(log_d_deg, -20.0, 10.0))
    alpha = np.exp(np.clip(log_alpha, -20.0, 10.0))
    kK_act = np.exp(np.clip(log_kK_act, -20.0, 10.0))
    kK_deact = np.exp(np.clip(log_kK_deact, -20.0, 10.0))
    k_off = np.exp(np.clip(log_k_off, -20.0, 10.0))
    beta_g = math.exp(clip_scalar(log_beta_g, -20.0, 10.0))
    beta_l = math.exp(clip_scalar(log_beta_l, -20.0, 10.0))
    gamma_S_p = 2.0 * np.tanh(raw_gamma[0])
    gamma_A_S = 2.0 * np.tanh(raw_gamma[1])
    gamma_A_p = 2.0 * np.tanh(raw_gamma[2])
    gamma_K_net = 2.0 * np.tanh(raw_gamma[3])

    return (k_act, k_deact, s_prod, d_deg, beta_g, beta_l, alpha,
            kK_act, kK_deact, k_off, gamma_S_p, gamma_A_S, gamma_A_p, gamma_K_net)


@njit(cache=True)
def network_rhs_nb_core_distributive(x, t, theta,
                        Cg, Cl, site_prot_idx, K_site_kin, R, L_alpha,
                        kin_to_prot_idx,
                        receptor_mask_prot, receptor_mask_kin,
                        K, M, N):
    """
    Numba-compiled RHS core (distributive mechanism) with:
      - explicit stimulus u(t) (step at t=0)
      - kinase ↔ protein coupling via kin_to_prot_idx
      - phosphorylation depending on protein abundance A
      - receptor vs downstream hierarchy via masks
    """

    (k_act, k_deact, s_prod, d_deg,
     beta_g, beta_l, alpha,
     kK_act, kK_deact, k_off,
     gamma_S_p, gamma_A_S, gamma_A_p, gamma_K_net) = decode_theta(theta, K, M, N)

    # unpack state
    S = x[0:K]
    A = x[K:2 * K]
    Kdyn = x[2 * K:2 * K + M]
    p = x[2 * K + M:2 * K + M + N]

    p = p.copy()
    Kdyn = Kdyn.copy()
    # -----------------------------------
    # external stimulus: step at t = 0
    # u(t) = 0 for t < 0 (pre-equilibration),
    # u(t) = 1 for t >= 0 (stimulated)
    # -----------------------------------
    # if t == 0.0:
    #     u = 0.0
    # else:
    #     u = 1.0

    u = 1.0 / (1.0 + np.exp(-(t) / 0.1))

    # ---------- site-level global + local context ----------
    coup_g = beta_g * (Cg @ p)  # (N,)
    coup_l = beta_l * (Cl @ p)  # (N,)
    coup = np.tanh(coup_g + coup_l)  # (N,)
    coup_pos = np.maximum(coup, 0.0)

    # per-protein aggregates
    num_p = np.zeros(K)
    num_c = np.zeros(K)
    den = np.zeros(K)

    for i in range(N):
        prot = site_prot_idx[i]
        num_p[prot] += p[i]
        num_c[prot] += coup[i]
        den[prot] += 1.0

    mean_p_per_prot = np.zeros(K)
    mean_ctx_per_prot = np.zeros(K)
    for k in range(K):
        if den[k] > 0.0:
            mean_p_per_prot[k] = num_p[k] / den[k]
            mean_ctx_per_prot[k] = num_c[k] / den[k]
        else:
            mean_p_per_prot[k] = 0.0
            mean_ctx_per_prot[k] = 0.0

    # ---------- 1) S dynamics ----------
    # baseline drive from phospho + context
    D_S = 1.0 + gamma_S_p * mean_p_per_prot + mean_ctx_per_prot

    # add external input only for receptor proteins
    for k in range(K):
        if receptor_mask_prot[k] == 1:
            D_S[k] += u

        if D_S[k] < 0.0:
            D_S[k] = 0.0

    dS = np.empty(K)
    for k in range(K):
        dS[k] = k_act[k] * D_S[k] * (1.0 - S[k]) - k_deact[k] * S[k]

    # ---------- 2) A dynamics ----------
    # abundance modulated by S (activation), but can be extended later
    s_eff = np.empty(K)
    for k in range(K):
        s_eff[k] = s_prod[k] * (1.0 + gamma_A_S * S[k])
        if s_eff[k] < 0.0:
            s_eff[k] = 0.0

    dA = np.empty(K)
    for k in range(K):
        dA[k] = s_eff[k] - d_deg[k] * A[k]

    # ---------- 3) Kdyn dynamics ----------
    # substrate signal
    u_sub = R @ p  # (M,)

    # kinase network / Laplacian
    if L_alpha.shape[0] > 0:
        u_net = -(L_alpha @ Kdyn)
    else:
        u_net = np.zeros(M)

    # base drive from substrate + network
    U = u_sub + gamma_K_net * u_net

    # couple kinase activity to host protein S and A
    for m in range(M):
        prot = kin_to_prot_idx[m]
        if prot >= 0:
            # S- and A-driven modulation (reuse gammas you already have)
            U[m] += gamma_A_S * S[prot] + gamma_A_p * A[prot]

        # external stimulus directly to receptor kinases
        if receptor_mask_kin[m] == 1:
            U[m] += u

    dK = np.empty(M)
    for m in range(M):
        # bounded activation via tanh
        act_term = np.tanh(U[m])
        dK[m] = kK_act[m] * act_term * (1.0 - Kdyn[m]) - kK_deact[m] * Kdyn[m]

    # ---------- 4) p dynamics (distributive, now A-dependent) ----------
    # effective on-rate, combining site→kinase, alpha, and Kdyn
    k_on_eff = K_site_kin @ (alpha * Kdyn)  # (N,)

    dp = np.empty(N)
    for i in range(N):
        # TEMP: ignore abundance dependence to decouple P from A for fitting
        # saturating abundance factor: A/(1+A) to avoid crazy rates
        # prot = site_prot_idx[i]
        # A_local = A[prot]
        # A_factor = A_local / (1.0 + A_local)
        # v_on  = k_on_eff[i] * (1.0 + coup_pos[i]) * A_factor * (1.0 - p[i])
        v_raw = k_on_eff[i] * (1.0 + coup_pos[i]) * (1.0 - p[i])
        # Saturate growth to kill runaway exponential
        v_on = v_raw / (1.0 + abs(v_raw))
        v_off = k_off[i] * p[i]
        v_off = v_off / (1.0 + v_off)
        dp[i] = v_on - v_off

    # pack
    dx = np.empty(2 * K + M + N)
    dx[0:K] = dS
    dx[K:2 * K] = dA
    dx[2 * K:2 * K + M] = dK
    dx[2 * K + M:2 * K + M + N] = dp

    # ======= STABILITY CLIPPING =======
    for k in range(K):
        if S[k] < 0.0: S[k] = 0.0
        if S[k] > 1.0: S[k] = 1.0

    for k in range(K):
        if A[k] < 0.0: A[k] = 0.0
        if A[k] > 5.0: A[k] = 5.0  # abundance is normalized

    for m in range(M):
        if Kdyn[m] < 0.0: Kdyn[m] = 0.0
        if Kdyn[m] > 1.0: Kdyn[m] = 1.0

    for i in range(N):
        if p[i] < 0.0: p[i] = 0.0
        if p[i] > 1.0: p[i] = 1.0

    return dx


@njit(cache=True)
def network_rhs_nb_core_sequential(x, t, theta,
                                   Cg, Cl, site_prot_idx, K_site_kin, R, L_alpha,
                                   kin_to_prot_idx,
                                   receptor_mask_prot, receptor_mask_kin,
                                   K, M, N):
    """
    RHS core with SEQUENTIAL multi-site phosphorylation per protein.
    Later sites on the same protein require the previous site (in index
    order) to be phosphorylated to activate strongly.
    """

    (k_act, k_deact, s_prod, d_deg,
     beta_g, beta_l, alpha,
     kK_act, kK_deact, k_off,
     gamma_S_p, gamma_A_S, gamma_A_p, gamma_K_net) = decode_theta(theta, K, M, N)

    # unpack state
    S = x[0:K]
    A = x[K:2 * K]
    Kdyn = x[2 * K:2 * K + M]
    p = x[2 * K + M:2 * K + M + N]

    p = p.copy()
    Kdyn = Kdyn.copy()

    # smooth external stimulus
    u = 1.0 / (1.0 + np.exp(-(t) / 0.1))

    # ---------- site-level global + local context ----------
    coup_g = beta_g * (Cg @ p)      # (N,)
    coup_l = beta_l * (Cl @ p)      # (N,)
    coup = np.tanh(coup_g + coup_l) # (N,)
    coup_pos = np.maximum(coup, 0.0)

    # per-protein aggregates
    num_p = np.zeros(K)
    num_c = np.zeros(K)
    den   = np.zeros(K)

    for i in range(N):
        prot = site_prot_idx[i]
        num_p[prot] += p[i]
        num_c[prot] += coup[i]
        den[prot]   += 1.0

    mean_p_per_prot   = np.zeros(K)
    mean_ctx_per_prot = np.zeros(K)
    for k in range(K):
        if den[k] > 0.0:
            mean_p_per_prot[k]   = num_p[k] / den[k]
            mean_ctx_per_prot[k] = num_c[k] / den[k]
        else:
            mean_p_per_prot[k]   = 0.0
            mean_ctx_per_prot[k] = 0.0

    # ---------- 1) S dynamics ----------
    D_S = 1.0 + gamma_S_p * mean_p_per_prot + mean_ctx_per_prot

    for k in range(K):
        if receptor_mask_prot[k] == 1:
            D_S[k] += u
        if D_S[k] < 0.0:
            D_S[k] = 0.0

    dS = np.empty(K)
    for k in range(K):
        dS[k] = k_act[k] * D_S[k] * (1.0 - S[k]) - k_deact[k] * S[k]

    # ---------- 2) A dynamics ----------
    s_eff = np.empty(K)
    for k in range(K):
        s_eff[k] = s_prod[k] * (1.0 + gamma_A_S * S[k])
        if s_eff[k] < 0.0:
            s_eff[k] = 0.0

    dA = np.empty(K)
    for k in range(K):
        dA[k] = s_eff[k] - d_deg[k] * A[k]

    # ---------- 3) Kdyn dynamics ----------
    u_sub = R @ p  # (M,)

    if L_alpha.shape[0] > 0:
        u_net = -(L_alpha @ Kdyn)
    else:
        u_net = np.zeros(M)

    U = u_sub + gamma_K_net * u_net

    for m in range(M):
        prot = kin_to_prot_idx[m]
        if prot >= 0:
            U[m] += gamma_A_S * S[prot] + gamma_A_p * A[prot]
        if receptor_mask_kin[m] == 1:
            U[m] += u

    dK = np.empty(M)
    for m in range(M):
        act_term = np.tanh(U[m])
        dK[m] = kK_act[m] * act_term * (1.0 - Kdyn[m]) - kK_deact[m] * Kdyn[m]

    # ---------- 4) p dynamics (SEQUENTIAL) ----------
    # effective on-rate
    k_on_eff = K_site_kin @ (alpha * Kdyn)  # (N,)

    # precompute predecessor index per site (per protein)
    prev_idx = np.empty(N, dtype=np.int64)
    last_idx = np.full(K, -1, dtype=np.int64)

    for i in range(N):
        prot = site_prot_idx[i]
        prev_idx[i] = last_idx[prot]
        last_idx[prot] = i

    dp = np.empty(N)
    for i in range(N):
        prot = site_prot_idx[i]
        j_prev = prev_idx[i]

        if j_prev == -1:
            # first site on this protein: behaves like distributive
            gate = 1.0
        else:
            # later sites gated by previous site's occupancy
            gate = p[j_prev]

        v_raw = k_on_eff[i] * (1.0 + coup_pos[i]) * gate * (1.0 - p[i])
        v_on  = v_raw / (1.0 + abs(v_raw))
        v_off = k_off[i] * p[i]
        v_off = v_off / (1.0 + v_off)
        dp[i] = v_on - v_off

    # pack
    dx = np.empty(2 * K + M + N)
    dx[0:K]               = dS
    dx[K:2 * K]           = dA
    dx[2 * K:2 * K + M]   = dK
    dx[2 * K + M:]        = dp

    # stability clipping on state variables (not on dx)
    for k in range(K):
        if S[k] < 0.0: S[k] = 0.0
        if S[k] > 1.0: S[k] = 1.0

    for k in range(K):
        if A[k] < 0.0: A[k] = 0.0
        if A[k] > 5.0: A[k] = 5.0

    for m in range(M):
        if Kdyn[m] < 0.0: Kdyn[m] = 0.0
        if Kdyn[m] > 1.0: Kdyn[m] = 1.0

    for i in range(N):
        if p[i] < 0.0: p[i] = 0.0
        if p[i] > 1.0: p[i] = 1.0

    return dx

@njit(cache=True)
def network_rhs_nb_core_random(x, t, theta,
                               Cg, Cl, site_prot_idx, K_site_kin, R, L_alpha,
                               kin_to_prot_idx,
                               receptor_mask_prot, receptor_mask_kin,
                               K, M, N):
    """
    RHS core with RANDOM / cooperative multi-site phosphorylation.
    Sites on the same protein do not have a fixed order; their on-rate
    increases with the mean phosphorylation level of that protein.
    """

    (k_act, k_deact, s_prod, d_deg,
     beta_g, beta_l, alpha,
     kK_act, kK_deact, k_off,
     gamma_S_p, gamma_A_S, gamma_A_p, gamma_K_net) = decode_theta(theta, K, M, N)

    # unpack state
    S = x[0:K]
    A = x[K:2 * K]
    Kdyn = x[2 * K:2 * K + M]
    p = x[2 * K + M:2 * K + M + N]

    p = p.copy()
    Kdyn = Kdyn.copy()

    # smooth external stimulus
    u = 1.0 / (1.0 + np.exp(-(t) / 0.1))

    # ---------- site-level global + local context ----------
    coup_g = beta_g * (Cg @ p)      # (N,)
    coup_l = beta_l * (Cl @ p)      # (N,)
    coup = np.tanh(coup_g + coup_l) # (N,)
    coup_pos = np.maximum(coup, 0.0)

    # per-protein aggregates
    num_p = np.zeros(K)
    num_c = np.zeros(K)
    den   = np.zeros(K)

    for i in range(N):
        prot = site_prot_idx[i]
        num_p[prot] += p[i]
        num_c[prot] += coup[i]
        den[prot]   += 1.0

    mean_p_per_prot   = np.zeros(K)
    mean_ctx_per_prot = np.zeros(K)
    for k in range(K):
        if den[k] > 0.0:
            mean_p_per_prot[k]   = num_p[k] / den[k]
            mean_ctx_per_prot[k] = num_c[k] / den[k]
        else:
            mean_p_per_prot[k]   = 0.0
            mean_ctx_per_prot[k] = 0.0

    # ---------- 1) S dynamics ----------
    D_S = 1.0 + gamma_S_p * mean_p_per_prot + mean_ctx_per_prot

    for k in range(K):
        if receptor_mask_prot[k] == 1:
            D_S[k] += u
        if D_S[k] < 0.0:
            D_S[k] = 0.0

    dS = np.empty(K)
    for k in range(K):
        dS[k] = k_act[k] * D_S[k] * (1.0 - S[k]) - k_deact[k] * S[k]

    # ---------- 2) A dynamics ----------
    s_eff = np.empty(K)
    for k in range(K):
        s_eff[k] = s_prod[k] * (1.0 + gamma_A_S * S[k])
        if s_eff[k] < 0.0:
            s_eff[k] = 0.0

    dA = np.empty(K)
    for k in range(K):
        dA[k] = s_eff[k] - d_deg[k] * A[k]

    # ---------- 3) Kdyn dynamics ----------
    u_sub = R @ p  # (M,)

    if L_alpha.shape[0] > 0:
        u_net = -(L_alpha @ Kdyn)
    else:
        u_net = np.zeros(M)

    U = u_sub + gamma_K_net * u_net

    for m in range(M):
        prot = kin_to_prot_idx[m]
        if prot >= 0:
            U[m] += gamma_A_S * S[prot] + gamma_A_p * A[prot]
        if receptor_mask_kin[m] == 1:
            U[m] += u

    dK = np.empty(M)
    for m in range(M):
        act_term = np.tanh(U[m])
        dK[m] = kK_act[m] * act_term * (1.0 - Kdyn[m]) - kK_deact[m] * Kdyn[m]

    # ---------- 4) p dynamics (RANDOM/cooperative) ----------
    k_on_eff = K_site_kin @ (alpha * Kdyn)  # (N,)

    dp = np.empty(N)
    for i in range(N):
        prot = site_prot_idx[i]
        # cooperative factor: more phospho on same protein → faster new sites
        coop = 1.0 + mean_p_per_prot[prot]

        v_raw = k_on_eff[i] * (1.0 + coup_pos[i]) * coop * (1.0 - p[i])
        v_on  = v_raw / (1.0 + abs(v_raw))
        v_off = k_off[i] * p[i]
        v_off = v_off / (1.0 + v_off)
        dp[i] = v_on - v_off

    # pack
    dx = np.empty(2 * K + M + N)
    dx[0:K]               = dS
    dx[K:2 * K]           = dA
    dx[2 * K:2 * K + M]   = dK
    dx[2 * K + M:]        = dp

    # stability clipping on state variables (not on dx)
    for k in range(K):
        if S[k] < 0.0: S[k] = 0.0
        if S[k] > 1.0: S[k] = 1.0

    for k in range(K):
        if A[k] < 0.0: A[k] = 0.0
        if A[k] > 5.0: A[k] = 5.0

    for m in range(M):
        if Kdyn[m] < 0.0: Kdyn[m] = 0.0
        if Kdyn[m] > 1.0: Kdyn[m] = 1.0

    for i in range(N):
        if p[i] < 0.0: p[i] = 0.0
        if p[i] > 1.0: p[i] = 1.0

    return dx

def network_rhs(x, t, theta, Cg, Cl, site_prot_idx, K_site_kin, R, L_alpha, kin_to_prot_idx,
                receptor_mask_prot, receptor_mask_kin, mech="dist"):
    K, M, N = ModelDims.K, ModelDims.M, ModelDims.N
    if mech == "dist":
        return network_rhs_nb_core_distributive(x, t, theta, Cg, Cl, site_prot_idx, K_site_kin, R, L_alpha,
                                                kin_to_prot_idx, receptor_mask_prot, receptor_mask_kin, K, M, N)
    elif mech == "seq":
        return network_rhs_nb_core_sequential(x, t, theta, Cg, Cl, site_prot_idx, K_site_kin, R, L_alpha,
                                              kin_to_prot_idx, receptor_mask_prot, receptor_mask_kin, K, M, N)
    elif mech == "rand":
        return network_rhs_nb_core_random(x, t, theta, Cg, Cl, site_prot_idx, K_site_kin, R, L_alpha, kin_to_prot_idx,
                                          receptor_mask_prot, receptor_mask_kin, K, M, N)
    else:
        raise ValueError(f"Unknown mechanism: {mech}")
