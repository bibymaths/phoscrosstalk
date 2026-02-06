#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
core_mechanisms_fast.py

High-performance RHS kernels for phospho-network ODEs.

Key improvements (without deleting any mechanism or dispatch logic):
1) Minimizes per-call allocations: writes directly into dx, avoids mean arrays, avoids coup_pos array.
2) Fixes sequential gating overhead: no prev_idx construction per call (O(N)) uses per-protein streaming state.
3) Adds fastmath=True to hot kernels.
4) Optional “workspace” mode: decode theta once per simulation + reuse buffers (major speedup for optimization loops).
5) Supports dense or CSR sparse matrices (Numba-friendly CSR struct + kernels).

Usage patterns:
A) Keep your existing solver call style (x,t,theta,...):
   - Use *_dense or *_csr variants based on matrix type.

B) Best performance (theta decoded once):
   - Build workspace (DenseWorkspace or CSRWorkspace), call ws.set_theta(theta),
     then integrate using ws.rhs_*(...) (still returns a fresh dx by default if you want safety).

Notes:
- This module does NOT remove your clipping logic, but it avoids extra full-array passes where feasible.
- Be careful: mutating x inside RHS is generally solver-hostile. Here clipping is applied to local copies
  for Kdyn and p S and A are not mutated by default.
"""

from __future__ import annotations

import math
import numpy as np
from numba import njit, int64, float64
from numba.experimental import jitclass

from phoscrosstalk.config import ModelDims


# -------------------------
# small utilities
# -------------------------

@njit(cache=True, fastmath=True)
def clip_scalar(x, lo, hi):
    if x < lo:
        return lo
    elif x > hi:
        return hi
    return x


@njit(cache=True, fastmath=True)
def decode_theta(theta, K, M, N):
    idx0 = 0
    log_k_act = theta[idx0:idx0 + K];
    idx0 += K
    log_k_deact = theta[idx0:idx0 + K];
    idx0 += K
    log_s_prod = theta[idx0:idx0 + K];
    idx0 += K
    log_d_deg = theta[idx0:idx0 + K];
    idx0 += K

    log_beta_g = theta[idx0];
    idx0 += 1
    log_beta_l = theta[idx0];
    idx0 += 1

    log_alpha = theta[idx0:idx0 + M];
    idx0 += M
    log_kK_act = theta[idx0:idx0 + M];
    idx0 += M
    log_kK_deact = theta[idx0:idx0 + M];
    idx0 += M

    log_k_off = theta[idx0:idx0 + N];
    idx0 += N
    raw_gamma = theta[idx0:idx0 + 4]

    # clip then exp
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

    gamma_S_p = 2.0 * math.tanh(raw_gamma[0])
    gamma_A_S = 2.0 * math.tanh(raw_gamma[1])
    gamma_A_p = 2.0 * math.tanh(raw_gamma[2])
    gamma_K_net = 2.0 * math.tanh(raw_gamma[3])

    return (k_act, k_deact, s_prod, d_deg, beta_g, beta_l, alpha,
            kK_act, kK_deact, k_off, gamma_S_p, gamma_A_S, gamma_A_p, gamma_K_net)


# -------------------------
# Dense matvec kernels
# -------------------------

@njit(cache=True, fastmath=True)
def dense_mv(A, x, out):
    # out = A @ x
    nrow = A.shape[0]
    ncol = A.shape[1]
    for i in range(nrow):
        s = 0.0
        Ai = A[i]
        for j in range(ncol):
            s += Ai[j] * x[j]
        out[i] = s


@njit(cache=True, fastmath=True)
def dense_mv_inplace_add(A, x, out, scale):
    # out += scale * (A @ x)
    nrow = A.shape[0]
    ncol = A.shape[1]
    for i in range(nrow):
        s = 0.0
        Ai = A[i]
        for j in range(ncol):
            s += Ai[j] * x[j]
        out[i] += scale * s


# -------------------------
# CSR sparse support (Numba-friendly)
# -------------------------

csr_spec = [
    ("data", float64[:]),
    ("indices", int64[:]),
    ("indptr", int64[:]),
    ("n_rows", int64),
    ("n_cols", int64),
]


@jitclass(csr_spec)
class CSRMatrix:
    def __init__(self, data, indices, indptr, n_rows, n_cols):
        self.data = data
        self.indices = indices
        self.indptr = indptr
        self.n_rows = n_rows
        self.n_cols = n_cols


def csr_from_scipy(A_csr) -> CSRMatrix:
    """
    Convert scipy.sparse.csr_matrix to CSRMatrix (Numba jitclass).
    Call this outside njit (python-side).
    """
    data = np.asarray(A_csr.data, dtype=np.float64)
    indices = np.asarray(A_csr.indices, dtype=np.int64)
    indptr = np.asarray(A_csr.indptr, dtype=np.int64)
    n_rows, n_cols = A_csr.shape
    return CSRMatrix(data, indices, indptr, n_rows, n_cols)


@njit(cache=True, fastmath=True)
def csr_mv(A, x, out):
    # out = A @ x
    for i in range(A.n_rows):
        s = 0.0
        start = A.indptr[i]
        end = A.indptr[i + 1]
        for k in range(start, end):
            s += A.data[k] * x[A.indices[k]]
        out[i] = s


@njit(cache=True, fastmath=True)
def csr_mv_inplace_add(A, x, out, scale):
    # out += scale * (A @ x)
    for i in range(A.n_rows):
        s = 0.0
        start = A.indptr[i]
        end = A.indptr[i + 1]
        for k in range(start, end):
            s += A.data[k] * x[A.indices[k]]
        out[i] += scale * s


# -------------------------
# Core RHS: DENSE
# -------------------------

@njit(cache=True, fastmath=True)
def _rhs_common_dense(x, t,
                      # decoded params
                      k_act, k_deact, s_prod, d_deg,
                      beta_g, beta_l, alpha,
                      kK_act, kK_deact, k_off,
                      gamma_S_p, gamma_A_S, gamma_A_p, gamma_K_net,
                      # model structures
                      Cg, Cl, site_prot_idx, K_site_kin, R, L_alpha,
                      kin_to_prot_idx, receptor_mask_prot, receptor_mask_kin,
                      K, M, N,
                      mech_code,
                      # reusable buffers
                      p_buf, coup_buf, num_p, num_c, den, u_sub, u_net, k_on_eff, last_occ, has_prev,
                      dx_out):
    """
    One dense kernel that handles mech_code:
      0 dist
      1 seq
      2 rand
    Writes into dx_out (preallocated).
    """

    S = x[0:K]
    A = x[K:2 * K]
    Kdyn0 = x[2 * K:2 * K + M]
    p0 = x[2 * K + M:2 * K + M + N]

    # Local copies for safe clipping if desired (do NOT mutate x)
    # (kept because you had p.copy(), Kdyn.copy())
    for m in range(M):
        v = Kdyn0[m]
        if v < 0.0: v = 0.0
        if v > 1.0: v = 1.0
        u_net[m] = v  # temporarily store clipped Kdyn in u_net buffer
    Kdyn = u_net  # alias

    # clip p0 into p_buf
    for i in range(N):
        v = p0[i]
        if v < 0.0: v = 0.0
        if v > 1.0: v = 1.0
        p_buf[i] = v
    p = p_buf

    # smooth external stimulus
    u = 1.0 / (1.0 + math.exp(-(t) / 0.1))

    # coup = tanh(beta_g*(Cg@p) + beta_l*(Cl@p))
    # compute coupling into coup_buf
    dense_mv(Cg, p, dx_out[0:N])  # temp
    for i in range(N):
        dx_out[i] *= beta_g
    dense_mv_inplace_add(Cl, p, dx_out[0:N], beta_l)

    for i in range(N):
        coup_buf[i] = math.tanh(dx_out[i])

    coup = coup_buf

    # aggregates per protein
    for k in range(K):
        num_p[k] = 0.0
        num_c[k] = 0.0
        den[k] = 0.0

    for i in range(N):
        prot = site_prot_idx[i]
        num_p[prot] += p[i]
        num_c[prot] += coup[i]
        den[prot] += 1.0

    # 1) S and 2) A directly into dx_out
    for k in range(K):
        if den[k] > 0.0:
            mp = num_p[k] / den[k]
            mc = num_c[k] / den[k]
        else:
            mp = 0.0
            mc = 0.0

        D_S = 1.0 + gamma_S_p * mp + mc
        if receptor_mask_prot[k] == 1:
            D_S += u
        if D_S < 0.0:
            D_S = 0.0

        dx_out[k] = k_act[k] * D_S * (1.0 - S[k]) - k_deact[k] * S[k]

        s_eff = s_prod[k] * (1.0 + gamma_A_S * S[k])
        if s_eff < 0.0:
            s_eff = 0.0
        dx_out[K + k] = s_eff - d_deg[k] * A[k]

    # 3) Kdyn dynamics
    dense_mv(R, p, u_sub)  # u_sub = R @ p  (M,)

    if L_alpha.shape[0] > 0 and gamma_K_net != 0.0:
        dense_mv(L_alpha, Kdyn, u_net)  # u_net = L_alpha @ Kdyn
        for m in range(M):
            u_net[m] = -u_net[m]
    else:
        for m in range(M):
            u_net[m] = 0.0

    for m in range(M):
        U = u_sub[m] + gamma_K_net * u_net[m]

        prot = kin_to_prot_idx[m]
        if prot >= 0:
            U += gamma_A_S * S[prot] + gamma_A_p * A[prot]
        if receptor_mask_kin[m] == 1:
            U += u

        act_term = math.tanh(U)
        dx_out[2 * K + m] = kK_act[m] * act_term * (1.0 - Kdyn[m]) - kK_deact[m] * Kdyn[m]

    # 4) p dynamics
    # k_on_eff = K_site_kin @ (alpha*Kdyn)
    for m in range(M):
        u_sub[m] = alpha[m] * Kdyn[m]  # reuse u_sub as alpha*Kdyn
    dense_mv(K_site_kin, u_sub, k_on_eff)

    base = 2 * K + M

    if mech_code == 0:
        # distributive
        for i in range(N):
            cp = coup[i]
            if cp < 0.0:
                cp = 0.0

            v_raw = k_on_eff[i] * (1.0 + cp) * (1.0 - p[i])
            v_on = v_raw / (1.0 + math.fabs(v_raw))

            v_off = k_off[i] * p[i]
            v_off = v_off / (1.0 + v_off)

            dx_out[base + i] = v_on - v_off

    elif mech_code == 1:
        # sequential: streaming predecessor occupancy per protein
        for k in range(K):
            last_occ[k] = 0.0
            has_prev[k] = 0.0

        for i in range(N):
            prot = site_prot_idx[i]
            gate = 1.0 if has_prev[prot] == 0.0 else last_occ[prot]

            cp = coup[i]
            if cp < 0.0:
                cp = 0.0

            v_raw = k_on_eff[i] * (1.0 + cp) * gate * (1.0 - p[i])
            v_on = v_raw / (1.0 + math.fabs(v_raw))

            v_off = k_off[i] * p[i]
            v_off = v_off / (1.0 + v_off)

            dx_out[base + i] = v_on - v_off

            has_prev[prot] = 1.0
            last_occ[prot] = p[i]

    else:
        # random/cooperative: coop = 1 + mean_p_per_prot[prot]
        # compute mean_p_per_prot on demand via num_p/den
        for i in range(N):
            prot = site_prot_idx[i]
            if den[prot] > 0.0:
                coop = 1.0 + (num_p[prot] / den[prot])
            else:
                coop = 1.0

            cp = coup[i]
            if cp < 0.0:
                cp = 0.0

            v_raw = k_on_eff[i] * (1.0 + cp) * coop * (1.0 - p[i])
            v_on = v_raw / (1.0 + math.fabs(v_raw))

            v_off = k_off[i] * p[i]
            v_off = v_off / (1.0 + v_off)

            dx_out[base + i] = v_on - v_off

    return dx_out


@njit(cache=True, fastmath=True)
def rhs_dense_onecall(x, t, theta,
                      Cg, Cl, site_prot_idx, K_site_kin, R, L_alpha,
                      kin_to_prot_idx,
                      receptor_mask_prot, receptor_mask_kin,
                      K, M, N,
                      mech_code):
    """
    Dense RHS with per-call theta decode (slower than workspace, but drop-in replacement).
    """
    (k_act, k_deact, s_prod, d_deg,
     beta_g, beta_l, alpha,
     kK_act, kK_deact, k_off,
     gamma_S_p, gamma_A_S, gamma_A_p, gamma_K_net) = decode_theta(theta, K, M, N)

    p_buf = np.empty(N)
    coup_buf = np.empty(N)
    num_p = np.empty(K)
    num_c = np.empty(K)
    den = np.empty(K)
    u_sub = np.empty(M)
    u_net = np.empty(M)
    k_on_eff = np.empty(N)
    last_occ = np.empty(K)
    has_prev = np.empty(K)
    dx = np.empty(2 * K + M + N)

    return _rhs_common_dense(x, t,
                             k_act, k_deact, s_prod, d_deg,
                             beta_g, beta_l, alpha,
                             kK_act, kK_deact, k_off,
                             gamma_S_p, gamma_A_S, gamma_A_p, gamma_K_net,
                             Cg, Cl, site_prot_idx, K_site_kin, R, L_alpha,
                             kin_to_prot_idx, receptor_mask_prot, receptor_mask_kin,
                             K, M, N, mech_code,
                             p_buf, coup_buf, num_p, num_c, den, u_sub, u_net, k_on_eff, last_occ, has_prev,
                             dx)


# -------------------------
# Core RHS: CSR sparse
# -------------------------

@njit(cache=True, fastmath=True)
def _rhs_common_csr(x, t,
                    k_act, k_deact, s_prod, d_deg,
                    beta_g, beta_l, alpha,
                    kK_act, kK_deact, k_off,
                    gamma_S_p, gamma_A_S, gamma_A_p, gamma_K_net,
                    Cg, Cl, site_prot_idx, K_site_kin, R, L_alpha,
                    kin_to_prot_idx, receptor_mask_prot, receptor_mask_kin,
                    K, M, N,
                    mech_code,
                    p_buf, coup_buf, num_p, num_c, den, u_sub, u_net, k_on_eff, last_occ, has_prev,
                    dx_out):
    S = x[0:K]
    A = x[K:2 * K]
    Kdyn0 = x[2 * K:2 * K + M]
    p0 = x[2 * K + M:2 * K + M + N]

    # clip local copies into buffers
    for m in range(M):
        v = Kdyn0[m]
        if v < 0.0: v = 0.0
        if v > 1.0: v = 1.0
        u_net[m] = v
    Kdyn = u_net

    for i in range(N):
        v = p0[i]
        if v < 0.0: v = 0.0
        if v > 1.0: v = 1.0
        p_buf[i] = v
    p = p_buf

    u = 1.0 / (1.0 + math.exp(-(t) / 0.1))

    # coup = tanh(beta_g*(Cg@p) + beta_l*(Cl@p))
    csr_mv(Cg, p, dx_out[0:N])
    for i in range(N):
        dx_out[i] *= beta_g
    csr_mv_inplace_add(Cl, p, dx_out[0:N], beta_l)

    for i in range(N):
        coup_buf[i] = math.tanh(dx_out[i])
    coup = coup_buf

    for k in range(K):
        num_p[k] = 0.0
        num_c[k] = 0.0
        den[k] = 0.0

    for i in range(N):
        prot = site_prot_idx[i]
        num_p[prot] += p[i]
        num_c[prot] += coup[i]
        den[prot] += 1.0

    for k in range(K):
        if den[k] > 0.0:
            mp = num_p[k] / den[k]
            mc = num_c[k] / den[k]
        else:
            mp = 0.0
            mc = 0.0

        D_S = 1.0 + gamma_S_p * mp + mc
        if receptor_mask_prot[k] == 1:
            D_S += u
        if D_S < 0.0:
            D_S = 0.0

        dx_out[k] = k_act[k] * D_S * (1.0 - S[k]) - k_deact[k] * S[k]

        s_eff = s_prod[k] * (1.0 + gamma_A_S * S[k])
        if s_eff < 0.0:
            s_eff = 0.0
        dx_out[K + k] = s_eff - d_deg[k] * A[k]

    csr_mv(R, p, u_sub)

    if L_alpha.n_rows > 0 and gamma_K_net != 0.0:
        csr_mv(L_alpha, Kdyn, u_net)
        for m in range(M):
            u_net[m] = -u_net[m]
    else:
        for m in range(M):
            u_net[m] = 0.0

    for m in range(M):
        U = u_sub[m] + gamma_K_net * u_net[m]
        prot = kin_to_prot_idx[m]
        if prot >= 0:
            U += gamma_A_S * S[prot] + gamma_A_p * A[prot]
        if receptor_mask_kin[m] == 1:
            U += u

        act_term = math.tanh(U)
        dx_out[2 * K + m] = kK_act[m] * act_term * (1.0 - Kdyn[m]) - kK_deact[m] * Kdyn[m]

    for m in range(M):
        u_sub[m] = alpha[m] * Kdyn[m]

    csr_mv(K_site_kin, u_sub, k_on_eff)

    base = 2 * K + M

    if mech_code == 0:
        for i in range(N):
            cp = coup[i]
            if cp < 0.0:
                cp = 0.0
            v_raw = k_on_eff[i] * (1.0 + cp) * (1.0 - p[i])
            v_on = v_raw / (1.0 + math.fabs(v_raw))
            v_off = k_off[i] * p[i]
            v_off = v_off / (1.0 + v_off)
            dx_out[base + i] = v_on - v_off

    elif mech_code == 1:
        for k in range(K):
            last_occ[k] = 0.0
            has_prev[k] = 0.0
        for i in range(N):
            prot = site_prot_idx[i]
            gate = 1.0 if has_prev[prot] == 0.0 else last_occ[prot]
            cp = coup[i]
            if cp < 0.0:
                cp = 0.0
            v_raw = k_on_eff[i] * (1.0 + cp) * gate * (1.0 - p[i])
            v_on = v_raw / (1.0 + math.fabs(v_raw))
            v_off = k_off[i] * p[i]
            v_off = v_off / (1.0 + v_off)
            dx_out[base + i] = v_on - v_off
            has_prev[prot] = 1.0
            last_occ[prot] = p[i]
    else:
        for i in range(N):
            prot = site_prot_idx[i]
            if den[prot] > 0.0:
                coop = 1.0 + (num_p[prot] / den[prot])
            else:
                coop = 1.0
            cp = coup[i]
            if cp < 0.0:
                cp = 0.0
            v_raw = k_on_eff[i] * (1.0 + cp) * coop * (1.0 - p[i])
            v_on = v_raw / (1.0 + math.fabs(v_raw))
            v_off = k_off[i] * p[i]
            v_off = v_off / (1.0 + v_off)
            dx_out[base + i] = v_on - v_off

    return dx_out


@njit(cache=True, fastmath=True)
def rhs_csr_onecall(x, t, theta,
                    Cg, Cl, site_prot_idx, K_site_kin, R, L_alpha,
                    kin_to_prot_idx,
                    receptor_mask_prot, receptor_mask_kin,
                    K, M, N,
                    mech_code):
    (k_act, k_deact, s_prod, d_deg,
     beta_g, beta_l, alpha,
     kK_act, kK_deact, k_off,
     gamma_S_p, gamma_A_S, gamma_A_p, gamma_K_net) = decode_theta(theta, K, M, N)

    p_buf = np.empty(N)
    coup_buf = np.empty(N)
    num_p = np.empty(K)
    num_c = np.empty(K)
    den = np.empty(K)
    u_sub = np.empty(M)
    u_net = np.empty(M)
    k_on_eff = np.empty(N)
    last_occ = np.empty(K)
    has_prev = np.empty(K)
    dx = np.empty(2 * K + M + N)

    return _rhs_common_csr(x, t,
                           k_act, k_deact, s_prod, d_deg,
                           beta_g, beta_l, alpha,
                           kK_act, kK_deact, k_off,
                           gamma_S_p, gamma_A_S, gamma_A_p, gamma_K_net,
                           Cg, Cl, site_prot_idx, K_site_kin, R, L_alpha,
                           kin_to_prot_idx, receptor_mask_prot, receptor_mask_kin,
                           K, M, N, mech_code,
                           p_buf, coup_buf, num_p, num_c, den, u_sub, u_net, k_on_eff, last_occ, has_prev,
                           dx)


# -------------------------
# Workspace mode (biggest win for optimization loops)
# -------------------------

dense_ws_spec = [
    ("K", int64), ("M", int64), ("N", int64),
    ("k_act", float64[:]), ("k_deact", float64[:]), ("s_prod", float64[:]), ("d_deg", float64[:]),
    ("alpha", float64[:]), ("kK_act", float64[:]), ("kK_deact", float64[:]), ("k_off", float64[:]),
    ("beta_g", float64), ("beta_l", float64),
    ("gamma_S_p", float64), ("gamma_A_S", float64), ("gamma_A_p", float64), ("gamma_K_net", float64),
    # buffers
    ("p_buf", float64[:]), ("coup_buf", float64[:]), ("num_p", float64[:]), ("num_c", float64[:]), ("den", float64[:]),
    ("u_sub", float64[:]), ("u_net", float64[:]), ("k_on_eff", float64[:]),
    ("last_occ", float64[:]), ("has_prev", float64[:]),
    ("dx", float64[:]),
]


@jitclass(dense_ws_spec)
class DenseWorkspace:
    def __init__(self, K, M, N):
        self.K = K
        self.M = M
        self.N = N

        self.k_act = np.ones(K)
        self.k_deact = np.ones(K)
        self.s_prod = np.ones(K)
        self.d_deg = np.ones(K)

        self.alpha = np.ones(M)
        self.kK_act = np.ones(M)
        self.kK_deact = np.ones(M)
        self.k_off = np.ones(N)

        self.beta_g = 1.0
        self.beta_l = 1.0
        self.gamma_S_p = 0.0
        self.gamma_A_S = 0.0
        self.gamma_A_p = 0.0
        self.gamma_K_net = 0.0

        self.p_buf = np.empty(N)
        self.coup_buf = np.empty(N)
        self.num_p = np.empty(K)
        self.num_c = np.empty(K)
        self.den = np.empty(K)
        self.u_sub = np.empty(M)
        self.u_net = np.empty(M)
        self.k_on_eff = np.empty(N)
        self.last_occ = np.empty(K)
        self.has_prev = np.empty(K)
        self.dx = np.empty(2 * K + M + N)

    def set_theta(self, theta):
        (k_act, k_deact, s_prod, d_deg,
         beta_g, beta_l, alpha,
         kK_act, kK_deact, k_off,
         gamma_S_p, gamma_A_S, gamma_A_p, gamma_K_net) = decode_theta(theta, self.K, self.M, self.N)

        self.k_act[:] = k_act
        self.k_deact[:] = k_deact
        self.s_prod[:] = s_prod
        self.d_deg[:] = d_deg

        self.alpha[:] = alpha
        self.kK_act[:] = kK_act
        self.kK_deact[:] = kK_deact
        self.k_off[:] = k_off

        self.beta_g = beta_g
        self.beta_l = beta_l

        self.gamma_S_p = gamma_S_p
        self.gamma_A_S = gamma_A_S
        self.gamma_A_p = gamma_A_p
        self.gamma_K_net = gamma_K_net

    def rhs(self, x, t,
            Cg, Cl, site_prot_idx, K_site_kin, R, L_alpha,
            kin_to_prot_idx, receptor_mask_prot, receptor_mask_kin,
            mech_code,
            return_copy):
        out = _rhs_common_dense(x, t,
                                self.k_act, self.k_deact, self.s_prod, self.d_deg,
                                self.beta_g, self.beta_l, self.alpha,
                                self.kK_act, self.kK_deact, self.k_off,
                                self.gamma_S_p, self.gamma_A_S, self.gamma_A_p, self.gamma_K_net,
                                Cg, Cl, site_prot_idx, K_site_kin, R, L_alpha,
                                kin_to_prot_idx, receptor_mask_prot, receptor_mask_kin,
                                self.K, self.M, self.N, mech_code,
                                self.p_buf, self.coup_buf, self.num_p, self.num_c, self.den,
                                self.u_sub, self.u_net, self.k_on_eff, self.last_occ, self.has_prev,
                                self.dx)
        if return_copy == 1:
            return out.copy()
        return out


csr_ws_spec = [
    ("K", int64), ("M", int64), ("N", int64),
    ("k_act", float64[:]), ("k_deact", float64[:]), ("s_prod", float64[:]), ("d_deg", float64[:]),
    ("alpha", float64[:]), ("kK_act", float64[:]), ("kK_deact", float64[:]), ("k_off", float64[:]),
    ("beta_g", float64), ("beta_l", float64),
    ("gamma_S_p", float64), ("gamma_A_S", float64), ("gamma_A_p", float64), ("gamma_K_net", float64),
    ("p_buf", float64[:]), ("coup_buf", float64[:]), ("num_p", float64[:]), ("num_c", float64[:]), ("den", float64[:]),
    ("u_sub", float64[:]), ("u_net", float64[:]), ("k_on_eff", float64[:]),
    ("last_occ", float64[:]), ("has_prev", float64[:]),
    ("dx", float64[:]),
]


@jitclass(csr_ws_spec)
class CSRWorkspace:
    def __init__(self, K, M, N):
        self.K = K
        self.M = M
        self.N = N

        self.k_act = np.ones(K)
        self.k_deact = np.ones(K)
        self.s_prod = np.ones(K)
        self.d_deg = np.ones(K)

        self.alpha = np.ones(M)
        self.kK_act = np.ones(M)
        self.kK_deact = np.ones(M)
        self.k_off = np.ones(N)

        self.beta_g = 1.0
        self.beta_l = 1.0
        self.gamma_S_p = 0.0
        self.gamma_A_S = 0.0
        self.gamma_A_p = 0.0
        self.gamma_K_net = 0.0

        self.p_buf = np.empty(N)
        self.coup_buf = np.empty(N)
        self.num_p = np.empty(K)
        self.num_c = np.empty(K)
        self.den = np.empty(K)
        self.u_sub = np.empty(M)
        self.u_net = np.empty(M)
        self.k_on_eff = np.empty(N)
        self.last_occ = np.empty(K)
        self.has_prev = np.empty(K)
        self.dx = np.empty(2 * K + M + N)

    def set_theta(self, theta):
        (k_act, k_deact, s_prod, d_deg,
         beta_g, beta_l, alpha,
         kK_act, kK_deact, k_off,
         gamma_S_p, gamma_A_S, gamma_A_p, gamma_K_net) = decode_theta(theta, self.K, self.M, self.N)

        self.k_act[:] = k_act
        self.k_deact[:] = k_deact
        self.s_prod[:] = s_prod
        self.d_deg[:] = d_deg

        self.alpha[:] = alpha
        self.kK_act[:] = kK_act
        self.kK_deact[:] = kK_deact
        self.k_off[:] = k_off

        self.beta_g = beta_g
        self.beta_l = beta_l
        self.gamma_S_p = gamma_S_p
        self.gamma_A_S = gamma_A_S
        self.gamma_A_p = gamma_A_p
        self.gamma_K_net = gamma_K_net

    def rhs(self, x, t,
            Cg, Cl, site_prot_idx, K_site_kin, R, L_alpha,
            kin_to_prot_idx, receptor_mask_prot, receptor_mask_kin,
            mech_code,
            return_copy):
        out = _rhs_common_csr(x, t,
                              self.k_act, self.k_deact, self.s_prod, self.d_deg,
                              self.beta_g, self.beta_l, self.alpha,
                              self.kK_act, self.kK_deact, self.k_off,
                              self.gamma_S_p, self.gamma_A_S, self.gamma_A_p, self.gamma_K_net,
                              Cg, Cl, site_prot_idx, K_site_kin, R, L_alpha,
                              kin_to_prot_idx, receptor_mask_prot, receptor_mask_kin,
                              self.K, self.M, self.N, mech_code,
                              self.p_buf, self.coup_buf, self.num_p, self.num_c, self.den,
                              self.u_sub, self.u_net, self.k_on_eff, self.last_occ, self.has_prev,
                              self.dx)
        if return_copy == 1:
            return out.copy()
        return out


# -------------------------
# Compatibility layer: keep your dispatch logic
# -------------------------

def network_rhs(x, t, theta,
                Cg, Cl, site_prot_idx, K_site_kin, R, L_alpha, kin_to_prot_idx,
                receptor_mask_prot, receptor_mask_kin,
                mech="dist",
                sparse=False,
                K=None, M=None, N=None):
    """
    Python-level dispatch that keeps your mechanism string and adds a sparse flag.
    - If sparse=False: expects dense numpy arrays
    - If sparse=True: expects CSRMatrix jitclass for each matrix
    """
    if K is None: K = ModelDims.K
    if M is None: M = ModelDims.M
    if N is None: N = ModelDims.N

    if mech == "dist":
        mech_code = 0
    elif mech == "seq":
        mech_code = 1
    elif mech == "rand":
        mech_code = 2
    else:
        raise ValueError(f"Unknown mechanism: {mech}")

    if not sparse:
        return rhs_dense_onecall(x, t, theta,
                                 Cg, Cl, site_prot_idx, K_site_kin, R, L_alpha,
                                 kin_to_prot_idx, receptor_mask_prot, receptor_mask_kin,
                                 K, M, N,
                                 mech_code)
    else:
        return rhs_csr_onecall(x, t, theta,
                               Cg, Cl, site_prot_idx, K_site_kin, R, L_alpha,
                               kin_to_prot_idx, receptor_mask_prot, receptor_mask_kin,
                               K, M, N,
                               mech_code)


@njit(cache=True, fastmath=True)
def rhs_nb_dispatch_dense(
        x, t, theta,
        Cg, Cl,
        site_prot_idx,
        K_site_kin, R,
        L_alpha,
        kin_to_prot_idx,
        receptor_mask_prot,
        receptor_mask_kin,
        K, M, N,
        mech_code  # 0: dist, 1: seq, 2: rand
):
    return rhs_dense_onecall(x, t, theta,
                             Cg, Cl, site_prot_idx, K_site_kin, R, L_alpha,
                             kin_to_prot_idx, receptor_mask_prot, receptor_mask_kin,
                             K, M, N,
                             mech_code)


@njit(cache=True, fastmath=True)
def rhs_nb_dispatch_csr(
        x, t, theta,
        Cg, Cl,
        site_prot_idx,
        K_site_kin, R,
        L_alpha,
        kin_to_prot_idx,
        receptor_mask_prot,
        receptor_mask_kin,
        K, M, N,
        mech_code  # 0: dist, 1: seq, 2: rand
):
    return rhs_csr_onecall(x, t, theta,
                           Cg, Cl, site_prot_idx, K_site_kin, R, L_alpha,
                           kin_to_prot_idx, receptor_mask_prot, receptor_mask_kin,
                           K, M, N,
                           mech_code)
