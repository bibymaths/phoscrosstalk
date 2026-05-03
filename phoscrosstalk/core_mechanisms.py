#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
core_mechanisms.py

Shared parameter-decoding utilities for the phospho-network model.

This module retains only the functions needed by downstream modules
(analysis.py, equations.py, optimization.py, post_processing.py, app.py):

  * clip_scalar  – fast scalar clip used internally by decode_theta.
  * decode_theta – decode the flat optimisation vector into biological params.

All Numba RHS kernels, workspace classes, CSR helpers, and dispatch
functions that existed in earlier versions have been removed because the
ODE simulation now uses JAX/Diffrax (see jax_mechanisms.py and
simulation.py).  The public API of this module is unchanged for all
callers that only import decode_theta.
"""

from __future__ import annotations

import math
import numpy as np
from numba import njit


# -------------------------
# small utilities
# -------------------------


@njit(cache=True, fastmath=True)
def clip_scalar(x, lo, hi):
    """
    Fast scalar clipping function compatible with Numba.

    Args:
        x (float): Value to clip.
        lo (float): Lower bound.
        hi (float): Upper bound.

    Returns:
        float: Clipped value.
    """
    if x < lo:
        return lo
    elif x > hi:
        return hi
    return x


@njit(cache=True, fastmath=True)
def decode_theta(theta, K, M, N):
    """
    Decodes the flat parameter vector ``theta`` into individual biological parameters.

    ``k_act`` and ``s_prod`` are no longer optimisation variables; they are
    derived from experimental data (mRNA / kinase signals) at run-time.  The
    parameter vector therefore has dimension ``2*K + 2 + 3*M + N + 4``.

    Performs unpacking, clipping to valid log-ranges, and exponentiation.

    Args:
        theta (np.ndarray): Flat parameter vector of length ``2*K+2+3*M+N+4``.
        K (int): Number of proteins.
        M (int): Number of kinases.
        N (int): Number of phosphosites.

    Returns:
        tuple: A tuple of 12 entries:
               (k_deact, d_deg, beta_g, beta_l, alpha,
                kK_act, kK_deact, k_off,
                gamma_S_p, gamma_A_S, gamma_A_p, gamma_K_net)
    """
    idx0 = 0
    log_k_deact = theta[idx0 : idx0 + K]
    idx0 += K
    log_d_deg = theta[idx0 : idx0 + K]
    idx0 += K

    log_beta_g = theta[idx0]
    idx0 += 1
    log_beta_l = theta[idx0]
    idx0 += 1

    log_alpha = theta[idx0 : idx0 + M]
    idx0 += M
    log_kK_act = theta[idx0 : idx0 + M]
    idx0 += M
    log_kK_deact = theta[idx0 : idx0 + M]
    idx0 += M

    log_k_off = theta[idx0 : idx0 + N]
    idx0 += N
    raw_gamma = theta[idx0 : idx0 + 4]

    # clip then exp
    k_deact = np.exp(np.clip(log_k_deact, -20.0, 10.0))
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
