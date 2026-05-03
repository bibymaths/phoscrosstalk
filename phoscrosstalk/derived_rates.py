"""
derived_rates.py

JAX-compatible factory functions for time-varying derived rate vectors.

``k_act(t)`` and ``s_prod(t)`` are no longer fitted parameters; they are
computed from experimental data (mRNA time-series and kinase–substrate
network) and injected into the ODE right-hand side as closures.

Public API
----------
make_k_act_fn(t_rna, rna_data, tf_prot_weights, interp_mode)
    Build a JAX function ``k_act_fn(t) -> jnp.array(shape=(K,))``
    representing the TF-driven protein activation rate at time *t*.

make_s_prod_fn(t_protein, Y_data, R_kin_site, kin_to_prot_idx, K, M,
               s_prod_fn_type, interp_mode)
    Build a JAX function ``s_prod_fn(t) -> jnp.array(shape=(K,))``
    representing the kinase-signal-driven synthesis rate at time *t*.

Both functions default gracefully to constant vectors (1.0 and 0.1
respectively) when the required input data are not available.
"""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Interpolation helpers (JAX-traceable, no Python branching on traced values)
# ---------------------------------------------------------------------------


_INTERPOLATION_EPSILON = 1e-12

def _piecewise_constant(t, times_j: jnp.ndarray, values_j: jnp.ndarray) -> jnp.ndarray:
    """
    Piecewise-constant (zero-order hold) interpolation.

    Args:
        t:        Scalar time value (JAX-traced).
        times_j:  (T,) sorted time points.
        values_j: (D, T) matrix of values; one column per time point.

    Returns:
        (D,) array – the column of *values_j* at the last time ≤ *t*.
    """
    T = times_j.shape[0]
    idx = jnp.searchsorted(times_j, t, side="right") - 1
    idx = jnp.clip(idx, 0, T - 1)
    return values_j[:, idx]


def _linear_interp(t, times_j: jnp.ndarray, values_j: jnp.ndarray) -> jnp.ndarray:
    """
    Piecewise-linear interpolation.

    Args:
        t:        Scalar time value (JAX-traced).
        times_j:  (T,) sorted time points.
        values_j: (D, T) matrix of values; one column per time point.

    Returns:
        (D,) array – linearly interpolated values at *t*.
    """
    T = times_j.shape[0]
    idx = jnp.searchsorted(times_j, t, side="right") - 1
    idx = jnp.clip(idx, 0, T - 2)
    t0 = times_j[idx]
    t1 = times_j[idx + 1]
    v0 = values_j[:, idx]
    v1 = values_j[:, idx + 1]
    alpha = jnp.clip((t - t0) / (t1 - t0 + _INTERPOLATION_EPSILON), 0.0, 1.0)
    return v0 + alpha * (v1 - v0)


def _interp_fn(interp_mode: str):
    """Return the interpolation helper matching *interp_mode*."""
    if interp_mode == "linear":
        return _linear_interp
    return _piecewise_constant  # default: piecewise_constant


# ---------------------------------------------------------------------------
# Softplus / linear scaling helpers
# ---------------------------------------------------------------------------


def _softplus(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.log1p(jnp.exp(x))


def _identity(x: jnp.ndarray) -> jnp.ndarray:
    return x


def _scale_fn(name: str):
    if name == "linear":
        return _identity
    return _softplus  # default


# ---------------------------------------------------------------------------
# k_act factory
# ---------------------------------------------------------------------------


def make_k_act_fn(
    t_rna: np.ndarray | None,
    rna_data: np.ndarray | None,
    tf_prot_weights: np.ndarray | None,
    K: int,
    interp_mode: str = "piecewise_constant",
):
    """
    Build a JAX closure ``k_act_fn(t) -> jnp.array(shape=(K,))``.

    Each element ``k_act[p]`` is the weighted sum of TF mRNA fold-changes
    at time *t* for protein *p*::

        k_act(p, t) = Σ_{tf} tf_prot_weights[p, tf] · x_tf(t)

    If any of the required inputs are absent (``None``), the function returns
    a constant vector of ones (neutral activation rate).

    Args:
        t_rna:           (T_rna,) mRNA time points.
        rna_data:        (n_genes, T_rna) mRNA fold-change matrix.
        tf_prot_weights: (K, n_genes) weight matrix mapping genes→proteins.
                         ``tf_prot_weights[p, g]`` = contribution of gene *g*
                         as a TF for protein *p*.
        K:               Number of proteins.
        interp_mode:     ``"piecewise_constant"`` or ``"linear"``.

    Returns:
        A JAX function ``fn(t) -> jnp.array(shape=(K,))``.
    """
    if t_rna is None or rna_data is None or tf_prot_weights is None:
        # No TF/mRNA data: constant neutral activation
        _ones = jnp.ones(K, dtype=jnp.float32)

        def _k_act_const(t):
            return _ones

        return _k_act_const

    # Pre-compute weighted mRNA signals per protein: (K, T_rna)
    # signal[p, t] = Σ_g tf_prot_weights[p, g] * rna_data[g, t]
    signal = np.asarray(tf_prot_weights, dtype=np.float32) @ np.asarray(
        rna_data, dtype=np.float32
    )  # (K, T_rna)
    signal_j = jnp.asarray(signal, dtype=jnp.float32)
    times_j = jnp.asarray(t_rna, dtype=jnp.float32)
    _interp = _interp_fn(interp_mode)

    def _k_act_fn(t):
        return _interp(t, times_j, signal_j)

    return _k_act_fn


# ---------------------------------------------------------------------------
# s_prod factory
# ---------------------------------------------------------------------------


def make_s_prod_fn(
    t_protein: np.ndarray,
    Y_data: np.ndarray,
    R_kin_site: np.ndarray,
    kin_to_prot_idx: np.ndarray,
    K: int,
    M: int,
    s_prod_fn_type: str = "softplus",
    interp_mode: str = "piecewise_constant",
):
    """
    Build a JAX closure ``s_prod_fn(t) -> jnp.array(shape=(K,))``.

    Derives per-protein synthesis rate from observed kinase–substrate signals::

        activity(k, t) = R_kin_site[k, :] @ Y_data[:, t_idx]
        s_prod(p, t)   = f( Σ_{k : kin_to_prot_idx[k]==p} activity(k, t) )

    where *f* is ``softplus`` (default) or ``linear``.

    If *Y_data* is empty or *R_kin_site* has no rows, the function returns a
    constant small vector (0.1).

    Args:
        t_protein:       (T,) protein/phospho time points.
        Y_data:          (N_sites, T) observed phospho data.
        R_kin_site:      (M, N_sites) kinase-to-site weight matrix.
        kin_to_prot_idx: (M,) integer array mapping kinases to proteins (−1 if
                         no mapping).
        K:               Number of proteins.
        M:               Number of kinases.
        s_prod_fn_type:  ``"softplus"`` or ``"linear"``.
        interp_mode:     ``"piecewise_constant"`` or ``"linear"``.

    Returns:
        A JAX function ``fn(t) -> jnp.array(shape=(K,))``.
    """
    if Y_data is None or Y_data.size == 0 or R_kin_site.shape[0] == 0:
        _const = jnp.full(K, 0.1, dtype=jnp.float32)

        def _s_prod_const(t):
            return _const

        return _s_prod_const

    # Compute kinase activity time series: (M, T)
    kin_activity = np.asarray(R_kin_site, dtype=np.float32) @ np.asarray(
        Y_data, dtype=np.float32
    )  # (M, T)

    # Aggregate to proteins: (K, T)
    prot_signal = np.zeros((K, Y_data.shape[1]), dtype=np.float32)
    for k_idx in range(M):
        p_idx = int(kin_to_prot_idx[k_idx])
        if 0 <= p_idx < K:
            prot_signal[p_idx] += kin_activity[k_idx]

    _f = _scale_fn(s_prod_fn_type)
    prot_signal_j = jnp.asarray(prot_signal, dtype=jnp.float32)
    times_j = jnp.asarray(t_protein, dtype=jnp.float32)
    _interp = _interp_fn(interp_mode)

    def _s_prod_fn(t):
        raw = _interp(t, times_j, prot_signal_j)
        return _f(raw)

    return _s_prod_fn
