"""
JAX-compatible factory functions for time-varying derived rate vectors.

k_act(t) and s_prod(t) are derived rate functions injected into the ODE RHS.
They are not fitted parameters.

The raw omics/network signals are not returned directly. They are first
converted into bounded, positive rate vectors using JAX-compatible transfer
functions. This prevents RNA fold-change or phosphosite-derived kinase drive
from becoming unconstrained ODE forcing.

Public API
----------
make_k_act_fn(...)
    Build k_act_fn(t) -> jnp.ndarray(shape=(K,)).

make_s_prod_fn(...)
    Build s_prod_fn(t) -> jnp.ndarray(shape=(K,)).

build_data_interpolations(...)
    Build NumPy/SciPy diagnostic interpolation callables only. These are not
    JAX-traceable and must not be passed into the ODE RHS or optimisation loss.

Design notes
------------
ODE RHS interpolation uses only JAX-traceable helpers:
    - piecewise_constant
    - linear

The bounded transfer function is JAX-native and uses jnp.where instead of
Python branching on traced values.

Diagnostic interpolation uses NumPy/SciPy callables and remains separate.
"""

from __future__ import annotations

from collections.abc import Callable

import jax.numpy as jnp
import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_INTERPOLATION_EPSILON = 1e-12
_DEFAULT_K_ACT_BASAL = 1.0
_DEFAULT_K_ACT_SCALE = 2.0
_DEFAULT_S_PROD_BASAL = 0.1
_DEFAULT_S_PROD_SCALE = 2.0


# ---------------------------------------------------------------------------
# Array validation / preparation helpers
# ---------------------------------------------------------------------------

def _as_float_np(x, name: str) -> np.ndarray:
    """Convert input to finite float64 NumPy array where possible."""
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim == 0:
        raise ValueError(f"{name} must be an array, got scalar.")
    return arr


def _validate_time_vector(times: np.ndarray, name: str = "times") -> np.ndarray:
    """
    Validate and return a sorted finite 1D time vector.

    The code assumes time points are already sorted. This function checks that
    assumption instead of silently sorting, because silently sorting would also
    require reordering data columns.
    """
    t = np.asarray(times, dtype=np.float64)

    if t.ndim != 1:
        raise ValueError(f"{name} must be 1D. Got shape {t.shape}.")

    if t.size == 0:
        raise ValueError(f"{name} is empty.")

    if not np.all(np.isfinite(t)):
        raise ValueError(f"{name} contains non-finite values.")

    if t.size > 1 and np.any(np.diff(t) < 0):
        raise ValueError(f"{name} must be sorted in non-decreasing order.")

    return t


def _validate_matrix_time_shape(
    values: np.ndarray,
    times: np.ndarray,
    values_name: str,
    times_name: str,
) -> None:
    """Check that values has shape (D, T) matching len(times)."""
    if values.ndim != 2:
        raise ValueError(f"{values_name} must be 2D with shape (D, T). Got {values.shape}.")

    if values.shape[1] != len(times):
        raise ValueError(
            f"{values_name}.shape[1] must match len({times_name}). "
            f"Got {values.shape[1]} and {len(times)}."
        )


def _nan_to_zero(arr: np.ndarray) -> np.ndarray:
    """
    Replace NaN/inf in raw ODE forcing inputs.

    This is intentional for JAX RHS forcing: missing upstream signal should not
    propagate NaNs through the ODE solver. Diagnostic interpolation preserves
    NaNs separately in build_data_interpolations().
    """
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


# ---------------------------------------------------------------------------
# JAX-traceable interpolation helpers
# ---------------------------------------------------------------------------

def _piecewise_constant(t, times_j: jnp.ndarray, values_j: jnp.ndarray) -> jnp.ndarray:
    """
    Zero-order hold interpolation.

    Args:
        t:        Scalar JAX-traced time.
        times_j:  (T,) sorted time vector.
        values_j: (D, T) value matrix.

    Returns:
        (D,) vector at the last observed time <= t.
    """
    T = times_j.shape[0]
    idx = jnp.searchsorted(times_j, t, side="right") - 1
    idx = jnp.clip(idx, 0, T - 1)
    return values_j[:, idx]


def _linear_interp(t, times_j: jnp.ndarray, values_j: jnp.ndarray) -> jnp.ndarray:
    """
    JAX-traceable piecewise-linear interpolation.

    Args:
        t:        Scalar JAX-traced time.
        times_j:  (T,) sorted time vector.
        values_j: (D, T) value matrix.

    Returns:
        (D,) linearly interpolated vector.
    """
    T = times_j.shape[0]

    # Degenerate one-point input: behave like constant interpolation.
    if T == 1:
        return values_j[:, 0]

    idx = jnp.searchsorted(times_j, t, side="right") - 1
    idx = jnp.clip(idx, 0, T - 2)

    t0 = times_j[idx]
    t1 = times_j[idx + 1]
    v0 = values_j[:, idx]
    v1 = values_j[:, idx + 1]

    alpha = jnp.clip(
        (t - t0) / (t1 - t0 + _INTERPOLATION_EPSILON),
        0.0,
        1.0,
    )

    return v0 + alpha * (v1 - v0)


def _interp_fn(interp_mode: str) -> Callable:
    """
    Return JAX-traceable interpolation function.

    Supported:
        - "piecewise_constant"
        - "linear"

    Unknown values fall back to piecewise_constant.
    """
    if interp_mode == "linear":
        return _linear_interp

    return _piecewise_constant


# ---------------------------------------------------------------------------
# JAX-compatible transfer functions
# ---------------------------------------------------------------------------

def _softplus(x: jnp.ndarray) -> jnp.ndarray:
    """
    Numerically stable softplus.

    Equivalent to log(1 + exp(x)), but stable for large positive x.
    """
    return jnp.log1p(jnp.exp(-jnp.abs(x))) + jnp.maximum(x, 0.0)


def _identity(x: jnp.ndarray) -> jnp.ndarray:
    """Return raw input unchanged."""
    return x


def _positive_floor(x: jnp.ndarray, floor: float = 1e-8) -> jnp.ndarray:
    """Clamp values to a small positive floor."""
    return jnp.maximum(x, floor)


def _softclip_unit(u_raw: jnp.ndarray) -> jnp.ndarray:
    """
    Squash raw regulatory input into (-1, 1).

    u = raw / (1 + |raw|)

    This prevents extreme RNA/TF or kinase/phosphosite projections from directly
    exploding the derived rates.
    """
    return u_raw / (1.0 + jnp.abs(u_raw))


def _bounded_activation_repression(
    u_raw: jnp.ndarray,
    basal: float,
    scale: float,
    eps: float = 1e-6,
) -> jnp.ndarray:
    """
    Bounded activation/repression transfer function.

    Let:
        u = u_raw / (1 + |u_raw|)

    For u >= 0:
        rate = basal * (1 + scale * u / (1 + u))

    For u < 0:
        rate = basal / (1 + scale * |u|)

    Properties:
        - JAX-traceable.
        - Positive output.
        - Bounded activation.
        - Bounded repression.
        - No Python branching on traced values.
    """
    u = _softclip_unit(u_raw)

    activation = basal * (1.0 + (scale * u) / (1.0 + u + eps))
    repression = basal / (1.0 + scale * jnp.abs(u) + eps)

    return jnp.where(u >= 0.0, activation, repression)


def _rate_transform_fn(
    name: str,
    basal: float,
    scale: float,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """
    Return a JAX-compatible transform from raw regulatory drive to rate.

    Supported names:
        - "bounded": bounded activation/repression transfer
        - "softplus": positive softplus transform
        - "linear": raw input unchanged
        - "positive": positive floor clamp

    Unknown values default to "bounded".
    """
    name = str(name or "bounded").lower()

    if name == "linear":
        return _identity

    if name == "softplus":
        return _softplus

    if name == "positive":
        def _fn(x):
            return _positive_floor(x)
        return _fn

    def _bounded_fn(x):
        return _bounded_activation_repression(
            u_raw=x,
            basal=float(basal),
            scale=float(scale),
        )

    return _bounded_fn


# Backward-compatible alias.
def _scale_fn(name: str):
    """
    Backward-compatible transform resolver.

    Existing code may still expect _scale_fn("softplus") or _scale_fn("linear").
    New code should prefer _rate_transform_fn().
    """
    name = str(name or "softplus").lower()

    if name == "linear":
        return _identity

    if name == "positive":
        return _positive_floor

    return _softplus


# ---------------------------------------------------------------------------
# Optional preprocessing of raw derived signals before JAX conversion
# ---------------------------------------------------------------------------

def _exp_smooth_signal(
    signal: np.ndarray,
    times: np.ndarray,
    tau: float,
) -> np.ndarray:
    """
    Exponential low-pass smoothing for sparse omics-derived forcing signals.

    Args:
        signal: (D, T) raw signal matrix.
        times:  (T,) time vector.
        tau:    Time constant in same units as times.

    Returns:
        (D, T) smoothed signal matrix.
    """
    signal = np.asarray(signal, dtype=np.float64)
    times = _validate_time_vector(times)

    _validate_matrix_time_shape(signal, times, "signal", "times")

    if signal.shape[1] == 1:
        return signal.copy()

    tau = max(float(tau), _INTERPOLATION_EPSILON)

    out = np.zeros_like(signal, dtype=np.float64)
    out[:, 0] = signal[:, 0]

    for j in range(1, len(times)):
        dt = max(float(times[j] - times[j - 1]), 0.0)
        alpha = 1.0 - np.exp(-dt / tau)
        out[:, j] = out[:, j - 1] + alpha * (signal[:, j] - out[:, j - 1])

    return out


def _preprocess_derived_signal(
    signal: np.ndarray,
    times: np.ndarray,
    method: str = "none",
    tau: float = 30.0,
    clip_min: float | None = None,
    clip_max: float | None = None,
) -> np.ndarray:
    """
    Preprocess raw omics/network-derived signals before JAX interpolation.

    This is done outside the JAX-traced RHS.

    Supported methods:
        - "none"
        - "exponential_smoothing"

    PCHIP/spline preprocessing is intentionally not included here to keep this
    file dependency-light. Use build_data_interpolations() for diagnostic curves.
    """
    method = str(method or "none").lower()

    signal = np.asarray(signal, dtype=np.float64)
    times = _validate_time_vector(times)

    _validate_matrix_time_shape(signal, times, "signal", "times")

    signal = _nan_to_zero(signal)

    if method in {"none", "raw"}:
        out = signal.copy()
    elif method in {"exponential_smoothing", "exp_smoothing", "lowpass", "low_pass"}:
        out = _exp_smooth_signal(signal, times, tau=tau)
    else:
        raise ValueError(
            f"Unknown derived signal preprocessing method: {method!r}. "
            "Expected one of: 'none', 'exponential_smoothing'."
        )

    if clip_min is not None or clip_max is not None:
        lo = -np.inf if clip_min is None else float(clip_min)
        hi = np.inf if clip_max is None else float(clip_max)
        out = np.clip(out, lo, hi)

    return out


# ---------------------------------------------------------------------------
# k_act factory
# ---------------------------------------------------------------------------

def make_k_act_fn(
        t_rna: np.ndarray | None,
        rna_data: np.ndarray | None,
        tf_prot_weights: np.ndarray | None,
        K: int,
        interp_mode: str = "piecewise_constant",
        protein_self_rna_idx: np.ndarray | None = None,
        rate_fn_type: str = "bounded",
        basal: float = _DEFAULT_K_ACT_BASAL,
        scale: float = _DEFAULT_K_ACT_SCALE,
        preprocess: str = "none",
        tau: float = 60.0,
        clip_min: float | None = None,
        clip_max: float | None = None,
):
    """
    Build a JAX closure k_act_fn(t) -> jnp.ndarray(shape=(K,)).

    This mirrors the mathematical structure of make_s_prod_fn():

        regulator_activity[g, t] = rna_data[g, t]

        prot_signal[p, t] =
            sum_g tf_prot_weights[p, g] * regulator_activity[g, t]

        k_act[p, t] =
            transfer(interpolate(preprocess(prot_signal[p, :]), t))

    Biological interpretation:
        - RNA is not used directly as k_act(t).
        - RNA first acts as a TF/regulator activity proxy.
        - TF/gene-to-protein weights project regulator activity onto each protein.
        - Proteins without TF upstream support can fall back to their own RNA
          trajectory when protein_self_rna_idx is provided.
        - The raw regulatory drive is converted into a bounded positive
          activation rate by the transfer function.

    Args:
        t_rna:                (T_rna,) mRNA time points.
        rna_data:             (n_genes, T_rna) mRNA matrix.
        tf_prot_weights:      (K, n_genes) TF/gene -> protein weights.
        K:                    Number of proteins.
        interp_mode:          "piecewise_constant" or "linear".
        protein_self_rna_idx: Optional (K,) fallback mapping into rna_data rows.
        rate_fn_type:         "bounded", "softplus", "linear", or "positive".
        basal:                Basal output rate for bounded transform.
        scale:                Max regulatory scale for bounded transform.
        preprocess:           "none" or "exponential_smoothing".
        tau:                  Time constant for exponential smoothing.
        clip_min:             Optional raw-drive lower clipping bound.
        clip_max:             Optional raw-drive upper clipping bound.

    Returns:
        Callable JAX function fn(t) -> jnp.ndarray(shape=(K,)).
    """
    K = int(K)

    if K <= 0:
        raise ValueError(f"K must be positive. Got {K}.")

    # No RNA/TF information: neutral basal activation.
    if t_rna is None or rna_data is None or tf_prot_weights is None:
        _const = jnp.full(K, float(basal), dtype=jnp.float64)

        def _k_act_const(t):
            return _const

        return _k_act_const

    times = _validate_time_vector(t_rna, "t_rna")

    rna_data_np = _as_float_np(rna_data, "rna_data")
    tf_weights_np = _as_float_np(tf_prot_weights, "tf_prot_weights")

    _validate_matrix_time_shape(rna_data_np, times, "rna_data", "t_rna")

    if tf_weights_np.ndim != 2:
        raise ValueError(
            f"tf_prot_weights must be 2D with shape (K, n_genes). "
            f"Got {tf_weights_np.shape}."
        )

    if tf_weights_np.shape[0] != K:
        raise ValueError(
            f"tf_prot_weights.shape[0] must equal K. "
            f"Got {tf_weights_np.shape[0]} and K={K}."
        )

    if tf_weights_np.shape[1] != rna_data_np.shape[0]:
        raise ValueError(
            "tf_prot_weights.shape[1] must match rna_data.shape[0]. "
            f"Got {tf_weights_np.shape[1]} and {rna_data_np.shape[0]}."
        )

    rna_data_np = _nan_to_zero(rna_data_np)
    tf_weights_np = _nan_to_zero(tf_weights_np)

    # ------------------------------------------------------------------
    # 1. Regulator activity drive
    # ------------------------------------------------------------------
    # RNA is treated as observed TF/regulator activity proxy:
    #
    #     regulator_activity[g, t] = rna_data[g, t]
    #
    # This mirrors make_s_prod_fn(), where:
    #
    #     kin_activity[m, t] = R_kin_site[m, :] @ Y_data[:, t]
    #
    regulator_activity = rna_data_np  # (n_genes, T_rna)

    # ------------------------------------------------------------------
    # 2. Aggregate regulator drive to protein-level activation drive
    # ------------------------------------------------------------------
    # Equivalent to:
    #
    #     prot_signal[p, t] =
    #         sum_g tf_prot_weights[p, g] * regulator_activity[g, t]
    #
    # Shape:
    #     tf_weights_np:      (K, n_genes)
    #     regulator_activity: (n_genes, T_rna)
    #     prot_signal:        (K, T_rna)
    #
    prot_signal = tf_weights_np @ regulator_activity

    # ------------------------------------------------------------------
    # 3. Biological fallback for proteins without TF upstream support
    # ------------------------------------------------------------------
    # This preserves the previous first-principles logic:
    #
    #   if protein has TF upstream edges:
    #       use TF-projected RNA drive
    #   elif own RNA is available:
    #       use own RNA trajectory as self-regulatory activation proxy
    #   else:
    #       raw drive = 0, so bounded transform returns basal
    #
    if protein_self_rna_idx is not None:
        self_rna_idx = np.asarray(protein_self_rna_idx, dtype=int)

        if self_rna_idx.shape[0] != K:
            raise ValueError(
                f"protein_self_rna_idx must have length K={K}. "
                f"Got shape {self_rna_idx.shape}."
            )

        for p_idx in range(K):
            row_has_tf_input = bool(np.any(np.abs(tf_weights_np[p_idx]) > 0.0))

            if not row_has_tf_input:
                s_idx = int(self_rna_idx[p_idx])

                if 0 <= s_idx < regulator_activity.shape[0]:
                    prot_signal[p_idx, :] = regulator_activity[s_idx, :]
                else:
                    # Raw drive zero gives basal after bounded transfer.
                    prot_signal[p_idx, :] = 0.0

    # ------------------------------------------------------------------
    # 4. Same preprocessing logic as make_s_prod_fn()
    # ------------------------------------------------------------------
    prot_signal = _preprocess_derived_signal(
        signal=prot_signal,
        times=times,
        method=preprocess,
        tau=tau,
        clip_min=clip_min,
        clip_max=clip_max,
    )

    prot_signal_j = jnp.asarray(prot_signal, dtype=jnp.float64)
    times_j = jnp.asarray(times, dtype=jnp.float64)

    # ------------------------------------------------------------------
    # 5. Same interpolation + transfer logic as make_s_prod_fn()
    # ------------------------------------------------------------------
    _interp = _interp_fn(interp_mode)
    _rate_transform = _rate_transform_fn(
        name=rate_fn_type,
        basal=float(basal),
        scale=float(scale),
    )

    def _k_act_fn(t):
        raw = _interp(t, times_j, prot_signal_j)
        return _rate_transform(raw)

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
        s_prod_fn_type: str = "bounded",
        interp_mode: str = "piecewise_constant",
        basal: float = _DEFAULT_S_PROD_BASAL,
        scale: float = _DEFAULT_S_PROD_SCALE,
        preprocess: str = "none",
        tau: float = 15.0,
        clip_min: float | None = None,
        clip_max: float | None = None,
):
    """
    Build a JAX closure s_prod_fn(t) -> jnp.ndarray(shape=(K,)).

    Raw construction:
        kin_activity[m, t] = R_kin_site[m, :] @ Y_data[:, t]
        prot_signal[p, t] = sum_{m: kin_to_prot_idx[m] == p} kin_activity[m, t]

    Final returned rate:
        s_prod(t) = transfer(interpolate(prot_signal, t))

    This means phosphosite-derived kinase activity is not injected directly as
    s_prod(t). It is converted into a bounded/positive synthesis-rate drive.

    Args:
        t_protein:       (T,) protein/phospho time points.
        Y_data:          (N_sites, T) phosphosite matrix.
        R_kin_site:      (M, N_sites) kinase-site weight matrix.
        kin_to_prot_idx: (M,) kinase -> protein mapping.
        K:               Number of proteins.
        M:               Number of kinases.
        s_prod_fn_type:  "bounded", "softplus", "linear", or "positive".
        interp_mode:     "piecewise_constant" or "linear".
        basal:           Basal output rate for bounded transform.
        scale:           Max regulatory scale for bounded transform.
        preprocess:      "none" or "exponential_smoothing".
        tau:             Time constant for exponential smoothing.
        clip_min:        Optional raw-drive lower clipping bound.
        clip_max:        Optional raw-drive upper clipping bound.

    Returns:
        Callable JAX function fn(t) -> jnp.ndarray(shape=(K,)).
    """
    K = int(K)
    M = int(M)

    if K <= 0:
        raise ValueError(f"K must be positive. Got {K}.")

    if M < 0:
        raise ValueError(f"M must be non-negative. Got {M}.")

    if Y_data is None or np.asarray(Y_data).size == 0:
        _const = jnp.full(K, float(basal), dtype=jnp.float64)

        def _s_prod_const(t):
            return _const

        return _s_prod_const

    if R_kin_site is None or np.asarray(R_kin_site).size == 0:
        _const = jnp.full(K, float(basal), dtype=jnp.float64)

        def _s_prod_const(t):
            return _const

        return _s_prod_const

    times = _validate_time_vector(t_protein, "t_protein")

    Y_np = _as_float_np(Y_data, "Y_data")
    R_np = _as_float_np(R_kin_site, "R_kin_site")
    kin_to_prot_idx_np = np.asarray(kin_to_prot_idx, dtype=int)

    _validate_matrix_time_shape(Y_np, times, "Y_data", "t_protein")

    if R_np.ndim != 2:
        raise ValueError(
            f"R_kin_site must be 2D with shape (M, N_sites). Got {R_np.shape}."
        )

    if R_np.shape[0] == 0:
        _const = jnp.full(K, float(basal), dtype=jnp.float64)

        def _s_prod_const(t):
            return _const

        return _s_prod_const

    if R_np.shape[1] != Y_np.shape[0]:
        raise ValueError(
            f"R_kin_site.shape[1] must match Y_data.shape[0]. "
            f"Got {R_np.shape[1]} and {Y_np.shape[0]}."
        )

    if M != R_np.shape[0]:
        raise ValueError(
            f"M must match R_kin_site.shape[0]. Got M={M}, "
            f"R_kin_site.shape[0]={R_np.shape[0]}."
        )

    if kin_to_prot_idx_np.shape[0] != M:
        raise ValueError(
            f"kin_to_prot_idx must have length M={M}. "
            f"Got shape {kin_to_prot_idx_np.shape}."
        )

    Y_np = _nan_to_zero(Y_np)
    R_np = _nan_to_zero(R_np)

    # Raw kinase activity drive: (M, T)
    kin_activity = R_np @ Y_np

    # Aggregate kinase drive to protein-level synthesis drive: (K, T)
    prot_signal = np.zeros((K, Y_np.shape[1]), dtype=np.float64)

    for k_idx in range(M):
        p_idx = int(kin_to_prot_idx_np[k_idx])
        if 0 <= p_idx < K:
            prot_signal[p_idx, :] += kin_activity[k_idx, :]

    prot_signal = _preprocess_derived_signal(
        signal=prot_signal,
        times=times,
        method=preprocess,
        tau=tau,
        clip_min=clip_min,
        clip_max=clip_max,
    )

    prot_signal_j = jnp.asarray(prot_signal, dtype=jnp.float64)
    times_j = jnp.asarray(times, dtype=jnp.float64)

    _interp = _interp_fn(interp_mode)
    _rate_transform = _rate_transform_fn(
        name=s_prod_fn_type,
        basal=float(basal),
        scale=float(scale),
    )

    def _s_prod_fn(t):
        raw = _interp(t, times_j, prot_signal_j)
        return _rate_transform(raw)

    return _s_prod_fn


# ---------------------------------------------------------------------------
# Diagnostic / visualisation interpolation
# ---------------------------------------------------------------------------

def build_data_interpolations(
        t_obs: np.ndarray,
        P_data: np.ndarray | None = None,
        A_data: np.ndarray | None = None,
        rna_data: np.ndarray | None = None,
        method: str = "linear",
        fill_forward_nans_at_end: bool = False,
        replace_nans_at_start: str | None = None,
) -> dict:
    """
    Build continuous interpolation callables for observed datasets.

    This function is for diagnostic / visualisation only.

    It does not affect:
        - optimisation loss
        - ODE RHS
        - sparse observed arrays
        - JAX-traced code paths

    Args:
        t_obs:      (T,) observed time points.
        P_data:     Optional (N_sites, T) phosphosite data.
        A_data:     Optional (K_obs, T) abundance data.
        rna_data:   Optional (n_genes, T) RNA data.
        method:     "linear" or "cubic_hermite".
                    "cubic_hermite" uses scipy.interpolate.PchipInterpolator.
        fill_forward_nans_at_end:
                    Forward-fill trailing NaNs in interpolation-only copy.
        replace_nans_at_start:
                    None, "zero", or "first_valid".

    Returns:
        dict with interpolation callables and metadata.
    """
    t = _validate_time_vector(t_obs, "t_obs")
    method = str(method or "linear").lower()

    nan_fill_log: list[str] = []

    def _prep_row(row: np.ndarray, label: str) -> np.ndarray:
        """
        Apply optional NaN handling to a single interpolation-only row.

        This never modifies the original input array.
        """
        row = np.asarray(row, dtype=np.float64).copy()

        if replace_nans_at_start is not None:
            first_valid = next((i for i, v in enumerate(row) if np.isfinite(v)), None)

            if first_valid is None:
                nan_fill_log.append(f"  {label}: all values are NaN; no fill applied.")
                return row

            if first_valid > 0:
                if replace_nans_at_start == "zero":
                    row[:first_valid] = 0.0
                    nan_fill_log.append(
                        f"  {label}: filled {first_valid} leading NaN(s) with 0."
                    )
                elif replace_nans_at_start == "first_valid":
                    row[:first_valid] = row[first_valid]
                    nan_fill_log.append(
                        f"  {label}: filled {first_valid} leading NaN(s) with "
                        f"first_valid={row[first_valid]:.4g}."
                    )
                else:
                    raise ValueError(
                        "replace_nans_at_start must be None, 'zero', or 'first_valid'. "
                        f"Got {replace_nans_at_start!r}."
                    )

        if fill_forward_nans_at_end:
            last_valid = next(
                (i for i in range(len(row) - 1, -1, -1) if np.isfinite(row[i])),
                None,
            )

            if last_valid is not None and last_valid < len(row) - 1:
                row[last_valid + 1:] = row[last_valid]
                n_trailing = len(row) - 1 - last_valid
                nan_fill_log.append(
                    f"  {label}: forward-filled {n_trailing} trailing NaN(s) "
                    f"from t={t[last_valid]:.3g}."
                )

        return row

    def _make_nan_callable(D: int):
        def _nan_fn(t_q):
            tq = np.asarray(t_q, dtype=np.float64)
            if tq.ndim == 0:
                return np.full(D, np.nan, dtype=np.float64)
            return np.full((D, tq.size), np.nan, dtype=np.float64)

        return _nan_fn

    def _make_interp(data: np.ndarray, label_prefix: str):
        """
        Return callable fn(t_query) -> np.ndarray for matrix data.

        Important:
            Interpolation is built row-wise using only each row's finite points.
            Missing values are not converted to zero.
        """
        arr = np.asarray(data, dtype=np.float64)

        if arr.ndim != 2:
            nan_fill_log.append(
                f"  {label_prefix}: expected 2D array, got shape {arr.shape}; "
                "returning None."
            )
            return None

        D, T = arr.shape

        if T != len(t):
            nan_fill_log.append(
                f"  {label_prefix}: time dimension mismatch. "
                f"data.shape[1]={T}, len(t_obs)={len(t)}; returning None."
            )
            return None

        rows_prepped = [
            _prep_row(arr[d], f"{label_prefix}[{d}]")
            for d in range(D)
        ]

        if method == "cubic_hermite":
            try:
                from scipy.interpolate import PchipInterpolator
            except ImportError as exc:
                raise ImportError(
                    "SciPy is required for method='cubic_hermite'. "
                    "Install it with `pip install scipy`."
                ) from exc

            interps = []

            for d, row in enumerate(rows_prepped):
                finite = np.isfinite(row)

                if finite.sum() < 2:
                    nan_fill_log.append(
                        f"  {label_prefix}[{d}]: fewer than 2 finite points; "
                        "returning NaN for this row."
                    )
                    interps.append(None)
                    continue

                interps.append(
                    PchipInterpolator(
                        t[finite],
                        row[finite],
                        extrapolate=False,
                    )
                )

            def _cubic_fn(t_q):
                tq = np.asarray(t_q, dtype=np.float64)
                scalar_input = tq.ndim == 0
                tq_eval = np.atleast_1d(tq)

                out = []

                for interp in interps:
                    if interp is None:
                        vals = np.full(tq_eval.shape, np.nan, dtype=np.float64)
                    else:
                        vals = np.asarray(interp(tq_eval), dtype=np.float64)
                    out.append(vals)

                stacked = np.stack(out, axis=0)

                if scalar_input:
                    return stacked[:, 0]

                return stacked

            return _cubic_fn

        if method != "linear":
            nan_fill_log.append(
                f"  {label_prefix}: unknown method={method!r}; using linear."
            )

        interps = []

        for d, row in enumerate(rows_prepped):
            finite = np.isfinite(row)

            if finite.sum() < 2:
                nan_fill_log.append(
                    f"  {label_prefix}[{d}]: fewer than 2 finite points; "
                    "returning NaN for this row."
                )
                interps.append(None)
                continue

            interps.append((t[finite], row[finite]))

        def _linear_fn(t_q):
            tq = np.asarray(t_q, dtype=np.float64)
            scalar_input = tq.ndim == 0
            tq_eval = np.atleast_1d(tq)

            out = []

            for item in interps:
                if item is None:
                    vals = np.full(tq_eval.shape, np.nan, dtype=np.float64)
                else:
                    tv, rv = item
                    vals = np.interp(tq_eval, tv, rv, left=np.nan, right=np.nan)
                out.append(vals)

            stacked = np.stack(out, axis=0)

            if scalar_input:
                return stacked[:, 0]

            return stacked

        return _linear_fn

    result: dict = {
        "t_obs": t,
        "P_interp": None,
        "A_interp": None,
        "rna_interp": None,
        "method": method,
        "nan_fill_log": nan_fill_log,
        "original_arrays_unchanged": True,
    }

    if P_data is not None:
        arr = np.asarray(P_data, dtype=np.float64)
        if arr.size > 0:
            result["P_interp"] = _make_interp(arr, "P_data")

    if A_data is not None:
        arr = np.asarray(A_data, dtype=np.float64)
        if arr.size > 0:
            result["A_interp"] = _make_interp(arr, "A_data")

    if rna_data is not None:
        arr = np.asarray(rna_data, dtype=np.float64)
        if arr.size > 0:
            result["rna_interp"] = _make_interp(arr, "rna_data")

    return result