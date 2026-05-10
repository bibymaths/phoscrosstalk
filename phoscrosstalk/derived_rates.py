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

build_data_interpolations(t_obs, P_data, A_data, rna_data, method,
                          fill_forward_nans_at_end, replace_nans_at_start)
    Build continuous NumPy/SciPy interpolation objects for observed datasets.
    **Diagnostic/visualisation only** – output is never fed into the ODE or
    the optimisation loss.  See docstring for details.

Both factory functions default gracefully to constant vectors (1.0 and 0.1
respectively) when the required input data are not available.

Interpolation backend design (ODE RHS)
---------------------------------------
The ODE right-hand side interpolates derived-rate inputs at every solver step
using JAX-traceable, piecewise implementations (_piecewise_constant /
_linear_interp).  These are kept as the sole ODE-time interpolation backend
because:

* They are fully JAX-traceable with no Python-level branching on traced values.
* Replacing them with diffrax.CubicInterpolation inside the traced RHS would
  require passing Diffrax interpolation objects through JIT boundaries, which
  is currently not supported without rewriting the entire RHS as an Equinox
  module or similar approach.
* Performance: piecewise-constant/linear interpolation is negligible overhead
  compared to the ODE integration cost.

Note: If Diffrax gains a JAX-traceable cubic Hermite callable that
can be passed as a static argument through jit without triggering re-tracing,
replace _piecewise_constant/_linear_interp with it in the ODE RHS.

For **exported/diagnostic** continuous representations of observed data, use
``build_data_interpolations()`` below – that function uses SciPy's ``interp1d``
(or a simple NumPy-based linear fallback) and returns standard Python callables,
which must NOT be passed into jitted code.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

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
    protein_self_rna_idx: np.ndarray | None = None,
):
    """
    Build a JAX closure ``k_act_fn(t) -> jnp.array(shape=(K,))``.

    Each element ``k_act[p]`` is determined by the following priority::

        if protein has TF upstream edges (non-zero row in tf_prot_weights):
            k_act_p(t) = Σ_{tf} tf_prot_weights[p, tf] · x_tf(t)
        elif protein_self_rna_idx[p] >= 0 (protein has its own RNA observation):
            k_act_p(t) = rna_data[protein_self_rna_idx[p], t]
        else:
            k_act_p(t) = 1.0  (neutral constant)

    If any of the required inputs are absent (``None``), the function returns
    a constant vector of ones (neutral activation rate).

    Args:
        t_rna:                (T_rna,) mRNA time points.
        rna_data:             (n_genes, T_rna) mRNA fold-change matrix.
        tf_prot_weights:      (K, n_genes) weight matrix mapping genes→proteins.
                              ``tf_prot_weights[p, g]`` = contribution of gene *g*
                              as a TF for protein *p*.
        K:                    Number of proteins.
        interp_mode:          ``"piecewise_constant"`` or ``"linear"``.
        protein_self_rna_idx: (K,) int array. For proteins without TF upstream
                              edges, ``protein_self_rna_idx[p]`` is the index
                              into *rna_data* rows for a self-RNA fallback signal.
                              Use ``-1`` to indicate no self-RNA (constant 1.0).
                              Pass ``None`` to disable the fallback entirely.

    Returns:
        (callable): A JAX function ``fn(t) -> jnp.array(shape=(K,))``.
    """
    if t_rna is None or rna_data is None or tf_prot_weights is None:
        # No TF/mRNA data: constant neutral activation
        _ones = jnp.ones(K, dtype=jnp.float64)

        def _k_act_const(t):
            return _ones

        return _k_act_const

    rna_data_np = np.asarray(rna_data, dtype=np.float64)
    tf_weights_np = np.asarray(tf_prot_weights, dtype=np.float64)

    # Pre-compute weighted mRNA signals per protein: (K, T_rna)
    # signal[p, t] = Σ_g tf_prot_weights[p, g] * rna_data[g, t]
    signal = tf_weights_np @ rna_data_np  # (K, T_rna)

    # Apply per-protein fallback for proteins without TF upstream edges.
    # A protein has TF input if its tf_prot_weights row is non-zero.
    if protein_self_rna_idx is not None:
        self_rna_idx = np.asarray(protein_self_rna_idx, dtype=int)
        rna_data_np.shape[1]
        for p_idx in range(K):
            row_sum = float(tf_weights_np[p_idx].sum())
            if row_sum == 0.0:
                s_idx = int(self_rna_idx[p_idx])
                if 0 <= s_idx < rna_data_np.shape[0]:
                    # Use the protein's own RNA trajectory as activation signal
                    signal[p_idx, :] = rna_data_np[s_idx, :]
                else:
                    # No self-RNA available: neutral constant 1.0
                    signal[p_idx, :] = 1.0

    signal_j = jnp.asarray(signal, dtype=jnp.float64)
    times_j = jnp.asarray(t_rna, dtype=jnp.float64)
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
        (callable): A JAX function ``fn(t) -> jnp.array(shape=(K,))``.
    """
    if Y_data is None or Y_data.size == 0 or R_kin_site.shape[0] == 0:
        _const = jnp.full(K, 0.1, dtype=jnp.float64)

        def _s_prod_const(t):
            return _const

        return _s_prod_const

    # Compute kinase activity time series: (M, T)
    kin_activity = np.asarray(R_kin_site, dtype=np.float64) @ np.asarray(
        Y_data, dtype=np.float64
    )  # (M, T)

    # Aggregate to proteins: (K, T)
    prot_signal = np.zeros((K, Y_data.shape[1]), dtype=np.float64)
    for k_idx in range(M):
        p_idx = int(kin_to_prot_idx[k_idx])
        if 0 <= p_idx < K:
            prot_signal[p_idx] += kin_activity[k_idx]

    _f = _scale_fn(s_prod_fn_type)
    prot_signal_j = jnp.asarray(prot_signal, dtype=jnp.float64)
    times_j = jnp.asarray(t_protein, dtype=jnp.float64)
    _interp = _interp_fn(interp_mode)

    def _s_prod_fn(t):
        raw = _interp(t, times_j, prot_signal_j)
        return _f(raw)

    return _s_prod_fn


# ---------------------------------------------------------------------------
# Diagnostic / visualisation data interpolation (NOT used in the ODE or loss)
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
    """Build continuous interpolation callables for observed datasets.

    This function is for **diagnostic / visualisation** purposes only.
    It does NOT affect the optimisation loss, does NOT expand the training
    targets, and must NOT be passed into any JAX-traced or jitted code path.

    The original sparse observed arrays (``P_data``, ``A_data``, ``rna_data``)
    are never modified.  NaN handling for interpolation is performed on a
    working copy, and the chosen handling strategy is clearly documented in
    the returned metadata dict.

    Args:
        t_obs:      (T,) observed time points (must be sorted and finite).
        P_data:     (N_sites, T) phosphosite observed data.  May contain NaN.
        A_data:     (K_obs, T) protein abundance observed data.  May contain NaN.
        rna_data:   (n_genes, T_rna) mRNA data.  May contain NaN.
                    Typically uses a different time vector; pass ``t_rna``
                    as the first argument when interpolating RNA data separately.
        method:     Interpolation method: ``"linear"`` (safe for NaN-heavy data)
                    or ``"cubic_hermite"`` (SciPy cubic Hermite spline via
                    ``scipy.interpolate.PchipInterpolator``).
        fill_forward_nans_at_end:
                    When True, forward-fill the last valid value for NaN tails
                    (applies to the interpolation-only copy; does NOT modify the
                    original sparse arrays used in the loss).
        replace_nans_at_start:
                    Strategy for leading NaNs (before the first valid point).
                    ``None`` → leave as NaN;
                    ``"zero"`` → fill with 0.0;
                    ``"first_valid"`` → repeat the first valid value.
                    Never modifies the original training arrays.

    Returns:
        (dict): Keys ``"t_obs"``, ``"P_interp"``, ``"A_interp"``, ``"rna_interp"``,
            ``"method"``, ``"nan_fill_log"``, ``"original_arrays_unchanged"``.
            All interp values are callables ``fn(t_q) -> np.ndarray`` or ``None``.

    Raises:
        ImportError: If ``method="cubic_hermite"`` and SciPy is not installed.
    """
    t = np.asarray(t_obs, dtype=np.float64)
    nan_fill_log: list[str] = []

    def _prep_row(row: np.ndarray, label: str) -> np.ndarray:
        """Apply NaN handling to a single (T,) series on a working copy."""
        row = row.copy().astype(np.float64)
        # Handle leading NaNs
        if replace_nans_at_start is not None:
            first_valid = next((i for i, v in enumerate(row) if np.isfinite(v)), None)
            if first_valid is None:
                # All NaN – nothing to do
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
        # Handle trailing NaNs
        if fill_forward_nans_at_end:
            last_valid = next(
                (i for i in range(len(row) - 1, -1, -1) if np.isfinite(row[i])), None
            )
            if last_valid is not None and last_valid < len(row) - 1:
                row[last_valid + 1 :] = row[last_valid]
                n_trailing = len(row) - 1 - last_valid
                nan_fill_log.append(
                    f"  {label}: forward-filled {n_trailing} trailing NaN(s) "
                    f"from t={t[last_valid]:.3g}."
                )
        return row

    def _make_interp(data: np.ndarray, label_prefix: str):
        """Return a callable fn(t_query) -> np.ndarray for a (D, T) matrix."""
        D, T = data.shape
        # Build per-row callables; rows with insufficient finite points fall
        # back to NaN so the caller can detect missing data.
        rows_prepped = []
        for d in range(D):
            rows_prepped.append(_prep_row(data[d], f"{label_prefix}[{d}]"))

        # Find valid (finite) time indices across all rows (union)
        any_finite = np.zeros(T, dtype=bool)
        for row in rows_prepped:
            any_finite |= np.isfinite(row)
        valid_t_idx = np.where(any_finite)[0]

        if len(valid_t_idx) < 2:
            # Cannot interpolate with fewer than 2 points
            nan_fill_log.append(
                f"  {label_prefix}: fewer than 2 valid time points; "
                "returning NaN callable."
            )

            def _nan_fn(t_q):
                return np.full(D, np.nan)

            return _nan_fn

        t_valid = t[valid_t_idx]

        if method == "cubic_hermite":
            try:
                from scipy.interpolate import PchipInterpolator
            except ImportError as exc:
                raise ImportError(
                    "SciPy is required for method='cubic_hermite'. "
                    "Install it with `pip install scipy`."
                ) from exc
            interps = []
            for row in rows_prepped:
                row_valid = row[valid_t_idx]
                # Replace any remaining NaN in valid_t_idx positions with 0 for
                # the interpolator (not for the original data).
                row_for_interp = np.where(np.isfinite(row_valid), row_valid, 0.0)
                interps.append(PchipInterpolator(t_valid, row_for_interp, extrapolate=False))

            def _cubic_fn(t_q):
                out = np.stack([interp(t_q) for interp in interps], axis=0)
                # Shape: (D,) for scalar t_q, (D, len(t_q)) for array t_q
                return out

            return _cubic_fn
        else:
            # Default: linear interpolation using numpy
            interps = []
            for row in rows_prepped:
                row_valid = row[valid_t_idx]
                row_for_interp = np.where(np.isfinite(row_valid), row_valid, 0.0)
                interps.append((t_valid, row_for_interp))

            def _linear_fn(t_q, _interps=interps):
                # Shape: (D,) for scalar t_q, (D, len(t_q)) for array t_q
                return np.array([
                    np.interp(t_q, tv, rv, left=np.nan, right=np.nan)
                    for tv, rv in _interps
                ])

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
        if arr.size > 0 and arr.ndim == 2 and arr.shape[1] == len(t):
            result["P_interp"] = _make_interp(arr, "P_data")

    if A_data is not None:
        arr = np.asarray(A_data, dtype=np.float64)
        if arr.size > 0 and arr.ndim == 2 and arr.shape[1] == len(t):
            result["A_interp"] = _make_interp(arr, "A_data")

    if rna_data is not None:
        arr = np.asarray(rna_data, dtype=np.float64)
        if arr.size > 0 and arr.ndim == 2:
            # rna_data may use a different time axis; we use t_obs here as
            # passed (callers should pass the RNA-specific time vector as t_obs
            # if RNA data uses a different grid).
            if arr.shape[1] == len(t):
                result["rna_interp"] = _make_interp(arr, "rna_data")

    return result
