"""
Weighting functions for phospho-network model simulations.
"""

from __future__ import annotations

import numpy as np

_EPS = 1e-12


def _as_2d_float_matrix(x: np.ndarray | None, name: str) -> np.ndarray:
    """
    Convert input to a finite 2D float matrix.

    Empty or None inputs return shape (0, 0).
    """
    if x is None:
        return np.zeros((0, 0), dtype=float)

    arr = np.asarray(x, dtype=float)

    if arr.size == 0:
        return np.zeros((0, 0), dtype=float)

    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2D matrix; got shape {arr.shape}.")

    if not np.isfinite(arr).all():
        bad = int((~np.isfinite(arr)).sum())
        raise ValueError(f"{name} contains {bad} non-finite value(s).")

    return arr


def _as_time_vector(t: np.ndarray, name: str) -> np.ndarray:
    """
    Convert time points to a finite 1D float vector.
    """
    arr = np.asarray(t, dtype=float)

    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1D time vector; got shape {arr.shape}.")

    if arr.size < 1:
        raise ValueError(f"{name} must contain at least one time point.")

    if not np.isfinite(arr).all():
        bad = int((~np.isfinite(arr)).sum())
        raise ValueError(f"{name} contains {bad} non-finite value(s).")

    return arr


def _normalize_mean_one(w: np.ndarray) -> np.ndarray:
    """
    Normalize weights to mean 1.0 while preserving empty arrays.
    """
    w = np.asarray(w, dtype=float)

    if w.size == 0:
        return w

    mean = float(np.mean(w))
    if not np.isfinite(mean) or abs(mean) < _EPS:
        return np.ones_like(w, dtype=float)

    return w / mean


def _compute_noise_weights(
        X: np.ndarray | None,
        *,
        name: str,
        min_weight: float = 0.1,
        max_weight: float = 20.0,
) -> np.ndarray:
    """
    Compute entity-level inverse-noise weights from temporal jaggedness.

    The noise estimate is based on first differences in log1p space. Noisier
    trajectories receive lower weights. The returned vector is clipped and
    normalized to mean 1.0.

    Args:
        X:
            Matrix with shape (n_entities, n_timepoints).
        name:
            Name used in error messages.
        min_weight:
            Lower clipping bound.
        max_weight:
            Upper clipping bound.

    Returns:
        Weight vector with shape (n_entities,).
    """
    X = _as_2d_float_matrix(X, name)

    if X.size == 0:
        return np.zeros((0,), dtype=float)

    if X.shape[1] < 2:
        return np.ones((X.shape[0],), dtype=float)

    logX = np.log1p(np.clip(X, 1e-3, None))
    diff = np.diff(logX, axis=1)
    sigma = np.sqrt(np.mean(diff * diff, axis=1) + 1e-8)

    w = 1.0 / (sigma * sigma + 1e-4)
    w = np.clip(w, min_weight, max_weight)
    return _normalize_mean_one(w)


def _time_weights_uniform(t: np.ndarray) -> np.ndarray:
    """
    Equal temporal weights.
    """
    t = _as_time_vector(t, "t")
    return np.ones_like(t, dtype=float)


def _time_weights_early_emphasis(
        t: np.ndarray,
        *,
        t_mid: float | None = None,
        strength: float = 2.0,
) -> np.ndarray:
    """
    Early-time emphasis using a smooth exponential decay.

    Larger ``strength`` gives stronger weighting to early time points.
    """
    t = _as_time_vector(t, "t")

    if strength <= 0:
        raise ValueError(f"strength must be positive; got {strength}.")

    if t_mid is None:
        t_mid = float(np.median(t))

    if not np.isfinite(t_mid) or t_mid <= 0:
        positive = t[t > 0]
        t_mid = float(np.median(positive)) if positive.size else 1.0

    scale = np.log(strength + _EPS)
    w = np.exp(-scale * (t / (t_mid + _EPS)))
    return _normalize_mean_one(w)


def _time_weights_early_emphasis_moderate(t: np.ndarray) -> np.ndarray:
    """
    Moderate early-time emphasis.
    """
    return _time_weights_early_emphasis(t, strength=1.5)


def _time_weights_late_emphasis(
        t: np.ndarray,
        *,
        t_mid: float | None = None,
        strength: float = 2.0,
) -> np.ndarray:
    """
    Late-time emphasis. Useful when long-term convergence matters.
    """
    t = _as_time_vector(t, "t")

    if strength <= 0:
        raise ValueError(f"strength must be positive; got {strength}.")

    if t_mid is None:
        t_mid = float(np.median(t))

    if not np.isfinite(t_mid) or t_mid <= 0:
        positive = t[t > 0]
        t_mid = float(np.median(positive)) if positive.size else 1.0

    scale = np.log(strength + _EPS)
    w = np.exp(scale * (t / (t_mid + _EPS)))
    return _normalize_mean_one(w)


def _time_weights_by_scheme(t: np.ndarray, scheme: str) -> np.ndarray:
    """
    Build temporal weights for a named scheme.
    """
    if scheme == "uniform":
        return _time_weights_uniform(t)

    if scheme == "early_emphasis":
        return _time_weights_early_emphasis(t, strength=2.0)

    if scheme == "early_emphasis_moderate":
        return _time_weights_early_emphasis_moderate(t)

    if scheme == "late_emphasis":
        return _time_weights_late_emphasis(t, strength=2.0)

    if scheme == "flat_no_noise":
        return np.ones_like(_as_time_vector(t, "t"), dtype=float)

    raise ValueError(
        "Unknown weighting scheme: "
        f"{scheme!r}. Expected one of: uniform, early_emphasis, "
        "early_emphasis_moderate, late_emphasis, flat_no_noise."
    )


def build_weight_matrices(
        t: np.ndarray,
        Y: np.ndarray,
        A_data: np.ndarray | None = None,
        *,
        t_mrna: np.ndarray | None = None,
        rna_data: np.ndarray | None = None,
        scheme: str = "uniform",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build phosphosite, protein-abundance, and mRNA loss weight matrices.

    Args:
        t:
            Protein/phosphosite time points, shape (T,).
        Y:
            Phosphosite data matrix, shape (N_sites, T).
        A_data:
            Optional protein abundance matrix, shape (K_obs, T).
        t_mrna:
            Optional mRNA time points, shape (T_rna,).
        rna_data:
            Optional mRNA matrix, shape (G_or_matched, T_rna).
        scheme:
            Weighting scheme:
                - ``uniform``
                - ``early_emphasis``
                - ``early_emphasis_moderate``
                - ``late_emphasis``
                - ``flat_no_noise``

    Returns:
        tuple:
            - W_data: phosphosite weights, shape (N_sites, T)
            - W_data_prot: protein weights, shape (K_obs, T), or (0, T)
            - W_data_mrna: mRNA weights, shape (G_or_matched, T_rna), or (0, T_rna)

    Notes:
        ``flat_no_noise`` disables entity-level noise weighting and returns all
        ones for available modalities.
    """
    t = _as_time_vector(t, "t")
    Y = _as_2d_float_matrix(Y, "Y")

    if Y.shape[1] != t.shape[0]:
        raise ValueError(
            f"Y has {Y.shape[1]} columns but t has {t.shape[0]} time points."
        )

    A = _as_2d_float_matrix(A_data, "A_data")
    if A.size > 0 and A.shape[1] != t.shape[0]:
        raise ValueError(
            f"A_data has {A.shape[1]} columns but t has {t.shape[0]} time points."
        )

    if t_mrna is None:
        if rna_data is not None and np.asarray(rna_data).size > 0:
            raise ValueError("rna_data was provided but t_mrna is None.")
        t_mrna_arr = np.zeros((0,), dtype=float)
    else:
        t_mrna_arr = _as_time_vector(t_mrna, "t_mrna")

    RNA = _as_2d_float_matrix(rna_data, "rna_data")
    if RNA.size > 0 and RNA.shape[1] != t_mrna_arr.shape[0]:
        raise ValueError(
            f"rna_data has {RNA.shape[1]} columns but t_mrna has "
            f"{t_mrna_arr.shape[0]} time points."
        )

    # Time weights
    w_time = _time_weights_by_scheme(t, scheme)
    w_time = _normalize_mean_one(w_time)

    if t_mrna_arr.size > 0:
        w_time_mrna = _time_weights_by_scheme(t_mrna_arr, scheme)
        w_time_mrna = _normalize_mean_one(w_time_mrna)
    else:
        w_time_mrna = np.zeros((0,), dtype=float)

    # Entity weights
    if scheme == "flat_no_noise":
        w_site = np.ones((Y.shape[0],), dtype=float)
        w_prot = np.ones((A.shape[0],), dtype=float) if A.size > 0 else np.zeros((0,))
        w_mrna = (
            np.ones((RNA.shape[0],), dtype=float) if RNA.size > 0 else np.zeros((0,))
        )
    else:
        w_site = _compute_noise_weights(Y, name="Y")
        w_prot = (
            _compute_noise_weights(A, name="A_data") if A.size > 0 else np.zeros((0,))
        )
        w_mrna = (
            _compute_noise_weights(RNA, name="rna_data")
            if RNA.size > 0
            else np.zeros((0,))
        )

    w_site = _normalize_mean_one(w_site)
    w_prot = _normalize_mean_one(w_prot)
    w_mrna = _normalize_mean_one(w_mrna)

    # Full matrices
    W_data = np.outer(w_site, w_time)

    if A.size > 0:
        W_data_prot = np.outer(w_prot, w_time)
    else:
        W_data_prot = np.zeros((0, t.shape[0]), dtype=float)

    if RNA.size > 0:
        W_data_mrna = np.outer(w_mrna, w_time_mrna)
    else:
        W_data_mrna = np.zeros((0, t_mrna_arr.shape[0]), dtype=float)

    return W_data, W_data_prot, W_data_mrna
