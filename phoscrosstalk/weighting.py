
from __future__ import annotations

import numpy as np


def _compute_site_noise_weights(Y: np.ndarray) -> np.ndarray:
    """
    Site-level weights from temporal noise in log-space.

    Y: (N_sites, T) raw FC (non-negative or small epsilon added)
    Returns:
        w_site: (N_sites,) normalized ~ O(1)
    """
    logY = np.log1p(np.clip(Y, 1e-3, None))
    diff = np.diff(logY, axis=1)  # (N, T-1)
    sigma_site = np.sqrt((diff ** 2).mean(axis=1) + 1e-8)

    w_site = 1.0 / (sigma_site ** 2 + 1e-4)
    w_site = np.clip(w_site, 0.1, 20.0)
    w_site /= w_site.mean()
    return w_site


def _compute_protein_noise_weights(A_data: np.ndarray | None) -> np.ndarray:
    """
    Protein-level weights from temporal noise in log-space.

    A_data: (K_obs, T) raw FC, or None / empty if no protein data.

    Returns:
        w_prot: (K_obs,) normalized ~ O(1), or empty array if no data.
    """
    if A_data is None or A_data.size == 0:
        return np.zeros((0,), dtype=float)

    logA = np.log1p(np.clip(A_data, 1e-3, None))
    diffA = np.diff(logA, axis=1)  # (K_obs, T-1)
    sigma_prot = np.sqrt((diffA ** 2).mean(axis=1) + 1e-8)

    w_prot = 1.0 / (sigma_prot ** 2 + 1e-4)
    w_prot = np.clip(w_prot, 0.1, 20.0)
    w_prot /= w_prot.mean()
    return w_prot


def _time_weights_uniform(t: np.ndarray) -> np.ndarray:
    w_time = np.ones_like(t, dtype=float)
    w_time /= max(w_time.mean(), 1e-12)
    return w_time


def _time_weights_early_emphasis(t: np.ndarray,
                                 t_mid: float | None = None,
                                 strength: float = 2.0) -> np.ndarray:
    """
    Early-emphasis via a decreasing function over time.

    strength ~ how much heavier earliest point is vs latest.
    t_mid: optional "half weight" time. If None, uses median(t).
    """
    t = np.asarray(t, dtype=float)
    if t_mid is None:
        t_mid = float(np.median(t))

    # Smooth logistic decay from ~1 at t << t_mid to ~1/strength at t >> t_mid
    # Then renormalize to mean 1.
    eps = 1e-12
    scale = np.log(strength + eps)
    # weight(t) ~ exp(-scale * (t / t_mid)) in effect
    w = np.exp(-scale * (t / (t_mid + eps)))
    w /= max(w.mean(), eps)
    return w


def _time_weights_early_emphasis_moderate(t: np.ndarray) -> np.ndarray:
    """
    A milder early-emphasis preset, just calls _time_weights_early_emphasis
    with smaller strength.
    """
    return _time_weights_early_emphasis(t, strength=1.5)


def build_weight_matrices(
    t: np.ndarray,
    Y: np.ndarray,
    A_data: np.ndarray | None = None,
    scheme: str = "uniform",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build (W_data, W_data_prot) for a given weighting scheme.

    Parameters
    ----------
    t : (T,)
        Time points.
    Y : (N_sites, T)
        Phosphosite data (raw FC).
    A_data : (K_obs, T) or None
        Protein abundance data (raw FC) if available.
    scheme : str
        One of:
            - "uniform"                     : uniform time, noise-based sites/proteins
            - "early_emphasis"              : strong emphasis on early time points
            - "early_emphasis_moderate"     : milder early-emphasis
            - "flat_no_noise"               : fully uniform across time AND sites

    Returns
    -------
    W_data : (N_sites, T)
        Site-level weights.
    W_data_prot : (K_obs, T)
        Protein-level weights (empty if no A_data).
    """

    t = np.asarray(t, dtype=float)
    Y = np.asarray(Y, dtype=float)

    # --- base per-site and per-protein weights from noise ---
    w_site = _compute_site_noise_weights(Y)
    w_prot = _compute_protein_noise_weights(A_data)

    # --- time weights according to scheme ---
    if scheme == "uniform":
        w_time = _time_weights_uniform(t)

    elif scheme == "early_emphasis":
        w_time = _time_weights_early_emphasis(t, strength=2.0)

    elif scheme == "early_emphasis_moderate":
        w_time = _time_weights_early_emphasis_moderate(t)

    elif scheme == "flat_no_noise":
        # ignore noise, everything = 1
        w_site = np.ones(Y.shape[0], dtype=float)
        if A_data is not None and A_data.size > 0:
            w_prot = np.ones(A_data.shape[0], dtype=float)
        w_time = np.ones_like(t, dtype=float)

    else:
        raise ValueError(f"Unknown weighting scheme: {scheme}")

    # Normalize to mean ~1 to keep overall scale consistent
    w_site /= max(w_site.mean(), 1e-12)
    if w_prot.size > 0:
        w_prot /= max(w_prot.mean(), 1e-12)
    w_time /= max(w_time.mean(), 1e-12)

    # --- outer products to full matrices ---
    W_data = np.outer(w_site, w_time)  # (N_sites, T)

    if A_data is not None and A_data.size > 0:
        W_data_prot = np.outer(w_prot, w_time)  # (K_obs, T)
    else:
        W_data_prot = np.zeros((0, t.shape[0]), dtype=float)

    return W_data, W_data_prot
