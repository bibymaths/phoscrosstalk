from __future__ import annotations

import numpy as np


def _compute_site_noise_weights(Y: np.ndarray) -> np.ndarray:
    """
    Calculate site-specific weights based on the temporal noise of the signal.

    Estimates noise by calculating the variance of the first differences in log-space.
    Noisier sites (high variance in step-to-step changes) are assigned lower weights
    (inverse variance weighting), normalized to a mean of 1.0.

    Args:
        Y (np.ndarray): Raw phosphosite data matrix (N_sites x T).

    Returns:
        np.ndarray: Vector of weights (N_sites,), clipped and normalized.
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
    Calculate protein-specific weights based on temporal noise.

    Similar to `_compute_site_noise_weights`, this down-weights protein trajectories
    that exhibit high jaggedness/noise in log-space.

    Args:
        A_data (np.ndarray | None): Raw protein abundance data matrix (K_obs x T).

    Returns:
        np.ndarray: Vector of weights (K_obs,), or an empty array if no data is provided.
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
    """
    Generate uniform temporal weights (all time points weighted equally).

    Args:
        t (np.ndarray): Time points vector.

    Returns:
        np.ndarray: Weight vector of ones.
    """
    w_time = np.ones_like(t, dtype=float)
    w_time /= max(w_time.mean(), 1e-12)
    return w_time


def _time_weights_early_emphasis(t: np.ndarray,
                                 t_mid: float | None = None,
                                 strength: float = 2.0) -> np.ndarray:
    """
    Generate temporal weights that decay over time, emphasizing early kinetics.

    Useful for signaling data where the initial response (transient phase) is often
    more information-rich than the late steady state. Uses an exponential decay function.

    Args:
        t (np.ndarray): Time points vector.
        t_mid (float, optional): Time point where weighting is reduced. Defaults to median(t).
        strength (float): Factor determining the steepness of the decay.

    Returns:
        np.ndarray: Temporal weight vector.
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
    A preset for early-emphasis weighting with a milder decay strength (1.5).

    Args:
        t (np.ndarray): Time points vector.

    Returns:
        np.ndarray: Temporal weight vector.
    """
    return _time_weights_early_emphasis(t, strength=1.5)


def build_weight_matrices(
        t: np.ndarray,
        Y: np.ndarray,
        A_data: np.ndarray | None = None,
        scheme: str = "uniform",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Construct full weight matrices for the loss function based on the selected scheme.

    Combines entity-level weights (based on signal noise) and temporal weights (based on
    the selected scheme) via an outer product. This results in specific weights for every
    data point in the time-series.

    Schemes:
    - **uniform**: Time points equal; noisy sites down-weighted.
    - **early_emphasis**: Early time points weighted higher; noisy sites down-weighted.
    - **flat_no_noise**: All weights set to 1.0 (noise ignored).

    Args:
        t (np.ndarray): Time points.
        Y (np.ndarray): Phosphosite data matrix.
        A_data (np.ndarray | None): Protein data matrix.
        scheme (str): Weighting strategy identifier.

    Returns:
        tuple:
            - W_data (np.ndarray): Weight matrix for phosphosites (N_sites x T).
            - W_data_prot (np.ndarray): Weight matrix for proteins (K_obs x T).
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
