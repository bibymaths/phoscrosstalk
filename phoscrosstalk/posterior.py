"""
posterior.py
============
Monte-Carlo (MCMC) posterior inference for phoscrosstalk model parameters.

Uses BlackJax (NUTS / HMC) to sample the posterior distribution of estimated
mechanistic parameters (theta).  The module is designed to be used after the
multi-start optimisation step and is independent of the neural ODE mode.

Typical usage in ``main.py``::

    from phoscrosstalk.posterior import run_posterior_inference

    # Works in both normal and neural_ode modes.
    posterior_results = run_posterior_inference(
        outdir=outdir,
        theta_best=theta_best,
        log_posterior_fn=...,   # or use the convenience builder below
        posterior_cfg=cfg.posterior,  # optional; falls back to defaults
    )

The log-posterior function can be constructed with
:func:`make_log_posterior_fn` from a problem instance.

Outputs written to ``{outdir}/posterior/``:

* ``posterior_samples.npz``  – thinned parameter samples ``(n_samples, dim)``.
* ``posterior_summary.tsv``  – per-parameter mean, std, 2.5%, 97.5% credible
  intervals and effective sample size (ESS).
* ``posterior_trace.png``    – trace plots (first 8 parameters).
* ``posterior_pairs.png``    – pairplot for the first 8 parameters.
* ``posterior_metadata.json`` – sampler settings and diagnostics.

Requirements
------------
    blackjax >= 1.0

If BlackJax is not installed the module raises ``ImportError`` with a
descriptive message pointing to the dependency.
"""

from __future__ import annotations

import json
import logging
import os
import time
import types
from typing import Any, Callable

import numpy as np

logger = logging.getLogger("phoscrosstalk")


# ---------------------------------------------------------------------------
# Helper: graceful BlackJax import
# ---------------------------------------------------------------------------


def _require_blackjax():
    """Return the ``blackjax`` module; raise ``ImportError`` when absent."""
    try:
        import blackjax  # noqa: PLC0415
        return blackjax
    except ImportError as err:
        raise ImportError(
            "BlackJax is required for posterior inference but is not installed.  "
            "Install it with:  pip install blackjax>=1.0"
        ) from err


# ---------------------------------------------------------------------------
# Default posterior config
# ---------------------------------------------------------------------------


def _default_posterior_cfg() -> types.SimpleNamespace:
    return types.SimpleNamespace(
        enabled=False,
        num_warmup=500,
        num_samples=1000,
        step_size=1e-3,
        thin_factor=10,
        target_acceptance_rate=0.8,
        seed=42,
        max_tree_depth=10,
        # Number of *leading* parameters to include in trace / pairs plots.
        plot_params=8,
    )


def _merge_cfg(user_cfg) -> types.SimpleNamespace:
    """Merge *user_cfg* (may be None or a SimpleNamespace) with defaults."""
    defaults = _default_posterior_cfg()
    if user_cfg is None:
        return defaults
    merged = types.SimpleNamespace(**vars(defaults))
    for key, val in vars(user_cfg).items():
        setattr(merged, key, val)
    return merged


# ---------------------------------------------------------------------------
# Log-posterior builder
# ---------------------------------------------------------------------------


def make_log_posterior_fn(
    *,
    residuals_fn: Callable,
    theta_lower: np.ndarray,
    theta_upper: np.ndarray,
    theta_prior_mean: np.ndarray | None = None,
    theta_prior_std: np.ndarray | None = None,
    sigma_noise: float = 0.1,
) -> Callable:
    """Build a JAX-compatible log-posterior function for NUTS sampling.

    The log-posterior is::

        log p(theta | data)
            = log likelihood(data | theta)
            + log prior(theta)
            + log domain(theta)

    where:

    * **Log likelihood** is a Gaussian with fixed observation noise
      ``sigma_noise`` applied to the residuals returned by *residuals_fn*.
    * **Log prior** is a Gaussian centred on *theta_prior_mean* (defaults to
      the midpoint of the bounds) with standard deviation *theta_prior_std*
      (defaults to ``(upper - lower) / 4``).
    * **Log domain** is ``-inf`` when any parameter is outside
      ``[theta_lower, theta_upper]``, otherwise 0.

    Args:
        residuals_fn:      ``theta → 1-D array of residuals``.  Must be JAX-
                           traceable (i.e. use ``jnp`` operations internally).
        theta_lower:       Lower bounds ``(dim,)``.
        theta_upper:       Upper bounds ``(dim,)``.
        theta_prior_mean:  Prior mean ``(dim,)``.  Defaults to midpoint.
        theta_prior_std:   Prior std ``(dim,)``.  Defaults to range / 4.
        sigma_noise:       Observation noise standard deviation (scalar).

    Returns:
        Callable: ``log_posterior_fn(theta) → scalar JAX float``.
    """
    import jax  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415

    xl = jnp.asarray(theta_lower, dtype=jnp.float64)
    xu = jnp.asarray(theta_upper, dtype=jnp.float64)
    mu = jnp.asarray(
        theta_prior_mean if theta_prior_mean is not None else 0.5 * (xl + xu),
        dtype=jnp.float64,
    )
    sigma_p = jnp.asarray(
        theta_prior_std if theta_prior_std is not None else (xu - xl) / 4.0,
        dtype=jnp.float64,
    )
    sigma_n = jnp.asarray(sigma_noise, dtype=jnp.float64)

    def log_posterior_fn(theta: jax.Array) -> jax.Array:
        # Domain check: -inf if outside bounds.
        in_bounds = jnp.all((theta >= xl) & (theta <= xu))

        # Gaussian log-likelihood.
        resid = jnp.asarray(residuals_fn(theta), dtype=jnp.float64)
        log_lik = -0.5 * jnp.sum(resid ** 2) / (sigma_n ** 2)

        # Gaussian prior.
        log_prior = -0.5 * jnp.sum(((theta - mu) / sigma_p) ** 2)

        total = log_lik + log_prior
        return jnp.where(in_bounds, total, jnp.asarray(-jnp.inf, dtype=jnp.float64))

    return log_posterior_fn


# ---------------------------------------------------------------------------
# Main inference entry point
# ---------------------------------------------------------------------------


def run_posterior_inference(
    *,
    outdir: str,
    theta_best: np.ndarray,
    log_posterior_fn: Callable,
    posterior_cfg=None,
    proteins: list[str] | None = None,
    theta_names: list[str] | None = None,
) -> dict[str, Any]:
    """Run NUTS posterior inference and save results.

    The sampler runs in two phases:

    1. **Warm-up** (``num_warmup`` steps, dual-averaging step-size
       adaptation).  Warm-up samples are discarded.
    2. **Sampling** (``num_samples`` steps, fixed step size from warm-up).
       Samples are thinned by ``thin_factor`` before saving.

    Args:
        outdir:            Root results directory; outputs go to
                           ``{outdir}/posterior/``.
        theta_best:        MAP / best-fit parameter vector ``(dim,)`` used
                           as the initial chain position.
        log_posterior_fn:  JAX-traceable ``theta → scalar`` log-posterior.
                           Build with :func:`make_log_posterior_fn` or
                           supply a custom function.
        posterior_cfg:     Optional config ``SimpleNamespace``; merged with
                           defaults from :func:`_default_posterior_cfg`.
        proteins:          Optional list of protein names (for output labels).
        theta_names:       Optional list of parameter names ``(dim,)``.  When
                           absent, uses ``"theta_0", "theta_1", ...``.

    Returns:
        dict: Keys:

        * ``"samples"``       – ``np.ndarray`` of thinned samples ``(n, dim)``.
        * ``"acceptance_rate"`` – mean NUTS acceptance rate.
        * ``"outdir"``        – path to ``{outdir}/posterior/``.
        * ``"summary"``       – ``pd.DataFrame`` with per-parameter statistics.
    """
    blackjax = _require_blackjax()

    import jax  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415
    import pandas as pd  # noqa: PLC0415

    cfg = _merge_cfg(posterior_cfg)

    post_dir = os.path.join(outdir, "posterior")
    os.makedirs(post_dir, exist_ok=True)

    dim = len(theta_best)
    if theta_names is None:
        theta_names = [f"theta_{i}" for i in range(dim)]

    key = jax.random.PRNGKey(int(cfg.seed))
    init_position = jnp.asarray(theta_best, dtype=jnp.float64)

    # ------------------------------------------------------------------ #
    # 1. Warm-up with dual-averaging step-size adaptation (NUTS)          #
    # ------------------------------------------------------------------ #
    logger.info(
        "[posterior] Starting NUTS warm-up: %d steps (seed=%d, initial_step_size=%.3g)",
        int(cfg.num_warmup),
        int(cfg.seed),
        float(cfg.step_size),
    )
    t_start = time.time()

    warmup = blackjax.window_adaptation(
        blackjax.nuts,
        log_posterior_fn,
        target_acceptance_rate=float(cfg.target_acceptance_rate),
    )

    key, warmup_key = jax.random.split(key)
    (warmup_state, warmup_params), _ = warmup.run(
        warmup_key,
        init_position,
        num_steps=int(cfg.num_warmup),
    )

    adapted_step_size = float(warmup_params.get("step_size", cfg.step_size))
    inverse_mass_matrix = warmup_params.get(
        "inverse_mass_matrix", jnp.ones(dim, dtype=jnp.float64)
    )

    logger.info(
        "[posterior] Warm-up complete in %.1f s.  Adapted step size: %.4g",
        time.time() - t_start,
        adapted_step_size,
    )

    # ------------------------------------------------------------------ #
    # 2. Sampling                                                          #
    # ------------------------------------------------------------------ #
    logger.info(
        "[posterior] Drawing %d NUTS samples (thin_factor=%d) …",
        int(cfg.num_samples),
        int(cfg.thin_factor),
    )

    nuts_kernel = blackjax.nuts(
        log_posterior_fn,
        adapted_step_size,
        inverse_mass_matrix,
    )

    state = nuts_kernel.init(warmup_state.position)
    step_fn = jax.jit(nuts_kernel.step)

    def _one_step(state, rng_key):
        state, info = step_fn(rng_key, state)
        return state, (state, info)

    key, sample_key = jax.random.split(key)
    keys = jax.random.split(sample_key, int(cfg.num_samples))

    t_sample = time.time()
    _, (states, infos) = jax.lax.scan(_one_step, state, keys)
    sample_elapsed = time.time() - t_sample

    logger.info("[posterior] Sampling done in %.1f s.", sample_elapsed)

    # ------------------------------------------------------------------ #
    # 3. Extract and thin samples                                          #
    # ------------------------------------------------------------------ #
    all_positions = np.asarray(states.position)   # (num_samples, dim)
    acceptance_rate = float(jnp.mean(infos.acceptance_rate))
    logger.info("[posterior] Mean acceptance rate: %.3f", acceptance_rate)

    thin = max(1, int(cfg.thin_factor))
    thinned = all_positions[::thin]  # (n_thinned, dim)
    n_thinned = len(thinned)
    logger.info("[posterior] Thinned to %d samples.", n_thinned)

    # ------------------------------------------------------------------ #
    # 4. Summary statistics                                                #
    # ------------------------------------------------------------------ #
    summary_rows = []
    for i, name in enumerate(theta_names):
        col = thinned[:, i]
        finite = col[np.isfinite(col)]
        if len(finite) == 0:
            summary_rows.append({
                "parameter": name, "mean": float("nan"), "std": float("nan"),
                "q2.5": float("nan"), "q97.5": float("nan"), "ess": 0,
            })
            continue
        mean_val = float(np.mean(finite))
        std_val = float(np.std(finite))
        q025 = float(np.percentile(finite, 2.5))
        q975 = float(np.percentile(finite, 97.5))
        # Naive ESS (batch means approximation)
        ess = _naive_ess(finite)
        summary_rows.append({
            "parameter": name,
            "mean": mean_val,
            "std": std_val,
            "q2.5": q025,
            "q97.5": q975,
            "ess": ess,
        })

    df_summary = pd.DataFrame(summary_rows)
    summary_path = os.path.join(post_dir, "posterior_summary.tsv")
    df_summary.to_csv(summary_path, sep="\t", index=False)
    logger.info("[posterior] Saved summary to %s", summary_path)

    # ------------------------------------------------------------------ #
    # 5. Save samples NPZ                                                  #
    # ------------------------------------------------------------------ #
    npz_path = os.path.join(post_dir, "posterior_samples.npz")
    np.savez(
        npz_path,
        samples=thinned,
        theta_names=np.array(theta_names, dtype=object),
        theta_best=np.asarray(theta_best, dtype=np.float64),
        acceptance_rate=np.array(acceptance_rate),
    )
    logger.info("[posterior] Saved samples to %s", npz_path)

    # ------------------------------------------------------------------ #
    # 6. Metadata JSON                                                     #
    # ------------------------------------------------------------------ #
    meta = {
        "num_warmup": int(cfg.num_warmup),
        "num_samples": int(cfg.num_samples),
        "thin_factor": thin,
        "n_thinned_samples": n_thinned,
        "initial_step_size": float(cfg.step_size),
        "adapted_step_size": adapted_step_size,
        "target_acceptance_rate": float(cfg.target_acceptance_rate),
        "actual_acceptance_rate": acceptance_rate,
        "seed": int(cfg.seed),
        "dim": dim,
        "sampler": "NUTS (BlackJax)",
        "warmup_time_s": float(time.time() - t_start - sample_elapsed),
        "sample_time_s": float(sample_elapsed),
    }
    meta_path = os.path.join(post_dir, "posterior_metadata.json")
    with open(meta_path, "w") as fh:
        json.dump(meta, fh, indent=2)
    logger.info("[posterior] Saved metadata to %s", meta_path)

    # ------------------------------------------------------------------ #
    # 7. Plots                                                             #
    # ------------------------------------------------------------------ #
    try:
        _plot_posterior(post_dir, thinned, theta_names, int(cfg.plot_params))
    except Exception as exc:
        logger.warning("[posterior] Plot generation failed: %s", exc)

    return {
        "samples": thinned,
        "acceptance_rate": acceptance_rate,
        "outdir": post_dir,
        "summary": df_summary,
    }


# ---------------------------------------------------------------------------
# Posterior prediction
# ---------------------------------------------------------------------------


def posterior_predict(
    *,
    samples: np.ndarray,
    simulate_fn: Callable,
    t_eval: np.ndarray,
    thin_factor: int = 1,
    credible_intervals: tuple[float, float] = (2.5, 97.5),
) -> dict[str, np.ndarray]:
    """Draw posterior-predictive trajectories.

    For each (thinned) sample, calls *simulate_fn(theta)* and stacks the
    resulting trajectories.  Returns the mean and credible interval across
    samples.

    Args:
        samples:            Posterior samples ``(n_samples, dim)``.
        simulate_fn:        ``theta → dict`` with at least ``"P_sim"``,
                            ``"A_sim"`` arrays.  May return any keys.
        t_eval:             Time points used in *simulate_fn*.
        thin_factor:        Sub-sample every *thin_factor*-th sample to reduce
                            computation.
        credible_intervals: (lower, upper) percentile bounds (default 2.5/97.5).

    Returns:
        dict: Keys per simulated quantity (``"P_sim"``, ``"A_sim"``, etc.):

        * ``"mean"``   – ``(entity, time)`` mean across samples.
        * ``"lower"``  – ``(entity, time)`` lower CI.
        * ``"upper"``  – ``(entity, time)`` upper CI.
        * ``"samples"`` – ``(n_used, entity, time)`` all trajectories.
    """
    thin = max(1, int(thin_factor))
    use_samples = samples[::thin]
    all_sims: list[dict] = []

    for theta in use_samples:
        try:
            result = simulate_fn(np.asarray(theta))
            all_sims.append({k: np.asarray(v, dtype=float) for k, v in result.items()})
        except Exception:
            continue

    if not all_sims:
        logger.warning("[posterior] posterior_predict: no valid simulations produced.")
        return {}

    output: dict[str, dict[str, np.ndarray]] = {}
    for key in all_sims[0]:
        stack = np.stack([s[key] for s in all_sims], axis=0)  # (n, entity, time)
        lo, hi = credible_intervals
        output[key] = {
            "mean": np.mean(stack, axis=0),
            "lower": np.percentile(stack, lo, axis=0),
            "upper": np.percentile(stack, hi, axis=0),
            "samples": stack,
        }

    return output


def plot_posterior_predictive(
    outdir: str,
    *,
    pred: dict,
    t_eval: np.ndarray,
    entity_names: list[str],
    key: str = "P_sim",
    observed: np.ndarray | None = None,
    t_observed: np.ndarray | None = None,
    max_entities: int = 12,
) -> None:
    """Plot posterior-predictive mean ± 95 % CI vs. observations.

    Args:
        outdir:       Directory to save ``posterior_predictive_{key}.png``.
        pred:         Output of :func:`posterior_predict`.
        t_eval:       Dense time axis used in ``pred``.
        entity_names: Labels for each entity (proteins, sites, etc.).
        key:          Which simulated quantity to plot (``"P_sim"`` etc.).
        observed:     Optional observed data ``(n_entities, T_obs)``.
        t_observed:   Time axis for *observed*.
        max_entities: Cap on how many entities are plotted (by dynamic range).
    """
    import matplotlib  # noqa: PLC0415
    matplotlib.use("Agg")
    from matplotlib import pyplot as _plt  # noqa: PLC0415

    os.makedirs(outdir, exist_ok=True)

    if key not in pred:
        logger.warning("[posterior] Key %r not in posterior predictive dict.", key)
        return

    mean_arr = np.asarray(pred[key]["mean"])
    lower_arr = np.asarray(pred[key]["lower"])
    upper_arr = np.asarray(pred[key]["upper"])
    t_arr = np.asarray(t_eval)

    n_entities = min(len(entity_names), mean_arr.shape[0])

    # Select up to max_entities by dynamic range.
    ranges = np.ptp(mean_arr[:n_entities], axis=1)
    top_idx = np.argsort(ranges)[::-1][:max_entities]
    n_plot = len(top_idx)
    if n_plot == 0:
        return

    ncols = min(4, n_plot)
    nrows = (n_plot + ncols - 1) // ncols
    fig, axes = _plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
    axes_flat = axes.flatten()

    for ai, ei in enumerate(top_idx):
        ax = axes_flat[ai]
        color = _plt.cm.tab10(ai % 10)
        name = entity_names[ei] if ei < len(entity_names) else f"entity_{ei}"

        ax.plot(t_arr, mean_arr[ei], "-", lw=2, color=color, label="Posterior mean")
        ax.fill_between(t_arr, lower_arr[ei], upper_arr[ei],
                        alpha=0.25, color=color, label="95% CI")

        if observed is not None and ei < observed.shape[0]:
            t_obs_arr = np.asarray(t_observed) if t_observed is not None else t_arr
            y_obs = np.asarray(observed[ei], dtype=float)
            m = np.isfinite(y_obs)
            if np.any(m):
                ax.scatter(t_obs_arr[m], y_obs[m], s=40, color=color, zorder=5, label="Observed")

        ax.set_title(name, fontsize=10, fontweight="bold")
        ax.set_xlabel("Time")
        ax.set_ylabel(key)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.25)

    for ax in axes_flat[n_plot:]:
        ax.set_visible(False)

    fig.suptitle(f"Posterior Predictive — {key}", fontsize=13, fontweight="bold")
    _plt.tight_layout()
    _path = os.path.join(outdir, f"posterior_predictive_{key}.png")
    fig.savefig(_path, dpi=300)
    _plt.close(fig)
    logger.info("[posterior] Saved %s", _path)


# ---------------------------------------------------------------------------
# Internal utilities
# ---------------------------------------------------------------------------


def _naive_ess(chain: np.ndarray) -> int:
    """Compute a naive ESS from the lag-1 autocorrelation."""
    n = len(chain)
    if n < 4:
        return n
    mu = np.mean(chain)
    var = np.var(chain)
    if var < 1e-15:
        return n
    ac1 = float(np.sum((chain[:-1] - mu) * (chain[1:] - mu)) / ((n - 1) * var))
    rho = max(0.0, min(ac1, 0.9999))
    ess = int(n * (1.0 - rho) / (1.0 + rho))
    return max(1, ess)


def _plot_posterior(outdir: str, thinned: np.ndarray, names: list[str], n_plot: int) -> None:
    """Save trace and pairs plots for the first *n_plot* parameters."""
    import matplotlib  # noqa: PLC0415
    matplotlib.use("Agg")
    from matplotlib import pyplot as _plt  # noqa: PLC0415

    n_show = min(n_plot, thinned.shape[1])
    if n_show == 0:
        return

    # Trace plot
    ncols = min(4, n_show)
    nrows = (n_show + ncols - 1) // ncols
    fig, axes = _plt.subplots(nrows, ncols, figsize=(5 * ncols, 3 * nrows), squeeze=False)
    axes_flat = axes.flatten()
    for i in range(n_show):
        ax = axes_flat[i]
        ax.plot(thinned[:, i], lw=0.8, color=_plt.cm.tab10(i % 10))
        ax.set_title(names[i] if i < len(names) else f"theta_{i}", fontsize=9)
        ax.set_xlabel("sample")
        ax.grid(alpha=0.2)
    for ax in axes_flat[n_show:]:
        ax.set_visible(False)
    fig.suptitle("Posterior trace plots", fontsize=12, fontweight="bold")
    _plt.tight_layout()
    fig.savefig(os.path.join(outdir, "posterior_trace.png"), dpi=200)
    _plt.close(fig)

    # Pairs plot (corner-style)
    if n_show >= 2:
        fig, axes = _plt.subplots(n_show, n_show, figsize=(3 * n_show, 3 * n_show))
        if n_show == 1:
            axes = [[axes]]
        for row in range(n_show):
            for col in range(n_show):
                ax = axes[row][col]
                if col > row:
                    ax.set_visible(False)
                    continue
                if row == col:
                    ax.hist(thinned[:, row], bins=30, color=_plt.cm.tab10(row % 10), alpha=0.7)
                    ax.set_ylabel(names[row] if row < len(names) else f"theta_{row}", fontsize=7)
                else:
                    ax.scatter(thinned[:, col], thinned[:, row], s=2, alpha=0.3,
                               color=_plt.cm.tab10(row % 10))
                if row == n_show - 1:
                    ax.set_xlabel(names[col] if col < len(names) else f"theta_{col}", fontsize=7)
        fig.suptitle("Posterior pairplot", fontsize=12, fontweight="bold")
        _plt.tight_layout()
        fig.savefig(os.path.join(outdir, "posterior_pairs.png"), dpi=200)
        _plt.close(fig)
