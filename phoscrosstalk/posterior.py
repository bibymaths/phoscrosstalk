# SPDX-License-Identifier: MIT
"""
Monte-Carlo posterior inference for PhosCrosstalk model parameters.

This module uses BlackJAX NUTS to sample the posterior distribution of
mechanistic model parameters after optimisation.

Runtime design
--------------
CPU/XLA threading is NOT configured here. It must be configured before JAX is
imported, usually from main.py via phoscrosstalk.runtime_env.

This module consumes the CPU plan from runtime_env.plan_posterior_runtime()
to decide how many independent chains to run in parallel.

Expected main.py usage
----------------------
    from phoscrosstalk.posterior import (
        run_posterior_inference,
        make_log_posterior_fn,
    )

    log_post_fn = make_log_posterior_fn(...)

    run_posterior_inference(
        outdir=outdir,
        theta_best=theta_best,
        log_posterior_fn=log_post_fn,
        posterior_cfg=cfg.posterior,
        theta_names=theta_names,
    )

Outputs
-------
Written to {outdir}/posterior/:

    posterior_samples.npz
    posterior_summary.tsv
    posterior_metadata.json
    posterior_theta_names.json
    posterior_trace.png          optional
    posterior_pairs.png          optional
"""

from __future__ import annotations

import json
import os
import time
import types
from typing import Any, Callable

import numpy as np

from phoscrosstalk.logger import get_logger
from phoscrosstalk.runtime_env import plan_posterior_runtime

logger = get_logger()


# ---------------------------------------------------------------------------
# BlackJAX import
# ---------------------------------------------------------------------------


def _require_blackjax():
    """Return blackjax or raise a clear ImportError."""
    try:
        import blackjax  # noqa: PLC0415

        return blackjax
    except ImportError as err:
        raise ImportError(
            "BlackJAX is required for posterior inference. "
            "Install it with: pip install blackjax>=1.0"
        ) from err


# ---------------------------------------------------------------------------
# Default posterior config
# ---------------------------------------------------------------------------


def _default_posterior_cfg() -> types.SimpleNamespace:
    """
    Default posterior inference configuration.

    Notes
    -----
    num_chains and threads_per_chain may be integers or "auto".
    The actual values are resolved through runtime_env.plan_posterior_runtime().
    """
    return types.SimpleNamespace(
        enabled=False,
        sampler="nuts",
        num_warmup=500,
        num_samples=1000,
        step_size=1e-3,
        thin_factor=10,
        target_acceptance_rate=0.8,
        seed=42,
        max_tree_depth=8,
        sigma_noise=0.1,
        plot_params=8,
        save_plots=True,
        # Runtime-aware parallelism.
        num_chains="auto",
        threads_per_chain="auto",
        reserve_cores=0,
        use_physical_cores=True,
        chain_jitter=1e-4,
        # Warmup strategy.
        # "shared": adapt once from theta_best, then run vectorized chains.
        # This is much faster and avoids compiling one warmup per chain.
        warmup_strategy="shared",
    )


def _merge_cfg(user_cfg) -> types.SimpleNamespace:
    """Merge user posterior config with defaults."""
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
    """
    Build a JAX-compatible log-posterior function.

    The posterior is:

        log p(theta | data)
            = log p(data | theta)
            + log p(theta)
            + log domain(theta)

    Likelihood:
        Gaussian fixed-noise likelihood over residuals_fn(theta).

    Prior:
        Gaussian prior centred at theta_prior_mean. If absent, the midpoint
        of bounds is used. If theta_prior_std is absent, range / 4 is used.

    Domain:
        Returns -inf outside [theta_lower, theta_upper].
    """
    import jax  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415

    xl = jnp.asarray(theta_lower, dtype=jnp.float64)
    xu = jnp.asarray(theta_upper, dtype=jnp.float64)

    if xl.ndim != 1 or xu.ndim != 1:
        raise ValueError(
            "theta_lower and theta_upper must be 1-D arrays. "
            f"Got theta_lower={xl.shape}, theta_upper={xu.shape}."
        )

    if xl.shape != xu.shape:
        raise ValueError(
            "theta_lower and theta_upper must have identical shape. "
            f"Got {xl.shape} and {xu.shape}."
        )

    if np.any(np.asarray(xu) <= np.asarray(xl)):
        raise ValueError(
            "All theta_upper values must be greater than theta_lower values."
        )

    if sigma_noise <= 0:
        raise ValueError(
            f"sigma_noise must be > 0. Got sigma_noise={sigma_noise!r}."
        )

    eps = jnp.asarray(1e-8, dtype=jnp.float64)
    sigma_n = jnp.asarray(sigma_noise, dtype=jnp.float64)

    mu = jnp.asarray(
        theta_prior_mean if theta_prior_mean is not None else 0.5 * (xl + xu),
        dtype=jnp.float64,
    )

    sigma_p_raw = jnp.asarray(
        theta_prior_std if theta_prior_std is not None else (xu - xl) / 4.0,
        dtype=jnp.float64,
    )

    if mu.shape != xl.shape:
        raise ValueError(
            "theta_prior_mean must have the same shape as bounds. "
            f"Got {mu.shape}, expected {xl.shape}."
        )

    if sigma_p_raw.shape != xl.shape:
        raise ValueError(
            "theta_prior_std must have the same shape as bounds. "
            f"Got {sigma_p_raw.shape}, expected {xl.shape}."
        )

    sigma_p = jnp.maximum(sigma_p_raw, eps)

    def log_posterior_fn(theta: jax.Array) -> jax.Array:
        theta = jnp.asarray(theta, dtype=jnp.float64)

        in_bounds = jnp.all((theta >= xl) & (theta <= xu))

        resid = jnp.asarray(residuals_fn(theta), dtype=jnp.float64)
        resid = jnp.ravel(resid)

        resid_finite = jnp.all(jnp.isfinite(resid))
        theta_finite = jnp.all(jnp.isfinite(theta))

        ssr = jnp.sum(resid ** 2)

        log_lik = -0.5 * ssr / (sigma_n ** 2)
        log_prior = -0.5 * jnp.sum(((theta - mu) / sigma_p) ** 2)

        total = log_lik + log_prior
        total_finite = jnp.isfinite(total)

        valid = in_bounds & theta_finite & resid_finite & total_finite

        return jnp.where(
            valid,
            total,
            jnp.asarray(-jnp.inf, dtype=jnp.float64),
        )

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
        theta_names: list[str] | None = None,
        theta_lower: np.ndarray | None = None,
        theta_upper: np.ndarray | None = None,
) -> dict[str, Any]:
    """
    Run runtime-aware vectorized multi-chain BlackJAX NUTS posterior inference.

    Parameters
    ----------
    outdir:
        Root output directory.

    theta_best:
        MAP / best-fit parameter vector. Used as the warmup initial position.

    log_posterior_fn:
        JAX-traceable theta -> scalar log-posterior function.

    posterior_cfg:
        Config object, typically cfg.posterior.

    theta_names:
        Optional parameter names.

    theta_lower:
        Optional lower bounds for parameters. If None, no bounds are applied.

    theta_upper:
        Optional upper bounds for parameters. If None, no bounds are applied.

    Returns
    -------
    dict with keys:
        samples
        acceptance_rate
        outdir
        summary
        metadata
    """
    blackjax = _require_blackjax()

    import jax  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415
    import pandas as pd  # noqa: PLC0415

    cfg = _merge_cfg(posterior_cfg)

    if str(getattr(cfg, "sampler", "nuts")).lower() != "nuts":
        raise ValueError(
            f"[posterior] Unsupported sampler={getattr(cfg, 'sampler', None)!r}. "
            "Only 'nuts' is currently implemented."
        )

    post_dir = os.path.join(outdir, "posterior")
    os.makedirs(post_dir, exist_ok=True)

    theta_best = np.asarray(theta_best, dtype=np.float64)

    if theta_best.ndim != 1:
        raise ValueError(
            f"theta_best must be a 1-D parameter vector. Got shape {theta_best.shape}."
        )

    dim = theta_best.shape[0]

    if theta_lower is not None and theta_upper is not None:
        theta_lower = np.asarray(theta_lower, dtype=np.float64)
        theta_upper = np.asarray(theta_upper, dtype=np.float64)

        if theta_lower.shape != theta_best.shape or theta_upper.shape != theta_best.shape:
            raise ValueError(
                "[posterior] theta_lower/theta_upper must match theta_best shape. "
                f"theta_best={theta_best.shape}, "
                f"theta_lower={theta_lower.shape}, "
                f"theta_upper={theta_upper.shape}."
            )

        if not np.all(np.isfinite(theta_lower)) or not np.all(np.isfinite(theta_upper)):
            raise ValueError("[posterior] theta_lower/theta_upper contain non-finite values.")

        if not np.all(theta_upper > theta_lower):
            raise ValueError("[posterior] All theta_upper values must be > theta_lower.")

        xl_jax = jnp.asarray(theta_lower, dtype=jnp.float64)
        xu_jax = jnp.asarray(theta_upper, dtype=jnp.float64)
    else:
        xl_jax = None
        xu_jax = None

    if theta_names is None:
        theta_names = [f"theta_{i}" for i in range(dim)]

    if len(theta_names) != dim:
        raise ValueError(
            f"theta_names length must match theta dimension. "
            f"Got len(theta_names)={len(theta_names)}, dim={dim}."
        )

    if int(cfg.num_warmup) <= 0:
        raise ValueError(f"[posterior] num_warmup must be > 0. Got {cfg.num_warmup}.")

    if int(cfg.num_samples) <= 0:
        raise ValueError(f"[posterior] num_samples must be > 0. Got {cfg.num_samples}.")

    if int(cfg.thin_factor) <= 0:
        raise ValueError(f"[posterior] thin_factor must be > 0. Got {cfg.thin_factor}.")

    if int(cfg.max_tree_depth) <= 0:
        raise ValueError(
            f"[posterior] max_tree_depth must be > 0. Got {cfg.max_tree_depth}."
        )

    if not (0.0 < float(cfg.target_acceptance_rate) < 1.0):
        raise ValueError(
            "[posterior] target_acceptance_rate must be in (0, 1). "
            f"Got {cfg.target_acceptance_rate}."
        )

    posterior_plan = plan_posterior_runtime(
        cpu_threads=getattr(cfg, "cpu_threads", "auto"),
        num_chains=getattr(cfg, "num_chains", "auto"),
        threads_per_chain=getattr(cfg, "threads_per_chain", "auto"),
        reserve_cores=getattr(cfg, "reserve_cores", 0),
        use_physical_cores=getattr(cfg, "use_physical_cores", True),
    )

    num_chains = int(posterior_plan.n_chains)

    if num_chains <= 0:
        raise ValueError(
            f"[posterior/runtime] Planned num_chains must be > 0. Got {num_chains}."
        )

    logger.info(
        "[posterior/runtime] CPU plan: total_cpus=%d, chains=%d, "
        "threads_per_chain=%d, source=%s",
        int(posterior_plan.total_available_cpus),
        int(posterior_plan.n_chains),
        int(posterior_plan.threads_per_chain),
        str(posterior_plan.topo.source),
    )

    key = jax.random.PRNGKey(int(cfg.seed))
    init_position = jnp.asarray(theta_best, dtype=jnp.float64)

    log_posterior_fn = jax.jit(log_posterior_fn)

    logger.info("[posterior] Compiling log-posterior...")
    t_compile = time.time()
    compile_val = log_posterior_fn(init_position)
    compile_val.block_until_ready()
    compile_elapsed = time.time() - t_compile

    if not bool(np.isfinite(float(compile_val))):
        raise RuntimeError(
            "[posterior] Initial log-posterior at theta_best is not finite. "
            f"log_posterior(theta_best)={float(compile_val)}. "
            "Check bounds, residuals, ODE simulation, or sigma_noise."
        )

    logger.info(
        "[posterior] Log-posterior compiled in %.2f s. Initial value=%.4e",
        compile_elapsed,
        float(compile_val),
    )

    # ------------------------------------------------------------------ #
    # 1. Shared NUTS warmup                                               #
    # ------------------------------------------------------------------ #
    # We adapt once from theta_best, then vectorize independent chains using
    # the adapted step size and mass matrix. This is much faster on CPU than
    # adapting every chain independently, and it keeps the implementation robust.
    # ------------------------------------------------------------------ #
    logger.info(
        "[posterior] Starting shared NUTS warmup: steps=%d, seed=%d, "
        "target_acceptance=%.3f",
        int(cfg.num_warmup),
        int(cfg.seed),
        float(cfg.target_acceptance_rate),
    )

    warmup = blackjax.window_adaptation(
        blackjax.nuts,
        log_posterior_fn,
        target_acceptance_rate=float(cfg.target_acceptance_rate),
        max_num_doublings=int(cfg.max_tree_depth),
    )

    t_warmup_start = time.time()
    key, warmup_key = jax.random.split(key)

    (warmup_state, warmup_params), _warmup_info = warmup.run(
        warmup_key,
        init_position,
        num_steps=int(cfg.num_warmup),
    )

    warmup_state.position.block_until_ready()
    warmup_elapsed = time.time() - t_warmup_start

    adapted_step_size = jnp.asarray(
        warmup_params.get("step_size", cfg.step_size),
        dtype=jnp.float64,
    )

    inverse_mass_matrix = jnp.asarray(
        warmup_params.get(
            "inverse_mass_matrix",
            jnp.ones(dim, dtype=jnp.float64),
        ),
        dtype=jnp.float64,
    )

    logger.info(
        "[posterior] Warmup complete in %.2f s. Adapted step size=%.4g.",
        warmup_elapsed,
        float(adapted_step_size),
    )

    # ------------------------------------------------------------------ #
    # 2. Vectorized chain initialization                                  #
    # ------------------------------------------------------------------ #
    chain_jitter = float(getattr(cfg, "chain_jitter", 1e-4))

    if chain_jitter < 0:
        raise ValueError(
            f"[posterior] chain_jitter must be >= 0. Got {chain_jitter}."
        )

    key, init_key = jax.random.split(key)

    if chain_jitter > 0:
        init_positions = warmup_state.position[None, :] + chain_jitter * jax.random.normal(
            init_key,
            shape=(num_chains, dim),
            dtype=jnp.float64,
        )
    else:
        init_positions = jnp.repeat(
            warmup_state.position[None, :],
            repeats=num_chains,
            axis=0,
        )

    # Keep chain starts inside the posterior support.
    # This prevents chain jitter from pushing parameters outside hard bounds.
    if xl_jax is not None and xu_jax is not None:
        span = xu_jax - xl_jax
        eps = jnp.maximum(1e-10 * span, jnp.asarray(1e-12, dtype=jnp.float64))
        init_positions = jnp.clip(init_positions, xl_jax + eps, xu_jax - eps)

    nuts_kernel = blackjax.nuts(
        log_posterior_fn,
        adapted_step_size,
        inverse_mass_matrix,
        max_num_doublings=int(cfg.max_tree_depth),
    )

    init_many = jax.jit(jax.vmap(nuts_kernel.init))
    states0 = init_many(init_positions)

    # ------------------------------------------------------------------ #
    # 3. Vectorized sampling                                              #
    # ------------------------------------------------------------------ #
    logger.info(
        "[posterior] Sampling: samples=%d × chains=%d, thin_factor=%d, "
        "max_tree_depth=%d",
        int(cfg.num_samples),
        int(num_chains),
        int(cfg.thin_factor),
        int(cfg.max_tree_depth),
    )

    key, sample_key = jax.random.split(key)

    keys = jax.random.split(
        sample_key,
        int(cfg.num_samples) * num_chains,
    ).reshape(int(cfg.num_samples), num_chains, 2)

    def _one_chain_step(rng_key, state):
        new_state, info = nuts_kernel.step(rng_key, state)
        return new_state, info

    one_step_many = jax.vmap(_one_chain_step, in_axes=(0, 0))

    def _scan_step(states, rng_keys):
        new_states, infos = one_step_many(rng_keys, states)
        return new_states, (new_states.position, infos.acceptance_rate)

    @jax.jit
    def _sample_many_chains(states_init, rng_keys):
        return jax.lax.scan(_scan_step, states_init, rng_keys)

    t_sample_start = time.time()
    _, (positions, acceptance_rates) = _sample_many_chains(states0, keys)

    positions.block_until_ready()
    sample_elapsed = time.time() - t_sample_start

    logger.info("[posterior] Sampling complete in %.2f s.", sample_elapsed)

    # positions shape: (num_samples, num_chains, dim)
    all_positions = np.asarray(positions, dtype=np.float64).reshape(-1, dim)
    acceptance_rate = float(jnp.mean(acceptance_rates))

    if not np.all(np.isfinite(all_positions)):
        raise RuntimeError(
            "[posterior] Non-finite parameter samples detected. "
            "Check posterior geometry, ODE stability, and NUTS settings."
        )

    logger.info(
        "[posterior] Mean acceptance rate across chains: %.3f",
        acceptance_rate,
    )

    # ------------------------------------------------------------------ #
    # 4. Thin samples                                                     #
    # ------------------------------------------------------------------ #
    thin = max(1, int(cfg.thin_factor))
    thinned = all_positions[::thin]
    n_thinned = len(thinned)

    if n_thinned == 0:
        raise RuntimeError(
            "[posterior] No samples remain after thinning. "
            f"num_samples={cfg.num_samples}, num_chains={num_chains}, thin_factor={thin}."
        )

    logger.info(
        "[posterior] Raw samples=%d, thinned samples=%d.",
        len(all_positions),
        n_thinned,
    )

    # ------------------------------------------------------------------ #
    # 5. Summary statistics                                               #
    # ------------------------------------------------------------------ #
    summary_rows = []

    for i, name in enumerate(theta_names):
        col = thinned[:, i]
        finite = col[np.isfinite(col)]

        if len(finite) == 0:
            raise RuntimeError(
                f"[posterior] Parameter {name!r} has no finite posterior samples."
            )

        summary_rows.append(
            {
                "parameter": name,
                "mean": float(np.mean(finite)),
                "std": float(np.std(finite)),
                "q2.5": float(np.percentile(finite, 2.5)),
                "q97.5": float(np.percentile(finite, 97.5)),
                "ess": int(_naive_ess(finite)),
            }
        )

    df_summary = pd.DataFrame(summary_rows)

    summary_path = os.path.join(post_dir, "posterior_summary.tsv")
    df_summary.to_csv(summary_path, sep="\t", index=False)
    logger.info("[posterior] Saved summary to %s", summary_path)

    # ------------------------------------------------------------------ #
    # 6. Save samples                                                     #
    # ------------------------------------------------------------------ #
    npz_path = os.path.join(post_dir, "posterior_samples.npz")

    np.savez(
        npz_path,
        samples=thinned,
        raw_samples_shape=np.array(positions.shape, dtype=np.int64),
        theta_names=np.array(theta_names, dtype=str),
        theta_best=np.asarray(theta_best, dtype=np.float64),
        acceptance_rate=np.array(acceptance_rate, dtype=np.float64),
        num_chains=np.array(num_chains, dtype=np.int64),
    )

    names_path = os.path.join(post_dir, "posterior_theta_names.json")
    with open(names_path, "w", encoding="utf-8") as fh:
        json.dump(theta_names, fh, indent=2)

    logger.info("[posterior] Saved samples to %s", npz_path)

    # ------------------------------------------------------------------ #
    # 7. Metadata                                                         #
    # ------------------------------------------------------------------ #
    meta = {
        "sampler": "NUTS (BlackJAX, shared warmup, vectorized chains)",
        "dim": int(dim),
        "seed": int(cfg.seed),
        "num_warmup": int(cfg.num_warmup),
        "num_samples_per_chain": int(cfg.num_samples),
        "num_chains": int(num_chains),
        "raw_total_samples": int(len(all_positions)),
        "thin_factor": int(thin),
        "n_thinned_samples": int(n_thinned),
        "initial_step_size": float(cfg.step_size),
        "adapted_step_size": float(adapted_step_size),
        "target_acceptance_rate": float(cfg.target_acceptance_rate),
        "actual_acceptance_rate": float(acceptance_rate),
        "max_tree_depth": int(cfg.max_tree_depth),
        "chain_jitter": float(chain_jitter),
        "compile_time_s": float(compile_elapsed),
        "warmup_time_s": float(warmup_elapsed),
        "sample_time_s": float(sample_elapsed),
        "runtime_total_available_cpus": int(posterior_plan.total_available_cpus),
        "runtime_num_chains": int(posterior_plan.n_chains),
        "runtime_threads_per_chain": int(posterior_plan.threads_per_chain),
        "runtime_cpu_source": str(posterior_plan.topo.source),
        "runtime_logical_cpus": int(posterior_plan.topo.logical_cpus),
        "runtime_physical_cores": (
            None
            if posterior_plan.topo.physical_cores is None
            else int(posterior_plan.topo.physical_cores)
        ),
        "runtime_slurm_cpus": (
            None
            if posterior_plan.topo.slurm_cpus is None
            else int(posterior_plan.topo.slurm_cpus)
        ),
        "runtime_affinity_cpus": (
            None
            if posterior_plan.topo.affinity_cpus is None
            else int(posterior_plan.topo.affinity_cpus)
        ),
    }

    meta_path = os.path.join(post_dir, "posterior_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)

    logger.info("[posterior] Saved metadata to %s", meta_path)

    # ------------------------------------------------------------------ #
    # 8. Optional plots                                                   #
    # ------------------------------------------------------------------ #
    if bool(getattr(cfg, "save_plots", True)):
        _plot_posterior(post_dir, thinned, theta_names, int(cfg.plot_params))

    return {
        "samples": thinned,
        "acceptance_rate": acceptance_rate,
        "outdir": post_dir,
        "summary": df_summary,
        "metadata": meta,
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
) -> dict[str, Any]:
    """
    Draw posterior-predictive trajectories.

    This function intentionally fails fast if simulate_fn fails for any sample.
    Silent skipping makes posterior predictive intervals biased and difficult
    to debug.
    """
    samples = np.asarray(samples, dtype=np.float64)

    if samples.ndim != 2:
        raise ValueError(
            f"samples must be 2-D with shape (n_samples, dim). Got {samples.shape}."
        )

    thin = max(1, int(thin_factor))
    use_samples = samples[::thin]

    if len(use_samples) == 0:
        raise ValueError(
            f"No posterior samples selected. samples={samples.shape}, thin_factor={thin}."
        )

    all_sims: list[dict[str, np.ndarray]] = []

    for idx, theta in enumerate(use_samples):
        result = simulate_fn(np.asarray(theta, dtype=np.float64))

        if not isinstance(result, dict):
            raise TypeError(
                f"simulate_fn must return a dict. Sample index {idx} returned "
                f"{type(result).__name__}."
            )

        all_sims.append(
            {key: np.asarray(value, dtype=float) for key, value in result.items()}
        )

    if not all_sims:
        raise RuntimeError("[posterior] posterior_predict produced no simulations.")

    output: dict[str, Any] = {}

    keys = list(all_sims[0].keys())

    for key in keys:
        shapes = [sim[key].shape for sim in all_sims]

        if len(set(shapes)) != 1:
            raise ValueError(
                f"[posterior] Inconsistent simulation shapes for key {key!r}: {shapes}"
            )

        stack = np.stack([sim[key] for sim in all_sims], axis=0)

        lo, hi = credible_intervals

        output[key] = {
            "mean": np.mean(stack, axis=0),
            "lower": np.percentile(stack, lo, axis=0),
            "upper": np.percentile(stack, hi, axis=0),
            "samples": stack,
        }

    output["t_eval"] = np.asarray(t_eval)

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
    """
    Plot posterior-predictive mean and credible interval.
    """
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")

    from matplotlib import pyplot as plt  # noqa: PLC0415

    os.makedirs(outdir, exist_ok=True)

    if key not in pred:
        raise KeyError(
            f"[posterior] Key {key!r} not found in posterior prediction output. "
            f"Available keys: {list(pred.keys())}"
        )

    mean_arr = np.asarray(pred[key]["mean"], dtype=float)
    lower_arr = np.asarray(pred[key]["lower"], dtype=float)
    upper_arr = np.asarray(pred[key]["upper"], dtype=float)
    t_arr = np.asarray(t_eval, dtype=float)

    if mean_arr.ndim != 2:
        raise ValueError(
            f"[posterior] Expected pred[{key!r}]['mean'] to be 2-D. "
            f"Got shape {mean_arr.shape}."
        )

    if mean_arr.shape != lower_arr.shape or mean_arr.shape != upper_arr.shape:
        raise ValueError(
            "[posterior] Posterior predictive mean/lower/upper shapes differ: "
            f"mean={mean_arr.shape}, lower={lower_arr.shape}, upper={upper_arr.shape}."
        )

    if mean_arr.shape[1] != len(t_arr):
        raise ValueError(
            f"[posterior] Time-axis mismatch for key {key!r}: "
            f"mean_arr.shape={mean_arr.shape}, len(t_eval)={len(t_arr)}."
        )

    n_entities = min(len(entity_names), mean_arr.shape[0])

    if n_entities == 0:
        raise ValueError("[posterior] No entities available for posterior plot.")

    ranges = np.ptp(mean_arr[:n_entities], axis=1)
    top_idx = np.argsort(ranges)[::-1][:max_entities]

    n_plot = len(top_idx)
    ncols = min(4, n_plot)
    nrows = (n_plot + ncols - 1) // ncols

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(5 * ncols, 4 * nrows),
        squeeze=False,
    )

    axes_flat = axes.flatten()

    for ai, ei in enumerate(top_idx):
        ax = axes_flat[ai]
        color = plt.cm.tab10(ai % 10)
        name = entity_names[ei] if ei < len(entity_names) else f"entity_{ei}"

        ax.plot(
            t_arr,
            mean_arr[ei],
            "-",
            lw=2,
            color=color,
            label="Posterior mean",
        )

        ax.fill_between(
            t_arr,
            lower_arr[ei],
            upper_arr[ei],
            alpha=0.25,
            color=color,
            label="95% CI",
        )

        if observed is not None and ei < observed.shape[0]:
            obs_arr = np.asarray(observed, dtype=float)
            t_obs_arr = np.asarray(t_observed, dtype=float) if t_observed is not None else t_arr

            if obs_arr.shape[1] != len(t_obs_arr):
                raise ValueError(
                    f"[posterior] Observed/time mismatch for posterior predictive plot: "
                    f"observed.shape={obs_arr.shape}, len(t_observed)={len(t_obs_arr)}."
                )

            y_obs = obs_arr[ei]
            mask = np.isfinite(y_obs)

            if np.any(mask):
                ax.scatter(
                    t_obs_arr[mask],
                    y_obs[mask],
                    s=40,
                    color=color,
                    zorder=5,
                    label="Observed",
                )

        ax.set_title(name, fontsize=10, fontweight="bold")
        ax.set_xlabel("Time")
        ax.set_ylabel(key)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.25)

    for ax in axes_flat[n_plot:]:
        ax.set_visible(False)

    fig.suptitle(f"Posterior Predictive — {key}", fontsize=13, fontweight="bold")
    plt.tight_layout()

    path = os.path.join(outdir, f"posterior_predictive_{key}.png")
    fig.savefig(path, dpi=300)
    plt.close(fig)

    logger.info("[posterior] Saved %s", path)


# ---------------------------------------------------------------------------
# Internal utilities
# ---------------------------------------------------------------------------


def _naive_ess(chain: np.ndarray) -> int:
    """
    Compute a simple lag-1 autocorrelation ESS estimate.

    This is intentionally lightweight. For serious diagnostics, use ArviZ.
    """
    chain = np.asarray(chain, dtype=float)
    n = len(chain)

    if n < 4:
        return n

    mu = np.mean(chain)
    var = np.var(chain)

    if var < 1e-15:
        return n

    ac1 = float(
        np.sum((chain[:-1] - mu) * (chain[1:] - mu)) / ((n - 1) * var)
    )

    rho = max(0.0, min(ac1, 0.9999))
    ess = int(n * (1.0 - rho) / (1.0 + rho))

    return max(1, ess)


def _plot_posterior(
        outdir: str,
        thinned: np.ndarray,
        names: list[str],
        n_plot: int,
) -> None:
    """
    Save trace and pair plots for leading posterior parameters.
    """
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")

    from matplotlib import pyplot as plt  # noqa: PLC0415

    thinned = np.asarray(thinned, dtype=float)

    if thinned.ndim != 2:
        raise ValueError(
            f"[posterior] thinned samples must be 2-D. Got {thinned.shape}."
        )

    n_show = min(int(n_plot), thinned.shape[1])

    if n_show <= 0:
        raise ValueError(f"[posterior] plot_params must be > 0. Got {n_plot}.")

    # Trace plot
    ncols = min(4, n_show)
    nrows = (n_show + ncols - 1) // ncols

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(5 * ncols, 3 * nrows),
        squeeze=False,
    )

    axes_flat = axes.flatten()

    for i in range(n_show):
        ax = axes_flat[i]
        ax.plot(thinned[:, i], lw=0.8, color=plt.cm.tab10(i % 10))
        ax.set_title(names[i] if i < len(names) else f"theta_{i}", fontsize=9)
        ax.set_xlabel("sample")
        ax.grid(alpha=0.2)

    for ax in axes_flat[n_show:]:
        ax.set_visible(False)

    fig.suptitle("Posterior trace plots", fontsize=12, fontweight="bold")
    plt.tight_layout()

    trace_path = os.path.join(outdir, "posterior_trace.png")
    fig.savefig(trace_path, dpi=200)
    plt.close(fig)

    logger.info("[posterior] Saved %s", trace_path)

    # Pair plot
    if n_show >= 2:
        fig, axes = plt.subplots(
            n_show,
            n_show,
            figsize=(3 * n_show, 3 * n_show),
            squeeze=False,
        )

        for row in range(n_show):
            for col in range(n_show):
                ax = axes[row][col]

                if col > row:
                    ax.set_visible(False)
                    continue

                if row == col:
                    ax.hist(
                        thinned[:, row],
                        bins=30,
                        color=plt.cm.tab10(row % 10),
                        alpha=0.7,
                    )
                    ax.set_ylabel(
                        names[row] if row < len(names) else f"theta_{row}",
                        fontsize=7,
                    )
                else:
                    ax.scatter(
                        thinned[:, col],
                        thinned[:, row],
                        s=2,
                        alpha=0.3,
                        color=plt.cm.tab10(row % 10),
                    )

                if row == n_show - 1:
                    ax.set_xlabel(
                        names[col] if col < len(names) else f"theta_{col}",
                        fontsize=7,
                    )

        fig.suptitle("Posterior pairplot", fontsize=12, fontweight="bold")
        plt.tight_layout()

        pair_path = os.path.join(outdir, "posterior_pairs.png")
        fig.savefig(pair_path, dpi=200)
        plt.close(fig)

        logger.info("[posterior] Saved %s", pair_path)