"""
hybrid_fit.py
Mutax-based hybrid optimization for the phospho-network model.

Pipeline
--------
Phase 0 – optional Latin Hypercube Sampling screen:
    Quickly explore the bounded parameter space and identify useful warm-start
    candidates.

Phase 1 – Mutax Differential Evolution global search:
    Global derivative-free search over the original bounded theta space.

Phase 2 – Optimistix local polish:
    Local least-squares refinement using the existing
    phoscrosstalk.optimization.run_single_optimisation path.

This module intentionally does not use evosax or qdax.

Design constraints
------------------
- Do not change model equations.
- Do not change objective semantics.
- Do not change f1/f2/f3/f4 definitions.
- Do not change parameter bounds.
- Keep downstream compatibility with main.py analysis:
    result.theta_opt
    result.total_loss
    result.f1, result.f2, result.f3, result.f4
    tuple unpacking as (theta_opt, total_loss, f1, f2, f3, f4)

Notes
-----
Mutax differential_evolution returns one final OptimizeResults object. It does
not expose a full final population in the documented API. Therefore local-polish
seeds are built from:
    1. Mutax global best solution.
    2. Optional LHS top seeds.
    3. Deterministic random starts from multistarts._generate_starts.
    4. Small bounded jitters around the Mutax best solution.

The actual local fitting remains the existing Optimistix implementation.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from phoscrosstalk.logger import get_logger
from phoscrosstalk.multistarts import _generate_starts
from phoscrosstalk.optimization import run_single_optimisation

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class HybridFitResult:
    """
    Return type for run_hybrid_fit.

    Supports both attribute access:

        result.theta_opt

    and tuple unpacking:

        theta_opt, total_loss, f1, f2, f3, f4 = result

    Attributes
    ----------
    theta_opt : np.ndarray
        Best polished parameter vector, shape (n_var,), float64.
    total_loss : float
        Total scalarized loss at theta_opt.
    f1 : float
        Phosphosite relative-signal loss.
    f2 : float
        Protein abundance loss.
    f3 : float
        Regularisation loss.
    f4 : float
        mRNA / R_rna loss. Zero when no RNA data are present.
    global_theta : np.ndarray
        Best parameter vector returned by Mutax before local polish.
    global_loss : float
        Scalarized loss of global_theta.
    polish_X : np.ndarray
        All successful polished candidates, shape (n_success, n_var).
    polish_F : np.ndarray
        Loss components for polished candidates, shape (n_success, 4).
    polish_J : np.ndarray
        Total losses for polished candidates, shape (n_success,).
    mutax_result : Any
        Raw Mutax OptimizeResults object.
    """

    theta_opt: np.ndarray
    total_loss: float
    f1: float
    f2: float
    f3: float
    f4: float
    global_theta: np.ndarray
    global_loss: float
    polish_X: np.ndarray
    polish_F: np.ndarray
    polish_J: np.ndarray
    mutax_result: Any | None = None

    def __iter__(self):
        yield self.theta_opt
        yield self.total_loss
        yield self.f1
        yield self.f2
        yield self.f3
        yield self.f4


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _as_float64_vector(x, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        raise ValueError(f"{name} must be a non-empty 1D vector.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values.")
    return arr


def _validate_bounds(xl: np.ndarray, xu: np.ndarray, n_var: int) -> tuple[np.ndarray, np.ndarray]:
    xl = _as_float64_vector(xl, "xl")
    xu = _as_float64_vector(xu, "xu")

    if xl.shape != xu.shape:
        raise ValueError(f"Bounds shape mismatch: xl={xl.shape}, xu={xu.shape}")

    if xl.size != int(n_var):
        raise ValueError(
            f"n_var={n_var} does not match bounds length {xl.size}."
        )

    if not np.all(xu > xl):
        bad = np.where(~(xu > xl))[0]
        raise ValueError(
            f"Invalid bounds: xu must be greater than xl for every parameter. "
            f"Bad indices: {bad[:20].tolist()}"
        )

    return xl, xu


def _loss_scalar_from_loss_fn(loss_fn, penalty: float = 1e12):
    """
    Build a JAX-compatible scalar objective for global search.

    loss_fn must have signature:

        loss_fn(theta, args) -> (total_loss, aux)

    as returned by optimization.make_loss_fn.
    """

    penalty_value = float(penalty)

    def objective(theta):
        theta = jnp.asarray(theta, dtype=jnp.float64)
        total, _aux = loss_fn(theta, None)
        total = jnp.asarray(total, dtype=jnp.float64)
        return jnp.where(
            jnp.isfinite(total),
            total,
            jnp.asarray(penalty_value, dtype=jnp.float64),
        )

    return objective


def evaluate_candidates(loss_fn, candidates: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Evaluate candidate theta vectors with the scalarized JAX loss.

    Returns
    -------
    candidates_sorted : np.ndarray
        Candidates sorted by ascending scalarized loss.
    losses_sorted : np.ndarray
        Corresponding losses.
    """
    candidates = np.asarray(candidates, dtype=np.float64)

    if candidates.ndim != 2:
        raise ValueError(
            f"candidates must be 2D with shape (n_candidates, n_var), got {candidates.shape}"
        )

    @jax.jit
    def _eval_batch(theta_batch):
        def _one(theta):
            total, _aux = loss_fn(theta, None)
            total = jnp.asarray(total, dtype=jnp.float64)
            return jnp.where(
                jnp.isfinite(total),
                total,
                jnp.asarray(1e12, dtype=jnp.float64),
            )

        return jax.vmap(_one)(theta_batch)

    losses = np.asarray(
        _eval_batch(jnp.asarray(candidates, dtype=jnp.float64)),
        dtype=np.float64,
    )

    order = np.argsort(losses)
    return candidates[order], losses[order]


def _deduplicate_candidates(
    candidates: np.ndarray,
    *,
    rtol: float = 1e-9,
    atol: float = 1e-12,
) -> np.ndarray:
    """
    Remove near-duplicate theta rows while preserving order.
    """
    candidates = np.asarray(candidates, dtype=np.float64)

    if candidates.size == 0:
        return candidates.reshape(0, 0)

    unique_rows: list[np.ndarray] = []
    for row in candidates:
        if not np.all(np.isfinite(row)):
            continue

        duplicate = False
        for old in unique_rows:
            if np.all(np.isclose(row, old, rtol=rtol, atol=atol)):
                duplicate = True
                break

        if not duplicate:
            unique_rows.append(row)

    if not unique_rows:
        return np.empty((0, candidates.shape[1]), dtype=np.float64)

    return np.vstack(unique_rows).astype(np.float64)


# ---------------------------------------------------------------------------
# Phase 0 – Latin Hypercube Sampling screen
# ---------------------------------------------------------------------------


def lhs_screen(
    loss_fn,
    xl: np.ndarray,
    xu: np.ndarray,
    n_samples: int = 512,
    top_p: int = 16,
    seed: int = 0,
    batch_size: int = 64,
) -> np.ndarray:
    """
    Optional Phase 0: Latin Hypercube Sampling over the bounded theta space.

    This is pure NumPy sampling plus batched JAX loss evaluation. It does not
    depend on scipy.

    Returns
    -------
    np.ndarray
        Top candidates sorted by scalarized loss, shape (top_p, n_var).
    """
    xl = np.asarray(xl, dtype=np.float64)
    xu = np.asarray(xu, dtype=np.float64)
    n_var = xl.size

    n_samples = max(1, int(n_samples))
    top_p = max(1, min(int(top_p), n_samples))
    batch_size = max(1, int(batch_size))

    rng = np.random.default_rng(seed)

    strata_width = 1.0 / n_samples
    lower_edges = np.arange(n_samples, dtype=np.float64) * strata_width

    unit_samples = np.empty((n_samples, n_var), dtype=np.float64)
    for d in range(n_var):
        perm = rng.permutation(n_samples)
        unit_samples[:, d] = lower_edges[perm] + rng.uniform(
            0.0,
            strata_width,
            size=n_samples,
        )

    samples = xl + unit_samples * (xu - xl)

    @jax.jit
    def _eval_batch(theta_batch):
        return jax.vmap(lambda th: loss_fn(th, None)[0])(theta_batch)

    pieces = []
    samples_j = jnp.asarray(samples, dtype=jnp.float64)
    for i in range(0, n_samples, batch_size):
        vals = _eval_batch(samples_j[i : i + batch_size])
        vals = jnp.where(
            jnp.isfinite(vals),
            vals,
            jnp.asarray(1e12, dtype=jnp.float64),
        )
        pieces.append(vals)

    losses = np.asarray(jnp.concatenate(pieces), dtype=np.float64)
    order = np.argsort(losses)[:top_p]

    return samples[order].astype(np.float64)


# ---------------------------------------------------------------------------
# Phase 1 – Mutax Differential Evolution
# ---------------------------------------------------------------------------


def run_mutax_de(
    loss_fn,
    xl: np.ndarray,
    xu: np.ndarray,
    *,
    strategy: str = "best1bin",
    maxiter: int = 200,
    popsize: int = 15,
    tol: float = 0.01,
    atol: float = 0.0,
    mutation: float | tuple[float, float] = (0.5, 1.0),
    recombination: float = 0.8,
    seed: int = 0,
    x0: np.ndarray | None = None,
    workers: int = 1,
    updating: str = "immediate",
    disp: bool = False,
    polish: bool = False,
):
    """
    Run Mutax Differential Evolution.

    Mutax performs global derivative-free search over the original theta bounds.
    Local polish is intentionally handled later by Optimistix, not by Mutax's
    optional BFGS polish.
    """
    try:
        from mutax import differential_evolution
    except ImportError as exc:
        raise ImportError(
            "mutax is required for the hybrid solver. Install it and remove "
            "evosax/qdax from the project dependencies."
        ) from exc

    if strategy not in {"best1bin", "rand1bin"}:
        logger.warning(
            "[fit]  hybrid_fit  unknown Mutax strategy %r; using 'best1bin'.",
            strategy,
        )
        strategy = "best1bin"

    xl_j = jnp.asarray(xl, dtype=jnp.float64)
    xu_j = jnp.asarray(xu, dtype=jnp.float64)
    bounds_j = jnp.stack([xl_j, xu_j], axis=1)

    objective = _loss_scalar_from_loss_fn(loss_fn)

    key = jax.random.PRNGKey(int(seed))

    x0_j = None
    if x0 is not None:
        x0_np = np.clip(np.asarray(x0, dtype=np.float64).reshape(-1), xl, xu)
        x0_j = jnp.asarray(x0_np, dtype=jnp.float64)

    logger.info(
        "[fit]  hybrid_fit  mutax_de  strategy=%s  maxiter=%d  popsize=%d  "
        "workers=%s  updating=%s",
        strategy,
        int(maxiter),
        int(popsize),
        str(workers),
        updating,
    )

    t0 = time.perf_counter()
    result = differential_evolution(
        objective,
        bounds_j,
        key=key,
        strategy=strategy,
        maxiter=int(maxiter),
        popsize=int(popsize),
        tol=float(tol),
        atol=float(atol),
        mutation=mutation,
        recombination=float(recombination),
        disp=bool(disp),
        polish=bool(polish),
        updating=updating,
        workers=workers,
        x0=x0_j,
        vectorized=False,
    )
    elapsed = time.perf_counter() - t0

    best_theta = np.asarray(result.x, dtype=np.float64)
    best_theta = np.clip(best_theta, xl, xu)

    try:
        best_loss = float(np.asarray(result.fun))
    except Exception:
        best_loss = float("nan")

    logger.info(
        "[fit]  hybrid_fit  mutax_de  done  fun=%.6e  nit=%s  nfev=%s  t=%.2fs",
        best_loss,
        getattr(result, "nit", "NA"),
        getattr(result, "nfev", "NA"),
        elapsed,
    )

    return best_theta, best_loss, result


def build_polish_candidates(
    *,
    de_best: np.ndarray,
    xl: np.ndarray,
    xu: np.ndarray,
    warm_seeds: np.ndarray | None,
    loss_fn,
    n_candidates: int,
    seed: int,
    random_starts: int = 8,
    jitter_count: int = 8,
    jitter_scale: float = 0.03,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build and rank local-polish candidates.

    Candidate sources:
    - Mutax DE best.
    - LHS warm seeds.
    - deterministic random starts via multistarts._generate_starts.
    - bounded Gaussian jitters around the DE best.

    Returns
    -------
    candidates : np.ndarray
        Top candidates sorted by scalarized loss.
    losses : np.ndarray
        Corresponding scalarized losses.
    """
    xl = np.asarray(xl, dtype=np.float64)
    xu = np.asarray(xu, dtype=np.float64)
    de_best = np.clip(np.asarray(de_best, dtype=np.float64).reshape(1, -1), xl, xu)

    blocks = [de_best]

    if warm_seeds is not None and np.asarray(warm_seeds).size > 0:
        blocks.append(np.asarray(warm_seeds, dtype=np.float64))

    if random_starts > 0:
        rand = np.asarray(_generate_starts(int(random_starts), xl, xu), dtype=np.float64)
        blocks.append(rand)

    if jitter_count > 0 and jitter_scale > 0:
        rng = np.random.default_rng(seed + 7919)
        span = xu - xl
        jitters = []
        for _ in range(int(jitter_count)):
            noise = rng.normal(0.0, float(jitter_scale), size=de_best.shape[1]) * span
            jitters.append(np.clip(de_best[0] + noise, xl, xu))
        blocks.append(np.asarray(jitters, dtype=np.float64))

    all_candidates = np.vstack(blocks)
    all_candidates = np.clip(all_candidates, xl, xu)
    all_candidates = _deduplicate_candidates(all_candidates)

    if all_candidates.size == 0:
        raise RuntimeError("No valid candidates available for local polish.")

    ranked, losses = evaluate_candidates(loss_fn, all_candidates)

    n_keep = max(1, min(int(n_candidates), len(ranked)))
    return ranked[:n_keep], losses[:n_keep]


# ---------------------------------------------------------------------------
# Phase 2 – Optimistix local polish
# ---------------------------------------------------------------------------


def run_lm_polish(
    residuals_fn,
    candidates: np.ndarray,
    xl: np.ndarray,
    xu: np.ndarray,
    *,
    lm_max_steps: int = 500,
    lm_rtol: float = 1e-8,
    lm_atol: float = 1e-8,
    optx_adjoint: str = "implicit",
    ls_solver: str = "lm",
    jac_mode: str = "fwd",
    verbose: bool = False,
) -> tuple[np.ndarray, float, float, float, float, float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Polish candidate seeds using the existing Optimistix path.

    Returns
    -------
    theta_best, total_best, f1, f2, f3, f4, X, F, J
    """
    candidates = np.asarray(candidates, dtype=np.float64)
    xl = np.asarray(xl, dtype=np.float64)
    xu = np.asarray(xu, dtype=np.float64)

    all_X: list[np.ndarray] = []
    all_F: list[list[float]] = []
    all_J: list[float] = []

    best = None
    best_loss = float("inf")

    for i, theta0 in enumerate(candidates):
        theta0 = np.clip(np.asarray(theta0, dtype=np.float64), xl, xu)

        logger.info(
            "[fit]  hybrid_fit  lm_polish  seed=%02d/%02d  starting",
            i + 1,
            len(candidates),
        )

        t0 = time.perf_counter()
        try:
            theta_opt, total, f1, f2, f3, f4 = run_single_optimisation(
                residuals_fn,
                theta0,
                max_steps=int(lm_max_steps),
                rtol=float(lm_rtol),
                atol=float(lm_atol),
                verbose=bool(verbose),
                optx_adjoint=optx_adjoint,
                ls_solver=ls_solver,
                jac_mode=jac_mode,
            )
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            logger.warning(
                "[fit]  hybrid_fit  lm_polish  seed=%02d/%02d  FAILED  t=%.2fs  error=%s",
                i + 1,
                len(candidates),
                elapsed,
                exc,
            )
            continue

        elapsed = time.perf_counter() - t0

        theta_opt = np.asarray(theta_opt, dtype=np.float64)
        total = float(total)
        f1 = float(f1)
        f2 = float(f2)
        f3 = float(f3)
        f4 = float(f4)

        all_X.append(theta_opt)
        all_F.append([f1, f2, f3, f4])
        all_J.append(total)

        logger.info(
            "[fit]  hybrid_fit  lm_polish  seed=%02d/%02d  "
            "loss=%.6e  f1=%.4e  f2=%.4e  f3=%.4e  f4=%.4e  t=%.2fs",
            i + 1,
            len(candidates),
            total,
            f1,
            f2,
            f3,
            f4,
            elapsed,
        )

        if total < best_loss:
            best_loss = total
            best = (theta_opt, total, f1, f2, f3, f4)

            logger.info(
                "[fit]  hybrid_fit  lm_polish  new_best  seed=%02d  loss=%.6e",
                i + 1,
                best_loss,
            )

    if best is None or not all_X:
        raise RuntimeError(
            "All hybrid local-polish runs failed. Check DE bounds, model dimensions, "
            "ODE solver stability, and residual function construction."
        )

    X = np.asarray(all_X, dtype=np.float64)
    F = np.asarray(all_F, dtype=np.float64)
    J = np.asarray(all_J, dtype=np.float64)

    theta_best, total_best, f1, f2, f3, f4 = best
    return theta_best, total_best, f1, f2, f3, f4, X, F, J


# ---------------------------------------------------------------------------
# Top-level orchestrator
# ---------------------------------------------------------------------------


def run_hybrid_fit(
    problem,  # kept for main.py/API compatibility; not required by Mutax itself
    loss_fn,
    residuals_fn,
    n_var: int,
    xl: np.ndarray,
    xu: np.ndarray,
    *,
    # Phase 0: LHS
    lhs_n_samples: int = 512,
    lhs_top_p: int = 16,
    skip_lhs: bool = False,
    lhs_batch_size: int = 64,
    # Phase 1: Mutax DE
    de_strategy: str = "best1bin",
    de_maxiter: int = 200,
    de_popsize: int = 15,
    de_tol: float = 0.01,
    de_atol: float = 0.0,
    de_mutation: float | tuple[float, float] = (0.5, 1.0),
    de_recombination: float = 0.8,
    de_workers: int = 1,
    de_updating: str = "immediate",
    de_polish: bool = False,
    # Backward-compatible aliases from old evosax config/CLI
    es_algo: str | None = None,
    es_popsize: int | None = None,
    es_n_generations: int | None = None,
    es_sigma_init: float | None = None,  # accepted, intentionally unused
    es_top_k: int | None = None,
    # Phase 2: Optimistix polish
    polish_top_k: int | None = None,
    lm_max_steps: int = 500,
    lm_rtol: float = 1e-8,
    lm_atol: float = 1e-8,
    optx_adjoint: str = "implicit",
    ls_solver: str = "lm",
    jac_mode: str = "fwd",
    # Candidate construction
    polish_random_starts: int = 8,
    polish_jitter_count: int = 8,
    polish_jitter_scale: float = 0.03,
    # General
    seed: int = 0,
    verbose: bool = False,
    **ignored_legacy_kwargs,
) -> HybridFitResult:
    """
    Run hybrid fitting:

        optional LHS -> Mutax Differential Evolution -> Optimistix polish

    Parameters are intentionally compatible with the previous hybrid call where
    possible. Old evosax names are accepted as aliases:
        es_popsize       -> de_popsize
        es_n_generations -> de_maxiter
        es_top_k         -> polish_top_k

    QDax-related kwargs are accepted via **ignored_legacy_kwargs and ignored with
    a warning, so old main.py/config calls can be cleaned incrementally.
    """
    del problem  # API compatibility only

    xl, xu = _validate_bounds(xl, xu, int(n_var))

    if ignored_legacy_kwargs:
        logger.warning(
            "[fit]  hybrid_fit  ignoring legacy kwargs no longer used by Mutax backend: %s",
            sorted(ignored_legacy_kwargs.keys()),
        )

    # Backward-compatible mapping from old evosax-style names.
    if es_popsize is not None:
        de_popsize = int(es_popsize)

    if es_n_generations is not None:
        de_maxiter = int(es_n_generations)

    if es_top_k is not None and polish_top_k is None:
        polish_top_k = int(es_top_k)

    if es_algo is not None:
        # Old values are no longer meaningful. Keep "de" and Mutax-native names.
        if es_algo in {"best1bin", "rand1bin"}:
            de_strategy = es_algo
        elif es_algo == "de":
            de_strategy = de_strategy
        else:
            logger.warning(
                "[fit]  hybrid_fit  old es_algo=%r is not used. "
                "Using Mutax de_strategy=%r.",
                es_algo,
                de_strategy,
            )

    if es_sigma_init is not None:
        logger.warning(
            "[fit]  hybrid_fit  es_sigma_init is ignored by Mutax Differential Evolution."
        )

    polish_top_k = max(1, int(polish_top_k if polish_top_k is not None else 5))

    logger.header("[*] Hybrid fitting: LHS → Mutax Differential Evolution → Optimistix polish")
    logger.info(
        "[fit]  hybrid_fit  n_var=%d  lhs=%s  de_strategy=%s  de_maxiter=%d  "
        "de_popsize=%d  polish_top_k=%d",
        int(n_var),
        "off" if skip_lhs else "on",
        de_strategy,
        int(de_maxiter),
        int(de_popsize),
        int(polish_top_k),
    )

    # ------------------------------------------------------------------ Phase 0
    warm_seeds = None
    if not skip_lhs:
        logger.info(
            "[fit]  hybrid_fit  phase=0/3  lhs_screen  n_samples=%d  top_p=%d",
            int(lhs_n_samples),
            int(lhs_top_p),
        )

        t0 = time.perf_counter()
        warm_seeds = lhs_screen(
            loss_fn,
            xl,
            xu,
            n_samples=int(lhs_n_samples),
            top_p=int(lhs_top_p),
            seed=int(seed),
            batch_size=int(lhs_batch_size),
        )
        elapsed = time.perf_counter() - t0

        warm_seeds, warm_losses = evaluate_candidates(loss_fn, warm_seeds)
        logger.info(
            "[fit]  hybrid_fit  phase=0/3  lhs_screen  done  best=%.6e  t=%.2fs",
            float(warm_losses[0]),
            elapsed,
        )

    # ------------------------------------------------------------------ Phase 1
    x0 = None
    if warm_seeds is not None and len(warm_seeds) > 0:
        x0 = warm_seeds[0]

    logger.info("[fit]  hybrid_fit  phase=1/3  mutax_de  starting")
    de_best, de_loss, mutax_result = run_mutax_de(
        loss_fn,
        xl,
        xu,
        strategy=de_strategy,
        maxiter=int(de_maxiter),
        popsize=int(de_popsize),
        tol=float(de_tol),
        atol=float(de_atol),
        mutation=de_mutation,
        recombination=float(de_recombination),
        seed=int(seed),
        x0=x0,
        workers=int(de_workers),
        updating=de_updating,
        disp=bool(verbose),
        polish=bool(de_polish),
    )

    # Re-evaluate through project loss_fn to ensure consistent scalar loss.
    de_best_ranked, de_best_loss_eval = evaluate_candidates(loss_fn, de_best[None, :])
    de_best = de_best_ranked[0]
    de_loss = float(de_best_loss_eval[0])

    logger.info(
        "[fit]  hybrid_fit  phase=1/3  mutax_de  best_loss=%.6e",
        de_loss,
    )

    # ------------------------------------------------------------------ Candidate ranking
    candidates, candidate_losses = build_polish_candidates(
        de_best=de_best,
        xl=xl,
        xu=xu,
        warm_seeds=warm_seeds,
        loss_fn=loss_fn,
        n_candidates=polish_top_k,
        seed=int(seed),
        random_starts=int(polish_random_starts),
        jitter_count=int(polish_jitter_count),
        jitter_scale=float(polish_jitter_scale),
    )

    logger.info(
        "[fit]  hybrid_fit  polish candidates: n=%d  best_pre_polish=%.6e",
        len(candidates),
        float(candidate_losses[0]),
    )

    # ------------------------------------------------------------------ Phase 2
    logger.info(
        "[fit]  hybrid_fit  phase=2/3  optimistix_polish  n_seeds=%d  max_steps=%d",
        len(candidates),
        int(lm_max_steps),
    )

    t0 = time.perf_counter()
    (
        theta_opt,
        total_loss,
        f1,
        f2,
        f3,
        f4,
        polish_X,
        polish_F,
        polish_J,
    ) = run_lm_polish(
        residuals_fn,
        candidates,
        xl,
        xu,
        lm_max_steps=int(lm_max_steps),
        lm_rtol=float(lm_rtol),
        lm_atol=float(lm_atol),
        optx_adjoint=optx_adjoint,
        ls_solver=ls_solver,
        jac_mode=jac_mode,
        verbose=bool(verbose),
    )
    elapsed = time.perf_counter() - t0

    logger.success(
        "[fit]  hybrid_fit  complete  loss=%.6e  f1=%.4e  f2=%.4e  "
        "f3=%.4e  f4=%.4e  polish_t=%.2fs",
        float(total_loss),
        float(f1),
        float(f2),
        float(f3),
        float(f4),
        elapsed,
    )

    return HybridFitResult(
        theta_opt=np.asarray(theta_opt, dtype=np.float64),
        total_loss=float(total_loss),
        f1=float(f1),
        f2=float(f2),
        f3=float(f3),
        f4=float(f4),
        global_theta=np.asarray(de_best, dtype=np.float64),
        global_loss=float(de_loss),
        polish_X=np.asarray(polish_X, dtype=np.float64),
        polish_F=np.asarray(polish_F, dtype=np.float64),
        polish_J=np.asarray(polish_J, dtype=np.float64),
        mutax_result=mutax_result,
    )