"""
hybrid_fit.py
Multi-phase hybrid optimization for the phospho-network model.

Provides a three-phase fitting pipeline:

  Phase 0 – Latin Hypercube Sampling (LHS) screen:
      Quickly explore the parameter space and identify promising seeds.

  Phase 1 – evosax global search (Sep-CMA-ES or CMA-ES):
      Global evolutionary strategy operating in normalized [0,1]^n_var space,
      optionally warm-started from LHS seeds.

  Phase 2 – Optimistix Levenberg-Marquardt polish:
      Local gradient-based refinement starting from the top-K candidates
      found by the global search.

  QDax MAP-Elites extension (mandatory):
      Quality-Diversity archive exploration of the parameter space using
      2D behaviour descriptors derived from biological half-lives.

Lazy imports ensure that ``import hybrid_fit`` succeeds even when ``evosax``
or ``qdax`` are not installed; ImportError is raised only when the relevant
function is called.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from evosax.algorithms import CMA_ES, Sep_CMA_ES
from evosax.core.restart import cma_cond, spread_cond, RestartParams, RestartState

from qdax.core.containers.mapelites_repertoire import (
    MapElitesRepertoire,
    compute_cvt_centroids,
)
from qdax.core.emitters.mutation_operators import isoline_variation
from qdax.core.map_elites import MAPElites

from phoscrosstalk.config import ModelDims
from phoscrosstalk.optimization import (
    run_single_optimisation,
)
from phoscrosstalk.logger import get_logger

logger = get_logger()

# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class HybridFitResult:
    """
    Return type for :func:`run_hybrid_fit`.

    Supports both attribute access (``result.theta_opt``) and 6-tuple
    unpacking (``theta_opt, total_loss, f1, f2, f3, f4 = result``).

    Attributes
    ----------
    theta_opt : np.ndarray
        Best-fit parameter vector, shape ``(n_var,)``, ``float64``.
    total_loss : float
        Sum of all weighted loss components at ``theta_opt``.
    f1 : float
        Phosphosite occupancy loss.
    f2 : float
        Protein abundance loss.
    f3 : float
        Regularisation loss.
    f4 : float
        mRNA / R_rna loss (zero when no RNA data present).
    qdax_repertoire : object
        QDax ``MapElitesRepertoire`` object.
    qdax_descriptors : np.ndarray
        Behaviour descriptors for filled niches, shape ``(n_filled, 2)``.
    qdax_fitnesses : np.ndarray
        Fitness values for filled niches, shape ``(n_filled,)``.
    qdax_genotypes : np.ndarray
        Parameter vectors for filled niches, shape ``(n_filled, n_var)``.
    """

    theta_opt: np.ndarray
    total_loss: float
    f1: float
    f2: float
    f3: float
    f4: float
    qdax_repertoire: object
    qdax_descriptors: np.ndarray
    qdax_fitnesses: np.ndarray
    qdax_genotypes: np.ndarray

    def __iter__(self):
        """Enable 6-tuple unpacking: ``theta_opt, total_loss, f1, f2, f3, f4 = result``."""
        yield self.theta_opt
        yield self.total_loss
        yield self.f1
        yield self.f2
        yield self.f3
        yield self.f4


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
    Phase 0: Latin Hypercube Sampling screen over the parameter space.

    Generates *n_samples* parameter vectors using a space-filling LHS design
    implemented in pure NumPy (no scipy dependency), evaluates the loss in
    JIT-compiled batches, and returns the *top_p* lowest-loss candidates.

    Parameters
    ----------
    loss_fn : callable (theta, args) -> (scalar_loss, aux)
        JAX-compatible scalarised loss function as returned by ``make_loss_fn``.
        Should be JIT-compiled (``jax.jit``) before being passed in.
    xl : np.ndarray
        Lower bounds for each parameter, shape ``(n_var,)``.
    xu : np.ndarray
        Upper bounds for each parameter, shape ``(n_var,)``.
    n_samples : int
        Number of LHS samples to generate.  Default: 512.
    top_p : int
        Number of top candidates to return.  Default: 16.
    seed : int
        Random seed for the NumPy LHS sampler.  Default: 0.
    batch_size : int
        Number of samples to evaluate per JIT-compiled vmap call.  Smaller
        batches reduce peak memory and JIT compile time.  Default: 64.

    Returns
    -------
    np.ndarray
        Top-*p* parameter vectors in original (un-normalised) space, sorted
        ascending by loss.  Shape ``(top_p, n_var)``, ``float64``.
    """
    n_var = len(xl)
    rng = np.random.default_rng(seed)

    # ----- Pure-NumPy Latin Hypercube Sampling -----
    # Divide [0,1] into n_samples equal strata per dimension;
    # draw one point uniformly from each stratum, then shuffle columns.
    strata_width = 1.0 / n_samples
    lower_edges = np.arange(n_samples, dtype=np.float64) * strata_width   # (n_samples,)
    unit_samples = np.empty((n_samples, n_var), dtype=np.float64)
    for d in range(n_var):
        perm = rng.permutation(n_samples)
        unit_samples[:, d] = lower_edges[perm] + rng.uniform(0.0, strata_width, n_samples)

    # Un-normalise to [xl, xu]
    span = xu - xl
    samples = xl + unit_samples * span   # (n_samples, n_var) float64

    # ----- JIT-compiled batched loss evaluation -----
    @jax.jit
    def eval_batch(batch_theta: jnp.ndarray) -> jnp.ndarray:
        return jax.vmap(lambda th: loss_fn(th, None)[0])(batch_theta)

    samples_f32 = jnp.asarray(samples, dtype=jnp.float32)
    loss_pieces = []
    for i in range(0, n_samples, batch_size):
        loss_pieces.append(eval_batch(samples_f32[i : i + batch_size]))
    losses_np = np.asarray(jnp.concatenate(loss_pieces), dtype=np.float64)

    # Return top_p seeds sorted ascending by loss
    order = np.argsort(losses_np)[:top_p]
    return samples[order].astype(np.float64)

# ---------------------------------------------------------------------------
# Phase 1 – evosax global search
# ---------------------------------------------------------------------------


def run_evosax(
    loss_fn,
    xl: np.ndarray,
    xu: np.ndarray,
    n_var: int,
    *,
    algo: str = "sep_cma_es",
    popsize: int = 64,
    n_generations: int = 200,
    sigma_init: float = 0.3,
    seed: int = 0,
    warm_start_pop: np.ndarray | None = None,
    verbose: bool = False,
    n_restarts: int = 3,
    chunk_size: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Phase 1: evosax evolutionary strategy global search.

    Operates internally in a normalised ``[0, 1]^n_var`` space and maps back
    to original parameter space for loss evaluation.  Optionally warm-starts
    the initial population from LHS seeds.  Supports BIPOP restarts and uses
    ``jax.lax.scan`` for the inner generation loop to reduce Python overhead.

    Parameters
    ----------
    loss_fn : callable (theta, args) -> (scalar_loss, aux)
        JAX-compatible scalarised loss function.  Must support ``jax.vmap``.
    xl : np.ndarray
        Lower bounds, shape ``(n_var,)``.
    xu : np.ndarray
        Upper bounds, shape ``(n_var,)``.
    n_var : int
        Parameter space dimensionality.
    algo : str
        Strategy name.  ``"sep_cma_es"`` (default) or ``"cma_es"``.
    popsize : int
        Population size.  Default: 64.
    n_generations : int
        Total number of generations to run.  Default: 200.
    sigma_init : float
        Initial step-size / mutation strength.  Default: 0.3.
    seed : int
        JAX random seed.  Default: 0.
    warm_start_pop : np.ndarray or None
        Seed population in original parameter space, shape ``(P, n_var)``.
        The first ``min(P, popsize)`` rows are injected via one manual
        ask→tell step before the scan loop begins.
    verbose : bool
        If ``True``, log progress every 50 generations.
    n_restarts : int
        BIPOP restart budget.  ``0`` disables restarts entirely.  Default: 3.
    chunk_size : int or None
        Number of generations per ``jax.lax.scan`` chunk.  ``None`` selects
        ``max(1, n_generations // (n_restarts + 1))`` automatically.

    Returns
    -------
    best_theta_original : np.ndarray
        Best parameter vector in original space, shape ``(n_var,)``, ``float64``.
    top_k_array : np.ndarray
        Entire final population un-normalised to original space,
        shape ``(popsize, n_var)``, ``float64``.
    """
    xl_j = jnp.asarray(xl, dtype=jnp.float32)
    xu_j = jnp.asarray(xu, dtype=jnp.float32)
    span = xu_j - xl_j

    def to_original(theta_norm: jnp.ndarray) -> jnp.ndarray:
        """Map [0,1]^n_var → [xl, xu]."""
        return xl_j + theta_norm * span

    def to_normalized(theta: np.ndarray) -> jnp.ndarray:
        """Map [xl, xu] → [0,1]^n_var (clipped)."""
        return jnp.clip((jnp.asarray(theta, dtype=jnp.float32) - xl_j) / span, 0.0, 1.0)

    # normed_fitness must be defined before the scan so it is captured as a
    # closed-over constant and never retraced inside lax.scan.
    @jax.jit
    def normed_fitness(pop_norm: jnp.ndarray) -> jnp.ndarray:
        """Evaluate loss for a normalised population batch."""
        pop_orig = jax.vmap(to_original)(pop_norm)
        losses, _ = jax.vmap(lambda theta: loss_fn(theta, None))(pop_orig)
        return losses

    # metrics_fn for best-fitness tracking inside the scan
    def metrics_fn(_key, population, fitness, state, params):  # noqa: ARG001
        return {"best_fitness": state.best_fitness}

    # Instantiate strategy (with metrics_fn if the installed version supports it)
    solution_init = jnp.zeros(n_var, dtype=jnp.float32)
    if algo == "cma_es":
        try:
            es = CMA_ES(
                population_size=popsize,
                solution=solution_init,
                metrics_fn=metrics_fn,
            )
        except TypeError:
            es = CMA_ES(population_size=popsize, solution=solution_init)
    else:
        # sep_cma_es is the default; 'de' and any unknown algo fall back to Sep_CMA_ES
        try:
            es = Sep_CMA_ES(
                population_size=popsize,
                solution=solution_init,
                metrics_fn=metrics_fn,
            )
        except TypeError:
            es = Sep_CMA_ES(population_size=popsize, solution=solution_init)

    mean_init = jnp.full(n_var, 0.5, dtype=jnp.float32)  # centre of [0,1] space

    # Start from default params; try to override sigma_init if the field exists.
    params = es.default_params
    if sigma_init is not None:
        if hasattr(params, "replace"):
            try:
                params = params.replace(sigma_init=sigma_init)
            except TypeError:
                logger.warning(
                    f"evosax Params {type(params)} has no field 'sigma_init'; "
                    "using default es.default_params."
                )
        elif hasattr(params, "_replace"):
            try:
                params = params._replace(sigma_init=sigma_init)
            except TypeError:
                logger.warning(
                    f"evosax Params {type(params)} has no field 'sigma_init'; "
                    "using default es.default_params."
                )
        else:
            logger.warning(
                f"evosax Params type {type(params)} does not support replace/_replace; "
                "using default es.default_params."
            )

    key = jax.random.PRNGKey(seed)
    key, init_key = jax.random.split(key)
    state = es.init(init_key, mean_init, params)

    # Warm-start: inject LHS seeds in one manual ask→tell step before the scan
    if warm_start_pop is not None:
        key, ask_key, tell_key = jax.random.split(key, 3)
        pop, state = es.ask(ask_key, state, params)
        n_inject = min(len(warm_start_pop), popsize)
        warm_norm = to_normalized(warm_start_pop[:n_inject])
        pop = pop.at[:n_inject].set(warm_norm)
        pop = jnp.clip(pop, 0.0, 1.0)
        losses = normed_fitness(pop)
        state, _ = es.tell(tell_key, pop, losses, state, params)

    # Scan step: one generation of ask → clip → evaluate → tell
    def step(carry, scan_key):
        state, params = carry
        key_ask, key_tell = jax.random.split(scan_key, 2)
        population, state = es.ask(key_ask, state, params)
        population = jnp.clip(population, 0.0, 1.0)
        losses = normed_fitness(population)
        state, metrics = es.tell(key_tell, population, losses, state, params)
        return (state, params), metrics

    # BIPOP outer loop: each iteration runs one chunk via lax.scan
    chunk = chunk_size or max(1, n_generations // (n_restarts + 1))
    total_gen = 0
    restart_count = 0
    global_best_fitness = float("inf")
    global_best_solution = None

    while total_gen < n_generations:
        remaining = n_generations - total_gen
        this_chunk = min(chunk, remaining)

        chunk_keys = jax.random.split(key, this_chunk + 1)
        key = chunk_keys[0]
        scan_keys = chunk_keys[1:]

        (state, _), _metrics_log = jax.lax.scan(step, (state, params), scan_keys)
        total_gen += this_chunk

        # Track global best across restarts (es.init resets state.best_*)
        current_best = float(state.best_fitness)
        if current_best < global_best_fitness:
            global_best_fitness = current_best
            global_best_solution = state.best_solution

        if restart_count < n_restarts:
            # Evaluate last-gen population to check restart conditions
            key, ask_key = jax.random.split(key)
            last_pop, _ = es.ask(ask_key, state, params)
            last_pop = jnp.clip(last_pop, 0.0, 1.0)
            last_losses = normed_fitness(last_pop)

            _restart_params = RestartParams()

            _restart_state = RestartState(restart_counter=jnp.int32(restart_count))

            should_restart = bool(
                spread_cond(last_pop, last_losses, state, params, _restart_state, _restart_params)
                | cma_cond(last_pop, last_losses, state, params, _restart_state, _restart_params)
            )

            if should_restart:
                try:
                    mean = es.get_mean(state)
                except AttributeError:
                    mean = state.mean
                key, subkey = jax.random.split(key)
                state = es.init(subkey, mean, params)
                restart_count += 1
                if verbose:
                    logger.info(
                        f"[evosax restart #{restart_count}] gen={total_gen}",
                        flush=True,
                    )

        if verbose and total_gen % 50 == 0:
            logger.info(
                f"[evosax gen {total_gen}] best={float(state.best_fitness):.6f}",
                flush=True,
            )

    # Extract best individual: prefer the global best tracked across all restarts
    if global_best_solution is not None:
        best_norm = global_best_solution
    else:
        best_norm = state.best_solution
    best_orig = np.asarray(to_original(best_norm), dtype=np.float64)

    # Full final population in original space
    key, ask_key = jax.random.split(key)
    final_pop, _ = es.ask(ask_key, state, params)
    final_pop = jnp.clip(final_pop, 0.0, 1.0)
    final_pop_orig = np.asarray(jax.vmap(to_original)(final_pop), dtype=np.float64)

    return best_orig, final_pop_orig


# ---------------------------------------------------------------------------
# Phase 2 – Optimistix LM polish
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
    verbose: bool = False,
) -> tuple[np.ndarray, float, float, float, float, float]:
    """
    Phase 2: Levenberg-Marquardt local polish for a set of candidate seeds.

    Runs :func:`~phoscrosstalk.optimization.run_single_optimisation` for each
    row in *candidates*, clipping each seed to ``[xl, xu]`` first, and returns
    the best result across all seeds.

    Parameters
    ----------
    residuals_fn : callable (theta, args) -> (residuals_1d, (f1, f2, f3, f4))
        Residual function as returned by ``make_residuals_fn``.
    candidates : np.ndarray
        Seed parameter vectors, shape ``(K, n_var)``.
    xl : np.ndarray
        Lower bounds, shape ``(n_var,)``.
    xu : np.ndarray
        Upper bounds, shape ``(n_var,)``.
    lm_max_steps : int
        Maximum LM iterations per seed.  Default: 500.
    lm_rtol : float
        Relative tolerance for LM convergence.  Default: 1e-8.
    lm_atol : float
        Absolute tolerance for LM convergence.  Default: 1e-8.
    verbose : bool
        If ``True``, log each seed's result.

    Returns
    -------
    tuple
        ``(theta_opt, total_loss, f1, f2, f3, f4)`` for the best seed, where
        ``theta_opt`` is ``np.ndarray`` shape ``(n_var,)`` and the losses are
        ``float`` scalars.
    """
    best_result = None
    best_loss = float("inf")

    for i, theta0 in enumerate(candidates):
        theta0_clipped = np.clip(np.asarray(theta0, dtype=np.float64), xl, xu)
        theta_opt, total, f1, f2, f3, f4 = run_single_optimisation(
            residuals_fn,
            theta0_clipped,
            max_steps=lm_max_steps,
            rtol=lm_rtol,
            atol=lm_atol,
            verbose=False,
        )
        if verbose:
            logger.info(
                f"  [LM seed {i}] total={total:.6f} f1={f1:.6f} "
                f"f2={f2:.6f} f3={f3:.6f} f4={f4:.6f}",
                flush=True,
            )
        if total < best_loss:
            best_loss = total
            best_result = (theta_opt, total, f1, f2, f3, f4)

    if best_result is None:
        raise RuntimeError("run_lm_polish received empty candidates array.")

    return best_result


# ---------------------------------------------------------------------------
# QDax MAP-Elites extension
# ---------------------------------------------------------------------------


def run_qdax_mapelites(
    loss_fn,
    xl: np.ndarray,
    xu: np.ndarray,
    n_var: int,
    theta_seed: np.ndarray,
    *,
    n_centroids: int = 1024,
    batch_size: int = 256,
    n_iterations: int = 2000,
    iso_sigma: float = 0.005,
    line_sigma: float = 0.05,
    seed: int = 0,
    verbose: bool = False,
) -> object:
    """
    QDax MAP-Elites quality-diversity archive exploration.

    Builds a Voronoi-tessellated MAP-Elites archive over a 2D behaviour
    descriptor space defined by biological half-lives, using the best
    parameter set from Phase 2 as a warm start.

    Behaviour descriptor
    --------------------
    ``BD = (log10(median_t_half_kinase), log10(median_t_half_protein))``

    Both axes are bounded in ``[log10(0.1), log10(1000)]``.

    Fitness
    -------
    ``fitness = -loss_fn(theta, None)[0]``

    MAP-Elites maximises fitness, so the loss is negated.

    Parameters
    ----------
    loss_fn : callable (theta, args) -> (scalar_loss, aux)
        JAX-compatible scalarised loss function.
    xl : np.ndarray
        Lower bounds, shape ``(n_var,)``.
    xu : np.ndarray
        Upper bounds, shape ``(n_var,)``.
    n_var : int
        Parameter space dimensionality.
    theta_seed : np.ndarray
        Best parameter vector from Phase 2 (warm start), shape ``(n_var,)``.
    n_centroids : int
        Number of Voronoi centroids for the CVT archive.  Default: 1024.
    batch_size : int
        Batch size for MAP-Elites evaluations.  Default: 256.
    n_iterations : int
        Number of MAP-Elites update iterations.  Default: 2000.
    iso_sigma : float
        Isotropic perturbation std for the isoline variation operator.
        Default: 0.005.
    line_sigma : float
        Line perturbation std for the isoline variation operator.
        Default: 0.05.
    seed : int
        JAX random seed.  Default: 0.
    verbose : bool
        If ``True``, log progress every 500 iterations.

    Returns
    -------
    object
        QDax ``MapElitesRepertoire`` after *n_iterations* iterations.
    """
    K = ModelDims.K
    M = ModelDims.M

    xl_j = jnp.asarray(xl, dtype=jnp.float32)
    xu_j = jnp.asarray(xu, dtype=jnp.float32)

    # ------------------------------------------------------------------
    # Behaviour descriptor function (pure JAX, float32)
    # ------------------------------------------------------------------
    # Parameter layout (from decode_theta / create_bounds):
    #   [0:K]       log_k_deact   (k_deact = exp(log_k_deact))
    #   [K:2K]      log_d_deg     (d_deg = exp(log_d_deg))
    #   [2K]        log_beta_g
    #   [2K+1]      log_beta_l
    #   [2K+2:2K+2+M]   log_alpha
    #   [2K+2+M:2K+2+2M] log_kK_act
    #   [2K+2+2M:2K+2+3M] log_kK_deact
    #   [2K+2+3M:2K+2+3M+N] log_k_off
    #   [2K+2+3M+N:2K+2+3M+N+4] raw_gamma

    log2 = math.log(2.0)
    _bd_min = math.log10(0.1)
    _bd_max = math.log10(1000.0)

    def descriptor_fn(theta: jnp.ndarray) -> jnp.ndarray:
        """
        Compute 2D behaviour descriptor from a parameter vector.

        Returns
        -------
        jnp.ndarray
            Shape ``(2,)`` with values in ``[log10(0.1), log10(1000)]``.
        """
        theta_f32 = jnp.asarray(theta, dtype=jnp.float32)
        # log_kK_deact is at indices [2K+2+2M : 2K+2+3M]
        log_kK_deact = theta_f32[2 * K + 2 + 2 * M : 2 * K + 2 + 3 * M]
        # log_d_deg is at indices [K : 2K]
        log_d_deg = theta_f32[K : 2 * K]

        kK_deact = jnp.exp(jnp.clip(log_kK_deact, -20.0, 10.0))
        d_deg = jnp.exp(jnp.clip(log_d_deg, -20.0, 10.0))

        t_half_kinase = jnp.float32(log2) / kK_deact  # shape (M,)
        t_half_protein = jnp.float32(log2) / d_deg  # shape (K,)

        # Median via sort
        median_tK = jnp.median(t_half_kinase)
        median_tP = jnp.median(t_half_protein)

        bd0 = jnp.log10(jnp.maximum(median_tK, 1e-9))
        bd1 = jnp.log10(jnp.maximum(median_tP, 1e-9))

        # Clip to BD bounds
        bd0 = jnp.clip(bd0, _bd_min, _bd_max)
        bd1 = jnp.clip(bd1, _bd_min, _bd_max)
        return jnp.stack([bd0, bd1])

    # ------------------------------------------------------------------
    # Fitness function (MAP-Elites maximises, so negate loss)
    # ------------------------------------------------------------------
    @jax.jit
    def scoring_fn(genotypes: jnp.ndarray, extra_scores=None):
        """Batch scoring: returns (fitnesses, descriptors, extra_scores, is_init)."""
        fitnesses = jax.vmap(lambda theta: -loss_fn(theta, None)[0])(genotypes)
        descriptors = jax.vmap(descriptor_fn)(genotypes)
        return fitnesses, descriptors, extra_scores

    # ------------------------------------------------------------------
    # CVT centroids over BD space
    # ------------------------------------------------------------------
    key = jax.random.PRNGKey(seed)
    key, centroid_key = jax.random.split(key)

    min_bd = jnp.array([_bd_min, _bd_min], dtype=jnp.float32)
    max_bd = jnp.array([_bd_max, _bd_max], dtype=jnp.float32)

    try:
        centroids, _ = compute_cvt_centroids(
            num_descriptors=2,
            num_init_cvt_samples=n_centroids * 10,
            num_centroids=n_centroids,
            minval=min_bd,
            maxval=max_bd,
            key=centroid_key,
        )
    except TypeError:
        centroids, _ = compute_cvt_centroids(
            num_descriptors=2,
            num_init_cvt_samples=n_centroids * 10,
            num_centroids=n_centroids,
            minval=min_bd,
            maxval=max_bd,
            key=centroid_key,
        )

    # ------------------------------------------------------------------
    # Initial population: perturb theta_seed
    # ------------------------------------------------------------------
    rng = np.random.default_rng(seed)
    perturb = rng.normal(0.0, iso_sigma, (batch_size, n_var)).astype(np.float32)
    init_pop_np = np.clip(
        np.asarray(theta_seed, dtype=np.float32)[None, :] + perturb,
        np.asarray(xl, dtype=np.float32),
        np.asarray(xu, dtype=np.float32),
    )
    init_genotypes = jnp.asarray(init_pop_np, dtype=jnp.float32)

    # ------------------------------------------------------------------
    # Emitter: isoline variation
    # ------------------------------------------------------------------
    def variation_fn(x: jnp.ndarray, repertoire, isoline_key: jax.Array) -> jnp.ndarray:
        """Apply isoline variation and clip to parameter bounds."""
        offspring, _ = isoline_variation(
            x,
            repertoire,
            isoline_key,
            iso_sigma=iso_sigma,
            line_sigma=line_sigma,
            minval=xl_j,
            maxval=xu_j,
        )
        return offspring

    # ------------------------------------------------------------------
    # Build MAP-Elites instance and initialise repertoire
    # ------------------------------------------------------------------
    map_elites = MAPElites(
        scoring_function=scoring_fn,
        emitter=None,  # we run the loop manually using variation_fn
        metrics_function=None,
    )

    key, score_key = jax.random.split(key)
    init_fitnesses, init_descriptors, _ = scoring_fn(init_genotypes)

    repertoire = MapElitesRepertoire.init(
        genotypes=init_genotypes,
        fitnesses=init_fitnesses,
        descriptors=init_descriptors,
        centroids=centroids,
    )

    logger.header("[*] QDax MAP-Elites exploration")

    # ------------------------------------------------------------------
    # MAP-Elites loop: JIT-compiled step to avoid retracing per iteration
    # ------------------------------------------------------------------
    @jax.jit
    def mapelites_step(repertoire, key):
        key, sample_key, var_key = jax.random.split(key, 3)
        parents = repertoire.sample(sample_key, batch_size)
        offspring = variation_fn(parents, repertoire, var_key)
        off_fit, off_desc, _ = scoring_fn(offspring)
        repertoire = repertoire.add(offspring, off_desc, off_fit)
        return repertoire, key

    for it in range(n_iterations):
        repertoire, key = mapelites_step(repertoire, key)
        if verbose and (it % 500 == 0 or it == n_iterations - 1):
            valid_mask = repertoire.fitnesses > -jnp.inf
            n_filled = int(valid_mask.sum())
            best_fit = float(jnp.max(repertoire.fitnesses[valid_mask]))
            logger.info(
                f"[QDax iter {it:5d}] filled={n_filled} best_fit={best_fit:.6f}",
                flush=True,
            )

    return repertoire


# ---------------------------------------------------------------------------
# Top-level orchestrator
# ---------------------------------------------------------------------------


def run_hybrid_fit(
    problem,
    loss_fn,
    residuals_fn,
    n_var: int,
    xl: np.ndarray,
    xu: np.ndarray,
    *,
    # Phase 0
    lhs_n_samples: int = 512,
    lhs_top_p: int = 16,
    skip_lhs: bool = False,
    # Phase 1
    es_algo: str = "sep_cma_es",
    es_popsize: int = 64,
    es_n_generations: int = 200,
    es_sigma_init: float = 0.3,
    es_top_k: int = 5,
    # Phase 2
    lm_max_steps: int = 500,
    lm_rtol: float = 1e-8,
    lm_atol: float = 1e-8,
    # QDax
    qdax_n_centroids: int = 1024,
    qdax_batch_size: int = 256,
    qdax_n_iterations: int = 2000,
    qdax_iso_sigma: float = 0.005,
    qdax_line_sigma: float = 0.05,
    # General
    seed: int = 0,
    verbose: bool = False,
) -> HybridFitResult:
    """
    Top-level hybrid fitting orchestrator.

    Executes the three-phase pipeline (LHS → evosax → LM) followed by the
    mandatory QDax MAP-Elites quality-diversity exploration.

    Parameters
    ----------
    problem : NetworkProblem
        Fully constructed problem instance (for data attributes).
    loss_fn : callable (theta, args) -> (scalar_loss, aux)
        Scalarised JAX loss function (from ``make_loss_fn``).
    residuals_fn : callable (theta, args) -> (residuals_1d, aux)
        Residual-vector function (from ``make_residuals_fn``).
    n_var : int
        Parameter space dimensionality.
    xl : np.ndarray
        Lower bounds, shape ``(n_var,)``.
    xu : np.ndarray
        Upper bounds, shape ``(n_var,)``.
    lhs_n_samples : int
        LHS sample count (Phase 0).  Default: 512.
    lhs_top_p : int
        Top-P seeds from LHS to forward to evosax.  Default: 16.
    skip_lhs : bool
        If ``True``, skip Phase 0.  Default: False.
    es_algo : str
        evosax strategy: ``"sep_cma_es"`` (default) or ``"cma_es"``.
    es_popsize : int
        evosax population size.  Default: 64.
    es_n_generations : int
        Number of evosax generations.  Default: 200.
    es_sigma_init : float
        Initial mutation step size.  Default: 0.3.
    es_top_k : int
        Number of top evosax candidates to pass to LM.  Default: 5.
    lm_max_steps : int
        Maximum LM iterations per seed.  Default: 500.
    lm_rtol : float
        LM relative tolerance.  Default: 1e-8.
    lm_atol : float
        LM absolute tolerance.  Default: 1e-8.
    qdax_n_centroids : int
        CVT centroids for MAP-Elites.  Default: 1024.
    qdax_batch_size : int
        MAP-Elites batch size.  Default: 256.
    qdax_n_iterations : int
        MAP-Elites iterations.  Default: 2000.
    qdax_iso_sigma : float
        Isoline variation iso_sigma.  Default: 0.005.
    qdax_line_sigma : float
        Isoline variation line_sigma.  Default: 0.05.
    seed : int
        Master random seed.  Default: 0.
    verbose : bool
        If ``True``, logger.info progress for all phases.

    Returns
    -------
    HybridFitResult
        Supports both ``result.theta_opt`` attribute access and
        ``theta_opt, total_loss, f1, f2, f3, f4 = result`` tuple unpacking.
    """
    # ------------------------------------------------------------------ Phase 0
    warm_seeds: np.ndarray | None = None
    if not skip_lhs:
        if verbose:
            logger.header("[*] Hybrid fitting pipeline: LHS → evosax → LM")
            logger.info("[hybrid_fit] Phase 0: LHS screen …", flush=True)
        warm_seeds = lhs_screen(
            loss_fn, xl, xu, n_samples=lhs_n_samples, top_p=lhs_top_p, seed=seed
        )

    # ------------------------------------------------------------------ Phase 1
    if verbose:
        logger.info("[hybrid_fit] Phase 1: evosax global search …", flush=True)
    best_theta, final_pop = run_evosax(
        loss_fn,
        xl,
        xu,
        n_var,
        algo=es_algo,
        popsize=es_popsize,
        n_generations=es_n_generations,
        sigma_init=es_sigma_init,
        seed=seed,
        warm_start_pop=warm_seeds,
        verbose=verbose,
    )

    # Rank final population by loss to choose LM seeds
    pop_f32 = jnp.asarray(final_pop, dtype=jnp.float32)
    pop_losses = np.asarray(
        jax.vmap(lambda theta: loss_fn(theta, None)[0])(pop_f32), dtype=np.float64
    )
    top_k_idx = np.argsort(pop_losses)[:es_top_k]
    top_k_seeds = final_pop[top_k_idx]

    # Always include best_theta as seed 0; avoid exact duplicates
    best_theta_row = best_theta[None, :]
    is_dup = np.all(
        np.isclose(top_k_seeds, best_theta_row, rtol=1e-9, atol=1e-12), axis=1
    )
    unique_top_k = top_k_seeds[~is_dup]
    lm_seeds = np.concatenate([best_theta_row, unique_top_k], axis=0)

    # ------------------------------------------------------------------ Phase 2
    if verbose:
        logger.info("[hybrid_fit] Phase 2: LM polish …", flush=True)
    lm_result = run_lm_polish(
        residuals_fn,
        lm_seeds,
        xl,
        xu,
        lm_max_steps=lm_max_steps,
        lm_rtol=lm_rtol,
        lm_atol=lm_atol,
        verbose=verbose,
    )
    theta_opt, total_loss, f1, f2, f3, f4 = lm_result

    # ------------------------------------------------------------------ QDax
    if verbose:
        logger.info("[hybrid_fit] QDax MAP-Elites …", flush=True)
    repertoire = run_qdax_mapelites(
        loss_fn,
        xl,
        xu,
        n_var,
        theta_opt,
        n_centroids=qdax_n_centroids,
        batch_size=qdax_batch_size,
        n_iterations=qdax_n_iterations,
        iso_sigma=qdax_iso_sigma,
        line_sigma=qdax_line_sigma,
        seed=seed,
        verbose=verbose,
    )

    # Extract filled niches
    valid_mask = np.asarray(repertoire.fitnesses > -jnp.inf)
    qdax_fitnesses = np.asarray(repertoire.fitnesses[valid_mask], dtype=np.float64)
    qdax_descriptors = np.asarray(repertoire.descriptors[valid_mask], dtype=np.float64)
    qdax_genotypes = np.asarray(repertoire.genotypes[valid_mask], dtype=np.float64)

    return HybridFitResult(
        theta_opt=theta_opt,
        total_loss=float(total_loss),
        f1=float(f1),
        f2=float(f2),
        f3=float(f3),
        f4=float(f4),
        qdax_repertoire=repertoire,
        qdax_descriptors=qdax_descriptors,
        qdax_fitnesses=qdax_fitnesses,
        qdax_genotypes=qdax_genotypes,
    )
