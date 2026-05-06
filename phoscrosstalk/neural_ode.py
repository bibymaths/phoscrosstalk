"""
neural_ode.py
Post-fit neural latent-rate refinement for the phospho-network model.

Implements a Mechanistic Graph-Constrained Latent Neural ODE.  After the
main mechanistic fit, the derived mechanistic rate trajectories ``k_act(t)``
and ``s_prod(t)`` are used as biological priors to initialise and regularise
**learned** latent rate functions ``k_hat_act(t)`` and ``s_hat_prod(t)``.

The learned neural rate generator improves latent biological dynamics while
staying anchored to the graph-derived mechanistic priors via a regularisation
term:

    L_total =
        data_weight_phospho   * phosphosite observed-time loss
      + data_weight_abundance * protein abundance observed-time loss
      + data_weight_mrna      * mRNA observed-time loss
      + prior_weight_k_act    * ||k_hat_act(t) - k_act_init(t)||²
      + prior_weight_s_prod   * ||s_hat_prod(t) - s_prod_init(t)||²

The mechanistic ODE equations and ``theta_best`` are kept fixed.
Only the neural rate generator parameters are optimised.

Public API
----------
run_neural_latent_rate_refinement(...)
    Main entry point called by ``main.py`` after the mechanistic fit.
"""

from __future__ import annotations

import json
import os
from typing import Optional

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optimistix as optx

from phoscrosstalk.config import ModelDims
from phoscrosstalk.logger import get_logger
from phoscrosstalk.mechanisms import compute_prev_site_idx, make_rhs
from phoscrosstalk.optimization import build_full_A0

logger = get_logger(__name__)

# Small constant added after softplus to guarantee strict positivity.
_EPS: float = 1e-8

# Penalty value for non-finite ODE results during neural training.
_NEURAL_PENALTY: float = 1e6


# ---------------------------------------------------------------------------
# Equinox module definitions
# ---------------------------------------------------------------------------


class LatentRateMLP(eqx.Module):
    """
    Small MLP that maps biological feature vector → one rate vector of shape (K,).

    Uses ``jax.nn.softplus`` hidden activations and an unconstrained final layer,
    then applies ``jax.nn.softplus(raw * out_scale) + eps`` to guarantee
    strictly positive output (biological rates must be non-negative).

    Attributes
    ----------
    mlp : eqx.nn.MLP
        The underlying MLP with softplus hidden activations.
    out_scale : jax.Array, shape (K,)
        Learnable per-output scaling factor applied before the final softplus.
    """

    mlp: eqx.nn.MLP
    out_scale: jax.Array

    def __init__(
        self,
        in_size: int,
        K: int,
        width: int,
        depth: int,
        key: jax.Array,
    ) -> None:
        self.mlp = eqx.nn.MLP(
            in_size=in_size,
            out_size=K,
            width_size=width,
            depth=depth,
            activation=jax.nn.softplus,
            final_activation=lambda x: x,
            use_bias=True,
            use_final_bias=True,
            dtype=jnp.float64,
            scan=True,
            key=key,
        )
        self.out_scale = jnp.ones(K, dtype=jnp.float64)

    def __call__(self, features: jax.Array) -> jax.Array:
        """
        Args:
            features: (in_size,) feature vector.

        Returns:
            (K,) strictly positive rate vector.
        """
        raw = self.mlp(features)
        return jax.nn.softplus(raw * self.out_scale) + jnp.asarray(
            _EPS, dtype=jnp.float64
        )


class NeuralRateGenerator(eqx.Module):
    """
    Two independent ``LatentRateMLP`` networks for ``k_hat_act(t)`` and
    ``s_hat_prod(t)``.

    The input feature vector is:
        [normalized_time, k_act_init(t), s_prod_init(t)]
    of shape ``(1 + K + K,)``.

    Attributes
    ----------
    k_act_net : LatentRateMLP
        Network for the protein activation rate.
    s_prod_net : LatentRateMLP
        Network for the protein synthesis rate.
    """

    k_act_net: LatentRateMLP
    s_prod_net: LatentRateMLP

    def __init__(self, K: int, width: int, depth: int, key: jax.Array) -> None:
        in_size = 1 + K + K  # [normalized_time, k_act_init(t), s_prod_init(t)]
        k_key, s_key = jax.random.split(key)
        self.k_act_net = LatentRateMLP(in_size, K, width, depth, k_key)
        self.s_prod_net = LatentRateMLP(in_size, K, width, depth, s_key)

    def __call__(self, features: jax.Array):
        """
        Args:
            features: (1 + K + K,) feature vector with normalized time, prior k_act, prior s_prod.

        Returns:
            Tuple of (k_hat_act, s_hat_prod), each shape (K,) and strictly positive.
        """
        k_hat = self.k_act_net(features)
        s_hat = self.s_prod_net(features)
        return k_hat, s_hat


# ---------------------------------------------------------------------------
# Neural loss function builder
# ---------------------------------------------------------------------------


def _make_neural_loss_fn(
    *,
    K: int,
    M: int,
    N: int,
    mechanism: str,
    theta_j: jax.Array,
    y0_j: jax.Array,
    t0_val: float,
    t1_val: float,
    t_eval_j: jax.Array,
    prot_idx_solver_j: jax.Array,
    prev_site_idx_j: jax.Array,
    Cg_j: jax.Array,
    Cl_j: jax.Array,
    K_sk_j: jax.Array,
    R_j: jax.Array,
    La_j: jax.Array,
    spi_j: jax.Array,
    k2p_j: jax.Array,
    rmp_j: jax.Array,
    rmk_j: jax.Array,
    P_data_j: jax.Array,
    A_scaled_j: jax.Array,
    prot_idx_j: jax.Array,
    W_data_j: jax.Array,
    W_prot_j: jax.Array,
    t_obs_j: jax.Array,
    k_act_init_obs_j: jax.Array,
    s_prod_init_obs_j: jax.Array,
    t_max: float,
    # mRNA (optional)
    has_mrna: bool,
    rna_j: Optional[jax.Array],
    mrna_idx_j: Optional[jax.Array],
    rna_prot_idx_j: Optional[jax.Array],
    W_rna_j: Optional[jax.Array],
    # per-modality loss weights
    w_phospho: float,
    w_abundance: float,
    w_mrna: float,
    prior_weight_k_act: float,
    prior_weight_s_prod: float,
    # ODE solver settings
    rtol: float,
    atol: float,
    dt0: float,
    max_steps: int,
    rna_relax: float,
    abundance_max: float,
    # derived-rate init fns (for neural input features inside ODE RHS)
    k_act_init_fn,
    s_prod_init_fn,
    # static for Equinox model reconstruction
    static,
):
    """
    Build and return the JIT-compilable neural loss function.

    The returned function has signature::

        neural_loss_fn(params, _args) -> (total_loss, (f_phospho, f_abund, f_mrna,
                                                        f_k_prior, f_s_prior))

    where ``params`` are the trainable Equinox array leaves of the
    ``NeuralRateGenerator``, and ``_args`` is unused (all data are in the closure).

    The ODE uses RecursiveCheckpointAdjoint for memory-efficient reverse-mode AD
    through the ODE solve.
    """
    n_p = max(1, int(P_data_j.size))
    n_A = max(1, int(A_scaled_j.size))
    has_abundance = A_scaled_j.size > 0
    n_rna = 1
    if has_mrna and rna_j is not None:
        n_rna = max(1, int(rna_j.size))

    t_max_j = jnp.asarray(t_max, dtype=jnp.float64)
    eps_j = jnp.asarray(_EPS, dtype=jnp.float64)
    penalty_j = jnp.asarray(_NEURAL_PENALTY, dtype=jnp.float64)

    # Fixed ODE args (theta and graph matrices are frozen at theta_best)
    ode_static_args = (
        theta_j,
        Cg_j,
        Cl_j,
        spi_j,
        K_sk_j,
        R_j,
        La_j,
        k2p_j,
        rmp_j,
        rmk_j,
        prev_site_idx_j,
    )

    # Stepsize controller
    sctrl = diffrax.PIDController(rtol=rtol, atol=atol)
    saveat = diffrax.SaveAt(ts=t_eval_j)
    adjoint = diffrax.RecursiveCheckpointAdjoint()

    def neural_loss_fn(params, _args):
        # Reconstruct the neural model from trainable array leaves + static parts.
        model = eqx.combine(params, static)

        # ------------------------------------------------------------------
        # Build neural rate closures that close over the traced `model`.
        # These are passed as k_act_fn and s_prod_fn to make_rhs.
        # JAX traces through them, computing gradients w.r.t. params.
        # ------------------------------------------------------------------
        def _k_hat_fn(t):
            t_norm = jnp.clip(t / t_max_j, 0.0, 1.0)
            k_prior = jnp.clip(k_act_init_fn(t), 0.0, None)
            s_prior = jnp.clip(s_prod_init_fn(t), 0.0, None)
            features = jnp.concatenate(
                [jnp.asarray([t_norm], dtype=jnp.float64), k_prior, s_prior]
            )
            k_hat, _ = model(features)
            return k_hat

        def _s_hat_fn(t):
            t_norm = jnp.clip(t / t_max_j, 0.0, 1.0)
            k_prior = jnp.clip(k_act_init_fn(t), 0.0, None)
            s_prior = jnp.clip(s_prod_init_fn(t), 0.0, None)
            features = jnp.concatenate(
                [jnp.asarray([t_norm], dtype=jnp.float64), k_prior, s_prior]
            )
            _, s_hat = model(features)
            return s_hat

        # Build mechanistic RHS with neural-generated rate functions.
        # The ODE structure (theta, matrices, equations) is unchanged.
        rhs_fn = make_rhs(
            K,
            M,
            N,
            mechanism,
            k_act_fn=_k_hat_fn,
            s_prod_fn=_s_hat_fn,
            rna_relax=rna_relax,
            abundance_max=abundance_max,
        )

        term = diffrax.ODETerm(rhs_fn)
        ode_solver = diffrax.Tsit5()

        sol = diffrax.diffeqsolve(
            term,
            ode_solver,
            t0=t0_val,
            t1=t1_val,
            dt0=dt0,
            y0=y0_j,
            args=ode_static_args,
            saveat=saveat,
            stepsize_controller=sctrl,
            max_steps=max_steps,
            adjoint=adjoint,
            throw=False,
        )

        xs = sol.ys  # (T_unified, 3*K+M+N)
        solve_ok = jnp.all(jnp.isfinite(xs)) & (
            sol.result == diffrax.RESULTS.successful
        )

        # Sample at protein/phospho time indices
        xs_prot = xs[prot_idx_solver_j, :]
        P_sim = jnp.clip(xs_prot[:, 3 * K + M :], 0.0, None).T  # (N, T_prot)
        A_sim = jnp.clip(xs_prot[:, 2 * K : 3 * K], 0.0, abundance_max).T  # (K, T_prot)

        # ------------------------------------------------------------------
        # Data losses
        # ------------------------------------------------------------------
        diff_p = P_sim - P_data_j  # (N, T_prot)
        f_phospho = jnp.sum(W_data_j * diff_p * diff_p) / jnp.asarray(
            n_p, dtype=jnp.float64
        )

        if has_abundance:
            A_sim_obs = A_sim[prot_idx_j, :]  # (K_obs, T_prot)
            diff_A = A_sim_obs - A_scaled_j
            f_abund = jnp.sum(W_prot_j * diff_A * diff_A) / jnp.asarray(
                n_A, dtype=jnp.float64
            )
        else:
            f_abund = jnp.asarray(0.0, dtype=jnp.float64)

        if has_mrna and rna_j is not None and mrna_idx_j is not None:
            xs_rna = xs[mrna_idx_j, :]
            R_sim_rna = jnp.clip(xs_rna[:, :K], 0.0, 20.0).T  # (K, T_rna)
            R_sim_matched = R_sim_rna[rna_prot_idx_j, :]  # (n_match, T_rna)
            diff_R = R_sim_matched - rna_j
            f_mrna = jnp.sum(W_rna_j * diff_R * diff_R) / jnp.asarray(
                n_rna, dtype=jnp.float64
            )
        else:
            f_mrna = jnp.asarray(0.0, dtype=jnp.float64)

        # ------------------------------------------------------------------
        # Prior regularisation: evaluate neural rates at observed time points
        # and compare to mechanistic priors.
        # k_act_init_obs_j shape: (K, T_obs)
        # s_prod_init_obs_j shape: (K, T_obs)
        # t_obs_j shape: (T_obs,)
        # ------------------------------------------------------------------
        T_obs = t_obs_j.shape[0]
        t_norms = jnp.clip(t_obs_j / t_max_j, 0.0, 1.0)[:, None]  # (T_obs, 1)
        k_priors_obs = k_act_init_obs_j.T  # (T_obs, K)
        s_priors_obs = s_prod_init_obs_j.T  # (T_obs, K)
        features_obs = jnp.concatenate(
            [t_norms, k_priors_obs, s_priors_obs], axis=1
        )  # (T_obs, 1+K+K)

        # vmap over time dimension
        k_hats_obs, s_hats_obs = jax.vmap(model)(features_obs)  # (T_obs, K), (T_obs, K)
        k_hats_obs = k_hats_obs.T  # (K, T_obs)
        s_hats_obs = s_hats_obs.T  # (K, T_obs)

        diff_k = k_hats_obs - k_act_init_obs_j  # (K, T_obs)
        diff_s = s_hats_obs - s_prod_init_obs_j  # (K, T_obs)
        n_rates = jnp.asarray(float(K * T_obs), dtype=jnp.float64)

        f_k_prior = jnp.sum(diff_k * diff_k) / n_rates
        f_s_prior = jnp.sum(diff_s * diff_s) / n_rates

        # ------------------------------------------------------------------
        # Total weighted loss
        # ------------------------------------------------------------------
        total = (
            jnp.asarray(w_phospho, dtype=jnp.float64) * f_phospho
            + jnp.asarray(w_abundance, dtype=jnp.float64) * f_abund
            + jnp.asarray(w_mrna, dtype=jnp.float64) * f_mrna
            + jnp.asarray(prior_weight_k_act, dtype=jnp.float64) * f_k_prior
            + jnp.asarray(prior_weight_s_prod, dtype=jnp.float64) * f_s_prior
        )

        # Penalise non-finite results without crashing
        total = jnp.where(jnp.isfinite(total), total, penalty_j)
        total = jnp.where(solve_ok, total, penalty_j)

        f_phospho = jnp.where(jnp.isfinite(f_phospho), f_phospho, penalty_j)
        f_abund = jnp.where(jnp.isfinite(f_abund), f_abund, penalty_j)
        f_mrna = jnp.where(jnp.isfinite(f_mrna), f_mrna, penalty_j)
        f_k_prior = jnp.where(jnp.isfinite(f_k_prior), f_k_prior, penalty_j)
        f_s_prior = jnp.where(jnp.isfinite(f_s_prior), f_s_prior, penalty_j)

        return total, (f_phospho, f_abund, f_mrna, f_k_prior, f_s_prior)

    return neural_loss_fn


# ---------------------------------------------------------------------------
# Dense simulation helper
# ---------------------------------------------------------------------------


def _neural_simulate_dense(
    *,
    model,
    K: int,
    M: int,
    N: int,
    mechanism: str,
    theta_j: jax.Array,
    y0_j: jax.Array,
    t0_val: float,
    t_dense: np.ndarray,
    prev_site_idx_j: jax.Array,
    Cg_j: jax.Array,
    Cl_j: jax.Array,
    K_sk_j: jax.Array,
    R_j: jax.Array,
    La_j: jax.Array,
    spi_j: jax.Array,
    k2p_j: jax.Array,
    rmp_j: jax.Array,
    rmk_j: jax.Array,
    t_max: float,
    k_act_init_fn,
    s_prod_init_fn,
    rtol: float,
    atol: float,
    dt0: float,
    max_steps: int,
    rna_relax: float,
    abundance_max: float,
) -> dict:
    """Run the neural ODE over a dense time grid and return solution arrays."""
    t_max_j = jnp.asarray(t_max, dtype=jnp.float64)

    def _k_hat_fn(t):
        t_norm = jnp.clip(t / t_max_j, 0.0, 1.0)
        k_prior = jnp.clip(k_act_init_fn(t), 0.0, None)
        s_prior = jnp.clip(s_prod_init_fn(t), 0.0, None)
        features = jnp.concatenate(
            [jnp.asarray([t_norm], dtype=jnp.float64), k_prior, s_prior]
        )
        k_hat, _ = model(features)
        return k_hat

    def _s_hat_fn(t):
        t_norm = jnp.clip(t / t_max_j, 0.0, 1.0)
        k_prior = jnp.clip(k_act_init_fn(t), 0.0, None)
        s_prior = jnp.clip(s_prod_init_fn(t), 0.0, None)
        features = jnp.concatenate(
            [jnp.asarray([t_norm], dtype=jnp.float64), k_prior, s_prior]
        )
        _, s_hat = model(features)
        return s_hat

    rhs_fn = make_rhs(
        K, M, N, mechanism,
        k_act_fn=_k_hat_fn, s_prod_fn=_s_hat_fn,
        rna_relax=rna_relax, abundance_max=abundance_max,
    )

    t_dense_j = jnp.asarray(t_dense, dtype=jnp.float64)
    t1_val = float(t_dense[-1])

    ode_static_args = (
        theta_j, Cg_j, Cl_j, spi_j, K_sk_j, R_j, La_j, k2p_j, rmp_j, rmk_j,
        prev_site_idx_j,
    )

    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(rhs_fn),
        diffrax.Tsit5(),
        t0=t0_val,
        t1=t1_val,
        dt0=dt0,
        y0=y0_j,
        args=ode_static_args,
        saveat=diffrax.SaveAt(ts=t_dense_j),
        stepsize_controller=diffrax.PIDController(rtol=rtol, atol=atol),
        max_steps=max_steps,
        throw=False,
    )

    xs = np.asarray(sol.ys)  # (T_dense, 3K+M+N)
    P_sim = np.clip(xs[:, 3 * K + M :].T, 0.0, None)  # (N, T_dense)
    A_sim = np.clip(xs[:, 2 * K : 3 * K].T, 0.0, abundance_max)  # (K, T_dense)
    return {"P_sim": P_sim, "A_sim": A_sim, "xs": xs}


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_neural_latent_rate_refinement(
    *,
    problem,
    theta_best: np.ndarray,
    k_act_fn,
    s_prod_fn,
    t: np.ndarray,
    P_scaled: np.ndarray,
    A_scaled: np.ndarray,
    prot_idx_for_A: np.ndarray,
    W_data: np.ndarray,
    W_data_prot: np.ndarray,
    proteins: list,
    sites: list,
    kinases: list,
    t_rna: Optional[np.ndarray],
    rna_obs_matched: Optional[np.ndarray],
    rna_model_prot_idx: Optional[np.ndarray],
    W_data_mrna_matched: Optional[np.ndarray],
    outdir: str,
    neural_cfg,
    mechanism: str = "dist",
    rna_relax: float = 0.1,
    abundance_max: float = 5.0,
    R_data0: Optional[np.ndarray] = None,
) -> None:
    """
    Run the post-fit neural latent-rate refinement stage.

    This function:
    1. Evaluates mechanistic ``k_act_init(t)`` and ``s_prod_init(t)`` at
       observed time points (prior trajectories).
    2. Builds a ``NeuralRateGenerator`` (two small MLPs) initialised with
       JAX random key from ``neural_cfg.seed``.
    3. Trains the neural parameters using Optimistix ``GradientDescent`` to
       minimise a weighted loss combining data fit and prior regularisation.
    4. Saves all outputs under ``<outdir>/neural_ode/``.

    The existing mechanistic fitted results (``theta_best``, ``fit_timeseries.tsv``,
    etc.) are **never modified or overwritten**.

    Args:
        problem: Fitted ``NetworkOptimizationProblem`` instance.
        theta_best: Best-fit mechanistic parameter vector (shape (n_var,)).
        k_act_fn: Mechanistic derived-rate closure ``k_act_fn(t) -> (K,)``.
        s_prod_fn: Mechanistic derived-rate closure ``s_prod_fn(t) -> (K,)``.
        t: Observed protein/phospho time points, shape (T,).
        P_scaled: Observed phospho data, shape (N, T).
        A_scaled: Observed protein abundance data, shape (K_obs, T).
        prot_idx_for_A: Integer mapping from abundance rows to protein indices.
        W_data: Phosphosite weight matrix, shape (N, T).
        W_data_prot: Protein abundance weight matrix, shape (K_obs, T).
        proteins: List of protein names, length K.
        sites: List of site labels, length N.
        kinases: List of kinase names, length M.
        t_rna: mRNA time points or None.
        rna_obs_matched: Matched mRNA observations or None, shape (n_match, T_rna).
        rna_model_prot_idx: Protein indices for matched mRNA rows or None.
        W_data_mrna_matched: mRNA weight matrix or None, shape (n_match, T_rna).
        outdir: Output directory (mechanistic outputs already saved here).
        neural_cfg: SimpleNamespace with neural ODE config values.
        mechanism: Phosphorylation mechanism string.
        rna_relax: RNA relaxation rate constant.
        abundance_max: Upper bound for protein abundance state.
        R_data0: Initial mRNA state vector or None.
    """
    K = ModelDims.K
    M = ModelDims.M
    N = ModelDims.N

    neural_outdir = os.path.join(outdir, "neural_ode")
    os.makedirs(neural_outdir, exist_ok=True)

    logger.info("[neural_ode] Starting post-fit neural latent-rate refinement.")
    logger.info(
        "[neural_ode] Config: width=%d, depth=%d, steps=%d, lr=%.4g, seed=%d",
        neural_cfg.width,
        neural_cfg.depth,
        neural_cfg.steps,
        neural_cfg.learning_rate,
        neural_cfg.seed,
    )

    # ------------------------------------------------------------------
    # 1. Evaluate mechanistic prior trajectories at observed time points
    # ------------------------------------------------------------------
    t_obs = np.sort(np.unique(t)).astype(np.float64)
    T_obs = len(t_obs)

    k_act_init_vals = np.stack(
        [np.asarray(k_act_fn(float(ti)), dtype=np.float64) for ti in t_obs], axis=1
    )  # (K, T_obs)
    s_prod_init_vals = np.stack(
        [np.asarray(s_prod_fn(float(ti)), dtype=np.float64) for ti in t_obs], axis=1
    )  # (K, T_obs)

    logger.info(
        "[neural_ode] Mechanistic prior trajectories computed at %d observed time points.",
        T_obs,
    )

    # ------------------------------------------------------------------
    # 2. Build initial ODE state (same construction as in make_residuals_fn)
    # ------------------------------------------------------------------
    T_prot = P_scaled.shape[1]
    A0_full = build_full_A0(K, T_prot, A_scaled, prot_idx_for_A)

    x0 = np.zeros(3 * K + M + N, dtype=np.float64)
    if R_data0 is not None:
        r_data = np.asarray(R_data0, dtype=np.float64)
        r0 = r_data[:, 0].copy() if r_data.ndim > 1 else r_data.copy()
        r0 = np.nan_to_num(r0, nan=1.0, posinf=5.0, neginf=0.0)
        x0[:K] = np.clip(r0, 0.0, 10.0)
    else:
        x0[:K] = 1.0
    a0 = np.nan_to_num(A0_full[:, 0], nan=1.0, posinf=5.0, neginf=0.0)
    x0[2 * K : 3 * K] = np.clip(a0, 0.0, 5.0)
    p0 = np.nan_to_num(P_scaled[:, 0], nan=0.0, posinf=10.0, neginf=0.0)
    x0[3 * K + M :] = np.clip(p0, 0.0, None)

    # ------------------------------------------------------------------
    # 3. Build JAX arrays for the neural loss function
    # ------------------------------------------------------------------
    # Unified time grid (protein ∪ mRNA)
    if t_rna is not None and len(t_rna) > 0:
        all_times = np.sort(np.union1d(t, t_rna)).astype(np.float64)
    else:
        all_times = np.sort(np.unique(t)).astype(np.float64)

    prot_time_idx = np.searchsorted(all_times, t)
    if t_rna is not None and len(t_rna) > 0:
        mrna_time_idx = np.searchsorted(all_times, t_rna)
    else:
        mrna_time_idx = None

    t0_val = float(all_times[0])
    t1_val = float(all_times[-1])
    t_max = float(all_times[-1]) if float(all_times[-1]) > 0 else 1.0

    # Read graph matrices from the problem object
    Cg = np.asarray(problem.Cg)
    Cl = np.asarray(problem.Cl)
    K_site_kin = np.asarray(problem.K_site_kin)
    R = np.asarray(problem.R)
    L_alpha = np.asarray(problem.L_alpha)
    site_prot_idx = np.asarray(problem.site_prot_idx, dtype=np.int32)
    kin_to_prot_idx = np.asarray(problem.kin_to_prot_idx, dtype=np.int32)
    receptor_mask_prot = np.asarray(problem.receptor_mask_prot)
    receptor_mask_kin = np.asarray(problem.receptor_mask_kin)

    prev_site_idx = compute_prev_site_idx(site_prot_idx, N)

    theta_j = jnp.asarray(theta_best, dtype=jnp.float64)
    y0_j = jnp.asarray(x0, dtype=jnp.float64)
    t_eval_j = jnp.asarray(all_times, dtype=jnp.float64)
    prot_idx_solver_j = jnp.asarray(prot_time_idx, dtype=jnp.int32)
    prev_site_idx_j = jnp.asarray(prev_site_idx, dtype=jnp.int32)

    Cg_j = jnp.asarray(Cg, dtype=jnp.float64)
    Cl_j = jnp.asarray(Cl, dtype=jnp.float64)
    K_sk_j = jnp.asarray(K_site_kin, dtype=jnp.float64)
    R_j = jnp.asarray(R, dtype=jnp.float64)
    La_j = jnp.asarray(L_alpha, dtype=jnp.float64)
    spi_j = jnp.asarray(site_prot_idx, dtype=jnp.int32)
    k2p_j = jnp.asarray(kin_to_prot_idx, dtype=jnp.int32)
    rmp_j = jnp.asarray(receptor_mask_prot, dtype=jnp.float64)
    rmk_j = jnp.asarray(receptor_mask_kin, dtype=jnp.float64)

    P_data_j = jnp.asarray(P_scaled, dtype=jnp.float64)
    A_scaled_j = jnp.asarray(A_scaled, dtype=jnp.float64)
    prot_idx_j = jnp.asarray(prot_idx_for_A, dtype=jnp.int32)
    W_data_j = jnp.asarray(W_data, dtype=jnp.float64)
    W_prot_j = jnp.asarray(W_data_prot, dtype=jnp.float64)

    t_obs_j = jnp.asarray(t_obs, dtype=jnp.float64)
    k_act_init_obs_j = jnp.asarray(k_act_init_vals, dtype=jnp.float64)
    s_prod_init_obs_j = jnp.asarray(s_prod_init_vals, dtype=jnp.float64)

    # mRNA arrays
    has_mrna = (
        t_rna is not None
        and rna_obs_matched is not None
        and len(t_rna) > 0
        and mrna_time_idx is not None
        and rna_model_prot_idx is not None
        and len(rna_model_prot_idx) > 0
    )

    if has_mrna:
        rna_j = jnp.asarray(rna_obs_matched, dtype=jnp.float64)
        mrna_idx_j = jnp.asarray(mrna_time_idx, dtype=jnp.int32)
        rna_prot_idx_j = jnp.asarray(rna_model_prot_idx, dtype=jnp.int32)
        n_matched = len(rna_model_prot_idx)
        T_rna = len(t_rna)
        W_rna_base = (
            np.asarray(W_data_mrna_matched, dtype=np.float64)
            if W_data_mrna_matched is not None
            else np.ones((n_matched, T_rna), dtype=np.float64)
        )
        W_rna_j = jnp.asarray(W_rna_base, dtype=jnp.float64)
    else:
        rna_j = mrna_idx_j = rna_prot_idx_j = W_rna_j = None

    # ------------------------------------------------------------------
    # 4. Build NeuralRateGenerator
    # ------------------------------------------------------------------
    key = jax.random.PRNGKey(int(neural_cfg.seed))
    neural_model = NeuralRateGenerator(
        K=K,
        width=int(neural_cfg.width),
        depth=int(neural_cfg.depth),
        key=key,
    )
    logger.info("[neural_ode] NeuralRateGenerator created (K=%d, width=%d, depth=%d).", K, neural_cfg.width, neural_cfg.depth)

    # Partition into trainable array leaves and static (non-array) parts
    params, static = eqx.partition(neural_model, eqx.is_array)

    # ------------------------------------------------------------------
    # 5. Build the JIT-compiled neural loss function
    # ------------------------------------------------------------------
    neural_loss_fn = _make_neural_loss_fn(
        K=K, M=M, N=N,
        mechanism=mechanism,
        theta_j=theta_j,
        y0_j=y0_j,
        t0_val=t0_val,
        t1_val=t1_val,
        t_eval_j=t_eval_j,
        prot_idx_solver_j=prot_idx_solver_j,
        prev_site_idx_j=prev_site_idx_j,
        Cg_j=Cg_j, Cl_j=Cl_j, K_sk_j=K_sk_j, R_j=R_j, La_j=La_j,
        spi_j=spi_j, k2p_j=k2p_j, rmp_j=rmp_j, rmk_j=rmk_j,
        P_data_j=P_data_j,
        A_scaled_j=A_scaled_j,
        prot_idx_j=prot_idx_j,
        W_data_j=W_data_j,
        W_prot_j=W_prot_j,
        t_obs_j=t_obs_j,
        k_act_init_obs_j=k_act_init_obs_j,
        s_prod_init_obs_j=s_prod_init_obs_j,
        t_max=t_max,
        has_mrna=has_mrna,
        rna_j=rna_j,
        mrna_idx_j=mrna_idx_j,
        rna_prot_idx_j=rna_prot_idx_j,
        W_rna_j=W_rna_j,
        w_phospho=float(neural_cfg.data_weight_phospho),
        w_abundance=float(neural_cfg.data_weight_abundance),
        w_mrna=float(neural_cfg.data_weight_mrna),
        prior_weight_k_act=float(neural_cfg.prior_weight_k_act),
        prior_weight_s_prod=float(neural_cfg.prior_weight_s_prod),
        rtol=float(neural_cfg.rtol),
        atol=float(neural_cfg.atol),
        dt0=float(neural_cfg.dt0),
        max_steps=int(neural_cfg.max_steps),
        rna_relax=float(rna_relax),
        abundance_max=float(abundance_max),
        k_act_init_fn=k_act_fn,
        s_prod_init_fn=s_prod_fn,
        static=static,
    )

    # ------------------------------------------------------------------
    # 6. Compute initial loss before training
    # ------------------------------------------------------------------
    logger.info("[neural_ode] Compiling neural loss function (first JIT call)...")
    loss_init, (f_p0, f_a0, f_r0, f_k0, f_s0) = neural_loss_fn(params, None)
    logger.info(
        "[neural_ode] Initial loss: total=%.4g | phospho=%.4g | abund=%.4g "
        "| mrna=%.4g | k_prior=%.4g | s_prior=%.4g",
        float(loss_init), float(f_p0), float(f_a0), float(f_r0),
        float(f_k0), float(f_s0),
    )

    # ------------------------------------------------------------------
    # 7. Train with Optimistix GradientDescent
    # ------------------------------------------------------------------
    logger.info("[neural_ode] Training neural rate generator for %d steps...", neural_cfg.steps)

    lr = float(neural_cfg.learning_rate)
    solver = optx.GradientDescent(
        learning_rate=lr,
        rtol=1e-100,  # Very tight — rely on max_steps for termination
        atol=1e-100,
    )

    train_result = optx.minimise(
        neural_loss_fn,
        solver,
        params,
        args=None,
        has_aux=True,
        max_steps=int(neural_cfg.steps),
        throw=False,
    )
    params_opt = train_result.value

    # ------------------------------------------------------------------
    # 8. Compute final loss after training
    # ------------------------------------------------------------------
    loss_final, (f_p_fin, f_a_fin, f_r_fin, f_k_fin, f_s_fin) = neural_loss_fn(
        params_opt, None
    )
    logger.info(
        "[neural_ode] Final loss:   total=%.4g | phospho=%.4g | abund=%.4g "
        "| mrna=%.4g | k_prior=%.4g | s_prior=%.4g",
        float(loss_final), float(f_p_fin), float(f_a_fin), float(f_r_fin),
        float(f_k_fin), float(f_s_fin),
    )

    # Reconstruct optimised model
    neural_model_opt = eqx.combine(params_opt, static)

    # ------------------------------------------------------------------
    # 9. Evaluate learned and prior rates at observed time points
    # ------------------------------------------------------------------
    T_obs = t_obs_j.shape[0]
    t_norms_obs = np.clip(t_obs / t_max, 0.0, 1.0)[:, None]  # (T_obs, 1)
    k_priors_obs_np = k_act_init_vals.T  # (T_obs, K)
    s_priors_obs_np = s_prod_init_vals.T  # (T_obs, K)
    features_obs_np = np.concatenate(
        [t_norms_obs, k_priors_obs_np, s_priors_obs_np], axis=1
    )  # (T_obs, 1+K+K)
    features_obs_j = jnp.asarray(features_obs_np, dtype=jnp.float64)

    # Evaluate learned rates
    k_hats_obs_j, s_hats_obs_j = jax.vmap(neural_model_opt)(features_obs_j)
    k_hats_obs = np.asarray(k_hats_obs_j)  # (T_obs, K)
    s_hats_obs = np.asarray(s_hats_obs_j)  # (T_obs, K)

    # ------------------------------------------------------------------
    # 10. Save neural_latent_rates.tsv and .npz
    # ------------------------------------------------------------------
    rate_rows = []
    for ti_idx, t_val in enumerate(t_obs):
        for p_idx, prot in enumerate(proteins):
            rate_rows.append({
                "rate_type": "k_act",
                "entity": prot,
                "time": float(t_val),
                "mechanistic_prior": float(k_act_init_vals[p_idx, ti_idx]),
                "neural_learned": float(k_hats_obs[ti_idx, p_idx]),
                "difference": float(k_hats_obs[ti_idx, p_idx] - k_act_init_vals[p_idx, ti_idx]),
            })
            rate_rows.append({
                "rate_type": "s_prod",
                "entity": prot,
                "time": float(t_val),
                "mechanistic_prior": float(s_prod_init_vals[p_idx, ti_idx]),
                "neural_learned": float(s_hats_obs[ti_idx, p_idx]),
                "difference": float(s_hats_obs[ti_idx, p_idx] - s_prod_init_vals[p_idx, ti_idx]),
            })

    import pandas as pd  # noqa: PLC0415

    df_rates = pd.DataFrame(rate_rows)
    rates_tsv = os.path.join(neural_outdir, "neural_latent_rates.tsv")
    df_rates.to_csv(rates_tsv, sep="\t", index=False)
    logger.info("[neural_ode] Saved %s", rates_tsv)

    np.savez(
        os.path.join(neural_outdir, "neural_latent_rates.npz"),
        t_obs=t_obs,
        proteins=np.array(proteins),
        k_act_mechanistic=k_act_init_vals,
        s_prod_mechanistic=s_prod_init_vals,
        k_act_neural=k_hats_obs.T,  # (K, T_obs) for consistency
        s_prod_neural=s_hats_obs.T,
    )

    # ------------------------------------------------------------------
    # 11. Neural fit timeseries at observed time points
    # ------------------------------------------------------------------
    sim_obs = _neural_simulate_dense(
        model=neural_model_opt,
        K=K, M=M, N=N,
        mechanism=mechanism,
        theta_j=theta_j,
        y0_j=y0_j,
        t0_val=t0_val,
        t_dense=t_obs,
        prev_site_idx_j=prev_site_idx_j,
        Cg_j=Cg_j, Cl_j=Cl_j, K_sk_j=K_sk_j, R_j=R_j, La_j=La_j,
        spi_j=spi_j, k2p_j=k2p_j, rmp_j=rmp_j, rmk_j=rmk_j,
        t_max=t_max,
        k_act_init_fn=k_act_fn,
        s_prod_init_fn=s_prod_fn,
        rtol=float(neural_cfg.rtol),
        atol=float(neural_cfg.atol),
        dt0=float(neural_cfg.dt0),
        max_steps=int(neural_cfg.max_steps),
        rna_relax=float(rna_relax),
        abundance_max=float(abundance_max),
    )

    ts_rows = []
    for ti_idx, t_val in enumerate(t_obs):
        for s_idx, site in enumerate(sites):
            ts_rows.append({
                "time": float(t_val),
                "entity_type": "phosphosite",
                "entity": site,
                "value_neural": float(sim_obs["P_sim"][s_idx, ti_idx]),
                "value_observed": float(P_scaled[s_idx, ti_idx]) if ti_idx < P_scaled.shape[1] else float("nan"),
            })
        for p_idx, prot in enumerate(proteins):
            ts_rows.append({
                "time": float(t_val),
                "entity_type": "abundance",
                "entity": prot,
                "value_neural": float(sim_obs["A_sim"][p_idx, ti_idx]),
                "value_observed": float("nan"),
            })

    df_ts = pd.DataFrame(ts_rows)
    ts_path = os.path.join(neural_outdir, "neural_fit_timeseries.tsv")
    df_ts.to_csv(ts_path, sep="\t", index=False)
    logger.info("[neural_ode] Saved %s", ts_path)

    # ------------------------------------------------------------------
    # 12. Dense timeseries (for dashboard visualisation)
    # ------------------------------------------------------------------
    if getattr(neural_cfg, "save_dense", True):
        n_dense = int(getattr(neural_cfg, "dense_n_points", 200))
        t_dense_arr = np.linspace(float(t_obs[0]), float(t_obs[-1]), n_dense)
        sim_dense = _neural_simulate_dense(
            model=neural_model_opt,
            K=K, M=M, N=N,
            mechanism=mechanism,
            theta_j=theta_j,
            y0_j=y0_j,
            t0_val=t0_val,
            t_dense=t_dense_arr,
            prev_site_idx_j=prev_site_idx_j,
            Cg_j=Cg_j, Cl_j=Cl_j, K_sk_j=K_sk_j, R_j=R_j, La_j=La_j,
            spi_j=spi_j, k2p_j=k2p_j, rmp_j=rmp_j, rmk_j=rmk_j,
            t_max=t_max,
            k_act_init_fn=k_act_fn,
            s_prod_init_fn=s_prod_fn,
            rtol=float(neural_cfg.rtol),
            atol=float(neural_cfg.atol),
            dt0=float(neural_cfg.dt0),
            max_steps=int(neural_cfg.max_steps),
            rna_relax=float(rna_relax),
            abundance_max=float(abundance_max),
        )

        dense_rows = []
        for ti_idx, t_val in enumerate(t_dense_arr):
            for s_idx, site in enumerate(sites):
                dense_rows.append({
                    "time": float(t_val),
                    "entity_type": "phosphosite",
                    "entity": site,
                    "value": float(sim_dense["P_sim"][s_idx, ti_idx]),
                    "series_type": "neural_refined_dense",
                })
            for p_idx, prot in enumerate(proteins):
                dense_rows.append({
                    "time": float(t_val),
                    "entity_type": "abundance",
                    "entity": prot,
                    "value": float(sim_dense["A_sim"][p_idx, ti_idx]),
                    "series_type": "neural_refined_dense",
                })

        df_dense = pd.DataFrame(dense_rows)
        dense_path = os.path.join(neural_outdir, "neural_fit_timeseries_dense.tsv")
        df_dense.to_csv(dense_path, sep="\t", index=False)
        logger.info("[neural_ode] Saved %s", dense_path)

    # ------------------------------------------------------------------
    # 13. Training losses TSV
    # ------------------------------------------------------------------
    df_losses = pd.DataFrame([
        {
            "step": 0,
            "neural_loss_total": float(loss_init),
            "neural_loss_phospho": float(f_p0),
            "neural_loss_abundance": float(f_a0),
            "neural_loss_mrna": float(f_r0),
            "neural_loss_k_act_prior": float(f_k0),
            "neural_loss_s_prod_prior": float(f_s0),
        },
        {
            "step": int(neural_cfg.steps),
            "neural_loss_total": float(loss_final),
            "neural_loss_phospho": float(f_p_fin),
            "neural_loss_abundance": float(f_a_fin),
            "neural_loss_mrna": float(f_r_fin),
            "neural_loss_k_act_prior": float(f_k_fin),
            "neural_loss_s_prod_prior": float(f_s_fin),
        },
    ])
    losses_path = os.path.join(neural_outdir, "neural_training_losses.tsv")
    df_losses.to_csv(losses_path, sep="\t", index=False)
    logger.info("[neural_ode] Saved %s", losses_path)

    # ------------------------------------------------------------------
    # 14. Metadata JSON
    # ------------------------------------------------------------------
    metadata = {
        "enabled": True,
        "width": int(neural_cfg.width),
        "depth": int(neural_cfg.depth),
        "steps": int(neural_cfg.steps),
        "seed": int(neural_cfg.seed),
        "optimizer": "optimistix.GradientDescent",
        "learning_rate": float(neural_cfg.learning_rate),
        "rtol": float(neural_cfg.rtol),
        "atol": float(neural_cfg.atol),
        "dt0": float(neural_cfg.dt0),
        "max_steps": int(neural_cfg.max_steps),
        "prior_weights": {
            "k_act": float(neural_cfg.prior_weight_k_act),
            "s_prod": float(neural_cfg.prior_weight_s_prod),
        },
        "data_weights": {
            "phospho": float(neural_cfg.data_weight_phospho),
            "abundance": float(neural_cfg.data_weight_abundance),
            "mrna": float(neural_cfg.data_weight_mrna),
        },
        "theta_fixed": True,
        "state_variables": ["R", "S", "A", "Kdyn", "P"],
        "initial_loss": float(loss_init),
        "final_loss": float(loss_final),
        "note": (
            "Post-fit latent-rate refinement. theta_best is held fixed. "
            "Only neural rate generator parameters are optimized."
        ),
    }
    meta_path = os.path.join(neural_outdir, "neural_metadata.json")
    with open(meta_path, "w") as fh:
        json.dump(metadata, fh, indent=2)
    logger.info("[neural_ode] Saved %s", meta_path)

    logger.success(
        "[neural_ode] Neural latent-rate refinement complete. "
        "Outputs saved to %s/",
        neural_outdir,
    )
