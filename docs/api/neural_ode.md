# Neural ODE

## Biological Role

The `neural_ode` module provides a post-fit refinement stage that replaces the analytically derived `k_act(t)` and `s_prod(t)` closures with learned neural network surrogates. This allows the model to capture rate dynamics that cannot be fully explained by observed TF/mRNA data alone, while remaining anchored to the mechanistic ODE structure.

## Implementation Overview

**Architecture:**

- `LatentRateMLP`: single Equinox MLP with `softplus` activations throughout. Input: `(t_norm, k_act_prior(t), s_prod_prior(t))` — size `1 + K + K`. Output: `K`-dimensional rate vector via `softplus(raw * out_scale) + eps`, ensuring positivity.
- `NeuralRateGenerator`: pairs two `LatentRateMLP` instances (one for `k_act`, one for `s_prod`).
- `JointNeuralMechanisticModel`: wraps `NeuralRateGenerator` together with a trainable copy of `theta` for fully joint optimisation.

**Training loop (`_train_with_optax`):** Python `for` loop over epochs; each step is JIT-compiled with `jax.jit`. Gradient clipping (`optax.clip_by_global_norm`) is applied before the update. Loss is logged every `log_every` epochs.

**Supported optimisers** (selected via `[neural_ode] optimiser`): `adabelief` (default), `adamw`, `adam`, `adan`, `radam`, `nadam`, `nadamw`, `lion`, `sgd`.

**Two training modes:**

1. **Frozen-theta**: only neural parameters are trained; `theta` is fixed at the mechanistic optimum.
2. **Joint**: `theta` and neural parameters are trained together via `JointNeuralMechanisticModel` with an additional `prior_weight_theta` penalty term.

**Neural loss:**
```
f_neural = w_phospho*MSE(P) + w_abundance*MSE(A) + w_mrna*MSE(R)
         + prior_weight_k_act  * ||k_hat - k_act_prior||²
         + prior_weight_s_prod * ||s_hat - s_prod_prior||²
```
Joint mode adds `prior_weight_theta * ||theta - theta_mech||²` and `prior_weight_traj`.

**`save_neural_ode_plots`** writes three PNG files to `output_dir`:
1. Training loss curve
2. Neural vs. prior `k_act(t)` comparison per protein
3. Neural vs. prior `s_prod(t)` comparison per protein

**`run_neural_latent_rate_refinement`** returns a 5-tuple:
```
(ts, ys, model_opt, loss_history, time_history)
```
where `ts` is the evaluation time array, `ys` is a dict with `"P_sim"` and `"A_sim"` keys,
`model_opt` is the trained Equinox model, `loss_history` is a list of per-step loss values,
and `time_history` is a list of per-step wall-clock times (seconds; empty for Optimistix path).

## Configuration Reference

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `[neural_ode] enabled` | `bool` | `false` | Run neural refinement after mechanistic fit |
| `[neural_ode] width` | `int` | `32` | MLP hidden layer width |
| `[neural_ode] depth` | `int` | `2` | MLP depth (number of hidden layers) |
| `[neural_ode] steps` | `int` | `500` | Number of GradientDescent / Optax training steps |
| `[neural_ode] learning_rate` | `float` | `1e-3` | Gradient-descent learning rate |
| `[neural_ode] prior_weight_k_act` | `float` | `1.0` | Regularisation weight toward mechanistic `k_act` prior |
| `[neural_ode] prior_weight_s_prod` | `float` | `1.0` | Regularisation weight toward mechanistic `s_prod` prior |
| `[neural_ode] data_weight_phospho` | `float` | `1.0` | Phosphosite data loss weight |
| `[neural_ode] data_weight_abundance` | `float` | `1.0` | Protein abundance data loss weight |
| `[neural_ode] data_weight_mrna` | `float` | `1.0` | mRNA data loss weight |
| `[neural_ode] seed` | `int` | `0` | JAX random seed for neural parameter initialisation |
| `[neural_ode] rtol` | `float` | `1e-5` | ODE relative tolerance during neural training |
| `[neural_ode] atol` | `float` | `1e-7` | ODE absolute tolerance during neural training |
| `[neural_ode] dt0` | `float` | `0.01` | Initial ODE step size during neural training |
| `[neural_ode] max_steps` | `int` | `65536` | Maximum ODE internal steps during neural training |
| `[neural_ode] save_dense` | `bool` | `true` | Write dense neural-refined timeseries for visualisation |
| `[neural_ode] dense_n_points` | `int` | `200` | Number of time points in the dense neural output grid |

## API Reference

```python
class LatentRateMLP(eqx.Module):
    """
    Single Equinox MLP producing a K-dimensional positive rate vector.

    Parameters
    ----------
    K     : int   – number of model proteins (output size)
    width : int   – hidden layer width
    depth : int   – number of hidden layers
    key   : jax.random.PRNGKey

    Forward call: __call__(features) -> (k_hat (K,), s_hat (K,))
      where features shape = (1 + K + K,) = (t_norm, k_act_prior, s_prod_prior)
    """

class NeuralRateGenerator(eqx.Module):
    """
    Container for k_act_net and s_prod_net (both LatentRateMLP instances).

    __call__(features) -> (k_hat (K,), s_hat (K,))
    """

class JointNeuralMechanisticModel(eqx.Module):
    """
    Neural generator + mechanistic theta for joint optimisation.

    Attributes
    ----------
    neural : NeuralRateGenerator
    theta  : jax.Array  (float64, shape = (2K+2+3M+N+4,))
    """

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
    t_rna: np.ndarray | None,
    rna_obs_matched: np.ndarray | None,
    rna_model_prot_idx: np.ndarray | None,
    W_data_mrna_matched: np.ndarray | None,
    outdir: str,
    neural_cfg,
    mechanism: str = "dist",
    rna_relax: float = 0.1,
    abundance_max: float = 5.0,
    R_data0: np.ndarray | None = None,
    jaxpr_out_dir=None,
) -> tuple:
    """
    Run post-fit neural latent-rate refinement.

    Returns
    -------
    (ts, ys, model_opt, loss_history, time_history)
      ts            : np.ndarray – time points used for evaluation
      ys            : dict with keys "P_sim" (N,T) and "A_sim" (K,T)
      model_opt     : trained Equinox neural model
      loss_history  : list[float] – total loss per logged step
      time_history  : list[float] – per-step wall-clock time (s)
    """

def save_neural_ode_plots(
    outdir: str,
    ts: np.ndarray,
    ys: dict,
    model,
    loss_history: list,
    time_history: list,
) -> None:
    """
    Write three diagnostic PNG files:
    - neural_ode_training_loss.png  – loss curve (log-y axis)
    - neural_ode_step_time.png      – per-step computation time (ms)
    - neural_ode_trajectories.png   – real vs model P_sim/A_sim trajectories
    """
```

## Known Limitations

- The `_train_with_optax` Python loop is not JIT-compiled at the epoch level; only individual gradient steps are JIT-compiled. Large `n_epochs` values with frequent logging will have Python-loop overhead.
- `joint_training=true` significantly increases memory usage because both the ODE adjoint and the neural parameter gradients must be held simultaneously.
- The `"adan"` and `"lion"` optimisers require `optax >= 0.1.7`; earlier versions will raise `AttributeError`.
- Neural refinement output closures (`k_act_neural_fn`, `s_prod_neural_fn`) are Equinox modules, not plain Python callables — they must be passed through `eqx.filter_jit` boundaries correctly.
