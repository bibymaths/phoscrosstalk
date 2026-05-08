# PINN / Universal ODE Mode

!!! warning "PINN mode bypasses mechanistic multistart and post-fit neuralODE."
    PINN mode is an **alternative run mode**, not an additional post-processing step.
    When `pinn.enabled = true`, the mechanistic multistart optimisation and the
    post-fit `neuralODE` workflow are **not run**.

---

## What is PINN mode?

PINN mode implements a **PINN / Universal ODE**-style augmentation of the
mechanistic PhosCrosstalk ODE.  The augmented dynamics are:

$$
\frac{dx}{dt} = f_{\text{mechanistic}}(x, t;\, \theta) + f_{\text{PINN}}(x, t;\, \phi)
$$

where:

- $f_{\text{mechanistic}}$ is the original mechanistic RHS (unchanged, from `mechanisms.py`).
- $f_{\text{PINN}}$ is a small neural network correction parameterised by $\phi$.
- $\theta$ are the mechanistic rate parameters (optimised jointly with $\phi$).

The mechanistic model remains the **primary scientific model**.
The neural term is a constrained residual correction for:

- missing regulatory edges
- unmeasured regulators
- missing nonlinear interactions
- topology limitations in the mechanistic graph

Large PINN residuals are **hypotheses** about missing biology, not proof.

---

## How does PINN mode differ from post-fit `neuralODE.py`?

| Feature | PINN mode (`phoscrosstalk.pinn`) | Post-fit neuralODE (`neuralODE.py`) |
|---------|----------------------------------|--------------------------------------|
| Purpose | Joint optimisation: theta + neural correction | Post-fit refinement of latent rate functions |
| Neural term | Additive correction to full ODE RHS | Neural latent rate generators (k_act, s_prod) |
| Timing | Replaces mechanistic multistart | Runs after mechanistic fit |
| Theta | Jointly optimised from scratch | Fixed at mechanistic theta_best |
| Multistart | No (single start only) | No |
| Config section | `[pinn]` | `[neural_ode]` |
| When to use | When mechanistic topology is suspected inadequate | When latent inputs need refinement |

---

## When to use PINN mode

- You suspect the mechanistic graph is missing important regulatory edges.
- You want to explore what unmeasured regulators might contribute.
- You want a data-driven correction to a known incomplete mechanistic model.
- You are **not** satisfied with the mechanistic fit and want a more flexible model.

## When NOT to use PINN mode

- You want interpretable mechanistic parameters only (use normal mode).
- You have already obtained a good mechanistic fit and want latent rate refinement
  (use `[neural_ode]` post-fit mode instead).
- You want to run multistart for robust parameter estimation (PINN mode is single-start only).

---

## Configuration

Add a `[pinn]` section to your `config.toml`:

```toml
[pinn]
enabled = true   # set to false to disable (default)

# Architecture
width_size = 64
depth      = 2
activation = "tanh"   # "tanh" | "relu" | "gelu" | "silu" | "softplus" | "elu"

# Regularisation
lambda_pinn = 0.1
regularize  = "residual_l2"   # "residual_l2" | "param_l2"

# Optimisation
max_steps      = 500
learning_rate  = 1e-3
rtol           = 1e-8
atol           = 1e-8
seed           = 0
print_every    = 50
grad_clip      = 1.0
optimizer      = "adam"

# Runtime
use_max_machine_threads = true
```

### Configuration reference

| Key | Default | Description |
|-----|---------|-------------|
| `enabled` | `false` | Enable PINN mode (bypasses mechanistic multistart and neuralODE) |
| `width_size` | `64` | Hidden layer width of the PINN MLP |
| `depth` | `2` | Number of hidden layers (≥ 1) |
| `activation` | `"tanh"` | Activation function |
| `lambda_pinn` | `0.1` | Weight for PINN regularisation term (≥ 0) |
| `regularize` | `"residual_l2"` | Regularisation strategy |
| `max_steps` | `500` | Number of Optax training steps |
| `learning_rate` | `1e-3` | Adam learning rate |
| `rtol` | `1e-8` | ODE solver relative tolerance during PINN training |
| `atol` | `1e-8` | ODE solver absolute tolerance during PINN training |
| `seed` | `0` | Random seed for PINN weight initialisation |
| `print_every` | `50` | Log progress every N steps |
| `grad_clip` | `1.0` | Global gradient norm clip (0 to disable) |
| `optimizer` | `"adam"` | Optax optimiser name |
| `use_max_machine_threads` | `true` | Use maximum available CPU budget for the single run |

---

## Output files

PINN mode writes the following files to `output_dir`:

| File | Description |
|------|-------------|
| `pinn_metadata.json` | Run metadata: mode, dimensions, config snapshot, final losses |
| `theta_opt.npy` | Fitted mechanistic parameter vector θ |
| `pinn_params/pinn_params.npz` | Fitted PINN neural parameters φ |
| `pinn_loss_components.tsv` | Final loss components: f1, f2, f3, f4, f_pinn_reg |
| `pinn_fit_timeseries.tsv` | Fitted trajectories (phosphosite and protein abundance) |
| `pinn_residuals.tsv` | Neural correction f_pinn(x, t) per state per time |
| `pinn_residual_summary.tsv` | Mean/max abs residual per (state_block, entity) |
| `pinn_residuals.png` | Heatmap of \|f_pinn(x, t)\| (state × time) |
| `pinn_loss_trajectory.png` | Training loss history per component |
| `pinn_fit_comparison.png` | Fitted vs observed phosphosite trajectories |
| `pareto_front.npz` | Dashboard-compatible stub with single-run loss |
| `pareto_stats.tsv` | Dashboard-compatible loss statistics stub |

### Loss components

| Component | Meaning |
|-----------|---------|
| `f1` | Phosphosite trajectory loss (weighted MSE) |
| `f2` | Protein abundance trajectory loss |
| `f3` | Mechanistic regularisation (L2 + Laplacian) |
| `f4` | mRNA / R_rna loss (0 when no RNA data) |
| `f_pinn_reg` | PINN neural regularisation term |

---

## Residual heatmap interpretation

The `pinn_residuals.png` heatmap shows rows = ODE state dimensions, columns = time.

State labels follow the format:

```
R_rna:<protein>
S:<protein>
A:<protein>
Kdyn:<kinase>
p:<site>
```

**Interpretation:**

- **Large `|f_pinn|`** → the neural term is contributing significantly to that state.
  This is a candidate missing mechanism (unmeasured regulator, missing edge, nonlinear interaction).
- **Small `|f_pinn|`** → the mechanistic RHS explains that state well.
  The neural correction is near zero as expected.

The `pinn_residual_summary.tsv` aggregates mean and max absolute residual per
(state_block, entity) for a biological overview.

---

## Regularisation strategies

### `residual_l2` (default)

Regularises by the mean L2 norm of the neural correction over the ODE trajectory:

$$
f_{\text{PINN reg}} = \lambda_{\text{pinn}} \cdot \text{mean}_t \|f_{\text{PINN}}(x(t), t)\|^2
$$

This ties the regularisation to the actual neural contribution.
Higher `lambda_pinn` constrains the neural term to stay small.

### `param_l2`

Regularises by the L2 norm of the PINN network parameters:

$$
f_{\text{PINN reg}} = \lambda_{\text{pinn}} \cdot \frac{1}{|\phi|}\sum_i \phi_i^2
$$

Cheaper (no additional trajectory evaluation needed) but less directly interpretable.

---

## Runtime behaviour

PINN mode uses a **single training run** — no multistart.

The existing `runtime_env.py` infrastructure is used to allocate the maximum
available CPU budget to this single run.  No extra workers are spawned.
`ProcessPoolExecutor` is not used in PINN mode.

If you need multistart, use normal mechanistic mode instead.

---

## Why multistart is disabled in PINN mode

Multistart in the mechanistic pipeline is used to escape local optima in a
high-dimensional non-convex landscape of pure mechanistic parameters.

PINN mode adds a neural network with many additional parameters.
Running multistart with a joint mechanistic+neural landscape would be:

1. Computationally prohibitive (each start requires training a neural network).
2. Likely unhelpful: the neural degrees of freedom often compensate for poor
   mechanistic initialisation rather than finding a better mechanistic solution.

Instead, PINN mode initialises θ at the centre of the parameter bounds and uses
a single-start Optax Adam training loop.

If you want a good mechanistic prior for PINN mode, run mechanistic fitting first
(normal mode), then use the best θ to initialise a subsequent PINN run.

---

## Backend limitations

PINN mode uses Optax Adam by default.
The `[pinn] optimizer` key accepts `"adam"` only in the current implementation.
The Optimistix least-squares backends require a flat residual-vector interface
that is not directly compatible with the (theta, pinn_model) PyTree trainable.

---

## API

::: phoscrosstalk.pinn.runner.run_pinn_pipeline

::: phoscrosstalk.pinn.model.PINNAugmentation

::: phoscrosstalk.pinn.rhs.make_combined_rhs

::: phoscrosstalk.pinn.loss.make_pinn_loss_fn
