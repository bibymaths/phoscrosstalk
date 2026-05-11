# Configuration

PhosCrosstalk is configured via `config.toml`. All values have built-in defaults
and can be overridden by CLI flags.

## Minimal `config.toml`

```toml
[paths]
data       = "data_timeseries/input1.csv"
ptm_intra  = "data_curated/processed/ptm_intra.db"
ptm_inter  = "data_curated/processed/ptm_inter.db"
output_dir = "results"

[model]
mechanism    = "dist"        # "dist" | "seq" | "rand"
scale_mode   = "none"        # "none" | "minmax" | "log-minmax"
length_scale = 50.0
weight_scheme = "uniform"

[optimisation]
n_starts  = 3
max_steps = 500
loss_type = "mse"
lambda_net = 0.0001
reg_lambda = 0.0001

[loss_weights]
phospho   = 1.0
abundance = 1.0
mrna      = 1.0
reg       = 1.0

[solver]
rtol = 1e-6
atol = 1e-9
max_steps = 16384

[time]
mrna_time_points = [4, 8, 15, 30, 60, 120, 240, 480, 960]
interpolation    = "piecewise_constant"

[derived_rates]
s_prod_fn = "softplus"
```

## Section reference

### `[paths]`

| Key               | Default    | Description                                       |
|-------------------|------------|---------------------------------------------------|
| `data`            | `""`       | Path to phosphosite / protein time-series CSV     |
| `ptm_intra`       | `""`       | PTMcode2 intra-protein crosstalk SQLite DB        |
| `ptm_inter`       | `""`       | PTMcode2 inter-protein crosstalk SQLite DB        |
| `output_dir`      | `"results"`| Output directory                                  |
| `rna_data`        | `""`       | mRNA time-series CSV (optional)                   |
| `tf_net`          | `""`       | TF → mRNA network CSV (optional)                  |
| `kinase_tsv`      | `""`       | Kinase-site prior TSV (optional)                  |
| `kea_ks_table`    | `""`       | KEA kinase-substrate table (optional)             |
| `unified_graph_pkl`| `""`      | Unified kinase graph pickle for network reg       |

### `[model]`

| Key                       | Default    | Description                                      |
|---------------------------|------------|--------------------------------------------------|
| `mechanism`               | `"dist"`   | Phosphorylation mechanism (`dist`/`seq`/`rand`)  |
| `scale_mode`              | `"none"`   | Data scaling (`none`/`minmax`/`log-minmax`)      |
| `length_scale`            | `50.0`     | Decay length for local sequence-based coupling   |
| `weight_scheme`           | `"uniform"`| Data weighting scheme                            |
| `receptors`               | `[]`       | Protein names treated as stimulated receptors    |
| `receptor_kinases`        | `[]`       | Kinase names treated as stimulated receptor kinases |
| `include_tfs_as_proteins` | `false`    | Include TF proteins when RNA/TF network present  |

### `[optimisation]`

| Key                       | Default        | Description                                     |
|---------------------------|----------------|-------------------------------------------------|
| `n_starts`                | `3`            | Number of multi-start initialisations           |
| `max_steps`               | `500`          | Max gradient steps per start                    |
| `ls_solver`               | `"lm"`         | Optimistix LS solver (`lm`/`gauss_newton`/…)    |
| `jac_mode`                | `"fwd"`        | Residual Jacobian mode (`fwd`/`bwd`)            |
| `optx_adjoint`            | `"implicit"`   | Optimistix adjoint (`implicit`/`checkpoint`)    |
| `loss_type`               | `"mse"`        | Loss metric type                                |
| `lambda_net`              | `0.0001`       | Network Laplacian regularization weight         |
| `reg_lambda`              | `0.0001`       | L2 parameter regularization weight              |
| `optimizer_backend`       | `"optimistix"` | Solver backend (see below)                      |
| `optimizer_backend_kwargs`| `{}`           | Backend-specific options (see below)            |

### `[loss_weights]`

| Key         | Default | Description                                    |
|-------------|---------|------------------------------------------------|
| `phospho`   | `1.0`   | Weight on phosphosite relative-signal loss     |
| `abundance` | `1.0`   | Weight on protein abundance loss               |
| `mrna`      | `1.0`   | Weight on mRNA (R(t)) loss                     |
| `reg`       | `1.0`   | Weight on regularization loss                  |

### `[solver]`

| Key        | Default  | Description                              |
|------------|----------|------------------------------------------|
| `rtol`     | `1e-6`   | Relative tolerance for Diffrax solver    |
| `atol`     | `1e-9`   | Absolute tolerance for Diffrax solver    |
| `max_steps`| `16384`  | Maximum Diffrax integration steps        |
| `ode_solver`| `"tsit5"`| Diffrax ODE solver name                 |
| `ode_adjoint`| `"forward"`| Diffrax adjoint method               |

### `[time]`

| Key                 | Default                           | Description                       |
|---------------------|-----------------------------------|-----------------------------------|
| `mrna_time_points`  | `[4,8,15,30,60,120,240,480,960]`  | mRNA time points (minutes)        |
| `interpolation`     | `"piecewise_constant"`            | Derived rate interpolation mode   |

### `[derived_rates]`

| Key          | Default      | Description                              |
|--------------|--------------|------------------------------------------|
| `s_prod_fn`  | `"softplus"` | Activation function for `s_prod(t)`     |

---

## Optimizer backends

The `optimizer_backend` field in `[optimisation]` selects which solver runs the
parameter fitting.  All backends return the same result tuple:
`(theta_opt, total_loss, f1, f2, f3, f4)`.

### `"optimistix"` (default)

Residual least-squares using
[Optimistix](https://github.com/patrick-kidger/optimistix).
Supports Levenberg-Marquardt (`ls_solver = "lm"`), Gauss-Newton,
dogleg, and indirect-LM.  This is the recommended production backend.

```toml
[optimisation]
optimizer_backend = "optimistix"
optimizer_backend_kwargs = {}
```

### `"jaxopt_lbfgsb"`

Bounded scalar minimisation via
[JAXopt](https://github.com/google/jaxopt) `ScipyBoundedMinimize` with SciPy
L-BFGS-B.  Hard box constraints; quasi-Newton convergence.

```toml
[optimisation]
optimizer_backend = "jaxopt_lbfgsb"
optimizer_backend_kwargs = { max_steps = 1000, tol = 1e-6, verbose = false }
```

### `"jaxopt_pgd"`

Projected-gradient descent via JAXopt `ProjectedGradient` with box projection.
Pure-JAX and JIT-able.

```toml
[optimisation]
optimizer_backend = "jaxopt_pgd"
optimizer_backend_kwargs = { max_steps = 3000, tol = 1e-6, stepsize = 1e-3, acceleration = true }
```

### `"scipy_jax"`

BFGS via `jax.scipy.optimize.minimize` with a smooth bound reparameterisation.

```toml
[optimisation]
optimizer_backend = "scipy_jax"
optimizer_backend_kwargs = { max_steps = 1000, gtol = 1e-5, bounds_strategy = "reparameterize" }
```

### JAXopt QP backends (experimental)

`"jaxopt_osqp"`, `"jaxopt_box_osqp"`, and `"jaxopt_eq_qp"` solve a **local
quadratic surrogate** of the nonlinear ODE loss, not the full nonlinear
problem.  They are intended as experimental local-step solvers and are **not
recommended** as drop-in replacements for Optimistix on large ODE models.

#### How they work

1. Compute the gradient `g` and Hessian `H` of the scalar loss at `theta0`
   using `jax.grad` and `jax.hessian`.
2. Build the quadratic surrogate:
   `Q = 0.5*(H+H.T) + ridge*I`, `c = g - Q @ theta0`.
3. Solve the resulting QP.
4. Optionally run a backtracking line search between `theta0` and the QP
   solution on the original nonlinear loss.
5. Clip the result to `[xl, xu]` and recompute the original loss diagnostics.

#### `"jaxopt_osqp"`

Uses JAXopt `OSQP`.  Box bounds are converted to linear inequalities
`[I; -I] @ theta <= [xu; -xl]`.

```toml
[optimisation]
optimizer_backend = "jaxopt_osqp"
optimizer_backend_kwargs = { qp_mode = "local_quadratic", ridge = 1e-6, line_search = true, line_search_steps = 8 }
```

#### `"jaxopt_box_osqp"`

Uses JAXopt `BoxOSQP`.  Box bounds are passed directly as `params_ineq`.

```toml
[optimisation]
optimizer_backend = "jaxopt_box_osqp"
optimizer_backend_kwargs = { qp_mode = "local_quadratic", ridge = 1e-6, line_search = true, line_search_steps = 8 }
```

#### `"jaxopt_eq_qp"`

Uses JAXopt `EqualityConstrainedQP`.  Box bounds are **not** supported
directly; they are enforced by post-solve clipping.  A warning is logged.

```toml
[optimisation]
optimizer_backend = "jaxopt_eq_qp"
optimizer_backend_kwargs = { qp_mode = "local_quadratic", ridge = 1e-6, line_search = true, line_search_steps = 8 }
```

#### QP `optimizer_backend_kwargs` reference

| Key                | Default              | Description                                      |
|--------------------|----------------------|--------------------------------------------------|
| `qp_mode`          | `"local_quadratic"`  | `"local_quadratic"` or `"explicit"`              |
| `ridge`            | `1e-6`               | Diagonal regularisation on Hessian               |
| `line_search`      | `true`               | Backtracking line search on original loss        |
| `line_search_steps`| `8`                  | Number of halving steps in line search           |

For `qp_mode = "explicit"`, supply explicit QP data directly:

```toml
# Not representable in TOML for large arrays; pass via Python API
optimizer_backend_kwargs = { qp_mode = "explicit" }
# Then set params_obj = (Q, c), params_eq = (A, b), params_ineq = (G, h)
# in the Python call to dispatch_optimisation / run_single_optimisation_jaxopt.
```

---

## NeuralODE latent-rate refinement

After the main mechanistic fit, an optional neural latent-rate refinement stage
can be enabled.  It trains a small MLP to refine the latent kinase activation
rates `k_act(t)` and production rates `s_prod(t)` using
[Equinox](https://github.com/patrick-kidger/equinox) and
[Optax](https://github.com/google-deepmind/optax).

```toml
[neural_ode]
enabled        = true
width          = 32
depth          = 2
steps          = 500
learning_rate  = 1e-3
```

See `phoscrosstalk/neural_ode.py` for full option reference.

---

## PINN / Universal ODE architecture

An optional Physics-Informed Neural Network (PINN) / Universal ODE mode
augments the mechanistic ODEs with a neural residual term.  When enabled,
the mechanistic multistart and post-fit NeuralODE steps are bypassed.

```toml
[pinn]
enabled      = true
width_size   = 64
depth        = 2
activation   = "tanh"
lambda_pinn  = 0.1
max_steps    = 500
learning_rate = 1e-3
```

See `phoscrosstalk/pinn/` for full option reference and `docs/pinn.md`.

---

## Optimizer-based uncertainty

After the main fit, optional optimizer-based uncertainty analysis estimates
parameter uncertainty via residual bootstrap refitting or profile-likelihood
identifiability analysis.

```toml
[posterior]
enabled  = true
method   = "bootstrap"   # "bootstrap" | "profile" | "both"
n_bootstrap = 100
```

See `phoscrosstalk/posterior.py` for the full option reference.

---

## Output directory structure

All results are written under `[paths] output_dir`:

```
results/
  theta_best.npy                 # Best-fit parameter vector
  fit_timeseries.tsv             # Long-format fitted trajectories
  fit_timeseries_dense.tsv       # Dense-grid forward simulation
  pareto_diagnostics.json        # Multi-start loss summary
  neural_ode/                    # NeuralODE refinement outputs (if enabled)
  pinn_bundle/                   # PINN model bundle (if enabled)
  posterior/                     # Uncertainty analysis outputs (if enabled)
```

!!! tip "CLI overrides"
    `--mechanism`, `--n-starts`, `--max-steps`, and `--outdir` override the
    corresponding TOML settings when specified on the command line.
