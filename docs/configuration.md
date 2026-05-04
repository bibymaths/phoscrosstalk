# Configuration

PhosCrosstalk is configured via `config.toml`. All values have built-in defaults
and can be overridden by CLI flags.

## Minimal `config.toml`

```toml
[paths]
data_dir   = "data_timeseries"
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

| Key          | Default             | Description                      |
|--------------|---------------------|----------------------------------|
| `data_dir`   | `"data_timeseries"` | Root data directory              |
| `output_dir` | `"results"`         | Default output directory         |

### `[model]`

| Key            | Default    | Description                                      |
|----------------|------------|--------------------------------------------------|
| `mechanism`    | `"dist"`   | Phosphorylation mechanism (`dist`/`seq`/`rand`)  |
| `scale_mode`   | `"none"`   | Data scaling (`none`/`minmax`/`log-minmax`)      |
| `length_scale` | `50.0`     | Decay length for local sequence-based coupling   |
| `weight_scheme`| `"uniform"`| Data weighting scheme                            |

### `[optimisation]`

| Key          | Default  | Description                                    |
|--------------|----------|------------------------------------------------|
| `n_starts`   | `3`      | Number of multi-start initialisations          |
| `max_steps`  | `500`    | Max gradient steps per start                   |
| `loss_type`  | `"mse"`  | Loss metric type                               |
| `lambda_net` | `0.0001` | Network Laplacian regularization weight        |
| `reg_lambda` | `0.0001` | L2 parameter regularization weight             |

### `[loss_weights]`

| Key         | Default | Description                                    |
|-------------|---------|------------------------------------------------|
| `phospho`   | `1.0`   | Weight on phosphosite occupancy loss           |
| `abundance` | `1.0`   | Weight on protein abundance loss               |
| `mrna`      | `1.0`   | Weight on mRNA (R(t)) loss                     |
| `reg`       | `1.0`   | Weight on regularization loss                  |

### `[solver]`

| Key        | Default  | Description                              |
|------------|----------|------------------------------------------|
| `rtol`     | `1e-6`   | Relative tolerance for Diffrax solver    |
| `atol`     | `1e-9`   | Absolute tolerance for Diffrax solver    |
| `max_steps`| `16384`  | Maximum Diffrax integration steps        |

### `[time]`

| Key                 | Default                           | Description                       |
|---------------------|-----------------------------------|-----------------------------------|
| `mrna_time_points`  | `[4,8,15,30,60,120,240,480,960]`  | mRNA time points (minutes)        |
| `interpolation`     | `"piecewise_constant"`            | Derived rate interpolation mode   |

### `[derived_rates]`

| Key          | Default      | Description                              |
|--------------|--------------|------------------------------------------|
| `s_prod_fn`  | `"softplus"` | Activation function for `s_prod(t)`     |

!!! tip "CLI overrides"
    `--mechanism`, `--n-starts`, `--max-steps`, and `--outdir` override the
    corresponding TOML settings when specified on the command line.
