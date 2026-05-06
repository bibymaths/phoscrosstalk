# Config

## Biological Role

The `config` module provides the global configuration loader and the `ModelDims` registry that carries the integer dimensions `K` (proteins), `M` (kinases), and `N` (phosphosites) needed by every other module. It is the first module that must be initialised in any pipeline run.

## Implementation Overview

`load_config` reads a TOML file from disk and deep-merges it with `_DEFAULTS`. Missing keys fall back to defaults silently; no error is raised for a missing file. The merged result is returned as a nested `SimpleNamespace` for attribute-style access.

`validate_config` returns `(errors: list, warnings: list)` — it does **not** raise. The caller decides whether to abort. In the standard entry point, a non-empty `errors` list triggers `SystemExit(1)`.

`ModelDims` is a thread-safe static registry (protected by `threading.Lock`). It must be populated via `ModelDims.set_dims(K, M, N)` before any simulation or optimisation call. Accessing `ModelDims.K`, `ModelDims.M`, or `ModelDims.N` before `set_dims` raises `RuntimeError`.

## Configuration Reference

### `[paths]`
| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `output_dir` | `str` | `"results"` | Root output directory |
| `data_dir` | `str` | `"data"` | Input data directory |
| `log_file` | `str\|null` | `null` | Log file path; null = stdout only |

### `[model]`
| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `mechanism` | `str` | `"dist"` | Phosphorylation mechanism: `dist`, `seq`, `rand` |
| `K` | `int\|null` | `null` | Number of proteins (set programmatically) |
| `M` | `int\|null` | `null` | Number of kinases (set programmatically) |
| `N` | `int\|null` | `null` | Number of phosphosites (set programmatically) |

### `[optimisation]`
| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `n_starts` | `int` | `32` | Multi-start runs |
| `max_steps` | `int` | `500` | Max iterations per run |
| `ls_solver` | `str` | `"lm"` | Solver: `lm`, `indirect_lm`, `dogleg`, `gauss_newton` |
| `optx_adjoint` | `str` | `"implicit"` | Adjoint: `implicit`, `checkpoint` |
| `jac_mode` | `str` | `"fwd"` | Jacobian: `fwd`, `bwd` |
| `atol` | `float` | `1e-5` | Convergence tolerance |
| `rtol` | `float` | `1e-5` | Relative convergence tolerance |

### `[loss_weights]`
| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `w_phospho` | `float` | `1.0` | Phosphosite MSLE weight |
| `w_abundance` | `float` | `0.5` | Abundance MSLE weight |
| `w_mrna` | `float` | `0.1` | mRNA MSE weight |
| `reg_lambda` | `float` | `1e-3` | L2 regularisation strength |
| `lambda_net` | `float` | `1e-4` | Network Laplacian regularisation |

### `[solver]`
| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `ode_solver` | `str` | `"tsit5"` | Diffrax solver name |
| `ode_adjoint` | `str` | `"forward"` | Adjoint method |
| `rtol` | `float` | `1e-4` | ODE relative tolerance |
| `atol` | `float` | `1e-5` | ODE absolute tolerance |
| `dt0` | `float` | `0.1` | Initial step size |
| `max_steps` | `int` | `16384` | Maximum ODE steps |

### `[time]`
| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `t0` | `float` | `0.0` | Simulation start |
| `t1` | `float` | `120.0` | Simulation end |
| `t_eval` | `list` | `[0,5,10,20,30,60,120]` | Observation time points |
| `interpolation` | `str` | `"piecewise_constant"` | Rate closure interpolation |

### `[derived_rates]`
| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `s_prod_fn` | `str` | `"softplus"` | s_prod output transform: `softplus` or `linear` |
| `rna_relax` | `float` | `0.1` | mRNA relaxation rate |

### `[bounds]`
Pairs of `*_min` / `*_max` float keys for: `log_k_deact`, `log_d_deg`, `log_beta`, `log_alpha`, `log_kK_act`, `log_kK_deact`, `log_k_off`, `gamma_raw`. See [Optimization](optimization.md) for defaults.

### `[analysis]`
| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `plot_format` | `str` | `"png"` | Plot format |
| `dpi` | `int` | `150` | Plot DPI |
| `plot_top_n` | `int` | `5` | Top N solutions to plot |

### `[simulation]`
| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `t_dense_n` | `int` | `200` | Dense trajectory points |
| `t_dense_min` | `float` | `0.0` | Dense grid start |
| `t_dense_max` | `float` | `120.0` | Dense grid end |

### `[neural_ode]`
See [Neural ODE](neural_ode.md) for the full table.

### `[data_interpolation]`
| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `enabled` | `bool` | `false` | Write dense observed-data interpolation files |
| `method` | `str` | `"linear"` | `linear` or `cubic_hermite` |
| `fill_forward_nans_at_end` | `bool` | `false` | Forward-fill trailing NaNs |
| `replace_nans_at_start` | `str\|null` | `null` | Leading NaN strategy: `null`, `zero`, `first_valid` |

### `[runtime]`
| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `n_jobs` | `int` | `1` | Parallel workers for multi-start |
| `device` | `str` | `"cpu"` | JAX device: `cpu` or `gpu` |
| `seed` | `int` | `42` | Global random seed |

### `[debug]`
| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `enabled` | `bool` | `false` | Enable debug logging |
| `log_every_n_steps` | `int` | `50` | Log interval during optimisation |

## API Reference

```python
class ModelDims:
    """Thread-safe static registry for model dimensions."""
    K: int | None
    M: int | None
    N: int | None

    @classmethod
    def set_dims(cls, K: int, M: int, N: int) -> None:
        """Set K, M, N atomically. Thread-safe via threading.Lock."""

def load_config(path: str | None = None) -> SimpleNamespace:
    """
    Deep-merge user TOML with _DEFAULTS and return as SimpleNamespace.
    Missing file does not raise — returns defaults silently.
    """

def validate_config(cfg: SimpleNamespace) -> tuple[list, list]:
    """
    Validate config values.

    Returns
    -------
    (errors, warnings) : tuple of two lists of str
    Does NOT raise — caller decides whether to abort.
    """
```

## Known Limitations

- `load_config` silently returns defaults when the config file is missing — pipeline runs with default settings without any warning.
- `validate_config` does not check `[bounds]` cross-consistency (e.g., `min > max`); such errors surface only at solve time.
- `ModelDims.set_dims` may be called multiple times; later calls overwrite earlier values with no warning — be careful in multi-experiment scripts.
