# Optimizer Backends

PhosCrosstalk supports multiple optimisation backends. The backend is selected
via the `[optimisation] optimizer_backend` key in `config.toml`.

## Available Backends

| Key | Module | Bounds strategy | Notes |
|-----|--------|-----------------|-------|
| `"optimistix"` | `optimization.py` | Native (Optimistix LM) | Default. Canonical residual-based LM. |
| `"jaxopt_lbfgsb"` | `optimizers/jaxopt_backend.py` | Hard box (SciPy L-BFGS-B) | Not JIT-able. |
| `"jaxopt_pgd"` | `optimizers/jaxopt_backend.py` | Projected GD | Pure-JAX; differentiable solution. |
| `"scipy_jax"` | `optimizers/scipy_jax_backend.py` | Reparameterisation | BFGS with sigmoid mapping. |
| `"scipy_jax_pen"` | `optimizers/scipy_jax_backend.py` | Quadratic penalty | BFGS with penalty; may violate bounds slightly. |
| `"optax_adam"` | `optimizers/optax_backend.py` | Projected (Adam) | First-order; projection after each step. |
| `"optax_sgd"` | `optimizers/optax_backend.py` | Projected (SGD) | First-order with momentum. |
| `"optax_lbfgs"` | `optimizers/optax_backend.py` | Projected (L-BFGS) | Quasi-Newton via Optax. |
| `"mpax"` | `optimizers/mpax_backend.py` | SQP with MPAX QP | Requires `JAX_ENABLE_X64=true`. |

## Common Return Signature

All backends return the same 6-tuple:

```python
(theta_opt: np.ndarray, total_loss: float, f1: float, f2: float, f3: float, f4: float)
```

where:

- `theta_opt` — fitted parameter vector (float64, clipped to `[xl, xu]`)
- `total_loss = f1 + f2 + f3 + f4`
- `f1` — phosphosite MSE loss
- `f2` — protein abundance MSE loss
- `f3` — regularisation loss
- `f4` — mRNA loss

## Selecting a Backend in config.toml

```toml
[optimisation]
optimizer_backend = "optax_adam"   # choose your backend
max_steps = 1000
```

## Dependencies

| Backend | Extra dependency |
|---------|-----------------|
| `optimistix` | `optimistix` (installed by default) |
| `jaxopt_lbfgsb`, `jaxopt_pgd` | `jaxopt` |
| `scipy_jax`, `scipy_jax_pen` | none (JAX built-in) |
| `optax_*` | `optax` (installed by default) |
| `mpax` | `mpax` |

## MPAX Limitations

The MPAX backend uses an SQP outer loop with MPAX solving bounded QP subproblems.
It requires `JAX_ENABLE_X64=true` (called automatically by `enable_x64()` in
`runtime_env.py` during normal runs). For large parameter dimensions, use
`diagonal_hessian=True` to reduce memory usage.

## API Reference

::: phoscrosstalk.optimizers
::: phoscrosstalk.optimizers.dispatch
::: phoscrosstalk.optimizers.jaxopt_backend
::: phoscrosstalk.optimizers.optax_backend
::: phoscrosstalk.optimizers.scipy_jax_backend
::: phoscrosstalk.optimizers.mpax_backend
