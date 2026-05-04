# PhosCrosstalk API

## Simulation

### `phoscrosstalk.simulation.simulate_ode`

Primary ODE simulation function using the Diffrax + JAX backend.

```python
simulate_ode(
    t_arr, P_data0, A_data0, theta,
    Cg, Cl, site_prot_idx, K_site_kin, R, L_alpha,
    kin_to_prot_idx, receptor_mask_prot, receptor_mask_kin,
    mechanism, full_output=False, return_full=False,
    rtol=1e-6, atol=1e-9, max_steps=16384, dt0=0.01,
    k_act_fn=None, s_prod_fn=None, t_extra=None,
    t_rna=None, R_data0=None, rna_relax=0.1,
)
```

Integrates the phospho-network ODE system using `diffrax.Tsit5` with adaptive
step-size control.  Returns simulated phosphosite occupancy `P_sim` (and
optionally all state components when `return_full=True`).

### `phoscrosstalk.simulation.simulate_p_scipy`

Backward-compatible alias for `simulate_ode`.  All arguments and return values
are identical.  The name is retained so that legacy scripts continue to work
after the SciPy / Numba backend was replaced by the Diffrax + JAX pipeline.

---

## Optimisation

### `phoscrosstalk.optimization.create_bounds`

```python
create_bounds(K: int, M: int, N: int) -> tuple[np.ndarray, np.ndarray, int]
```

Returns `(xl, xu, dim)` — lower bounds, upper bounds, and the total parameter
count `2*K + 2 + 3*M + N + 4` — for the flattened theta optimisation space.

### `phoscrosstalk.optimization.make_loss_fn`

```python
make_loss_fn(
    t, P_data, A_scaled, prot_idx_for_A, W_data, W_data_prot,
    Cg, Cl, site_prot_idx, K_site_kin, R, L_alpha,
    kin_to_prot_idx, receptor_mask_prot, receptor_mask_kin,
    mechanism, lambda_net, reg_lambda,
    w_phospho=1.0, w_abundance=1.0, w_reg=1.0,
    rtol=1e-6, atol=1e-9, max_steps=16384,
    k_act_fn=None, s_prod_fn=None,
    t_mrna=None, rna_data_scaled=None, w_mrna=1.0,
    rna_model_prot_idx=None, rna_obs_idx=None, rna_fit_genes=None,
    R_data0=None, W_data_rna=None, rna_relax=0.1,
) -> Callable
```

Builds a JAX-differentiable scalar loss function compatible with
`optimistix.minimise` and `jax.hessian`.  The returned `loss_fn(theta, args)`
returns `(total_loss, (f1, f2, f3, f4))`.

> **Note:** The internal Diffrax solver uses `Tsit5(scan_kind="bounded")` so
> that higher-order autodiff (e.g. `jax.hessian`) works correctly through the
> ODE solve.

### `phoscrosstalk.optimization.make_residuals_fn`

```python
make_residuals_fn(
    t, P_data, A_scaled, prot_idx_for_A, W_data, W_data_prot,
    Cg, Cl, site_prot_idx, K_site_kin, R, L_alpha,
    kin_to_prot_idx, receptor_mask_prot, receptor_mask_kin,
    mechanism, lambda_net, reg_lambda,
    w_phospho=1.0, w_abundance=1.0, w_reg=1.0,
    rtol=1e-6, atol=1e-9, max_steps=16384,
    k_act_fn=None, s_prod_fn=None,
    t_mrna=None, rna_data_scaled=None, w_mrna=1.0,
    rna_model_prot_idx=None, rna_obs_idx=None, rna_fit_genes=None,
    R_data0=None, W_data_mrna=None, rna_relax=0.1,
) -> Callable
```

Builds a JAX-differentiable residual-vector function compatible with
`optimistix.least_squares`.  The returned `residuals_fn(theta, args)` returns
`(residuals_1d, (f1, f2, f3, f4))`.  Uses `diffrax.ForwardMode` adjoint for
LM-compatible forward-mode AD.

### `phoscrosstalk.optimization.run_single_optimisation`

```python
run_single_optimisation(
    residuals_fn, theta0,
    max_steps=500, rtol=1e-8, atol=1e-8, verbose=False,
    *, ls_solver="lm", optx_adjoint="implicit",
) -> tuple[np.ndarray, float, float, float, float, float]
```

Runs one Optimistix least-squares optimisation pass.  Returns
`(theta_opt, total_loss, f1, f2, f3, f4)`.

### `phoscrosstalk.hybrid_fit.run_hybrid_fit`

Multi-stage hybrid fitting routine that combines a global search with local
refinement.  See the module docstring in `phoscrosstalk/hybrid_fit.py` for
argument details.

---

## Sensitivity analysis

### `phoscrosstalk.optimization.build_parameter_labels`

```python
build_parameter_labels(K: int, M: int, N: int) -> list[str]
```

Returns a list of `2*K + 2 + 3*M + N + 4` human-readable parameter labels
that match the exact theta layout:

| Slice        | Labels                                        |
|--------------|-----------------------------------------------|
| `[0 : K)`    | `log_k_deact[0]` … `log_k_deact[K-1]`        |
| `[K : 2K)`   | `log_d_deg[0]` … `log_d_deg[K-1]`            |
| `[2K : 2K+1)`   | `log_beta_g`                                  |
| `[2K+1 : 2K+2)` | `log_beta_l`                                  |
| `[2K+2 : …)` | `log_alpha[0]` … `log_alpha[M-1]`             |
| `[… : …)`    | `log_kK_act[0]` … `log_kK_act[M-1]`          |
| `[… : …)`    | `log_kK_deact[0]` … `log_kK_deact[M-1]`      |
| `[… : …)`    | `log_k_off[0]` … `log_k_off[N-1]`            |
| `[… : end)`  | `gamma_raw[0]` … `gamma_raw[3]`               |

### `phoscrosstalk.optimization.compute_second_order_sensitivities`

```python
compute_second_order_sensitivities(
    theta: np.ndarray,
    loss_fn: Callable,
    param_labels: Sequence[str],
    out_dir: str | os.PathLike,
    prefix: str = "loss_hessian",
    jit: bool = True,
) -> np.ndarray
```

Computes the Hessian of the scalarised loss with respect to theta using
`jax.hessian`, saves it as `.npy`, `.tsv`, and a heatmap `.png`, and returns
the float64 NumPy array.

#### Usage example

```python
from phoscrosstalk.optimization import (
    make_loss_fn,
    build_parameter_labels,
    compute_second_order_sensitivities,
)

# 1. Build the scalar loss function
loss_fn = make_loss_fn(
    t=t, P_data=P_data, A_scaled=A_scaled,
    prot_idx_for_A=prot_idx_for_A,
    W_data=W_data, W_data_prot=W_data_prot,
    Cg=Cg, Cl=Cl, site_prot_idx=site_prot_idx,
    K_site_kin=K_site_kin, R=R, L_alpha=L_alpha,
    kin_to_prot_idx=kin_to_prot_idx,
    receptor_mask_prot=receptor_mask_prot,
    receptor_mask_kin=receptor_mask_kin,
    mechanism="dist", lambda_net=1e-4, reg_lambda=1e-4,
)

# 2. Obtain optimised parameters from the fitting pipeline
theta_opt, *_ = run_single_optimisation(residuals_fn, theta0)

# 3. Build parameter labels
K, M, N = ModelDims.K, ModelDims.M, ModelDims.N
param_labels = build_parameter_labels(K, M, N)

# 4. Compute and save Hessian
H = compute_second_order_sensitivities(
    theta=theta_opt,
    loss_fn=loss_fn,
    param_labels=param_labels,
    out_dir="output/sensitivity",
    prefix="theta_loss_hessian",
)
# Produces:
#   output/sensitivity/theta_loss_hessian.npy
#   output/sensitivity/theta_loss_hessian.tsv
#   output/sensitivity/theta_loss_hessian_heatmap.png
```
