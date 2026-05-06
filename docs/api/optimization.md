# Optimization

## Biological Role

The `optimization` module performs multi-start nonlinear least-squares fitting of the mechanistic ODE parameters to phosphoproteomic time-series data. It uses the Optimistix Levenberg–Marquardt solver with JAX-JIT-compiled residual and loss functions.

## Implementation Overview

**Primary path:** `make_residuals_fn` → `optx.least_squares` (Levenberg–Marquardt). The residual vector is:

```
residuals = [
    sqrt(w_phospho * W)       * (P_sim - P_data),    shape (N*T,)
    sqrt(w_abundance * W_prot)* (A_sim - A_data),    shape (K_obs*T,)
    sqrt(w_mrna * W_mrna)     * (R_sim - R_obs),     shape (n_rna,)
    sqrt(reg_lambda)          * theta,                shape (n_var,)
    sqrt(lambda_net)          * L_alpha @ alpha,      shape (M,)
]
```

On ODE failure, each residual element is set to `_FAILED_SOLVE_PENALTY = 1e3`.

**Loss decomposition** (used by `make_loss_fn` and reported per run):

| Term | Formula | Weight key |
|------|---------|------------|
| `f1` – phosphosite MSLE | `mean(log1p(W * (P_sim - P_data)²))` | `w_phospho` |
| `f2` – abundance MSLE | `mean(log1p(W_prot * (A_sim - A_data)²))` | `w_abundance` |
| `f3` – regularisation | `reg_lambda*‖θ‖² + lambda_net*‖L·α‖²` | (direct) |
| `f4` – mRNA MSE | `mean(W_rna * (R_sim - R_obs)²)` | `w_mrna` |

`run_single_optimisation` wraps a single optimisation run: initialises `theta0` from Latin Hypercube sampling within `create_bounds`, runs the solver, and returns `(theta_opt, loss_final, converged, n_steps, wall_time)`.

`bio_score(theta)` penalises biologically implausible parameter ranges. It computes `(log10(median_t_half_kinase) - log10(10))² + (log10(median_t_half_protein) - log10(600))²`, targeting kinase half-life ≈ 10 min and protein half-life ≈ 600 min.

## Configuration Reference

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `[optimisation] n_starts` | `int` | `32` | Number of multi-start runs |
| `[optimisation] max_steps` | `int` | `500` | Max Optimistix iterations per run |
| `[optimisation] ls_solver` | `str` | `"lm"` | Solver: `lm`, `indirect_lm`, `dogleg`, `gauss_newton` |
| `[optimisation] optx_adjoint` | `str` | `"implicit"` | Optimistix adjoint: `implicit`, `checkpoint` |
| `[optimisation] jac_mode` | `str` | `"fwd"` | Jacobian mode: `fwd` (forward-mode AD) or `bwd` |
| `[optimisation] atol` | `float` | `1e-5` | Convergence tolerance |
| `[optimisation] rtol` | `float` | `1e-5` | Relative convergence tolerance |
| `[loss_weights] w_phospho` | `float` | `1.0` | Weight for phosphosite MSLE term |
| `[loss_weights] w_abundance` | `float` | `0.5` | Weight for abundance MSLE term |
| `[loss_weights] w_mrna` | `float` | `0.1` | Weight for mRNA MSE term (0 if no RNA data) |
| `[loss_weights] reg_lambda` | `float` | `1e-3` | L2 regularisation strength |
| `[loss_weights] lambda_net` | `float` | `1e-4` | Network Laplacian regularisation strength |
| `[bounds] log_k_deact_min/max` | `float` | `-4` / `2` | Log-scale bounds for protein deactivation |
| `[bounds] log_d_deg_min/max` | `float` | `-6` / `0` | Log-scale bounds for protein degradation |
| `[bounds] log_beta_min/max` | `float` | `-3` / `3` | Log-scale bounds for coupling strengths |
| `[bounds] log_alpha_min/max` | `float` | `-4` / `4` | Log-scale bounds for kinase amplitudes |
| `[bounds] log_k_off_min/max` | `float` | `-4` / `2` | Log-scale bounds for phosphatase rates |
| `[bounds] gamma_raw_min/max` | `float` | `-3` / `3` | Raw bounds for crosstalk couplings |

## API Reference

```python
def make_residuals_fn(
    P_data, A_data, rna_data,
    W, W_prot, W_mrna,
    w_phospho, w_abundance, w_mrna,
    reg_lambda, lambda_net,
    simulate_fn, L_alpha,
    args_static: tuple,
    K: int, M: int, N: int,
) -> callable:
    """Return JIT-compiled residual closure for optx.least_squares."""

def make_loss_fn(
    P_data, A_data, rna_data,
    W, W_prot, W_mrna,
    w_phospho, w_abundance, w_mrna,
    reg_lambda, lambda_net,
    simulate_fn, L_alpha,
    args_static: tuple,
    K: int, M: int, N: int,
) -> callable:
    """Return JIT-compiled scalar loss closure (for diagnostics/neural mode)."""

def run_single_optimisation(
    residuals_fn,
    theta_bounds: tuple,
    n_var: int,
    max_steps: int = 500,
    solver_name: str = "lm",
    atol: float = 1e-5,
    rtol: float = 1e-5,
    seed: int = 0,
) -> tuple:
    """
    Run one multi-start optimisation trial.

    Returns
    -------
    (theta_opt, loss_final, converged, n_steps, wall_time)
    """

def bio_score(theta, K: int, M: int) -> float:
    """NumPy bio-plausibility score (for post-hoc filtering)."""

def bio_score_jax(theta, K: int, M: int) -> jax.Array:
    """JAX-traceable variant of bio_score."""

def create_bounds(K: int, M: int, N: int, cfg) -> tuple:
    """
    Return (lower_bounds, upper_bounds) arrays from config.

    Returns
    -------
    (lb, ub) : tuple of jnp.ndarray, each shape (n_var,)
    """

def build_parameter_labels(K: int, M: int, N: int,
                           prot_names, kin_names, site_names) -> list[str]:
    """Return human-readable label list aligned with theta vector."""

def validate_problem_shapes(P_data, A_data, W, W_prot, K, M, N, T) -> None:
    """Raise ValueError if array shapes are inconsistent."""
```

## Known Limitations

- LHS sampling for `theta0` uses `scipy.stats.qmc.LatinHypercube`; reproducibility requires setting the global JAX random seed via `jax.random.PRNGKey` before calling `run_single_optimisation`.
- `make_loss_fn` uses `FAILED_SOLVE_PENALTY = 1e6` (scalar), while `make_residuals_fn` uses `_FAILED_SOLVE_PENALTY = 1e3` per element — these are not equivalent and may produce different convergence behaviour.
- The Dogleg and Gauss–Newton solvers (`dogleg`, `gauss_newton`) do not support `has_aux=True`; use `lm` or `indirect_lm` for the standard residual path.
