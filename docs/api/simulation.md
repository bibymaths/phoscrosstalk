# Simulation

## Biological Role

The `simulation` module wraps the Diffrax ODE solver to integrate the phosphorylation ODE system forward in time. It provides the primary entry point used during optimisation (`simulate`) and a post-fit dense-output wrapper (`simulate_dense`) for smooth trajectory visualisation.

## Implementation Overview

`simulate` builds the `diffrax.ODETerm` from the RHS returned by `make_rhs`, constructs the initial condition via `build_full_A0`, and calls `diffrax.diffeqsolve`. On solver failure (non-finite state or `diffrax.RESULTS.successful` is False) it returns a `_nan_result` filled with `jnp.nan`.

`simulate_dense` wraps `simulate(return_full=True)` and adds a `success` boolean key. It is never called during optimisation.

`build_full_A0` assembles the initial ODE state from per-compartment arrays:
```
y0 = [R_rna_0 (K,) | S_0 (K,) | A_0 (K,) | Kdyn_0 (M,) | p_0 (N,)]
```

## Configuration Reference

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `[solver] ode_solver` | `str` | `"tsit5"` | Diffrax solver: `tsit5`, `dopri5`, `dopri8`, `bosh3`, `kvaerno3`, `kvaerno4`, `kvaerno5` |
| `[solver] ode_adjoint` | `str` | `"forward"` | Adjoint method: `forward`, `checkpoint`, `direct`, `backsolve`, `none` |
| `[solver] rtol` | `float` | `1e-4` | Relative tolerance |
| `[solver] atol` | `float` | `1e-5` | Absolute tolerance |
| `[solver] dt0` | `float` | `0.1` | Initial step size |
| `[solver] max_steps` | `int` | `16384` | Maximum ODE steps |
| `[time] t0` | `float` | `0.0` | Integration start time |
| `[time] t1` | `float` | `120.0` | Integration end time |
| `[time] t_eval` | `list` | `[0,5,10,20,30,60,120]` | Observation time points |

## API Reference

```python
def simulate(
    theta,
    args: tuple,
    t0: float,
    t1: float,
    t_eval: np.ndarray,
    solver_name: str = "tsit5",
    adjoint: str = "forward",
    rtol: float = 1e-4,
    atol: float = 1e-5,
    dt0: float = 0.1,
    max_steps: int = 16384,
    return_full: bool = False,
) -> dict:
    """
    Integrate the ODE from t0 to t1 and return states at t_eval.

    Returns dict with keys:
        "P"       (N, T) – phosphosite trajectories
        "S"       (K, T) – protein signalling state
        "A"       (K, T) – protein abundance
        "Kdyn"    (M, T) – kinase activity
        "R_rna"   (K, T) – mRNA/transcription drive
        "success" bool   – only when return_full=True
    On failure returns _nan_result of the same shape.
    """

def simulate_dense(
    theta,
    args: tuple,
    t0: float,
    t1: float,
    t_dense: np.ndarray,
    **solver_kwargs,
) -> dict:
    """Post-fit dense simulation; wraps simulate(return_full=True)."""

def build_full_A0(
    R_rna_0, S_0, A_0, Kdyn_0, p_0
) -> jnp.ndarray:
    """Concatenate per-compartment initial conditions into flat y0."""
```

## Known Limitations

- Stiff kinetics may require implicit solvers (`kvaerno3`–`kvaerno5`); explicit solvers (`tsit5`, `dopri5`) may fail or take very small steps for stiff systems.
- `simulate_dense` is not JAX-JIT-compatible when `t_dense` is a Python list; pass a `jnp.ndarray`.
- ODE failure returns NaN arrays silently; callers must inspect the `success` key (available in `return_full=True` mode) to detect failures.
