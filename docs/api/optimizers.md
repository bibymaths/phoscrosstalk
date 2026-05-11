# Optimizer & Adjoint Reference — PhosCrosstalk

PhosCrosstalk supports multiple optimisation backends selectable via
`[optimisation] optimizer_backend` in `config.toml`.

This document is the authoritative reference for every exposed backend.
It records the working solver/controller/adjoint combination, bound
strategy, gradient path, JIT status, and known failure modes derived
directly from inspecting the backend implementations and the Diffrax API.

---

## 1. Project Optimisation Model

The optimised parameter vector `theta` has dimension `2K + 2 + 3M + N + 4`:

| Block       | Length | Encoding       | Meaning                              |
|-------------|--------|----------------|--------------------------------------|
| `k_deact`   | K      | log-space      | Protein deactivation rates           |
| `d_deg`     | K      | log-space      | Protein degradation rates            |
| `beta_g`    | 1      | log-space      | Global crosstalk coupling            |
| `beta_l`    | 1      | log-space      | Local crosstalk coupling             |
| `alpha`     | M      | log-space      | Kinase–site activation strengths     |
| `kK_act`    | M      | log-space      | Kinase activation rates              |
| `kK_deact`  | M      | log-space      | Kinase deactivation rates            |
| `k_off`     | N      | log-space      | Phosphosite dephosphorylation rates  |
| `gamma`     | 4      | raw (tanh)     | Signed regulatory coupling           |

`k_act` and `s_prod` are **not** optimised. They are derived from input data.

The objective is a residual-based least-squares problem:

```python
residuals = [
    sqrt(w_phospho)   * (P_sim - P_data),
    sqrt(w_abundance) * (A_sim - A_data),
    sqrt(w_mrna)      * (R_sim - R_obs),
    sqrt(reg_lambda)  * theta,
    sqrt(lambda_net)  * alpha_net_reg,
]

total_loss = sum(residuals**2) = f1 + f2 + f3 + f4
````

---

## 2. Diffrax Solver, Controller, and Adjoint Compatibility

### ODE Solvers

| Key        | Type     | Stiffness | Notes                             |
| ---------- | -------- | --------- | --------------------------------- |
| `tsit5`    | Explicit | Non-stiff | Fast first test; default fallback |
| `dopri5`   | Explicit | Non-stiff | Similar to Tsit5                  |
| `dopri8`   | Explicit | Non-stiff | High-order; expensive             |
| `bosh3`    | Explicit | Non-stiff | Low-order; rough debug runs       |
| `kvaerno3` | Implicit | Stiff     | ESDIRK; cheaper than Kvaerno4     |
| `kvaerno4` | Implicit | Stiff     | **Recommended for production**    |
| `kvaerno5` | Implicit | Stiff     | Higher-order; expensive           |

### Step-size Controller

All backends use:

```python
diffrax.PIDController(rtol=..., atol=...)
```

This is built by `make_stepsize_controller()` in `solver_config.py`.

The controller is always adaptive. There is no fixed-step solver path in this project.

### Diffrax Adjoints

| Key          | Diffrax class                  | Forward AD | Reverse AD | Notes                            |
| ------------ | ------------------------------ | ---------- | ---------- | -------------------------------- |
| `forward`    | `ForwardMode()`                | Yes        | No         | Required for `jac_mode="fwd"`    |
| `recursive`  | `RecursiveCheckpointAdjoint()` | No         | Yes        | Recommended for `jac_mode="bwd"` |
| `checkpoint` | `RecursiveCheckpointAdjoint()` | No         | Yes        | Alias for `recursive`            |
| `direct`     | `DirectAdjoint()`              | Yes        | Yes        | Debug/Hessian; usually slower    |
| `backsolve`  | `BacksolveAdjoint()`           | Approx.    | Yes        | Approximate; avoid for fitting   |

Critical compatibility rule:

```text
jac_mode = "fwd" requires ode_adjoint = "forward"
jac_mode = "bwd" requires ode_adjoint = "recursive" or "checkpoint"
```

Do not combine:

```text
jac_mode = "fwd"
ode_adjoint = "recursive"
```

`RecursiveCheckpointAdjoint` does not support forward-mode AD and will raise an error during JVP tracing.

For gradient-based backends such as `jaxopt_lbfgsb`, `jaxopt_pgd`, and `scipy_jax`, use:

```text
ode_adjoint = "recursive"
```

or:

```text
ode_adjoint = "checkpoint"
```

These backends use reverse-mode gradients through `jax.grad` or `jax.value_and_grad`.

---

## 3. Backend Compatibility Matrix

| Backend         | Requires grad?   | Native bounds? | Bound strategy             | Solver   | Controller    | Adjoint                  | JIT status | Recommendation |
| --------------- | ---------------- | -------------- | -------------------------- | -------- | ------------- | ------------------------ | ---------- | -------------- |
| `optimistix`    | Jacobian fwd/bwd | No             | LM trust region + clipping | Kvaerno4 | PIDController | `forward` or `recursive` | JIT-able   | Recommended    |
| `jaxopt_lbfgsb` | value + grad     | Yes            | SciPy L-BFGS-B hard bounds | Kvaerno4 | PIDController | `recursive`              | Not JIT    | Works          |
| `jaxopt_pgd`    | grad             | Projection     | `projection_box`           | Kvaerno4 | PIDController | `recursive`              | JIT-able   | Fallback       |
| `scipy_jax`     | value + grad     | Reparameterise | Sigmoid mapping            | Kvaerno4 | PIDController | `recursive`              | JIT-able   | Fallback       |

---

## 4. Required Working Combination for Each Backend

---

### optimistix

Recommended solver:

```text
kvaerno4
```

Use `tsit5` only for non-stiff sanity checks.

Recommended controller:

```text
PIDController
```

Recommended adjoint:

```text
ode_adjoint = "forward"
```

when:

```text
jac_mode = "fwd"
```

This is the production default.

Use:

```text
ode_adjoint = "recursive"
```

or:

```text
ode_adjoint = "checkpoint"
```

when:

```text
jac_mode = "bwd"
```

Bound handling:

Optimistix does not use native hard box bounds in this project. Instead, the Levenberg–Marquardt trust-region mechanism constrains step size, while `theta` is clipped inside `residuals_fn` to prevent ODE divergence in biologically implausible regions.

Gradient/Jacobian path:

Optimistix computes Jacobian-vector products or vector-Jacobian products through the ODE solve using the selected Diffrax adjoint.

With:

```text
jac_mode = "fwd"
```

the residual Jacobian is computed using forward-mode AD.

With:

```text
jac_mode = "bwd"
```

the residual Jacobian is computed using reverse-mode AD.

JIT behavior:

Optimistix is fully JIT-able. Its iteration loop is implemented using JAX control flow, and the ODE solve inside `residuals_fn` is JIT-compiled.

Expected failure modes:

| Symptom                                         | Cause                                           | Fix                                                  |
| ----------------------------------------------- | ----------------------------------------------- | ---------------------------------------------------- |
| ODE `max_steps` exceeded                        | Stiff dynamics or poor initial parameters       | Use `kvaerno4`; increase `max_steps`; tighten bounds |
| `RecursiveCheckpointAdjoint` forward-mode error | `jac_mode="fwd"` with `ode_adjoint="recursive"` | Set `ode_adjoint="forward"`                          |
| NaN or infinite residuals                       | Explicit solver on stiff system                 | Use `kvaerno4`                                       |
| Slow convergence                                | Poor scaling or weak trust-region progress      | Check bounds, weights, and initialisation            |

Status:

```text
Recommended
```

---

### jaxopt_lbfgsb

Recommended solver:

```text
kvaerno4
```

Recommended controller:

```text
PIDController
```

Recommended adjoint:

```text
ode_adjoint = "recursive"
```

or:

```text
ode_adjoint = "checkpoint"
```

`jaxopt.ScipyBoundedMinimize` uses `jax.value_and_grad` internally, so reverse-mode differentiation is required.

Do not use:

```text
ode_adjoint = "forward"
```

with this backend.

Bound handling:

`jaxopt_lbfgsb` uses native hard box constraints through SciPy L-BFGS-B.

The lower and upper bounds are passed as:

```python
bounds = (xl_j, xu_j)
```

to:

```python
solver.run(...)
```

Bounds are strictly enforced by the SciPy L-BFGS-B algorithm.

Gradient path:

`jaxopt.ScipyBoundedMinimize` wraps the JAX objective and calls:

```python
jax.value_and_grad(...)
```

to provide function values and gradients to SciPy.

Gradients flow through the ODE solve via the selected reverse-mode-compatible Diffrax adjoint.

JIT behavior:

This backend is not fully JIT-able because SciPy L-BFGS-B runs host-side through NumPy/SciPy routines.

Do not wrap the full optimisation call in:

```python
jax.jit(...)
```

Expected failure modes:

| Symptom                    | Cause                                        | Fix                                                |
| -------------------------- | -------------------------------------------- | -------------------------------------------------- |
| ForwardMode gradient error | `ode_adjoint="forward"` with `jax.grad`      | Use `recursive` or `checkpoint`                    |
| Float/type instability     | JAX x64 disabled                             | Enable x64 before JAX import                       |
| Very slow convergence      | Tolerance too strict for stiff ODE objective | Relax `tol`; improve initialisation                |
| Poor local optimum         | Non-convex ODE fitting landscape             | Multi-start; better bounds; improved initial theta |

Status:

```text
Works
```

---

### jaxopt_pgd

Recommended solver:

```text
kvaerno4
```

Recommended controller:

```text
PIDController
```

Recommended adjoint:

```text
ode_adjoint = "recursive"
```

or:

```text
ode_adjoint = "checkpoint"
```

`jaxopt.ProjectedGradient` uses reverse-mode AD internally.

Bound handling:

Projected gradient descent applies box projection after each gradient step:

```python
jaxopt.projection.projection_box(x, (xl, xu))
```

The projection parameters are passed through:

```python
hyperparams_proj = (xl_j, xu_j)
```

to:

```python
solver.run(...)
```

Gradient path:

`jaxopt.ProjectedGradient` calls:

```python
jax.grad(...)
```

on the scalar loss at each step.

Gradients flow through the ODE solve via the selected reverse-mode-compatible adjoint.

JIT behavior:

This backend is JIT-able. `ProjectedGradient` uses JAX-compatible update logic when JIT execution is enabled.

Expected failure modes:

| Symptom                    | Cause                                              | Fix                                    |
| -------------------------- | -------------------------------------------------- | -------------------------------------- |
| Divergence or oscillation  | Step size too large                                | Decrease `stepsize`                    |
| Very slow convergence      | Step size too small or poor conditioning           | Tune `stepsize`; enable acceleration   |
| ForwardMode gradient error | `ode_adjoint="forward"` with reverse-mode gradient | Use `recursive` or `checkpoint`        |
| Weak final accuracy        | First-order method on stiff nonlinear objective    | Prefer `optimistix` or `jaxopt_lbfgsb` |

Status:

```text
Fallback
```

---

### scipy_jax

Recommended solver:

```text
kvaerno4
```

Recommended controller:

```text
PIDController
```

Recommended adjoint:

```text
ode_adjoint = "recursive"
```

or:

```text
ode_adjoint = "checkpoint"
```

`jax.scipy.optimize.minimize` uses gradient-based BFGS and requires reverse-mode differentiation through the scalar objective.

Bound handling:

`scipy_jax` handles bounds through smooth reparameterisation.

The unconstrained optimisation variable is `phi`.

The bounded parameter vector is reconstructed as:

```python
theta = xl + (xu - xl) * sigmoid(phi)
```

The initial unconstrained parameter vector is computed using the inverse-logit transformation from `theta0`.

This gives a smooth differentiable mapping into the feasible box. Bounds are approached asymptotically and are not reached exactly unless clipped after the solve.

Gradient path:

`jax.scipy.optimize.minimize` calls value-and-gradient logic internally.

Gradients flow through:

```text
phi -> sigmoid(phi) -> theta -> ODE solve -> scalar loss
```

JIT behavior:

The objective passed to `jax.scipy.optimize.minimize` must be JIT-compatible.

The function must not contain Python side effects, non-JAX NumPy operations on traced values, or non-static control flow that depends on traced values.

Expected failure modes:

| Symptom                   | Cause                                      | Fix                                                         |
| ------------------------- | ------------------------------------------ | ----------------------------------------------------------- |
| `NotImplementedError`     | JAX only supports `method="BFGS"` here     | Use `method="BFGS"`                                         |
| Weak boundary convergence | Sigmoid saturates near bounds              | Improve initialisation; widen bounds; avoid extreme margins |
| Slow convergence          | Full BFGS on high-dimensional problem      | Prefer `jaxopt_lbfgsb`                                      |
| Gradient instability      | Parameters too close to sigmoid saturation | Use safer inverse-logit clipping                            |

Status:

```text
Fallback
```

---

## 5. Known Failure Modes and Fixes

| Symptom                                                       | Likely Cause                                             | Fix                                                      |
| ------------------------------------------------------------- | -------------------------------------------------------- | -------------------------------------------------------- |
| `RecursiveCheckpointAdjoint does not support forward-mode AD` | `jac_mode="fwd"` with `ode_adjoint="recursive"`          | Set `ode_adjoint="forward"`                              |
| `ForwardMode()` with `jax.grad` error                         | `ode_adjoint="forward"` used with gradient-based backend | Set `ode_adjoint="recursive"` or `"checkpoint"`          |
| ODE `max_steps` exceeded                                      | Stiff system or poor initialisation                      | Use `kvaerno4`; increase `max_steps`; tighten bounds     |
| NaN loss from start                                           | Biologically implausible initial `theta`                 | Decrease `rate_max`; clip `theta0`; inspect data scaling |
| L-BFGS-B type/value error                                     | JAX x64 disabled or dtype mismatch                       | Enable x64 before JAX import                             |
| `jax.scipy.optimize.minimize` `NotImplementedError`           | Unsupported method requested                             | Use only `method="BFGS"`                                 |
| Slow PGD convergence                                          | First-order method on ill-conditioned objective          | Prefer `optimistix` or `jaxopt_lbfgsb`                   |
| Explicit solver divergence                                    | Stiff phosphorylation/RNA/protein dynamics               | Use `kvaerno4`                                           |

---

## 6. Minimal Code Examples

### Selecting a backend in `config.toml`

```toml
[optimisation]
optimizer_backend = "jaxopt_lbfgsb"
max_steps         = 1000
verbose           = true
```

### Direct dispatch in Python

```python
from phoscrosstalk.optimization import create_bounds
from phoscrosstalk.optimizers.dispatch import dispatch_optimisation

xl, xu, dim = create_bounds(K, M, N, bounds=cfg.bounds)

theta_opt, loss, f1, f2, f3, f4 = dispatch_optimisation(
    backend="jaxopt_lbfgsb",
    loss_fn=my_loss_fn,        # (theta, args) -> (scalar, (f1, f2, f3, f4))
    theta0=theta_init,
    xl=xl,
    xu=xu,
    max_steps=1000,
    verbose=True,
)
```

### Optimistix canonical path

```python
dispatch_optimisation(
    backend="optimistix",
    loss_fn=residuals_fn,      # residual vector, not scalar loss
    theta0=theta_init,
    xl=xl,
    xu=xu,
    max_steps=5000,
    ls_solver="lm",
    jac_mode="fwd",
    ode_adjoint="forward",
    verbose=True,
)
```

### JAXopt L-BFGS-B path

```python
dispatch_optimisation(
    backend="jaxopt_lbfgsb",
    loss_fn=scalar_loss_fn,    # scalar loss with optional aux terms
    theta0=theta_init,
    xl=xl,
    xu=xu,
    max_steps=1000,
    ode_adjoint="recursive",
    verbose=True,
)
```

### JAXopt projected-gradient fallback

```python
dispatch_optimisation(
    backend="jaxopt_pgd",
    loss_fn=scalar_loss_fn,
    theta0=theta_init,
    xl=xl,
    xu=xu,
    max_steps=2000,
    ode_adjoint="recursive",
    verbose=True,
)
```

### JAX SciPy BFGS fallback

```python
dispatch_optimisation(
    backend="scipy_jax",
    loss_fn=scalar_loss_fn,
    theta0=theta_init,
    xl=xl,
    xu=xu,
    max_steps=1000,
    ode_adjoint="recursive",
    verbose=True,
)
```

---

## 7. Final Recommendation

For coupled RNA/protein/phosphosite ODE fitting in PhosCrosstalk:

1. **Start with `optimistix`.**

   Use:

   ```text
   ls_solver = "lm"
   jac_mode = "fwd"
   ode_adjoint = "forward"
   solver = "kvaerno4"
   ```

   This is the most appropriate default path for residual-based nonlinear least squares.

2. **Use `jaxopt_lbfgsb` as the first alternative.**

   This backend provides native hard box constraints through SciPy L-BFGS-B and generally behaves well for bounded scalar loss optimisation.

   Required adjoint:

   ```text
   ode_adjoint = "recursive"
   ```

   or:

   ```text
   ode_adjoint = "checkpoint"
   ```

3. **Use `jaxopt_pgd` as a conservative fallback.**

   It is useful when hard projection is desirable and quasi-Newton behaviour is unstable or unavailable.

   Expect slower convergence than `optimistix` or `jaxopt_lbfgsb`.

4. **Use `scipy_jax` only when JAXopt is unavailable.**

   The sigmoid reparameterisation is smooth and differentiable, but it is weaker near parameter bounds and generally less robust than native L-BFGS-B.

---

## 8. Verbose Progress Logging

Progress logging is controlled by:

```toml
[optimisation]
verbose = true
```

When:

```toml
verbose = false
```

all backends remain quiet except for final summary messages.

### What each backend logs

| Backend         | Per-step logging                   | Information logged                                        |
| --------------- | ---------------------------------- | --------------------------------------------------------- |
| `optimistix`    | Via Optimistix `verbose=True` flag | Internal Optimistix progress; final loss always logged    |
| `jaxopt_lbfgsb` | Final only                         | `nit`, `fun_val`; SciPy terminal information when enabled |
| `jaxopt_pgd`    | Final only                         | `nit`, `error` or terminal state information              |
| `scipy_jax`     | Final only                         | `success`, `nit`, `fun`                                   |

All logging uses the project-wide structured logger:

```python
phoscrosstalk.logger
```

No raw `print()` calls should be used inside backend implementations.

### Logging implementation

`optimistix` exposes the `verbose` flag directly to the solver constructor, for example:

```python
LevenbergMarquardt(verbose=True)
```

This can print Optimistix-internal iteration information.

`jaxopt_lbfgsb`, `jaxopt_pgd`, and `scipy_jax` generally expose terminal information through the solver result object. Per-step logging from inside compiled solver loops is not available without explicit callbacks, which are intentionally avoided in the current implementation.

### Verbose mode does not disable JIT

Verbose logging does not disable JIT compilation.

For JAX-compatible paths, the objective and ODE solve may still be compiled. Backend-level progress reporting is limited by whether the solver loop itself is Python-side, SciPy-side, or JAX-compiled.

---

## API Reference

::: phoscrosstalk.optimizers
::: phoscrosstalk.optimizers.dispatch
::: phoscrosstalk.optimizers.jaxopt_backend
::: phoscrosstalk.optimizers.scipy_jax_backend