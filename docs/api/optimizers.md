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

`k_act` and `s_prod` are **not** optimised — they are derived from input data.

The objective is a residual-based least-squares problem:

```
residuals = [sqrt(w_phospho) * (P_sim - P_data),
             sqrt(w_abundance) * (A_sim - A_data),
             sqrt(w_mrna) * (R_sim - R_obs),
             sqrt(reg_lambda) * theta,
             sqrt(lambda_net) * alpha_net_reg]
total_loss = sum(residuals**2) = f1 + f2 + f3 + f4
```

---

## 2. Diffrax Solver, Controller, and Adjoint Compatibility

### ODE Solvers

| Key           | Type     | Stiffness    | Notes                              |
|---------------|----------|--------------|------------------------------------|
| `tsit5`       | Explicit | Non-stiff    | Fast first test; default fallback  |
| `dopri5`      | Explicit | Non-stiff    | Similar to Tsit5                   |
| `dopri8`      | Explicit | Non-stiff    | High-order; expensive              |
| `bosh3`       | Explicit | Non-stiff    | Low-order; rough debug runs        |
| `kvaerno3`    | Implicit | Stiff        | ESDIRK; cheaper than Kvaerno4      |
| `kvaerno4`    | Implicit | Stiff        | **Recommended for production**     |
| `kvaerno5`    | Implicit | Stiff        | Higher-order; expensive            |

### Step-size Controller

All backends use `diffrax.PIDController(rtol=..., atol=...)`.
This is built by `make_stepsize_controller()` in `solver_config.py`
and is always adaptive — there is no fixed-step path in this project.

### Diffrax Adjoints

| Key          | Diffrax class                   | Forward AD | Reverse AD | Notes                            |
|--------------|---------------------------------|-----------|------------|----------------------------------|
| `forward`    | `ForwardMode()`                 | ✅ Yes    | ❌ No      | Required for `jac_mode="fwd"`    |
| `recursive`  | `RecursiveCheckpointAdjoint()`  | ❌ No     | ✅ Yes     | Recommended for `jac_mode="bwd"` |
| `checkpoint` | `RecursiveCheckpointAdjoint()`  | ❌ No     | ✅ Yes     | Alias for `recursive`            |
| `direct`     | `DirectAdjoint()`               | ✅ Yes    | ✅ Yes     | Debug/Hessian; usually slower    |
| `backsolve`  | `BacksolveAdjoint()`            | ⚠️ Approx | ✅ Yes     | Approximate; avoid for fitting   |

**Critical compatibility rule:**

- `jac_mode = "fwd"` (Optimistix forward Jacobian) **requires** `ode_adjoint = "forward"`.
- `jac_mode = "bwd"` (Optimistix reverse Jacobian) **requires** `ode_adjoint = "recursive"` or `"checkpoint"`.
- Do **not** combine `jac_mode = "fwd"` with `ode_adjoint = "recursive"`:
  `RecursiveCheckpointAdjoint` does not support forward-mode AD and will raise.

For gradient-based backends (`jaxopt`, `optax`, `scipy_jax`, `mpax`):
use `ode_adjoint = "recursive"` or `"checkpoint"` (reverse-mode).
`ForwardMode` can also work when `jax.grad` is used, but `recursive` is safer
and more memory-efficient.

---

## 3. Backend Compatibility Matrix

| Backend | Requires grad? | Native bounds? | Bound strategy | Solver | Controller | Adjoint | JIT status | Recommendation |
|---|---:|---:|---|---|---|---|---|---|
| `optimistix` | Jacobian (fwd/bwd) | No | LM trust region | Kvaerno4 | PIDController | `forward` or `recursive` | ✅ JIT-able | **Recommended** |
| `jaxopt_lbfgsb` | ✅ value+grad | ✅ Hard box (SciPy) | SciPy L-BFGS-B native | Kvaerno4 | PIDController | `recursive` | ❌ Not JIT | Works |
| `jaxopt_pgd` | ✅ grad | Projection | `jaxopt.projection_box` | Kvaerno4 | PIDController | `recursive` | ✅ JIT-able | Fallback |
| `scipy_jax` | ✅ grad (BFGS) | Reparameterise | sigmoid mapping | Kvaerno4 | PIDController | `recursive` | ✅ JIT-able | Fallback |
| `scipy_jax_pen` | ✅ grad (BFGS) | Penalty | quadratic penalty | Kvaerno4 | PIDController | `recursive` | ✅ JIT-able | Fallback |
| `optax_adam` | ✅ grad | Projection | `projection_box` post-step | Kvaerno4 | PIDController | `recursive` | ✅ JIT-able | Works |
| `optax_sgd` | ✅ grad | Projection | `projection_box` post-step | Kvaerno4 | PIDController | `recursive` | ✅ JIT-able | Works |
| `optax_lbfgs` | ✅ value+grad | Projection | `projection_box` post-step | Kvaerno4 | PIDController | `recursive` | ✅ JIT-able | Works |
| `mpax` | ✅ grad+Hessian | Trust region | SQP box step | Kvaerno4 | PIDController | `recursive` | ⚠️ Partial | Debug only |

---

## 4. Required Working Combination for Each Backend

### optimistix

- **Recommended solver:** `kvaerno4` (stiff) or `tsit5` (non-stiff sanity check)
- **Recommended controller:** `PIDController`
- **Recommended adjoint:**
  - `forward` (ForwardMode) when `jac_mode = "fwd"` ← **production default**
  - `recursive` (RecursiveCheckpointAdjoint) when `jac_mode = "bwd"`
- **Bound handling:** LM trust-region implicitly constrains step size.
  Hard box bounds are applied via `jnp.clip` inside `residuals_fn` to prevent
  ODE divergence in biologically implausible regions.
- **Gradient path:** Optimistix computes JVPs or VJPs through the ODE solve
  via the selected adjoint. `jac_mode = "fwd"` computes the residual Jacobian
  by forward-mode AD; `jac_mode = "bwd"` uses reverse-mode.
- **JIT behavior:** Fully JIT-able. Optimistix wraps its iteration in
  `lax.while_loop`; the ODE solve inside residuals_fn is JIT-compiled.
- **Expected failure modes:**
  - ODE `max_steps` exceeded → fall-back penalty residual returned.
  - Mismatch `jac_mode="fwd"` + `ode_adjoint="recursive"` → JAX tracing error at first JVP.
  - Stiff system with explicit solver → NaN/infinite residuals → optimizer diverges.
- **Status:** **Recommended**

---

### jaxopt_lbfgsb

- **Recommended solver:** `kvaerno4`
- **Recommended controller:** `PIDController`
- **Recommended adjoint:** `recursive` (RecursiveCheckpointAdjoint) —
  `jaxopt.ScipyBoundedMinimize` uses `jax.value_and_grad` internally, which
  requires reverse-mode. `forward` (ForwardMode) is incompatible.
- **Bound handling:** Native hard box constraints via SciPy L-BFGS-B. `xl` and
  `xu` are passed as `bounds=(xl_j, xu_j)` to `solver.run()`. Bounds are strictly
  enforced by the SciPy L-BFGS-B algorithm.
- **Gradient path:** `jaxopt.ScipyBoundedMinimize` wraps the JAX function and
  calls `jax.value_and_grad` to supply function values and gradients to SciPy.
  Gradients flow through the ODE solve via the selected adjoint.
- **JIT behavior:** **NOT JIT-able.** SciPy L-BFGS-B calls NumPy/SciPy host-side
  routines. Do not wrap the call in `jax.jit`.
- **Expected failure modes:**
  - `ode_adjoint = "forward"` + gradient call → forward-mode does not support
    `jax.grad`; raises a JAX error.
  - JAX x64 not enabled → L-BFGS-B may produce float32 parameters, causing
    numerical issues. Enable x64 mode (`enable_x64()` in `runtime_env.py`).
  - Very small `tol` with stiff ODE → slow convergence.
- **Status:** Works

---

### jaxopt_pgd

- **Recommended solver:** `kvaerno4`
- **Recommended controller:** `PIDController`
- **Recommended adjoint:** `recursive` — `jaxopt.ProjectedGradient` uses
  reverse-mode AD internally.
- **Bound handling:** `jaxopt.projection.projection_box(x, (xl, xu))` applied
  after each gradient step. `hyperparams_proj=(xl_j, xu_j)` passed to `solver.run()`.
- **Gradient path:** `jaxopt.ProjectedGradient` calls `jax.grad` on the scalar
  loss at each step. Gradients flow through the ODE solve via the adjoint.
- **JIT behavior:** JIT-able. `ProjectedGradient` uses `lax.while_loop` internally
  when `jit_update=True` (default). The result is differentiable through the
  solution via implicit-function theorem.
- **Expected failure modes:**
  - Small `stepsize` needed for stability; too large → divergence or oscillation.
  - Without `acceleration=True` (FISTA), convergence is slow on poorly conditioned
    problems.
  - `ode_adjoint = "forward"` → incompatible with reverse-mode gradient.
- **Status:** Fallback

---

### scipy_jax

- **Recommended solver:** `kvaerno4`
- **Recommended controller:** `PIDController`
- **Recommended adjoint:** `recursive` — `jax.scipy.optimize.minimize` (BFGS)
  uses `jax.value_and_grad` internally. Requires reverse-mode.
- **Bound handling:** Reparameterisation — `theta = xl + (xu - xl) * sigmoid(phi)`.
  Optimises over unconstrained `phi`. The sigmoid mapping is bijective and smooth;
  corners are approached asymptotically (never exactly reached). Initial `phi0`
  is computed via inverse logit.
- **Gradient path:** `jax.scipy.optimize.minimize` requires the objective to be
  JIT-able. It calls `jax.value_and_grad` internally; gradients flow through
  the sigmoid mapping and the ODE solve.
- **JIT behavior:** The objective function passed to `jax.scipy.optimize.minimize`
  must be JIT-able (no Python side-effects inside). The BFGS loop itself runs
  in Python host-side via `lax.while_loop`.
- **Expected failure modes:**
  - `jax.scipy.optimize.minimize` only supports `method="BFGS"` in JAX; passing
    any other method raises `NotImplementedError`.
  - Very small sigmoid margins near boundaries may cause numerical gradient issues.
  - BFGS is a full-space quasi-Newton method; does not scale as well as L-BFGS-B
    to very high-dimensional problems.
- **Status:** Fallback

---

### scipy_jax_pen

- **Recommended solver:** `kvaerno4`
- **Recommended controller:** `PIDController`
- **Recommended adjoint:** `recursive`
- **Bound handling:** Quadratic penalty — augmented loss adds
  `_PENALTY_COEFF * sum(max(xl - theta, 0)^2 + max(theta - xu, 0)^2)`.
  The solution is hard-clipped to `[xl, xu]` after optimisation.
  May sit slightly inside feasible region rather than exactly on boundary.
- **Gradient path:** Same as `scipy_jax`. Penalty terms are differentiable
  everywhere (smooth quadratic).
- **JIT behavior:** JIT-able for the objective; BFGS loop is host-side.
- **Expected failure modes:**
  - If `_PENALTY_COEFF` (1e4) is too small relative to the loss scale, bounds
    may be violated at convergence.
  - Hard clip post-solve may move the solution slightly, making the reported
    loss inconsistent with the clipped parameters (recomputed at clipped theta).
- **Status:** Fallback

---

### optax_adam

- **Recommended solver:** `kvaerno4`
- **Recommended controller:** `PIDController`
- **Recommended adjoint:** `recursive`
- **Bound handling:** `optax.projections.projection_box(theta, xl, xu)` applied
  after every gradient step. Guarantees feasibility at all iterations.
- **Gradient path:** `jax.value_and_grad(loss_with_aux, has_aux=True)(theta)` at
  each step. Gradients flow through the ODE solve. Runs in a Python `for` loop
  (not `lax.while_loop`) so per-step logging works naturally.
- **JIT behavior:** The loss function itself is JIT-compiled. The outer `for`
  loop is Python-side, which allows per-step logging but incurs Python overhead
  per step. For long runs, consider increasing `max_steps` and relying on
  convergence early-stopping.
- **Expected failure modes:**
  - Very small `learning_rate` → slow convergence; very large → overshooting and
    oscillation.
  - Adam does not adapt to curvature; on badly conditioned ODE problems it can
    get stuck.
  - No line search → may not converge to machine precision.
- **Status:** Works

---

### optax_sgd

- **Recommended solver:** `kvaerno4`
- **Recommended controller:** `PIDController`
- **Recommended adjoint:** `recursive`
- **Bound handling:** Same as `optax_adam` — `projection_box` after each step.
- **Gradient path:** Same as `optax_adam`.
- **JIT behavior:** Same as `optax_adam`.
- **Expected failure modes:**
  - Momentum (0.9 by default) can cause oscillation near the optimum.
  - SGD has no curvature information and is significantly slower than Adam or
    L-BFGS on smooth nonlinear problems.
  - Primarily useful as a debug/sanity-check backend.
- **Status:** Works

---

### optax_lbfgs

- **Recommended solver:** `kvaerno4`
- **Recommended controller:** `PIDController`
- **Recommended adjoint:** `recursive`
- **Bound handling:** `projection_box` after each L-BFGS step.
- **Gradient path:** `optax.value_and_grad_from_state(scalar_loss)` reads the
  cached value from the optimiser state to avoid redundant function evaluations.
  `optimizer.update(grad, state, theta, value=value, grad=grad, value_fn=scalar_loss)`
  is the Optax L-BFGS API (>= 0.2). Gradients flow through the ODE solve.
- **JIT behavior:** The outer loop is Python-side (same as Adam/SGD), enabling
  per-step logging. The loss function is JIT-compiled.
- **Expected failure modes:**
  - `optax.lbfgs()` requires Optax >= 0.2; older versions lack this optimizer.
  - Projection after each step breaks the L-BFGS curvature estimate because the
    true iterate is projected. This makes `optax_lbfgs` a heuristic method rather
    than a true L-BFGS-B; prefer `jaxopt_lbfgsb` for hard-bounded L-BFGS.
  - Line search in Optax L-BFGS is limited; may not satisfy Wolfe conditions on
    stiff ODE objectives.
- **Status:** Works

---

### mpax

- **Recommended solver:** `kvaerno4`
- **Recommended controller:** `PIDController`
- **Recommended adjoint:** `recursive` — MPAX SQP uses `jax.grad` and
  `jax.hessian`, both requiring reverse-mode.
- **Bound handling:** Trust-region box in step space:
  `l_qp = max(xl - theta, -trust_radius)`,
  `u_qp = min(xu - theta, trust_radius)`.
  After the QP step: `theta = clip(theta + d_theta, xl, xu)`.
- **Gradient path:** `jax.grad(scalar_loss)` and `jax.hessian(scalar_loss)` at
  each SQP step. Full Hessian is O(n²) memory; use `diagonal_hessian=True` for
  large networks.
- **JIT behavior:** Partially JIT-able. The MPAX `raPDHG` solver is a JAX-based
  QP solver; `jax.hessian` is JIT-compiled. The outer SQP `for` loop is
  Python-side, enabling per-step logging.
- **Expected failure modes:**
  - Requires `JAX_ENABLE_X64=true` — raises `RuntimeError` otherwise.
    `enable_x64()` in `runtime_env.py` is called automatically by `main.py`.
  - Full Hessian is prohibitively expensive for large `n = 2K + 2 + 3M + N + 4`.
    Use `diagonal_hessian=True` as a memory-efficient approximation.
  - MPAX is a convex QP solver. The nonlinear ODE objective is non-convex, so
    each QP subproblem is only a local quadratic approximation. The outer SQP
    loop may converge slowly or stall if the Hessian is far from positive-definite
    (regularisation via `hess_reg` mitigates this).
  - `trust_radius` must be tuned; too large → QP subproblem departs from local
    quadratic accuracy; too small → very slow convergence.
- **Status:** Debug only

---

## 5. Known Failure Modes and Fixes

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| `RecursiveCheckpointAdjoint does not support forward-mode AD` | `jac_mode="fwd"` + `ode_adjoint="recursive"` | Set `ode_adjoint="forward"` |
| `ForwardMode()` + `jax.grad` error | `ode_adjoint="forward"` with gradient-based backend | Set `ode_adjoint="recursive"` or `"checkpoint"` |
| ODE `max_steps` exceeded; penalty loss returned | Stiff system with explicit solver | Switch to `kvaerno4`; increase `max_steps` |
| NaN loss from start | Biologically implausible initial `theta` | Decrease `rate_max`; theta0 is clipped inside residuals_fn |
| L-BFGS-B `ValueError: not a float` | JAX x64 disabled | Call `enable_x64()` before JAX import |
| `jax.scipy.optimize.minimize` `NotImplementedError` | Non-BFGS method requested | Only `method="BFGS"` is supported |
| Slow `optax_lbfgs` convergence | Projection breaks L-BFGS curvature | Use `jaxopt_lbfgsb` for proper bounded L-BFGS |
| MPAX `RuntimeError` on x64 | `JAX_ENABLE_X64=false` | Ensure `enable_x64()` is called first |
| Very slow MPAX | Full Hessian on large network | Set `diagonal_hessian=True` |

---

## 6. Minimal Code Examples

### Selecting a backend in config.toml

```toml
[optimisation]
optimizer_backend = "jaxopt_lbfgsb"
max_steps         = 1000
verbose           = true
log_every         = 100   # only used by optax-style backends
```

### Direct dispatch (Python)

```python
from phoscrosstalk.optimization import create_bounds
from phoscrosstalk.optimizers.dispatch import dispatch_optimisation

xl, xu, dim = create_bounds(K, M, N, bounds=cfg.bounds)

theta_opt, loss, f1, f2, f3, f4 = dispatch_optimisation(
    backend   = "optax_adam",
    loss_fn   = my_loss_fn,         # (theta, args) -> (scalar, (f1,f2,f3,f4))
    theta0    = theta_init,
    xl        = xl,
    xu        = xu,
    max_steps = 1000,
    verbose   = True,
    log_every = 50,
)
```

### Optimistix (canonical path)

```python
dispatch_optimisation(
    backend      = "optimistix",
    loss_fn      = residuals_fn,    # residual vector, not scalar loss
    theta0       = theta_init,
    xl           = xl,
    xu           = xu,
    max_steps    = 5000,
    ls_solver    = "lm",
    jac_mode     = "fwd",
    optx_adjoint = "implicit",
    verbose      = True,
)
```

---

## 7. Final Recommendation

For coupled RNA/protein/phosphosite ODE fitting in PhosCrosstalk:

1. **Start with `optimistix`** (default). Use `ls_solver="lm"`,
   `jac_mode="fwd"`, `ode_adjoint="forward"` with `kvaerno4`.
   This is the most thoroughly tested path.

2. **Use `jaxopt_lbfgsb`** as the first alternative. L-BFGS-B has
   native hard box constraints and quasi-Newton convergence.
   Requires `ode_adjoint="recursive"`. Not JIT-able but usually fast.

3. **Use `optax_adam` or `optax_lbfgs`** for first-order exploration
   when `jaxopt_lbfgsb` fails or is unavailable.

4. **Use `scipy_jax`** when JAXopt is unavailable. The sigmoid
   reparameterisation is smooth and differentiable everywhere.

5. **Avoid `mpax`** for routine fitting. Reserve it for research into
   SQP-style curvature exploitation on small networks.

---

## 8. Verbose Progress Logging

Progress logging is controlled by `[optimisation] verbose = true` in `config.toml`.
When `verbose = false`, all backends remain quiet except final summary messages.

```toml
[optimisation]
verbose   = true
log_every = 100   # step interval for optax-style per-step logging
```

### What each backend logs

| Backend | Per-step logging | Information logged |
|---------|-----------------|-------------------|
| `optimistix` | Via Optimistix `verbose=True` flag | Internal Optimistix progress (noisy); final loss always logged |
| `jaxopt_lbfgsb` | Final only (SciPy L-BFGS-B) | `nit`, `fun_val`; SciPy `disp=True` also enabled |
| `jaxopt_pgd` | Final only | `nit`, `error` (step norm) |
| `scipy_jax` | Final only | `success`, `nit`, `fun` |
| `scipy_jax_pen` | Final only | `success`, `nit`, `fun` |
| `optax_adam` | Every `log_every` steps | `step`, `loss`, `f1`, `f2`, `f3`, `f4` |
| `optax_sgd` | Every `log_every` steps | `step`, `loss`, `f1`, `f2`, `f3`, `f4` |
| `optax_lbfgs` | Every `log_every` steps | `step`, `loss`, `f1`, `f2`, `f3`, `f4` |
| `mpax` | Every SQP step | `step`, `loss`, `f1`, `f2`, `f3`, `f4` |

All logging uses the project-wide structured logger (`phoscrosstalk.logger`),
not raw `print()` calls.

### Logging implementation

- **Optax backends** log inside the Python `for` loop using `logger.info()`.
  `log_every` controls the interval. No `jax.debug.print` is needed because
  the loop is Python-side.

- **JAXopt / scipy_jax backends** expose only final or terminal progress via
  the solver result object. Per-step logging from inside JIT-compiled loops is
  not available without out-of-JIT callbacks. The implementations log the richest
  available information at the end of the run.

- **Optimistix** exposes the `verbose` flag directly to the solver constructor
  (`LevenbergMarquardt(verbose=True)`), which prints Optimistix-internal step
  information.

- **MPAX** logs per SQP step because the outer loop is Python-side.

### Verbose mode does NOT disable JIT

JIT is not disabled for any backend when `verbose=True`. Verbose logging for
JIT-compiled functions uses Python-side post-step logging in the outer loop.
`jax.debug.print` is not used in the current implementation.

---

## API Reference

::: phoscrosstalk.optimizers
::: phoscrosstalk.optimizers.dispatch
::: phoscrosstalk.optimizers.jaxopt_backend
::: phoscrosstalk.optimizers.optax_backend
::: phoscrosstalk.optimizers.scipy_jax_backend
::: phoscrosstalk.optimizers.mpax_backend

