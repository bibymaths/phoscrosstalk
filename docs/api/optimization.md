# Optimisation

## Bounds

::: phoscrosstalk.optimization.create_bounds

## Scalar loss

::: phoscrosstalk.optimization.make_loss_fn

!!! note "Diffrax solver"
    The optimisation path should use a Diffrax solver setup compatible with JAX autodiff. If Hessian-based analysis is required, use `Tsit5(scan_kind="bounded")` and an adjoint compatible with the selected Optimistix solver.

## Residual function

::: phoscrosstalk.optimization.make_residuals_fn

!!! tip "Least-squares fitting"
    Prefer residual-vector fitting with `optimistix.least_squares` when using Levenberg-Marquardt. This follows the native Optimistix pattern for parameterised ODE fitting.

## Single optimisation run

::: phoscrosstalk.optimization.run_single_optimisation

## Parameter labels

::: phoscrosstalk.optimization.build_parameter_labels

## Second-order sensitivities

::: phoscrosstalk.optimization.compute_second_order_sensitivities

## Network problem wrapper

::: phoscrosstalk.optimization.NetworkProblem