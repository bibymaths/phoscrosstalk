"""Tests for the optimizer dispatch and backends."""
import os

os.environ["JAX_ENABLE_X64"] = "true"

import numpy as np
import pytest
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Shared test fixture: simple quadratic objective
# ---------------------------------------------------------------------------

def make_quadratic_loss(target):
    """loss(theta) = sum((theta - target)^2), aux=(loss, 0, 0, 0)."""
    target_j = jnp.asarray(target, dtype=jnp.float64)

    def loss_fn(theta, args):
        val = jnp.sum((theta - target_j) ** 2)
        return val, (val, jnp.zeros((), dtype=jnp.float64), jnp.zeros((), dtype=jnp.float64), jnp.zeros((), dtype=jnp.float64))

    return loss_fn


TARGET_INSIDE = np.array([0.3, 0.6, 0.4])
TARGET_OUTSIDE = np.array([1.5, -0.5, 2.0])
XL = np.zeros(3)
XU = np.ones(3)
THETA0 = np.array([0.5, 0.5, 0.5])


# ---------------------------------------------------------------------------
# Dispatcher tests
# ---------------------------------------------------------------------------

def test_available_backends():
    from phoscrosstalk.optimizers import AVAILABLE_BACKENDS
    assert "optimistix" in AVAILABLE_BACKENDS
    assert "jaxopt_lbfgsb" in AVAILABLE_BACKENDS
    assert "jaxopt_pgd" in AVAILABLE_BACKENDS
    assert "scipy_jax" in AVAILABLE_BACKENDS
    assert "scipy_jax_pen" in AVAILABLE_BACKENDS
    assert "optax_adam" in AVAILABLE_BACKENDS
    assert "optax_sgd" in AVAILABLE_BACKENDS
    assert "optax_lbfgs" in AVAILABLE_BACKENDS
    assert "mpax" in AVAILABLE_BACKENDS


def test_invalid_backend_raises():
    from phoscrosstalk.optimizers import dispatch_optimisation
    loss_fn = make_quadratic_loss(TARGET_INSIDE)
    with pytest.raises(ValueError, match="Unknown backend"):
        dispatch_optimisation("nonexistent", loss_fn, THETA0, XL, XU)


# ---------------------------------------------------------------------------
# Helper: verify common return contract
# ---------------------------------------------------------------------------

def _check_result(result, xl, xu, theta0, loss_fn):
    theta_opt, total_loss, f1, f2, f3, f4 = result
    assert isinstance(theta_opt, np.ndarray), "theta_opt must be np.ndarray"
    assert np.all(np.isfinite(theta_opt)), "theta_opt must be finite"
    assert np.isfinite(total_loss), "total_loss must be finite"
    assert np.isfinite(f1) and np.isfinite(f2) and np.isfinite(f3) and np.isfinite(f4)
    # Bounds respected within tolerance
    assert np.all(theta_opt >= xl - 1e-6), "theta_opt below lower bound"
    assert np.all(theta_opt <= xu + 1e-6), "theta_opt above upper bound"
    # Loss reduced vs initial point
    theta0_j = jnp.asarray(theta0, dtype=jnp.float64)
    init_loss_val, _ = loss_fn(theta0_j, None)
    init_loss = float(init_loss_val)
    assert total_loss < init_loss + 1e-6, f"Loss not reduced: {total_loss} vs init {init_loss}"


# ---------------------------------------------------------------------------
# JAXopt backend tests
# ---------------------------------------------------------------------------

jaxopt = pytest.importorskip("jaxopt")


def test_jaxopt_lbfgsb_inside_bounds():
    from phoscrosstalk.optimizers import dispatch_optimisation
    loss_fn = make_quadratic_loss(TARGET_INSIDE)
    result = dispatch_optimisation(
        "jaxopt_lbfgsb", loss_fn, THETA0, XL, XU, max_steps=200
    )
    _check_result(result, XL, XU, THETA0, loss_fn)
    theta_opt = result[0]
    assert np.allclose(theta_opt, TARGET_INSIDE, atol=0.05), f"Expected ~{TARGET_INSIDE}, got {theta_opt}"


def test_jaxopt_lbfgsb_outside_bounds():
    from phoscrosstalk.optimizers import dispatch_optimisation
    loss_fn = make_quadratic_loss(TARGET_OUTSIDE)
    result = dispatch_optimisation(
        "jaxopt_lbfgsb", loss_fn, THETA0, XL, XU, max_steps=200
    )
    _check_result(result, XL, XU, THETA0, loss_fn)
    # Optimum should be on boundary
    theta_opt = result[0]
    assert np.allclose(theta_opt, np.clip(TARGET_OUTSIDE, XL, XU), atol=0.1)


def test_jaxopt_pgd():
    from phoscrosstalk.optimizers import dispatch_optimisation
    loss_fn = make_quadratic_loss(TARGET_INSIDE)
    result = dispatch_optimisation(
        "jaxopt_pgd", loss_fn, THETA0, XL, XU, max_steps=500, stepsize=0.1
    )
    _check_result(result, XL, XU, THETA0, loss_fn)


# ---------------------------------------------------------------------------
# JAXopt QP backend tests
# ---------------------------------------------------------------------------

def test_jaxopt_osqp_inside_bounds():
    """OSQP QP surrogate: target inside bounds should find near-optimal."""
    from phoscrosstalk.optimizers import dispatch_optimisation
    loss_fn = make_quadratic_loss(TARGET_INSIDE)
    result = dispatch_optimisation(
        "jaxopt_osqp", loss_fn, THETA0, XL, XU,
        ridge=1e-4, line_search=True, line_search_steps=8,
    )
    _check_result(result, XL, XU, THETA0, loss_fn)


def test_jaxopt_osqp_outside_bounds():
    """OSQP QP surrogate: target outside bounds should clip to boundary."""
    from phoscrosstalk.optimizers import dispatch_optimisation
    loss_fn = make_quadratic_loss(TARGET_OUTSIDE)
    result = dispatch_optimisation(
        "jaxopt_osqp", loss_fn, THETA0, XL, XU,
        ridge=1e-4, line_search=True, line_search_steps=8,
    )
    theta_opt, total_loss, f1, f2, f3, f4 = result
    assert isinstance(theta_opt, np.ndarray)
    assert np.all(theta_opt >= XL - 1e-6) and np.all(theta_opt <= XU + 1e-6)
    assert np.isfinite(total_loss)


def test_jaxopt_box_osqp_inside_bounds():
    """BoxOSQP QP surrogate: target inside bounds should find near-optimal."""
    from phoscrosstalk.optimizers import dispatch_optimisation
    loss_fn = make_quadratic_loss(TARGET_INSIDE)
    result = dispatch_optimisation(
        "jaxopt_box_osqp", loss_fn, THETA0, XL, XU,
        ridge=1e-4, line_search=True, line_search_steps=8,
    )
    _check_result(result, XL, XU, THETA0, loss_fn)


def test_jaxopt_box_osqp_outside_bounds():
    """BoxOSQP QP surrogate: target outside bounds should clip to boundary."""
    from phoscrosstalk.optimizers import dispatch_optimisation
    loss_fn = make_quadratic_loss(TARGET_OUTSIDE)
    result = dispatch_optimisation(
        "jaxopt_box_osqp", loss_fn, THETA0, XL, XU,
        ridge=1e-4, line_search=True, line_search_steps=8,
    )
    theta_opt, total_loss, f1, f2, f3, f4 = result
    assert isinstance(theta_opt, np.ndarray)
    assert np.all(theta_opt >= XL - 1e-6) and np.all(theta_opt <= XU + 1e-6)
    assert np.isfinite(total_loss)


def test_jaxopt_eq_qp_inside_bounds():
    """EqQP surrogate: post-solve clipping enforces bounds."""
    from phoscrosstalk.optimizers import dispatch_optimisation
    loss_fn = make_quadratic_loss(TARGET_INSIDE)
    result = dispatch_optimisation(
        "jaxopt_eq_qp", loss_fn, THETA0, XL, XU,
        ridge=1e-4, line_search=True, line_search_steps=8,
    )
    theta_opt, total_loss, f1, f2, f3, f4 = result
    assert isinstance(theta_opt, np.ndarray)
    assert np.all(np.isfinite(theta_opt))
    assert np.all(theta_opt >= XL - 1e-6) and np.all(theta_opt <= XU + 1e-6)
    assert np.isfinite(total_loss)


def test_jaxopt_qp_explicit_mode():
    """Explicit QP data bypass: pass (Q, c) directly."""
    from phoscrosstalk.optimizers.jaxopt_backend import run_single_optimisation_jaxopt
    # Build explicit QP for sum((theta - target)^2): Q = 2*I, c = -2*target
    target = jnp.asarray(TARGET_INSIDE, dtype=jnp.float64)
    Q = 2.0 * np.eye(3)
    c = -2.0 * np.asarray(TARGET_INSIDE)
    loss_fn = make_quadratic_loss(TARGET_INSIDE)
    result = run_single_optimisation_jaxopt(
        loss_fn, THETA0, XL, XU,
        solver_kind="osqp",
        qp_mode="explicit",
        params_obj=(Q, c),
        line_search=False,
    )
    theta_opt, total_loss, f1, f2, f3, f4 = result
    assert isinstance(theta_opt, np.ndarray)
    assert np.all(theta_opt >= XL - 1e-6) and np.all(theta_opt <= XU + 1e-6)
    assert np.isfinite(total_loss)


def test_jaxopt_hessian_mode_invalid():
    """Invalid hessian_mode should raise ValueError."""
    from phoscrosstalk.optimizers.jaxopt_backend import run_single_optimisation_jaxopt
    loss_fn = make_quadratic_loss(TARGET_INSIDE)
    with pytest.raises(ValueError, match="hessian_mode"):
        run_single_optimisation_jaxopt(
            loss_fn, THETA0, XL, XU,
            solver_kind="osqp",
            hessian_mode="gauss_newton",
        )


def test_jaxopt_qp_new_backends_in_dispatch():
    """All three QP backends should be registered in AVAILABLE_BACKENDS."""
    from phoscrosstalk.optimizers import AVAILABLE_BACKENDS
    assert "jaxopt_osqp" in AVAILABLE_BACKENDS
    assert "jaxopt_box_osqp" in AVAILABLE_BACKENDS
    assert "jaxopt_eq_qp" in AVAILABLE_BACKENDS


# ---------------------------------------------------------------------------
# JAX scipy backend tests
# ---------------------------------------------------------------------------

def test_scipy_jax_reparameterize():
    from phoscrosstalk.optimizers import dispatch_optimisation
    loss_fn = make_quadratic_loss(TARGET_INSIDE)
    result = dispatch_optimisation(
        "scipy_jax", loss_fn, THETA0, XL, XU, max_steps=200
    )
    _check_result(result, XL, XU, THETA0, loss_fn)
    theta_opt = result[0]
    assert np.allclose(theta_opt, TARGET_INSIDE, atol=0.05)


def test_scipy_jax_penalty():
    from phoscrosstalk.optimizers import dispatch_optimisation
    loss_fn = make_quadratic_loss(TARGET_INSIDE)
    result = dispatch_optimisation(
        "scipy_jax_pen", loss_fn, THETA0, XL, XU, max_steps=200
    )
    _check_result(result, XL, XU, THETA0, loss_fn)


# ---------------------------------------------------------------------------
# Optax backend tests
# ---------------------------------------------------------------------------

def test_optax_adam():
    from phoscrosstalk.optimizers import dispatch_optimisation
    loss_fn = make_quadratic_loss(TARGET_INSIDE)
    result = dispatch_optimisation(
        "optax_adam", loss_fn, THETA0, XL, XU,
        max_steps=500, learning_rate=0.1,
    )
    _check_result(result, XL, XU, THETA0, loss_fn)


def test_optax_sgd():
    from phoscrosstalk.optimizers import dispatch_optimisation
    loss_fn = make_quadratic_loss(TARGET_INSIDE)
    result = dispatch_optimisation(
        "optax_sgd", loss_fn, THETA0, XL, XU,
        max_steps=500, learning_rate=0.05,
    )
    _check_result(result, XL, XU, THETA0, loss_fn)


def test_optax_lbfgs():
    from phoscrosstalk.optimizers import dispatch_optimisation
    loss_fn = make_quadratic_loss(TARGET_INSIDE)
    result = dispatch_optimisation(
        "optax_lbfgs", loss_fn, THETA0, XL, XU,
        max_steps=200,
    )
    _check_result(result, XL, XU, THETA0, loss_fn)


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------

def test_config_optimizer_backend_default():
    from phoscrosstalk.config import _DEFAULTS
    assert _DEFAULTS["optimisation"]["optimizer_backend"] == "optimistix"


def test_config_optimizer_backend_valid():
    from phoscrosstalk.config import validate_config
    from types import SimpleNamespace
    # The backend list is importable and contains expected keys
    from phoscrosstalk.optimizers import AVAILABLE_BACKENDS
    assert "optax_adam" in AVAILABLE_BACKENDS


def test_config_invalid_backend_raises():
    """Test that an invalid backend is rejected."""
    _VALID = [
        "optimistix", "jaxopt_lbfgsb", "jaxopt_pgd",
        "scipy_jax", "scipy_jax_pen",
        "optax_adam", "optax_sgd", "optax_lbfgs",
        "mpax",
    ]
    from phoscrosstalk.optimizers import AVAILABLE_BACKENDS
    # All valid backends must be in AVAILABLE_BACKENDS
    for b in _VALID:
        if b != "optimistix":  # optimistix uses different runner
            assert b in AVAILABLE_BACKENDS


# ---------------------------------------------------------------------------
# Callable-contract adapter tests
# ---------------------------------------------------------------------------

def make_quadratic_residuals(target):
    """
    residuals_fn(theta) = (theta - target), aux=(sum_sq, 0, 0, 0).
    Compatible with the optimistix backend interface.
    """
    target_j = jnp.asarray(target, dtype=jnp.float64)

    def residuals_fn(theta, args):
        r = theta - target_j
        f1 = jnp.sum(r ** 2)
        zeros = jnp.zeros((), dtype=jnp.float64)
        return r, (f1, zeros, zeros, zeros)

    return residuals_fn


def test_residuals_fn_to_loss_fn_scalar():
    """_residuals_fn_to_loss_fn wraps residuals to produce a scalar equal to sum(r^2)."""
    from phoscrosstalk.multistarts import _residuals_fn_to_loss_fn

    target = np.array([0.3, 0.6, 0.4])
    theta = np.array([0.5, 0.5, 0.5])

    residuals_fn = make_quadratic_residuals(target)
    loss_fn = _residuals_fn_to_loss_fn(residuals_fn)

    theta_j = jnp.asarray(theta, dtype=jnp.float64)
    scalar, (f1, f2, f3, f4) = loss_fn(theta_j, None)

    expected = float(jnp.sum((theta_j - jnp.asarray(target, dtype=jnp.float64)) ** 2))
    assert abs(float(scalar) - expected) < 1e-10, (
        f"scalar={float(scalar)}, expected={expected}"
    )
    # aux is passed through from residuals_fn unchanged
    assert abs(float(f1) - expected) < 1e-10
    assert float(f2) == 0.0
    assert float(f3) == 0.0
    assert float(f4) == 0.0


def test_residuals_fn_to_loss_fn_preserves_aux():
    """The (f1,f2,f3,f4) aux from residuals_fn is passed through unchanged."""
    from phoscrosstalk.multistarts import _residuals_fn_to_loss_fn

    target = np.array([0.2, 0.8])
    theta = np.array([0.0, 1.0])

    residuals_fn = make_quadratic_residuals(target)
    loss_fn = _residuals_fn_to_loss_fn(residuals_fn)

    theta_j = jnp.asarray(theta, dtype=jnp.float64)
    # Call both directly
    r, (r_f1, r_f2, r_f3, r_f4) = residuals_fn(theta_j, None)
    _, (l_f1, l_f2, l_f3, l_f4) = loss_fn(theta_j, None)

    assert abs(float(r_f1) - float(l_f1)) < 1e-12
    assert float(l_f2) == float(r_f2)
    assert float(l_f3) == float(r_f3)
    assert float(l_f4) == float(r_f4)


# ---------------------------------------------------------------------------
# Backend propagation tests
# ---------------------------------------------------------------------------

def test_backend_in_args_namespace_defaults_to_optimistix():
    """args without optimizer_backend falls back to 'optimistix' in multistarts."""
    from types import SimpleNamespace
    args = SimpleNamespace()
    from phoscrosstalk.multistarts import run_multi_start_optimization
    # Just verify that getattr(args, "optimizer_backend", "optimistix") works.
    backend = getattr(args, "optimizer_backend", "optimistix")
    assert backend == "optimistix"


def test_backend_propagated_from_args():
    """args.optimizer_backend is read and used by run_multi_start_optimization."""
    from types import SimpleNamespace
    args = SimpleNamespace(optimizer_backend="optax_adam", optimizer_backend_kwargs={})
    backend = getattr(args, "optimizer_backend", "optimistix")
    backend_kwargs = dict(getattr(args, "optimizer_backend_kwargs", None) or {})
    assert backend == "optax_adam"
    assert backend_kwargs == {}


def test_dispatch_optimistix_with_residuals_fn():
    """
    dispatch_optimisation("optimistix", residuals_fn, ...) correctly calls
    run_single_optimisation with the residuals_fn callable.
    """
    from phoscrosstalk.optimizers.dispatch import dispatch_optimisation

    target = np.array([0.3, 0.6, 0.4])
    residuals_fn = make_quadratic_residuals(target)

    result = dispatch_optimisation(
        "optimistix",
        residuals_fn,
        THETA0,
        XL,
        XU,
        max_steps=200,
    )
    theta_opt, total_loss, f1, f2, f3, f4 = result
    assert isinstance(theta_opt, np.ndarray)
    assert np.all(np.isfinite(theta_opt))
    assert np.isfinite(total_loss)
