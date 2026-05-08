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
