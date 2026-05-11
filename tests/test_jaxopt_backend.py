import numpy as np
import jax.numpy as jnp
import pytest

from phoscrosstalk.optimizers.jaxopt_backend import run_single_optimisation_jaxopt


def _loss_fn(theta, args):
    val = jnp.sum((theta - 0.2) ** 2)
    z = jnp.array(0.0, dtype=jnp.float64)
    return val, (val, z, z, z)


def test_jaxopt_lbfgsb_and_pgd_return_contract():
    theta0 = np.array([0.9, 0.1], dtype=np.float64)
    xl = np.zeros(2, dtype=np.float64)
    xu = np.ones(2, dtype=np.float64)
    for solver in ("lbfgsb", "projected_gradient"):
        theta_opt, total_loss, f1, f2, f3, f4 = run_single_optimisation_jaxopt(
            _loss_fn,
            theta0,
            xl,
            xu,
            solver_kind=solver,
            max_steps=50,
        )
        assert theta_opt.shape == theta0.shape
        assert np.isfinite(total_loss)
        assert np.isfinite([f1, f2, f3, f4]).all()
        assert np.all(theta_opt >= xl - 1e-8)
        assert np.all(theta_opt <= xu + 1e-8)


def test_invalid_solver_kind_raises():
    theta0 = np.array([0.5, 0.5], dtype=np.float64)
    xl = np.zeros(2, dtype=np.float64)
    xu = np.ones(2, dtype=np.float64)
    with pytest.raises(ValueError, match="Unknown solver_kind"):
        run_single_optimisation_jaxopt(
            _loss_fn, theta0, xl, xu, solver_kind="bad_solver"
        )


def test_jaxopt_box_osqp_simple_qp_contract():
    theta0 = np.array([0.9, 0.1], dtype=np.float64)
    xl = np.zeros(2, dtype=np.float64)
    xu = np.ones(2, dtype=np.float64)
    theta_opt, total_loss, f1, f2, f3, f4 = run_single_optimisation_jaxopt(
        _loss_fn,
        theta0,
        xl,
        xu,
        solver_kind="box_osqp",
        max_steps=50,
        line_search=False,
    )
    assert theta_opt.shape == theta0.shape
    assert np.all(theta_opt >= xl - 1e-8)
    assert np.all(theta_opt <= xu + 1e-8)
    assert np.isfinite(total_loss)
    assert np.isfinite([f1, f2, f3, f4]).all()
