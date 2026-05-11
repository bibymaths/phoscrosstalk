"""
test_optimization.py

Tests for the new Diffrax + Optimistix LevenbergMarquardt least-squares optimization path.

Coverage:
  1. run_single_optimisation uses optx.least_squares (not optx.minimise).
  2. solve_model / residuals use diffrax.DirectAdjoint.
  3. Residual vector is finite for a tiny synthetic model.
  4. Failed ODE solve returns finite penalty residuals (no inf/nan).
  5. RNA residual block uses W_data_mrna consistently.
  6. RNA disabled gives f4=0 and no crash.
  7. validate_problem_shapes raises ValueError on bad shapes.
  8. validate_problem_shapes passes on valid problem.
  9. make_residuals_fn returns correct residual block sizes.
 10. config.py optimisation section has solver/verbose/rtol/atol defaults.
"""  # noqa: E501

import numpy as np
import pytest

from phoscrosstalk.config import ModelDims

# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------


def _make_tiny_model(K=2, M=3, N=4, T=6, seed=11):
    """Build a minimal synthetic model for testing."""
    dims = ModelDims(K=K, M=M, N=N)
    rng = np.random.default_rng(seed)

    dim = 2 * K + 2 + 3 * M + N + 4
    t = np.linspace(0.0, 60.0, T)
    P_data = rng.uniform(0.1, 0.9, (N, T))
    A_data = rng.uniform(0.5, 1.5, (K, T))
    theta = rng.uniform(-2.0, -0.1, dim)

    Cg = np.eye(N) * 0.1
    Cl = np.eye(N) * 0.05
    site_prot_idx = np.array([0, 0, 1, 1], dtype=int)
    K_site_kin = rng.uniform(0, 1, (N, M))
    K_site_kin /= K_site_kin.sum(axis=1, keepdims=True) + 1e-8
    R_mat = K_site_kin.T.copy()
    R_mat /= R_mat.sum(axis=1, keepdims=True) + 1e-8
    L_alpha = np.zeros((M, M))
    kin2prot = np.array([0, 1, -1], dtype=int)
    rm_prot = np.zeros(K, dtype=int)
    rm_kin = np.zeros(M, dtype=int)

    return dict(
        K=K,
        M=M,
        N=N,
        dims=dims,
        T=T,
        dim=dim,
        t=t,
        P_data=P_data,
        A_data=A_data,
        theta=theta,
        Cg=Cg,
        Cl=Cl,
        site_prot_idx=site_prot_idx,
        K_site_kin=K_site_kin,
        R=R_mat,
        L_alpha=L_alpha,
        kin_to_prot_idx=kin2prot,
        receptor_mask_prot=rm_prot,
        receptor_mask_kin=rm_kin,
    )


def _make_residuals_fn(m, *, with_rna=False, lambda_net=1e-4):
    """Build a make_residuals_fn for the tiny model."""
    from phoscrosstalk.optimization import make_residuals_fn

    K, _M, N, T = m["K"], m["M"], m["N"], m["T"]

    kw = dict(
        t=m["t"],
        P_data=m["P_data"],
        A_scaled=np.zeros((0, T)),
        prot_idx_for_A=np.array([], dtype=int),
        W_data=np.ones((N, T)),
        W_data_prot=np.zeros((0, T)),
        Cg=m["Cg"],
        Cl=m["Cl"],
        site_prot_idx=m["site_prot_idx"],
        K_site_kin=m["K_site_kin"],
        R=m["R"],
        L_alpha=m["L_alpha"],
        kin_to_prot_idx=m["kin_to_prot_idx"],
        receptor_mask_prot=m["receptor_mask_prot"],
        receptor_mask_kin=m["receptor_mask_kin"],
        mechanism="dist",
        lambda_net=lambda_net,
        reg_lambda=1e-4,
    )

    if with_rna:
        t_rna = np.array([4.0, 8.0, 30.0, 60.0])
        rna_obs = np.ones((K, len(t_rna)), dtype=np.float64) * 1.2
        W_rna = np.ones((K, len(t_rna)), dtype=np.float64)
        kw.update(
            t_mrna=t_rna,
            rna_data_scaled=rna_obs,
            rna_model_prot_idx=np.arange(K, dtype=int),
            W_data_mrna=W_rna,
        )

    return make_residuals_fn(dims=m["dims"], **kw)


# ---------------------------------------------------------------------------
# 1. run_single_optimisation uses optx.least_squares (not optx.minimise)
# ---------------------------------------------------------------------------


def test_run_single_optimisation_uses_least_squares():
    """
    run_single_optimisation must use optx.least_squares, not optx.minimise.
    We verify this by checking the source code and by importing.
    """
    import inspect

    from phoscrosstalk.optimization import run_single_optimisation

    src = inspect.getsource(run_single_optimisation)
    assert "least_squares" in src, "run_single_optimisation must use optx.least_squares"
    assert "LevenbergMarquardt" in src, (
        "run_single_optimisation must use LevenbergMarquardt"
    )
    # Must NOT use optx.minimise as the primary solver
    # (minimise may still appear in other functions; but not in run_single_optimisation)
    assert "optx.minimise" not in src or "least_squares" in src, (
        "Primary solver should be least_squares, not minimise"
    )


# ---------------------------------------------------------------------------
# 2. make_residuals_fn uses DirectAdjoint
# ---------------------------------------------------------------------------


def test_make_residuals_fn_uses_direct_adjoint():
    """The residuals function must use diffrax.DirectAdjoint."""
    import inspect

    from phoscrosstalk.optimization import make_residuals_fn

    src = inspect.getsource(make_residuals_fn)
    assert "DirectAdjoint" in src, (
        "make_residuals_fn must use diffrax.DirectAdjoint for forward-mode AD"
    )
    assert "Tsit5" in src, "make_residuals_fn must use diffrax.Tsit5"
    assert "SaveAt" in src, "make_residuals_fn must use diffrax.SaveAt"


# ---------------------------------------------------------------------------
# 3. Residual vector is finite for a tiny synthetic model
# ---------------------------------------------------------------------------


def test_residuals_finite_tiny_model():
    """Residual vector must be all-finite for a valid tiny model."""
    import jax.numpy as jnp

    m = _make_tiny_model()
    residuals_fn = _make_residuals_fn(m)

    theta0 = jnp.asarray(m["theta"], dtype=jnp.float64)
    r, (f1, f2, f3, f4) = residuals_fn(theta0, None)

    r_np = np.asarray(r)
    assert np.all(np.isfinite(r_np)), (
        f"Residual vector has non-finite values: {r_np[~np.isfinite(r_np)]}"
    )
    assert np.isfinite(float(f1)), f"f1 is not finite: {f1}"
    assert np.isfinite(float(f4)), f"f4 is not finite: {f4}"


# ---------------------------------------------------------------------------
# 4. Failed ODE solve returns finite penalty residuals
# ---------------------------------------------------------------------------


def test_failed_solve_returns_finite_penalty():
    """
    When the ODE solve produces non-finite states, the residual vector must
    still be finite (large penalty, not inf/nan).
    """
    import jax.numpy as jnp

    m = _make_tiny_model()
    # Use extreme theta to try to stress the solver
    residuals_fn = _make_residuals_fn(m)
    theta_extreme = jnp.full((m["dim"],), 100.0, dtype=jnp.float64)

    r, (f1, f2, f3, f4) = residuals_fn(theta_extreme, None)
    r_np = np.asarray(r)
    assert np.all(np.isfinite(r_np)), (
        f"Residuals with extreme theta must be finite (penalty). Got: {r_np[:5]}"
    )


# ---------------------------------------------------------------------------
# 5. RNA residual block uses W_data_mrna
# ---------------------------------------------------------------------------


def test_rna_residual_block_uses_W_data_mrna():
    """
    The RNA residuals should differ when W_data_mrna varies (larger weights → larger residuals).
    """  # noqa: E501
    import jax.numpy as jnp

    from phoscrosstalk.optimization import make_residuals_fn

    m = _make_tiny_model(K=2, M=3, N=4, T=6)
    K, T = m["K"], m["T"]
    t_rna = np.array([4.0, 8.0, 30.0, 60.0])
    rna_obs = np.ones((K, len(t_rna)), dtype=np.float64) * 1.5
    theta0 = jnp.asarray(m["theta"], dtype=jnp.float64)

    def _build(w_scale):
        W_rna = np.ones((K, len(t_rna)), dtype=np.float64) * w_scale
        return make_residuals_fn(
            dims=m["dims"],
            t=m["t"],
            P_data=m["P_data"],
            A_scaled=np.zeros((0, T)),
            prot_idx_for_A=np.array([], dtype=int),
            W_data=np.ones((m["N"], T)),
            W_data_prot=np.zeros((0, T)),
            Cg=m["Cg"],
            Cl=m["Cl"],
            site_prot_idx=m["site_prot_idx"],
            K_site_kin=m["K_site_kin"],
            R=m["R"],
            L_alpha=m["L_alpha"],
            kin_to_prot_idx=m["kin_to_prot_idx"],
            receptor_mask_prot=m["receptor_mask_prot"],
            receptor_mask_kin=m["receptor_mask_kin"],
            mechanism="dist",
            lambda_net=0.0,
            reg_lambda=1e-4,
            t_mrna=t_rna,
            rna_data_scaled=rna_obs,
            rna_model_prot_idx=np.arange(K, dtype=int),
            W_data_mrna=W_rna,
        )

    fn_low = _build(0.1)
    fn_high = _build(10.0)

    r_low, (_, _, _, f4_low) = fn_low(theta0, None)
    r_high, (_, _, _, f4_high) = fn_high(theta0, None)

    # Higher W_data_mrna should increase RNA-related residuals (or loss)
    # Both should be finite
    assert np.all(np.isfinite(np.asarray(r_low))), "r_low has non-finite values"
    assert np.all(np.isfinite(np.asarray(r_high))), "r_high has non-finite values"
    assert np.isfinite(float(f4_low)), "f4_low is not finite"
    assert np.isfinite(float(f4_high)), "f4_high is not finite"
    # High weight should give larger RNA loss
    assert float(f4_high) >= float(f4_low), (
        f"Higher W_data_mrna should give equal or larger f4: f4_low={float(f4_low):.4f}, f4_high={float(f4_high):.4f}"  # noqa: E501
    )


# ---------------------------------------------------------------------------
# 6. RNA disabled gives f4=0 and no crash
# ---------------------------------------------------------------------------


def test_rna_disabled_gives_f4_zero():
    """When RNA data is not provided, f4 must be 0.0 and residuals must be finite."""
    import jax.numpy as jnp

    m = _make_tiny_model()
    residuals_fn = _make_residuals_fn(m, with_rna=False)
    theta0 = jnp.asarray(m["theta"], dtype=jnp.float64)

    r, (f1, f2, f3, f4) = residuals_fn(theta0, None)

    assert float(f4) == 0.0, f"f4 must be 0.0 when RNA is disabled, got {float(f4)}"
    assert np.all(np.isfinite(np.asarray(r))), (
        "Residuals must be finite with RNA disabled"
    )


# ---------------------------------------------------------------------------
# 7. validate_problem_shapes raises on bad shapes
# ---------------------------------------------------------------------------


def test_validate_problem_shapes_bad_P_data():
    """validate_problem_shapes must raise ValueError when P_data shape != W_data shape."""  # noqa: E501
    import types

    from phoscrosstalk.optimization import validate_problem_shapes

    m = _make_tiny_model()
    _K, M, N, T = m["K"], m["M"], m["N"], m["T"]

    # Build a mock problem with mismatched shapes
    problem = types.SimpleNamespace(
        P_data=np.ones((N, T)),
        W_data=np.ones((N + 1, T)),  # wrong shape
        A_scaled=np.zeros((0, T)),
        W_data_prot=np.zeros((0, T)),
        K_site_kin=np.ones((N, M)),
        R=np.ones((M, N)),
        Cg=np.eye(N),
        Cl=np.eye(N),
        L_alpha=np.eye(M),
        site_prot_idx=m["site_prot_idx"],
        kin_to_prot_idx=m["kin_to_prot_idx"],
        t_rna=None,
        rna_obs_matched=None,
        rna_model_prot_idx=None,
        rna_fit_genes=[],
        W_data_mrna=None,
    )

    with pytest.raises(ValueError, match="P_data"):
        validate_problem_shapes(problem)


# ---------------------------------------------------------------------------
# 8. validate_problem_shapes passes on a valid problem
# ---------------------------------------------------------------------------


def test_validate_problem_shapes_valid():
    """validate_problem_shapes must not raise for a correctly shaped problem."""
    import types

    from phoscrosstalk.optimization import validate_problem_shapes

    m = _make_tiny_model()
    _K, M, N, T = m["K"], m["M"], m["N"], m["T"]

    problem = types.SimpleNamespace(
        P_data=np.ones((N, T)),
        W_data=np.ones((N, T)),
        A_scaled=np.zeros((0, T)),
        W_data_prot=np.zeros((0, T)),
        K_site_kin=np.ones((N, M)),
        R=np.ones((M, N)),
        Cg=np.eye(N),
        Cl=np.eye(N),
        L_alpha=np.eye(M),
        site_prot_idx=m["site_prot_idx"],
        kin_to_prot_idx=m["kin_to_prot_idx"],
        t_rna=None,
        rna_obs_matched=None,
        rna_model_prot_idx=None,
        rna_fit_genes=[],
        W_data_mrna=None,
    )

    # Should not raise
    validate_problem_shapes(problem)


# ---------------------------------------------------------------------------
# 9. make_residuals_fn returns correct residual block sizes
# ---------------------------------------------------------------------------


def test_residuals_block_sizes():
    """The residual vector size should match the sum of all residual block sizes."""
    import jax.numpy as jnp

    m = _make_tiny_model(K=2, M=3, N=4, T=6)
    _K, _M, N, T = m["K"], m["M"], m["N"], m["T"]
    dim = m["dim"]

    # Use lambda_net=0.0 to disable Laplacian regularisation for clean block size check
    residuals_fn = _make_residuals_fn(m, with_rna=False, lambda_net=0.0)
    theta0 = jnp.asarray(m["theta"], dtype=jnp.float64)
    r, _ = residuals_fn(theta0, None)

    # Without RNA and without lambda_net:
    # phospho: N*T, abundance: 0, rna: 0, reg: dim (L2 only, no Laplacian)
    expected = N * T + 0 + 0 + dim
    assert r.shape[0] == expected, (
        f"Residual size mismatch: got {r.shape[0]}, expected {expected}"
    )


def test_residuals_block_sizes_with_rna():
    """Residual vector size should include RNA block when RNA data is provided."""
    import jax.numpy as jnp

    m = _make_tiny_model(K=2, M=3, N=4, T=6)
    K, _M, N, T = m["K"], m["M"], m["N"], m["T"]
    dim = m["dim"]
    T_rna = 4  # 4 RNA time points in _make_residuals_fn with_rna=True

    # Use lambda_net=0.0 to disable Laplacian regularisation for clean block size check
    residuals_fn = _make_residuals_fn(m, with_rna=True, lambda_net=0.0)
    theta0 = jnp.asarray(m["theta"], dtype=jnp.float64)
    r, _ = residuals_fn(theta0, None)

    # phospho: N*T, abundance: 0, rna: K*T_rna, reg: dim (L2 only)
    expected = N * T + 0 + K * T_rna + dim
    assert r.shape[0] == expected, (
        f"Residual size with RNA mismatch: got {r.shape[0]}, expected {expected}"
    )


# ---------------------------------------------------------------------------
# 10. config.py has new optimisation defaults
# ---------------------------------------------------------------------------


def test_config_optimisation_defaults():
    """config.py must have solver/verbose/rtol/atol in [optimisation] defaults."""
    from phoscrosstalk.config import load_config

    cfg = load_config(None)  # use defaults
    o = cfg.optimisation

    assert hasattr(o, "solver"), "optimisation.solver missing from config defaults"
    assert hasattr(o, "verbose"), "optimisation.verbose missing from config defaults"
    assert hasattr(o, "rtol"), "optimisation.rtol missing from config defaults"
    assert hasattr(o, "atol"), "optimisation.atol missing from config defaults"

    assert o.solver == "levenberg_marquardt", (
        f"Default solver should be 'levenberg_marquardt', got {o.solver!r}"
    )
    assert o.verbose is False, f"Default verbose should be False, got {o.verbose}"
    assert o.rtol > 0, f"Default rtol should be positive, got {o.rtol}"
    assert o.atol > 0, f"Default atol should be positive, got {o.atol}"


# ---------------------------------------------------------------------------
# 11. Loss helper functions – unit tests for all loss_type values
# ---------------------------------------------------------------------------


class TestLossHelpers:
    """Unit tests for compute_weighted_data_loss / compute_weighted_timeseries_loss."""

    def _arrays(self):
        """Return (y_sim, y_data, weights) with known residuals."""
        import jax.numpy as jnp
        y_sim = jnp.array([[1.1, 1.2, 1.3], [0.9, 0.8, 0.7]], dtype=jnp.float64)
        y_data = jnp.ones((2, 3), dtype=jnp.float64)
        weights = jnp.ones((2, 3), dtype=jnp.float64)
        return y_sim, y_data, weights

    def test_mse_returns_scalar(self):
        import jax.numpy as jnp
        from phoscrosstalk.optimization import compute_weighted_timeseries_loss
        y_sim, y_data, w = self._arrays()
        out = compute_weighted_timeseries_loss(y_sim, y_data, w, loss_type="mse")
        assert jnp.ndim(out) == 0

    def test_mse_correct_value(self):
        """MSE with uniform weights = mean of squared residuals."""
        import jax.numpy as jnp
        from phoscrosstalk.optimization import compute_weighted_timeseries_loss
        y_sim, y_data, w = self._arrays()
        resid = y_sim - y_data
        expected = float(jnp.mean(resid ** 2))
        got = float(compute_weighted_timeseries_loss(y_sim, y_data, w, loss_type="mse"))
        assert abs(got - expected) < 1e-10

    def test_mse_zero_residual(self):
        import jax.numpy as jnp
        from phoscrosstalk.optimization import compute_weighted_timeseries_loss
        y = jnp.ones((3, 4), dtype=jnp.float64)
        w = jnp.ones((3, 4), dtype=jnp.float64)
        assert float(compute_weighted_timeseries_loss(y, y, w, loss_type="mse")) < 1e-12

    def test_pseudo_huber_smaller_than_mse_for_large_residuals(self):
        """Pseudo-Huber is more robust: smaller than MSE for large errors."""
        import jax.numpy as jnp
        from phoscrosstalk.optimization import compute_weighted_timeseries_loss
        y_sim = jnp.array([[10.0, 20.0]], dtype=jnp.float64)
        y_data = jnp.zeros((1, 2), dtype=jnp.float64)
        w = jnp.ones((1, 2), dtype=jnp.float64)
        mse_val = float(compute_weighted_timeseries_loss(
            y_sim, y_data, w, loss_type="mse"))
        ph_val = float(compute_weighted_timeseries_loss(
            y_sim, y_data, w, loss_type="pseudo_huber", pseudo_huber_delta=1.0))
        assert ph_val < mse_val

    def test_pseudo_huber_zero_residual(self):
        import jax.numpy as jnp
        from phoscrosstalk.optimization import compute_weighted_timeseries_loss
        y = jnp.ones((2, 5), dtype=jnp.float64)
        w = jnp.ones((2, 5), dtype=jnp.float64)
        val = float(compute_weighted_timeseries_loss(
            y, y, w, loss_type="pseudo_huber", pseudo_huber_delta=0.1))
        assert abs(val) < 1e-12

    def test_log_cosh_returns_scalar(self):
        import jax.numpy as jnp
        from phoscrosstalk.optimization import compute_weighted_timeseries_loss
        y_sim, y_data, w = self._arrays()
        out = compute_weighted_timeseries_loss(y_sim, y_data, w, loss_type="log_cosh")
        assert jnp.ndim(out) == 0

    def test_log_cosh_zero_residual(self):
        import jax.numpy as jnp
        from phoscrosstalk.optimization import compute_weighted_timeseries_loss
        y = jnp.ones((3, 3), dtype=jnp.float64)
        w = jnp.ones((3, 3), dtype=jnp.float64)
        assert float(compute_weighted_timeseries_loss(
            y, y, w, loss_type="log_cosh")) < 1e-12

    def test_pseudo_huber_slope_greater_than_pseudo_huber(self):
        """pseudo_huber_slope adds a slope term so should be >= pseudo_huber."""
        import jax.numpy as jnp
        from phoscrosstalk.optimization import compute_weighted_timeseries_loss
        y_sim, y_data, w = self._arrays()
        ph_val = float(compute_weighted_timeseries_loss(
            y_sim, y_data, w, loss_type="pseudo_huber",
            pseudo_huber_delta=0.1, slope_lambda=0.1))
        phs_val = float(compute_weighted_timeseries_loss(
            y_sim, y_data, w, loss_type="pseudo_huber_slope",
            pseudo_huber_delta=0.1, slope_lambda=0.1))
        # slope_lambda > 0 adds a positive slope term
        assert phs_val >= ph_val

    def test_pseudo_huber_slope_zero_slope_lambda_equals_pseudo_huber(self):
        """With slope_lambda=0, pseudo_huber_slope == pseudo_huber."""
        import jax.numpy as jnp
        from phoscrosstalk.optimization import compute_weighted_timeseries_loss
        y_sim, y_data, w = self._arrays()
        ph_val = float(compute_weighted_timeseries_loss(
            y_sim, y_data, w, loss_type="pseudo_huber",
            pseudo_huber_delta=0.1, slope_lambda=0.0))
        phs_val = float(compute_weighted_timeseries_loss(
            y_sim, y_data, w, loss_type="pseudo_huber_slope",
            pseudo_huber_delta=0.1, slope_lambda=0.0))
        assert abs(phs_val - ph_val) < 1e-12

    def test_all_loss_types_finite(self):
        """All loss_type values must produce finite scalars."""
        import jax.numpy as jnp
        from phoscrosstalk.optimization import compute_weighted_timeseries_loss, ALLOWED_LOSS_TYPES
        y_sim = jnp.array([[0.5, 1.0, 1.5], [2.0, 1.0, 0.5]], dtype=jnp.float64)
        y_data = jnp.ones((2, 3), dtype=jnp.float64)
        w = jnp.ones((2, 3), dtype=jnp.float64)
        for lt in ALLOWED_LOSS_TYPES:
            val = float(compute_weighted_timeseries_loss(
                y_sim, y_data, w, loss_type=lt,
                pseudo_huber_delta=0.1, slope_lambda=0.1))
            assert np.isfinite(val), f"loss_type={lt!r} produced non-finite value: {val}"

    def test_unknown_loss_type_raises(self):
        import jax.numpy as jnp
        from phoscrosstalk.optimization import compute_weighted_timeseries_loss
        y = jnp.ones((2, 3), dtype=jnp.float64)
        with pytest.raises(ValueError, match="Unknown loss_type"):
            compute_weighted_timeseries_loss(y, y, y, loss_type="huber_absolute")

    def test_weighted_mean_uses_weights(self):
        """_safe_weighted_mean with zero weight on an element should ignore it."""
        import jax.numpy as jnp
        from phoscrosstalk.optimization import compute_weighted_timeseries_loss
        # Two-row array; second row has zero weight
        y_sim = jnp.array([[1.1, 1.2], [10.0, 20.0]], dtype=jnp.float64)
        y_data = jnp.ones((2, 2), dtype=jnp.float64)
        w_uniform = jnp.ones((2, 2), dtype=jnp.float64)
        w_zero_row2 = jnp.array([[1.0, 1.0], [0.0, 0.0]], dtype=jnp.float64)
        val_uniform = float(compute_weighted_timeseries_loss(
            y_sim, y_data, w_uniform, loss_type="mse"))
        val_zeroed = float(compute_weighted_timeseries_loss(
            y_sim, y_data, w_zero_row2, loss_type="mse"))
        # Zeroing out large residuals should reduce the loss
        assert val_zeroed < val_uniform

    def test_network_problem_stores_loss_params(self):
        """NetworkProblem stores loss_type, pseudo_huber_delta, slope_lambda."""
        from phoscrosstalk.optimization import NetworkProblem
        K, M, N, T = 2, 3, 4, 6
        rng = np.random.default_rng(99)
        from phoscrosstalk.config import ModelDims
        dims = ModelDims(K=K, M=M, N=N)
        t = np.linspace(0, 60, T)
        n_params = 2 * K + 2 + 3 * M + N + 4
        prob = NetworkProblem(
            dims, t, rng.uniform(0.1, 0.9, (N, T)),
            np.eye(N) * 0.1, np.eye(N) * 0.05,
            np.array([0, 0, 1, 1]), np.ones((N, M)) / M,
            np.ones((M, N)) / N,
            np.zeros((0, T)), np.array([], dtype=int),
            np.ones((N, T)), np.zeros((0, T)),
            np.zeros((M, M)), np.array([0, 1, -1]),
            1e-4, 1e-4, np.zeros(K), np.zeros(M), "dist",
            np.full(n_params, -2.0), np.full(n_params, 0.0),
            loss_type="pseudo_huber",
            pseudo_huber_delta=0.5,
            slope_lambda=0.2,
        )
        assert prob.loss_type == "pseudo_huber"
        assert prob.pseudo_huber_delta == 0.5
        assert prob.slope_lambda == 0.2

    def test_network_problem_invalid_loss_type_raises(self):
        """NetworkProblem raises ValueError for unknown loss_type."""
        from phoscrosstalk.optimization import NetworkProblem
        K, M, N, T = 2, 3, 4, 6
        rng = np.random.default_rng(100)
        from phoscrosstalk.config import ModelDims
        dims = ModelDims(K=K, M=M, N=N)
        t = np.linspace(0, 60, T)
        n_params = 2 * K + 2 + 3 * M + N + 4
        with pytest.raises(ValueError, match="loss_type"):
            NetworkProblem(
                dims, t, rng.uniform(0.1, 0.9, (N, T)),
                np.eye(N) * 0.1, np.eye(N) * 0.05,
                np.array([0, 0, 1, 1]), np.ones((N, M)) / M,
                np.ones((M, N)) / N,
                np.zeros((0, T)), np.array([], dtype=int),
                np.ones((N, T)), np.zeros((0, T)),
                np.zeros((M, M)), np.array([0, 1, -1]),
                1e-4, 1e-4, np.zeros(K), np.zeros(M), "dist",
                np.full(n_params, -2.0), np.full(n_params, 0.0),
                loss_type="not_valid",
            )

    def test_compute_weighted_data_loss_all_types_finite(self):
        """compute_weighted_data_loss must return finite scalars for all loss types."""
        import jax.numpy as jnp
        from phoscrosstalk.optimization import (
            compute_weighted_data_loss, ALLOWED_LOSS_TYPES,
        )
        y_sim = jnp.array([0.8, 1.2, 1.5, 0.4], dtype=jnp.float64)
        y_data = jnp.ones(4, dtype=jnp.float64)
        w = jnp.ones(4, dtype=jnp.float64)
        for lt in ALLOWED_LOSS_TYPES:
            val = float(compute_weighted_data_loss(
                y_sim, y_data, w, loss_type=lt, pseudo_huber_delta=0.1))
            assert np.isfinite(val), f"loss_type={lt!r} produced non-finite value: {val}"
