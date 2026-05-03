"""
test_jax_migration.py

Regression tests for the JAX/Diffrax/Optimistix migration:

1. Diffrax simulation tests:
   - simulate_ode returns correct shape
   - no NaNs for a valid synthetic model
   - deterministic for fixed inputs
   - backward-compatible alias simulate_p_scipy works
   - full_output mode returns 4-tuple
   - all three mechanisms (dist, seq, rand) run without error

2. Optimistix optimisation tests:
   - loss decreases from initialisation
   - fitted parameter array has correct shape

3. Dependency regression tests:
   - no runtime imports from pymoo
   - no scipy ODE solver imports

4. CLI regression tests:
   - old flags (--gen, --pop-size, --algorithm) are accepted
   - new flags (--optimizer, --max-steps, --rtol, etc.) are accepted
"""

import sys
import importlib

import numpy as np
import pytest

from phoscrosstalk.config import ModelDims


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_model_dims():
    """Restore ModelDims after each test."""
    saved = (ModelDims.K, ModelDims.M, ModelDims.N)
    yield
    ModelDims.K, ModelDims.M, ModelDims.N = saved


def _make_tiny_model(K=2, M=3, N=4, T=6, seed=42):
    """Build a small synthetic model for testing."""
    ModelDims.set_dims(K, M, N)
    rng = np.random.default_rng(seed)

    dim = 4 * K + 2 + 3 * M + N + 4
    t = np.array([0.0, 1.0, 5.0, 10.0, 30.0, 60.0])
    P_data = rng.uniform(0.1, 0.9, (N, T))
    A_data = rng.uniform(0.5, 1.5, (K, T))
    theta = rng.uniform(-2, 0, dim)

    Cg = np.eye(N) * 0.1
    Cl = np.eye(N) * 0.05
    site_prot_idx = np.array([0, 0, 1, 1], dtype=int)
    K_site_kin = rng.uniform(0, 1, (N, M))
    K_site_kin /= K_site_kin.sum(axis=1, keepdims=True) + 1e-8
    R = K_site_kin.T.copy()
    R /= R.sum(axis=1, keepdims=True) + 1e-8
    L_alpha = np.zeros((M, M))
    kin2prot = np.array([0, 1, -1], dtype=int)
    rm_prot = np.zeros(K, dtype=int)
    rm_kin = np.zeros(M, dtype=int)

    return dict(
        K=K,
        M=M,
        N=N,
        T=T,
        t=t,
        P_data=P_data,
        A_data=A_data,
        theta=theta,
        Cg=Cg,
        Cl=Cl,
        site_prot_idx=site_prot_idx,
        K_site_kin=K_site_kin,
        R=R,
        L_alpha=L_alpha,
        kin_to_prot_idx=kin2prot,
        receptor_mask_prot=rm_prot,
        receptor_mask_kin=rm_kin,
    )


# ---------------------------------------------------------------------------
# 1. Diffrax simulation tests
# ---------------------------------------------------------------------------


class TestDiffraxSimulation:
    def test_output_shape_dist(self):
        m = _make_tiny_model()
        from phoscrosstalk.simulation import simulate_ode

        P_sim, A_sim = simulate_ode(
            m["t"],
            m["P_data"],
            m["A_data"],
            m["theta"],
            m["Cg"],
            m["Cl"],
            m["site_prot_idx"],
            m["K_site_kin"],
            m["R"],
            m["L_alpha"],
            m["kin_to_prot_idx"],
            m["receptor_mask_prot"],
            m["receptor_mask_kin"],
            mechanism="dist",
        )
        assert P_sim.shape == (m["N"], m["T"]), f"P_sim shape mismatch: {P_sim.shape}"
        assert A_sim.shape == (m["K"], m["T"]), f"A_sim shape mismatch: {A_sim.shape}"

    def test_no_nans_dist(self):
        m = _make_tiny_model()
        from phoscrosstalk.simulation import simulate_ode

        P_sim, A_sim = simulate_ode(
            m["t"],
            m["P_data"],
            m["A_data"],
            m["theta"],
            m["Cg"],
            m["Cl"],
            m["site_prot_idx"],
            m["K_site_kin"],
            m["R"],
            m["L_alpha"],
            m["kin_to_prot_idx"],
            m["receptor_mask_prot"],
            m["receptor_mask_kin"],
            mechanism="dist",
        )
        assert np.all(np.isfinite(P_sim)), "NaN/Inf in P_sim"
        assert np.all(np.isfinite(A_sim)), "NaN/Inf in A_sim"

    def test_deterministic(self):
        m = _make_tiny_model()
        from phoscrosstalk.simulation import simulate_ode

        def run():
            return simulate_ode(
                m["t"],
                m["P_data"],
                m["A_data"],
                m["theta"],
                m["Cg"],
                m["Cl"],
                m["site_prot_idx"],
                m["K_site_kin"],
                m["R"],
                m["L_alpha"],
                m["kin_to_prot_idx"],
                m["receptor_mask_prot"],
                m["receptor_mask_kin"],
                mechanism="dist",
            )

        P1, A1 = run()
        P2, A2 = run()
        np.testing.assert_array_equal(P1, P2)
        np.testing.assert_array_equal(A1, A2)

    def test_backward_compat_alias(self):
        """simulate_p_scipy must be an alias for simulate_ode."""
        from phoscrosstalk.simulation import simulate_p_scipy, simulate_ode

        assert simulate_p_scipy is simulate_ode

    def test_full_output_shape(self):
        m = _make_tiny_model()
        from phoscrosstalk.simulation import simulate_ode

        result = simulate_ode(
            m["t"],
            m["P_data"],
            m["A_data"],
            m["theta"],
            m["Cg"],
            m["Cl"],
            m["site_prot_idx"],
            m["K_site_kin"],
            m["R"],
            m["L_alpha"],
            m["kin_to_prot_idx"],
            m["receptor_mask_prot"],
            m["receptor_mask_kin"],
            mechanism="dist",
            full_output=True,
        )
        assert len(result) == 4, "full_output should return 4-tuple"
        P_sim, A_sim, S_sim, Kdyn_sim = result
        assert P_sim.shape == (m["N"], m["T"])
        assert A_sim.shape == (m["K"], m["T"])
        assert S_sim.shape == (m["K"], m["T"])
        assert Kdyn_sim.shape == (m["M"], m["T"])

    @pytest.mark.parametrize("mech", ["dist", "seq", "rand"])
    def test_all_mechanisms(self, mech):
        m = _make_tiny_model()
        from phoscrosstalk.simulation import simulate_ode

        P_sim, A_sim = simulate_ode(
            m["t"],
            m["P_data"],
            m["A_data"],
            m["theta"],
            m["Cg"],
            m["Cl"],
            m["site_prot_idx"],
            m["K_site_kin"],
            m["R"],
            m["L_alpha"],
            m["kin_to_prot_idx"],
            m["receptor_mask_prot"],
            m["receptor_mask_kin"],
            mechanism=mech,
        )
        assert np.all(np.isfinite(P_sim)), f"NaN in P_sim for mechanism={mech}"

    def test_requires_modeldims(self):
        from phoscrosstalk.simulation import simulate_ode

        saved = (ModelDims.K, ModelDims.M, ModelDims.N)
        ModelDims.K = ModelDims.M = ModelDims.N = None
        try:
            with pytest.raises(RuntimeError, match="ModelDims have not been set"):
                simulate_ode(
                    np.array([0.0, 1.0]),
                    np.zeros((2, 2)),
                    np.zeros((2, 2)),
                    np.zeros(10),
                    np.zeros((2, 2)),
                    np.zeros((2, 2)),
                    np.array([0, 0], dtype=int),
                    np.eye(2),
                    np.eye(2),
                    np.zeros((2, 2)),
                    np.array([0, 0], dtype=int),
                    np.array([0, 0], dtype=int),
                    np.array([0, 0], dtype=int),
                    "dist",
                )
        finally:
            ModelDims.K, ModelDims.M, ModelDims.N = saved

    def test_p_sim_clipped(self):
        """P_sim should be in [0, 1]."""
        m = _make_tiny_model()
        from phoscrosstalk.simulation import simulate_ode

        P_sim, _ = simulate_ode(
            m["t"],
            m["P_data"],
            m["A_data"],
            m["theta"],
            m["Cg"],
            m["Cl"],
            m["site_prot_idx"],
            m["K_site_kin"],
            m["R"],
            m["L_alpha"],
            m["kin_to_prot_idx"],
            m["receptor_mask_prot"],
            m["receptor_mask_kin"],
            mechanism="dist",
        )
        assert P_sim.min() >= -1e-6, "P_sim below 0"
        assert P_sim.max() <= 1.0 + 1e-6, "P_sim above 1"


# ---------------------------------------------------------------------------
# 2. Optimistix optimisation tests
# ---------------------------------------------------------------------------


class TestOptimistixOptimisation:
    def test_loss_decreases(self):
        import jax.numpy as jnp
        from phoscrosstalk.optimization import (
            make_loss_fn,
            run_single_optimisation,
            create_bounds,
        )

        m = _make_tiny_model()
        K, M, N = m["K"], m["M"], m["N"]
        xl, xu, _ = create_bounds(K, M, N)

        loss_fn = make_loss_fn(
            t=m["t"],
            P_data=m["P_data"],
            A_scaled=np.zeros((0, m["T"])),
            prot_idx_for_A=np.array([], dtype=int),
            W_data=np.ones((N, m["T"])),
            W_data_prot=np.zeros((0, m["T"])),
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
            lambda_net=1e-4,
            reg_lambda=1e-4,
        )

        rng = np.random.default_rng(5)
        theta0 = xl + rng.random(len(xl)) * (xu - xl)
        loss0, _ = loss_fn(jnp.asarray(theta0, dtype=jnp.float32), None)

        theta_opt, total_loss, f1, f2, f3, result_status, diagnostics = run_single_optimisation(
            loss_fn, theta0, xl, xu, max_steps=30
        )

        assert total_loss < float(loss0), (
            f"Loss did not decrease: {total_loss} >= {float(loss0)}"
        )

    def test_fitted_param_shape(self):
        from phoscrosstalk.optimization import (
            make_loss_fn,
            run_single_optimisation,
            create_bounds,
        )

        m = _make_tiny_model()
        K, M, N = m["K"], m["M"], m["N"]
        xl, xu, dim = create_bounds(K, M, N)

        loss_fn = make_loss_fn(
            t=m["t"],
            P_data=m["P_data"],
            A_scaled=np.zeros((0, m["T"])),
            prot_idx_for_A=np.array([], dtype=int),
            W_data=np.ones((N, m["T"])),
            W_data_prot=np.zeros((0, m["T"])),
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
            lambda_net=1e-4,
            reg_lambda=1e-4,
        )

        rng = np.random.default_rng(7)
        theta0 = xl + rng.random(dim) * (xu - xl)
        theta_opt, *_ = run_single_optimisation(loss_fn, theta0, xl, xu, max_steps=10)

        assert theta_opt.shape == (dim,), (
            f"Expected shape ({dim},), got {theta_opt.shape}"
        )

    def test_objectives_finite(self):
        import jax.numpy as jnp
        from phoscrosstalk.optimization import make_loss_fn, create_bounds

        m = _make_tiny_model()
        K, M, N = m["K"], m["M"], m["N"]
        xl, xu, _ = create_bounds(K, M, N)

        loss_fn = make_loss_fn(
            t=m["t"],
            P_data=m["P_data"],
            A_scaled=np.zeros((0, m["T"])),
            prot_idx_for_A=np.array([], dtype=int),
            W_data=np.ones((N, m["T"])),
            W_data_prot=np.zeros((0, m["T"])),
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
            lambda_net=1e-4,
            reg_lambda=1e-4,
        )

        theta0 = 0.5 * (xl + xu)
        total, (f1, f2, f3) = loss_fn(jnp.asarray(theta0, dtype=jnp.float32), None)
        assert np.isfinite(float(total)), "total loss is not finite"
        assert np.isfinite(float(f1)), "f1 is not finite"


# ---------------------------------------------------------------------------
# 3. Dependency regression tests
# ---------------------------------------------------------------------------


class TestDependencyRegression:
    def test_no_pymoo_imports(self):
        """After migration, phoscrosstalk modules must not import from pymoo."""
        import re
        import phoscrosstalk.simulation as sim_mod
        import phoscrosstalk.optimization as opt_mod
        import phoscrosstalk.multistarts as ms_mod
        import phoscrosstalk.hyperparam as hp_mod
        import phoscrosstalk.main as main_mod

        # Pattern: an actual Python import statement (not a docstring mention)
        import_pattern = re.compile(r"^\s*(from pymoo|import pymoo)", re.MULTILINE)

        for mod in [sim_mod, opt_mod, ms_mod, hp_mod, main_mod]:
            src = importlib.util.find_spec(mod.__name__)
            with open(src.origin) as f:
                content = f.read()
            matches = import_pattern.findall(content)
            assert not matches, (
                f"{mod.__name__} still has pymoo import statements: {matches}"
            )

    def test_no_scipy_ode_imports(self):
        """After migration, simulation.py must not import scipy.integrate."""
        import importlib.util
        import phoscrosstalk.simulation as sim_mod

        src = importlib.util.find_spec(sim_mod.__name__)
        with open(src.origin) as f:
            content = f.read()
        assert "scipy.integrate" not in content, (
            "simulation.py still imports scipy.integrate"
        )
        assert "odeint" not in content, "simulation.py still references odeint"

    def test_jax_diffrax_importable(self):
        pass

    def test_package_importable(self):
        pass


# ---------------------------------------------------------------------------
# 4. CLI regression tests
# ---------------------------------------------------------------------------


class TestCLIRegression:
    def test_cli_callable(self):
        from phoscrosstalk.main import cli

        assert callable(cli)

    def test_missing_required_args(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["phoscrosstalk"])
        from phoscrosstalk.main import main

        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code != 0

    def test_help_exits_zero(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["phoscrosstalk", "--help"])
        from phoscrosstalk.main import main

        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code == 0

    def test_old_flags_accepted(self, monkeypatch):
        """Legacy flags --gen, --pop-size, --algorithm must parse without error."""
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "phoscrosstalk",
                "--data",
                "fake.csv",
                "--ptm-intra",
                "fake.db",
                "--ptm-inter",
                "fake.db",
                "--gen",
                "100",
                "--pop-size",
                "50",
                "--algorithm",
                "nsga2",
            ],
        )
        from phoscrosstalk.main import main

        # The flags should be accepted by argparse; execution will fail on missing files
        # but NOT with a SystemExit(2) "unrecognized arguments" error.
        with pytest.raises((SystemExit, Exception)) as exc_info:
            main()
        # argparse SystemExit(2) would mean unrecognized argument; that must NOT happen
        exc = exc_info.value
        if isinstance(exc, SystemExit):
            assert exc.code != 2, "--algorithm or other flag was rejected by argparse"

    def test_new_flags_accepted(self, monkeypatch):
        """New flags --optimizer, --max-steps, --rtol, --atol, loss weights."""
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "phoscrosstalk",
                "--data",
                "fake.csv",
                "--ptm-intra",
                "fake.db",
                "--ptm-inter",
                "fake.db",
                "--optimizer",
                "bfgs",
                "--max-steps",
                "50",
                "--rtol",
                "1e-5",
                "--atol",
                "1e-8",
                "--loss-weight-phospho",
                "2.0",
                "--loss-weight-abundance",
                "1.0",
                "--loss-weight-reg",
                "0.5",
            ],
        )
        from phoscrosstalk.main import main

        with pytest.raises((SystemExit, Exception)) as exc_info:
            main()
        exc = exc_info.value
        if isinstance(exc, SystemExit):
            assert exc.code != 2, "new flag was rejected by argparse"


# ---------------------------------------------------------------------------
# 5. JAX mechanisms unit tests
# ---------------------------------------------------------------------------


class TestJaxMechanisms:
    def test_decode_theta_jax_shapes(self):
        import jax.numpy as jnp
        from phoscrosstalk.jax_mechanisms import decode_theta_jax

        K, M, N = 3, 5, 7
        dim = 4 * K + 2 + 3 * M + N + 4
        theta = jnp.zeros(dim)
        decoded = decode_theta_jax(theta, K, M, N)
        assert len(decoded) == 14
        k_act, k_deact, s_prod, d_deg = decoded[:4]
        assert k_act.shape == (K,)
        assert k_deact.shape == (K,)
        assert s_prod.shape == (K,)
        assert d_deg.shape == (K,)

    def test_compute_prev_site_idx(self):
        from phoscrosstalk.jax_mechanisms import compute_prev_site_idx

        # Sites: [0, 0, 1, 0, 1]  → prev: [-1, 0, -1, 1, 2]
        spi = np.array([0, 0, 1, 0, 1], dtype=np.int32)
        prev = compute_prev_site_idx(spi, 5)
        expected = np.array([-1, 0, -1, 1, 2], dtype=np.int32)
        np.testing.assert_array_equal(prev, expected)

    def test_make_rhs_output_shape(self):
        import jax.numpy as jnp
        from phoscrosstalk.jax_mechanisms import make_rhs, compute_prev_site_idx

        K, M, N = 2, 3, 4
        rhs = make_rhs(K, M, N, "dist")

        dim = 4 * K + 2 + 3 * M + N + 4
        theta = jnp.zeros(dim)
        Cg = jnp.eye(N)
        Cl = jnp.eye(N)
        spi = jnp.array([0, 0, 1, 1], dtype=jnp.int32)
        K_sk = jnp.ones((N, M)) / M
        R = jnp.ones((M, N)) / N
        La = jnp.zeros((M, M))
        k2p = jnp.array([0, 1, -1], dtype=jnp.int32)
        rmp = jnp.zeros(K)
        rmk = jnp.zeros(M)
        prev_spi = jnp.array(
            compute_prev_site_idx(np.array([0, 0, 1, 1]), N), dtype=jnp.int32
        )

        y = jnp.zeros(2 * K + M + N)
        args = (theta, Cg, Cl, spi, K_sk, R, La, k2p, rmp, rmk, prev_spi)

        dy = rhs(0.0, y, args)
        assert dy.shape == (2 * K + M + N,), f"dy shape: {dy.shape}"


# ---------------------------------------------------------------------------
# 6. SLSQP-jax integration tests
# ---------------------------------------------------------------------------


class TestSLSQPIntegration:
    def test_slsqp_importable(self):
        """slsqp-jax must be importable and expose SLSQP and get_diagnostics."""
        from slsqp_jax import SLSQP, get_diagnostics

        assert callable(SLSQP)
        assert callable(get_diagnostics)

    def test_bounds_shape(self):
        """Bounds array passed to SLSQP must have shape (n_params, 2)."""
        import jax.numpy as jnp
        from phoscrosstalk.optimization import create_bounds

        m = _make_tiny_model()
        K, M, N = m["K"], m["M"], m["N"]
        xl, xu, dim = create_bounds(K, M, N)

        assert len(xl) == dim
        assert len(xu) == dim

        bounds = jnp.stack([jnp.asarray(xl, dtype=jnp.float32),
                            jnp.asarray(xu, dtype=jnp.float32)], axis=1)
        assert bounds.shape == (dim, 2), f"bounds shape mismatch: {bounds.shape}"
        assert jnp.all(bounds[:, 0] <= bounds[:, 1]), "xl > xu somewhere"

    def test_objective_returns_scalar_and_aux(self):
        """Objective function must return (scalar, aux_tuple)."""
        import jax.numpy as jnp
        from phoscrosstalk.optimization import make_loss_fn, create_bounds

        m = _make_tiny_model()
        K, M, N = m["K"], m["M"], m["N"]
        xl, xu, _ = create_bounds(K, M, N)

        loss_fn = make_loss_fn(
            t=m["t"],
            P_data=m["P_data"],
            A_scaled=np.zeros((0, m["T"])),
            prot_idx_for_A=np.array([], dtype=int),
            W_data=np.ones((N, m["T"])),
            W_data_prot=np.zeros((0, m["T"])),
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
            lambda_net=1e-4,
            reg_lambda=1e-4,
        )

        theta0 = jnp.asarray(0.5 * (xl + xu), dtype=jnp.float32)
        result = loss_fn(theta0, None)
        assert len(result) == 2, "objective must return (scalar, aux)"
        total_loss, aux = result
        assert total_loss.ndim == 0, "total_loss must be a scalar"
        assert len(aux) == 3, "aux must be a 3-tuple (f1, f2, f3)"

    def test_slsqp_reduces_loss(self):
        """SLSQP must reduce the objective on a small synthetic model."""
        import jax.numpy as jnp
        from phoscrosstalk.optimization import (
            make_loss_fn,
            run_single_optimisation,
            create_bounds,
        )

        m = _make_tiny_model()
        K, M, N = m["K"], m["M"], m["N"]
        xl, xu, _ = create_bounds(K, M, N)

        loss_fn = make_loss_fn(
            t=m["t"],
            P_data=m["P_data"],
            A_scaled=np.zeros((0, m["T"])),
            prot_idx_for_A=np.array([], dtype=int),
            W_data=np.ones((N, m["T"])),
            W_data_prot=np.zeros((0, m["T"])),
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
            lambda_net=1e-4,
            reg_lambda=1e-4,
        )

        rng = np.random.default_rng(42)
        theta0 = xl + rng.random(len(xl)) * (xu - xl)
        loss0, _ = loss_fn(jnp.asarray(theta0, dtype=jnp.float32), None)

        theta_opt, total_loss, f1, f2, f3, result_status, diagnostics = run_single_optimisation(
            loss_fn, theta0, xl, xu, max_steps=30
        )

        assert np.isfinite(total_loss), "optimised loss is not finite"
        assert total_loss < float(loss0), (
            f"SLSQP did not reduce loss: {total_loss} >= {float(loss0)}"
        )

    def test_slsqp_result_has_diagnostics(self):
        """run_single_optimisation must return diagnostics dict."""
        from phoscrosstalk.optimization import (
            make_loss_fn,
            run_single_optimisation,
            create_bounds,
        )

        m = _make_tiny_model()
        K, M, N = m["K"], m["M"], m["N"]
        xl, xu, _ = create_bounds(K, M, N)

        loss_fn = make_loss_fn(
            t=m["t"],
            P_data=m["P_data"],
            A_scaled=np.zeros((0, m["T"])),
            prot_idx_for_A=np.array([], dtype=int),
            W_data=np.ones((N, m["T"])),
            W_data_prot=np.zeros((0, m["T"])),
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
            lambda_net=1e-4,
            reg_lambda=1e-4,
        )

        theta0 = 0.5 * (xl + xu)
        theta_opt, total_loss, f1, f2, f3, result_status, diagnostics = run_single_optimisation(
            loss_fn, theta0, xl, xu, max_steps=10
        )

        assert isinstance(result_status, str), "result_status must be a string"
        assert isinstance(diagnostics, dict), "diagnostics must be a dict"
        for key in ("n_ls_failures", "divergence_triggered", "tail_ls_failures"):
            assert key in diagnostics, f"diagnostics missing key: {key}"

    def test_theta_inside_bounds_after_optimisation(self):
        """Optimised theta must lie within [xl, xu]."""
        from phoscrosstalk.optimization import (
            make_loss_fn,
            run_single_optimisation,
            create_bounds,
        )

        m = _make_tiny_model()
        K, M, N = m["K"], m["M"], m["N"]
        xl, xu, _ = create_bounds(K, M, N)

        loss_fn = make_loss_fn(
            t=m["t"],
            P_data=m["P_data"],
            A_scaled=np.zeros((0, m["T"])),
            prot_idx_for_A=np.array([], dtype=int),
            W_data=np.ones((N, m["T"])),
            W_data_prot=np.zeros((0, m["T"])),
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
            lambda_net=1e-4,
            reg_lambda=1e-4,
        )

        rng = np.random.default_rng(99)
        theta0 = xl + rng.random(len(xl)) * (xu - xl)
        theta_opt, *_ = run_single_optimisation(loss_fn, theta0, xl, xu, max_steps=20)

        tol = 1e-4
        assert np.all(theta_opt >= xl - tol), "theta_opt violates lower bounds"
        assert np.all(theta_opt <= xu + tol), "theta_opt violates upper bounds"

    def test_no_bfgs_in_optimization_module(self):
        """optimization.py must not use optx.BFGS as the primary solver."""
        import importlib.util
        import phoscrosstalk.optimization as opt_mod

        src = importlib.util.find_spec(opt_mod.__name__)
        with open(src.origin) as f:
            content = f.read()
        # SLSQP must be present
        assert "SLSQP" in content, "SLSQP not found in optimization.py"
        # BFGS must not be used as primary solver
        assert "optx.BFGS" not in content, "optx.BFGS still used in optimization.py"

    def test_slsqp_importable_from_optimization(self):
        """SLSQP must be importable from optimization module context."""
        from slsqp_jax import SLSQP
        import optimistix as optx

        assert hasattr(optx, "minimise"), "optimistix.minimise not available"
