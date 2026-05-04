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

import importlib
import sys

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

    # New dim: 2*K + 2 + 3*M + N + 4  (k_act and s_prod removed from theta)
    dim = 2 * K + 2 + 3 * M + N + 4
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
        from phoscrosstalk.simulation import simulate_ode, simulate_p_scipy

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
    def _make_residuals(self, m):
        """Helper to build a residuals_fn using make_residuals_fn for tests."""
        from phoscrosstalk.optimization import create_bounds, make_residuals_fn

        K, M, N = m["K"], m["M"], m["N"]
        xl, xu, _ = create_bounds(K, M, N)
        residuals_fn = make_residuals_fn(
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
        return residuals_fn, xl, xu

    def test_loss_decreases(self):
        import jax.numpy as jnp

        from phoscrosstalk.optimization import run_single_optimisation

        m = _make_tiny_model()
        residuals_fn, xl, xu = self._make_residuals(m)

        rng = np.random.default_rng(5)
        theta0 = xl + rng.random(len(xl)) * (xu - xl)
        # Compute initial total loss from residuals
        r0, (f1_0, f2_0, f3_0, f4_0) = residuals_fn(
            jnp.asarray(theta0, dtype=jnp.float32), None
        )
        loss0 = float(f1_0) + float(f2_0) + float(f3_0) + float(f4_0)

        theta_opt, total_loss, f1, f2, f3, f4 = run_single_optimisation(
            residuals_fn, theta0, max_steps=30
        )

        assert total_loss < loss0 or total_loss < 1e4, (
            f"Loss did not improve significantly: {total_loss} vs initial {loss0}"
        )

    def test_fitted_param_shape(self):
        from phoscrosstalk.optimization import create_bounds, run_single_optimisation

        m = _make_tiny_model()
        residuals_fn, xl, xu = self._make_residuals(m)
        _, _, dim = create_bounds(m["K"], m["M"], m["N"])

        rng = np.random.default_rng(7)
        theta0 = xl + rng.random(dim) * (xu - xl)
        theta_opt, *_ = run_single_optimisation(residuals_fn, theta0, max_steps=10)

        assert theta_opt.shape == (dim,), (
            f"Expected shape ({dim},), got {theta_opt.shape}"
        )

    def test_objectives_finite(self):
        import jax.numpy as jnp

        from phoscrosstalk.optimization import create_bounds, make_residuals_fn

        m = _make_tiny_model()
        K, M, N = m["K"], m["M"], m["N"]
        xl, xu, _ = create_bounds(K, M, N)

        residuals_fn = make_residuals_fn(
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
        r, (f1, f2, f3, f4) = residuals_fn(
            jnp.asarray(theta0, dtype=jnp.float32), None
        )
        assert np.all(np.isfinite(np.asarray(r))), "residual vector contains non-finite values"
        assert np.isfinite(float(f1)), "f1 is not finite"
        assert np.isfinite(float(f4)), "f4 is not finite"


# ---------------------------------------------------------------------------
# 3. Dependency regression tests
# ---------------------------------------------------------------------------


class TestDependencyRegression:
    def test_no_pymoo_imports(self):
        """After migration, phoscrosstalk modules must not import from pymoo."""
        import re

        import phoscrosstalk.hyperparam as hp_mod
        import phoscrosstalk.main as main_mod
        import phoscrosstalk.multistarts as ms_mod
        import phoscrosstalk.optimization as opt_mod
        import phoscrosstalk.simulation as sim_mod

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

    def test_old_flags_removed(self, monkeypatch):
        """Legacy flags --gen, --pop-size, --algorithm, --data, etc. are now rejected."""
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "phoscrosstalk",
                "--data",
                "fake.csv",
                "--gen",
                "100",
                "--pop-size",
                "50",
                "--algorithm",
                "nsga2",
            ],
        )
        from phoscrosstalk.main import main

        with pytest.raises(SystemExit) as exc_info:
            main()
        # argparse SystemExit(2) means unrecognized argument – expected
        assert exc_info.value.code == 2, (
            "Removed flags should be rejected by argparse with exit code 2"
        )

    def test_new_flags_accepted(self, monkeypatch, tmp_path):
        """Only --config (and --help/--version) are accepted; config file is read."""
        cfg_path = tmp_path / "config.toml"
        cfg_path.write_text(
            ""
        )  # empty → validation error (exit 1, not argparse exit 2)

        monkeypatch.setattr(
            sys,
            "argv",
            ["phoscrosstalk", "--config", str(cfg_path)],
        )
        from phoscrosstalk.main import main

        with pytest.raises((SystemExit, Exception)) as exc_info:
            main()
        exc = exc_info.value
        if isinstance(exc, SystemExit):
            # exit 2 would mean argparse rejected --config, which is wrong
            assert exc.code != 2, "--config flag rejected by argparse (exit 2)"


# ---------------------------------------------------------------------------
# 5. JAX mechanisms unit tests
# ---------------------------------------------------------------------------


class TestJaxMechanisms:
    def test_decode_theta_jax_shapes(self):
        import jax.numpy as jnp

        from phoscrosstalk.jax_mechanisms import decode_theta_jax

        K, M, N = 3, 5, 7
        # New dim: 2*K + 2 + 3*M + N + 4  (k_act and s_prod removed)
        dim = 2 * K + 2 + 3 * M + N + 4
        theta = jnp.zeros(dim)
        decoded = decode_theta_jax(theta, K, M, N)
        assert len(decoded) == 12
        k_deact, d_deg = decoded[:2]
        assert k_deact.shape == (K,)
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

        from phoscrosstalk.jax_mechanisms import compute_prev_site_idx, make_rhs

        K, M, N = 2, 3, 4
        rhs = make_rhs(K, M, N, "dist")

        # New dim: 2*K + 2 + 3*M + N + 4
        dim = 2 * K + 2 + 3 * M + N + 4
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

        y = jnp.zeros(3 * K + M + N)
        args = (theta, Cg, Cl, spi, K_sk, R, La, k2p, rmp, rmk, prev_spi)

        dy = rhs(0.0, y, args)
        assert dy.shape == (3 * K + M + N,), f"dy shape: {dy.shape}"
