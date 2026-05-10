"""
tests/test_pinn.py
Tests for the phoscrosstalk.pinn subpackage.

Uses small synthetic fixtures. Does not run the full mechanistic pipeline.
"""

from __future__ import annotations

import os
import types
import json

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Environment setup: must be done before any JAX import.
# ---------------------------------------------------------------------------

os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("JAX_PLATFORMS", "cpu")


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_small_dims(K=3, M=2, N=4):
    """Return (K, M, N) and a ModelDims instance."""
    from phoscrosstalk.config import ModelDims
    return K, M, N, ModelDims(K=K, M=M, N=N)


def _make_pinn_cfg(**overrides):
    defaults = dict(
        enabled=True,
        width_size=8,
        depth=1,
        activation="tanh",
        lambda_pinn=0.1,
        regularize="residual_l2",
        max_steps=3,
        learning_rate=1e-3,
        rtol=1e-3,
        atol=1e-5,
        seed=0,
        verbose=False,
        print_every=1,
        grad_clip=1.0,
        optimizer="adam",
        use_max_machine_threads=False,
    )
    defaults.update(overrides)
    return types.SimpleNamespace(**defaults)


def _make_synthetic_inputs(K=3, M=2, N=4, T=5):
    """Build minimal synthetic arrays for PINN tests."""
    rng = np.random.default_rng(42)
    t = np.linspace(0.0, 10.0, T)
    P_scaled = np.abs(rng.normal(1.0, 0.3, (N, T)))
    A_scaled = np.abs(rng.normal(1.0, 0.3, (K, T)))
    W_data     = np.ones((N, T))
    W_data_prot = np.ones((K, T))
    Cg = np.eye(N)
    Cl = np.eye(N)
    site_prot_idx   = np.zeros(N, dtype=np.int32)
    K_site_kin      = np.ones((N, M))
    R               = np.ones((M, N))
    L_alpha         = np.eye(M)
    kin_to_prot_idx = np.zeros(M, dtype=np.int32)
    prot_idx_for_A  = np.arange(K, dtype=np.int32)
    receptor_mask_prot = np.ones(K)
    receptor_mask_kin  = np.ones(M)
    return dict(
        t=t,
        P_scaled=P_scaled,
        A_scaled=A_scaled,
        W_data=W_data,
        W_data_prot=W_data_prot,
        Cg=Cg,
        Cl=Cl,
        site_prot_idx=site_prot_idx,
        K_site_kin=K_site_kin,
        R=R,
        L_alpha=L_alpha,
        kin_to_prot_idx=kin_to_prot_idx,
        prot_idx_for_A=prot_idx_for_A,
        receptor_mask_prot=receptor_mask_prot,
        receptor_mask_kin=receptor_mask_kin,
    )


# ===========================================================================
# 1. Config tests
# ===========================================================================


class TestPINNConfig:
    def test_pinn_disabled_defaults(self):
        """[pinn] enabled = false loads correctly from defaults."""
        from phoscrosstalk.config import load_config
        cfg = load_config(None)
        assert hasattr(cfg, "pinn"), "cfg.pinn must exist even without [pinn] section"
        assert cfg.pinn.enabled is False

    def test_pinn_enabled_from_toml(self, tmp_path):
        """[pinn] enabled = true loads correctly."""
        toml = tmp_path / "cfg.toml"
        toml.write_text(
            "[paths]\ndata = ''\nptm_intra = ''\nptm_inter = ''\n"
            "[pinn]\nenabled = true\nwidth_size = 16\ndepth = 2\n"
            "activation = 'tanh'\nlambda_pinn = 0.05\nregularize = 'param_l2'\n"
            "max_steps = 10\nlearning_rate = 1e-3\nrtol = 1e-6\natol = 1e-8\n"
            "seed = 1\nverbose = true\nprint_every = 5\ngrad_clip = 0.5\n"
            "optimizer = 'adam'\nuse_max_machine_threads = false\n"
        )
        from phoscrosstalk.config import load_config
        cfg = load_config(str(toml))
        assert cfg.pinn.enabled is True
        assert cfg.pinn.width_size == 16
        assert cfg.pinn.depth == 2
        assert cfg.pinn.lambda_pinn == pytest.approx(0.05)

    def test_pinn_invalid_width_raises(self):
        from phoscrosstalk.pinn.config import validate_pinn_config
        bad = types.SimpleNamespace(
            width_size=0, depth=1, activation="tanh",
            lambda_pinn=0.1, max_steps=10, regularize="residual_l2",
            rtol=1e-6, atol=1e-8,
        )
        with pytest.raises(ValueError, match="width_size"):
            validate_pinn_config(bad)

    def test_pinn_invalid_depth_raises(self):
        from phoscrosstalk.pinn.config import validate_pinn_config
        bad = types.SimpleNamespace(
            width_size=8, depth=0, activation="tanh",
            lambda_pinn=0.1, max_steps=10, regularize="residual_l2",
            rtol=1e-6, atol=1e-8,
        )
        with pytest.raises(ValueError, match="depth"):
            validate_pinn_config(bad)

    def test_pinn_invalid_lambda_raises(self):
        from phoscrosstalk.pinn.config import validate_pinn_config
        bad = types.SimpleNamespace(
            width_size=8, depth=1, activation="tanh",
            lambda_pinn=-0.1, max_steps=10, regularize="residual_l2",
            rtol=1e-6, atol=1e-8,
        )
        with pytest.raises(ValueError, match="lambda_pinn"):
            validate_pinn_config(bad)

    def test_pinn_invalid_activation_raises(self):
        from phoscrosstalk.pinn.config import validate_pinn_config
        bad = types.SimpleNamespace(
            width_size=8, depth=1, activation="linear",
            lambda_pinn=0.1, max_steps=10, regularize="residual_l2",
            rtol=1e-6, atol=1e-8,
        )
        with pytest.raises(ValueError, match="activation"):
            validate_pinn_config(bad)

    def test_pinn_invalid_max_steps_raises(self):
        from phoscrosstalk.pinn.config import validate_pinn_config
        bad = types.SimpleNamespace(
            width_size=8, depth=1, activation="tanh",
            lambda_pinn=0.1, max_steps=0, regularize="residual_l2",
            rtol=1e-6, atol=1e-8,
        )
        with pytest.raises(ValueError, match="max_steps"):
            validate_pinn_config(bad)

    def test_pinn_invalid_regularize_raises(self):
        from phoscrosstalk.pinn.config import validate_pinn_config
        bad = types.SimpleNamespace(
            width_size=8, depth=1, activation="tanh",
            lambda_pinn=0.1, max_steps=10, regularize="unknown",
            rtol=1e-6, atol=1e-8,
        )
        with pytest.raises(ValueError, match="regularize"):
            validate_pinn_config(bad)


# ===========================================================================
# 2. Model tests
# ===========================================================================


class TestPINNModel:
    def test_forward_shape(self):
        """PINN model forward pass returns shape (state_dim,)."""
        import jax
        import jax.numpy as jnp
        from phoscrosstalk.pinn.model import PINNAugmentation

        K, M, N = 3, 2, 4
        state_dim = 3 * K + M + N
        key = jax.random.PRNGKey(0)
        model = PINNAugmentation(state_dim=state_dim, width_size=8, depth=1,
                                  activation="tanh", key=key)
        x = jnp.zeros(state_dim, dtype=jnp.float64)
        t = jnp.asarray(0.0, dtype=jnp.float64)
        out = model(x, t)
        assert out.shape == (state_dim,), f"Expected ({state_dim},) got {out.shape}"

    def test_deterministic_init(self):
        """Same key gives identical model parameters."""
        import jax
        import jax.numpy as jnp
        from phoscrosstalk.pinn.model import PINNAugmentation

        state_dim = 10
        key = jax.random.PRNGKey(7)
        m1 = PINNAugmentation(state_dim=state_dim, width_size=4, depth=1, activation="tanh", key=key)
        m2 = PINNAugmentation(state_dim=state_dim, width_size=4, depth=1, activation="tanh", key=key)
        x = jnp.ones(state_dim, dtype=jnp.float64)
        t = jnp.asarray(1.0, dtype=jnp.float64)
        out1 = m1(x, t)
        out2 = m2(x, t)
        np.testing.assert_allclose(np.asarray(out1), np.asarray(out2))

    def test_output_finite(self):
        """Model output is finite for finite input."""
        import jax
        import jax.numpy as jnp
        from phoscrosstalk.pinn.model import PINNAugmentation

        state_dim = 6
        key = jax.random.PRNGKey(1)
        model = PINNAugmentation(state_dim=state_dim, width_size=8, depth=2,
                                  activation="tanh", key=key)
        rng = np.random.default_rng(0)
        x = jnp.asarray(rng.normal(size=state_dim), dtype=jnp.float64)
        t = jnp.asarray(5.0, dtype=jnp.float64)
        out = model(x, t)
        assert np.all(np.isfinite(np.asarray(out))), "PINN model output contains non-finite values"

    def test_all_activations(self):
        """All supported activation functions produce finite outputs."""
        import jax
        import jax.numpy as jnp
        from phoscrosstalk.pinn.model import PINNAugmentation, _ACTIVATIONS

        state_dim = 5
        x = jnp.ones(state_dim, dtype=jnp.float64)
        t = jnp.asarray(0.0, dtype=jnp.float64)
        for act in _ACTIVATIONS:
            key = jax.random.PRNGKey(0)
            model = PINNAugmentation(state_dim=state_dim, width_size=4, depth=1,
                                      activation=act, key=key)
            out = model(x, t)
            assert np.all(np.isfinite(np.asarray(out))), f"Non-finite output for activation={act}"


# ===========================================================================
# 3. RHS tests
# ===========================================================================


class TestPINNRHS:
    def test_combined_rhs_shape(self):
        """Combined RHS produces output of shape (state_dim,)."""
        import jax
        import jax.numpy as jnp
        from phoscrosstalk.pinn.rhs import make_combined_rhs
        from phoscrosstalk.pinn.model import PINNAugmentation
        from phoscrosstalk.mechanisms import compute_prev_site_idx

        K, M, N = 3, 2, 4
        state_dim = 3 * K + M + N
        combined = make_combined_rhs(K, M, N, "dist")

        key = jax.random.PRNGKey(0)
        pinn_model = PINNAugmentation(state_dim=state_dim, width_size=4, depth=1,
                                       activation="tanh", key=key)

        inps = _make_synthetic_inputs(K, M, N)
        Cg_j   = jnp.asarray(inps["Cg"],              dtype=jnp.float64)
        Cl_j   = jnp.asarray(inps["Cl"],              dtype=jnp.float64)
        K_sk_j = jnp.asarray(inps["K_site_kin"],      dtype=jnp.float64)
        R_j    = jnp.asarray(inps["R"],               dtype=jnp.float64)
        La_j   = jnp.asarray(inps["L_alpha"],         dtype=jnp.float64)
        spi_j  = jnp.asarray(inps["site_prot_idx"],   dtype=jnp.int32)
        k2p_j  = jnp.asarray(inps["kin_to_prot_idx"], dtype=jnp.int32)
        rmp_j  = jnp.asarray(inps["receptor_mask_prot"], dtype=jnp.float64)
        rmk_j  = jnp.asarray(inps["receptor_mask_kin"],  dtype=jnp.float64)
        psi_j  = jnp.asarray(compute_prev_site_idx(inps["site_prot_idx"].astype(np.int32), N),
                              dtype=jnp.int32)

        from phoscrosstalk.optimization import create_bounds
        K2, M2, N2 = K, M, N
        xl, xu, _ = create_bounds(K2, M2, N2)
        theta = jnp.asarray((xl + xu) / 2.0, dtype=jnp.float64)

        args = (theta, Cg_j, Cl_j, spi_j, K_sk_j, R_j, La_j, k2p_j, rmp_j, rmk_j, psi_j,
                pinn_model)
        y = jnp.zeros(state_dim, dtype=jnp.float64)
        t = jnp.asarray(0.0, dtype=jnp.float64)
        dy = combined(t, y, args)
        assert dy.shape == (state_dim,), f"Expected ({state_dim},) got {dy.shape}"

    def test_disabled_pinn_noop(self):
        """When pinn_model is None, combined RHS equals mechanistic RHS exactly."""
        import jax.numpy as jnp
        from phoscrosstalk.pinn.rhs import make_combined_rhs
        from phoscrosstalk.mechanisms import make_rhs, compute_prev_site_idx

        K, M, N = 2, 2, 3
        state_dim = 3 * K + M + N
        combined = make_combined_rhs(K, M, N, "dist")
        mech     = make_rhs(K, M, N, "dist")

        inps = _make_synthetic_inputs(K, M, N)
        from phoscrosstalk.optimization import create_bounds
        xl, xu, _ = create_bounds(K, M, N)
        theta = jnp.asarray((xl + xu) / 2.0, dtype=jnp.float64)

        Cg_j   = jnp.asarray(inps["Cg"],              dtype=jnp.float64)
        Cl_j   = jnp.asarray(inps["Cl"],              dtype=jnp.float64)
        K_sk_j = jnp.asarray(inps["K_site_kin"],      dtype=jnp.float64)
        R_j    = jnp.asarray(inps["R"],               dtype=jnp.float64)
        La_j   = jnp.asarray(inps["L_alpha"],         dtype=jnp.float64)
        spi_j  = jnp.asarray(inps["site_prot_idx"],   dtype=jnp.int32)
        k2p_j  = jnp.asarray(inps["kin_to_prot_idx"], dtype=jnp.int32)
        rmp_j  = jnp.asarray(inps["receptor_mask_prot"], dtype=jnp.float64)
        rmk_j  = jnp.asarray(inps["receptor_mask_kin"],  dtype=jnp.float64)
        psi_j  = jnp.asarray(compute_prev_site_idx(inps["site_prot_idx"].astype(np.int32), N),
                              dtype=jnp.int32)
        mech_args = (theta, Cg_j, Cl_j, spi_j, K_sk_j, R_j, La_j, k2p_j, rmp_j, rmk_j, psi_j)

        y = jnp.zeros(state_dim, dtype=jnp.float64)
        t = jnp.asarray(0.0, dtype=jnp.float64)

        dy_mech     = mech(t, y, mech_args)
        dy_combined = combined(t, y, (*mech_args, None))

        np.testing.assert_allclose(
            np.asarray(dy_mech), np.asarray(dy_combined), rtol=1e-12,
            err_msg="Combined RHS with None PINN should equal mechanistic RHS exactly."
        )


# ===========================================================================
# 4. Loss tests
# ===========================================================================


class TestPINNLoss:
    def test_loss_returns_finite(self):
        """PINN loss returns finite total and finite components."""
        import jax
        import jax.numpy as jnp
        from phoscrosstalk.pinn.loss import make_pinn_loss_fn
        from phoscrosstalk.pinn.model import PINNAugmentation

        K, M, N, dims = _make_small_dims()
        inps = _make_synthetic_inputs(K, M, N)

        from phoscrosstalk.optimization import create_bounds
        xl, xu, _ = create_bounds(K, M, N)
        theta0 = jnp.asarray((xl + xu) / 2.0, dtype=jnp.float64)

        key = jax.random.PRNGKey(0)
        pinn_model = PINNAugmentation(state_dim=3 * K + M + N, width_size=4, depth=1,
                                       activation="tanh", key=key)

        loss_fn = make_pinn_loss_fn(
            dims=dims, t=inps["t"], P_data=inps["P_scaled"], A_scaled=inps["A_scaled"],
            prot_idx_for_A=inps["prot_idx_for_A"], W_data=inps["W_data"],
            W_data_prot=inps["W_data_prot"], Cg=inps["Cg"], Cl=inps["Cl"],
            site_prot_idx=inps["site_prot_idx"], K_site_kin=inps["K_site_kin"],
            R=inps["R"], L_alpha=inps["L_alpha"], kin_to_prot_idx=inps["kin_to_prot_idx"],
            receptor_mask_prot=inps["receptor_mask_prot"],
            receptor_mask_kin=inps["receptor_mask_kin"],
            mechanism="dist", lambda_net=0.0, reg_lambda=0.0, lambda_pinn=0.1,
            rtol=1e-3, atol=1e-5, max_steps=512,
        )
        total, aux = loss_fn((theta0, pinn_model), None)
        assert float(total) >= 0.0
        assert np.isfinite(float(total)), "Total loss must be finite"
        f1, f2, f3, f4, fp = tuple(float(a) for a in aux)
        for name, val in [("f1", f1), ("f2", f2), ("f3", f3), ("f4", f4), ("f_pinn_reg", fp)]:
            assert np.isfinite(val), f"{name} is not finite"

    def test_f_pinn_reg_nonneg(self):
        """f_pinn_reg is non-negative."""
        import jax
        import jax.numpy as jnp
        from phoscrosstalk.pinn.loss import make_pinn_loss_fn
        from phoscrosstalk.pinn.model import PINNAugmentation

        K, M, N, dims = _make_small_dims()
        inps = _make_synthetic_inputs(K, M, N)
        from phoscrosstalk.optimization import create_bounds
        xl, xu, _ = create_bounds(K, M, N)
        theta0 = jnp.asarray((xl + xu) / 2.0, dtype=jnp.float64)
        key = jax.random.PRNGKey(0)
        pinn_model = PINNAugmentation(state_dim=3 * K + M + N, width_size=4, depth=1,
                                       activation="tanh", key=key)
        loss_fn = make_pinn_loss_fn(
            dims=dims, t=inps["t"], P_data=inps["P_scaled"], A_scaled=inps["A_scaled"],
            prot_idx_for_A=inps["prot_idx_for_A"], W_data=inps["W_data"],
            W_data_prot=inps["W_data_prot"], Cg=inps["Cg"], Cl=inps["Cl"],
            site_prot_idx=inps["site_prot_idx"], K_site_kin=inps["K_site_kin"],
            R=inps["R"], L_alpha=inps["L_alpha"], kin_to_prot_idx=inps["kin_to_prot_idx"],
            receptor_mask_prot=inps["receptor_mask_prot"],
            receptor_mask_kin=inps["receptor_mask_kin"],
            mechanism="dist", lambda_net=0.0, reg_lambda=0.0, lambda_pinn=0.1,
            rtol=1e-3, atol=1e-5, max_steps=512,
        )
        _, aux = loss_fn((theta0, pinn_model), None)
        fp = float(aux[4])
        assert fp >= 0.0, f"f_pinn_reg should be non-negative, got {fp}"

    def test_param_l2_regularize(self):
        """param_l2 regularize mode returns finite non-negative f_pinn_reg."""
        import jax
        import jax.numpy as jnp
        from phoscrosstalk.pinn.loss import make_pinn_loss_fn
        from phoscrosstalk.pinn.model import PINNAugmentation

        K, M, N, dims = _make_small_dims()
        inps = _make_synthetic_inputs(K, M, N)
        from phoscrosstalk.optimization import create_bounds
        xl, xu, _ = create_bounds(K, M, N)
        theta0 = jnp.asarray((xl + xu) / 2.0, dtype=jnp.float64)
        key = jax.random.PRNGKey(0)
        pinn_model = PINNAugmentation(state_dim=3 * K + M + N, width_size=4, depth=1,
                                       activation="tanh", key=key)
        loss_fn = make_pinn_loss_fn(
            dims=dims, t=inps["t"], P_data=inps["P_scaled"], A_scaled=inps["A_scaled"],
            prot_idx_for_A=inps["prot_idx_for_A"], W_data=inps["W_data"],
            W_data_prot=inps["W_data_prot"], Cg=inps["Cg"], Cl=inps["Cl"],
            site_prot_idx=inps["site_prot_idx"], K_site_kin=inps["K_site_kin"],
            R=inps["R"], L_alpha=inps["L_alpha"], kin_to_prot_idx=inps["kin_to_prot_idx"],
            receptor_mask_prot=inps["receptor_mask_prot"],
            receptor_mask_kin=inps["receptor_mask_kin"],
            mechanism="dist", lambda_net=0.0, reg_lambda=0.0, lambda_pinn=0.1,
            regularize="param_l2",
            rtol=1e-3, atol=1e-5, max_steps=512,
        )
        _, aux = loss_fn((theta0, pinn_model), None)
        fp = float(aux[4])
        assert np.isfinite(fp) and fp >= 0.0


# ===========================================================================
# 5. Runner tests
# ===========================================================================


class TestPINNRunner:
    def test_runner_saves_required_files(self, tmp_path):
        """PINN runner saves pinn_metadata.json and pareto_front.npz."""
        import types
        from phoscrosstalk.config import load_config
        from phoscrosstalk.pinn.runner import run_pinn_pipeline

        K, M, N, dims = _make_small_dims()
        inps = _make_synthetic_inputs(K, M, N)

        cfg = load_config(None)
        # Patch to use small steps for speed
        cfg.pinn = _make_pinn_cfg(max_steps=2)
        cfg.solver = types.SimpleNamespace(
            ode_solver="tsit5", ode_adjoint="recursive",
            rtol=1e-3, atol=1e-5, max_steps=512, dt0=0.1,
            root_find_max_steps=5,
        )
        cfg.simulation = types.SimpleNamespace(dense_n_points=10)

        proteins = [f"P{k}" for k in range(K)]
        kinases  = [f"K{m}" for m in range(M)]
        sites    = [f"S{n}" for n in range(N)]

        result = run_pinn_pipeline(
            cfg=cfg,
            dims=dims,
            t=inps["t"],
            P_scaled=inps["P_scaled"],
            A_scaled=inps["A_scaled"],
            prot_idx_for_A=inps["prot_idx_for_A"],
            W_data=inps["W_data"],
            W_data_prot=inps["W_data_prot"],
            Cg=inps["Cg"],
            Cl=inps["Cl"],
            site_prot_idx=inps["site_prot_idx"],
            K_site_kin=inps["K_site_kin"],
            R=inps["R"],
            L_alpha=inps["L_alpha"],
            kin_to_prot_idx=inps["kin_to_prot_idx"],
            receptor_mask_prot=inps["receptor_mask_prot"],
            receptor_mask_kin=inps["receptor_mask_kin"],
            mechanism="dist",
            outdir=str(tmp_path),
            proteins=proteins,
            kinases=kinases,
            sites=sites,
        )

        # Required outputs
        assert (tmp_path / "pinn_metadata.json").exists(), "pinn_metadata.json missing"
        assert (tmp_path / "pareto_front.npz").exists(), "pareto_front.npz missing"
        assert (tmp_path / "pareto_stats.tsv").exists(), "pareto_stats.tsv missing"
        assert (tmp_path / "pinn_loss_components.tsv").exists(), "pinn_loss_components.tsv missing"

        # Check metadata content
        with open(tmp_path / "pinn_metadata.json") as fh:
            meta = json.load(fh)
        assert meta["run_mode"] == "pinn"
        assert meta["K"] == K
        assert meta["M"] == M
        assert meta["N"] == N

        # Result dict structure
        assert "theta_opt" in result
        assert "pinn_model" in result
        assert "loss_components" in result
        lc = result["loss_components"]
        for key in ("f1", "f2", "f3", "f4", "f_pinn_reg"):
            assert key in lc, f"Missing loss component: {key}"

    def test_runner_does_not_use_multistart(self, tmp_path):
        """PINN runner uses a single training run (no multistart workers)."""
        import types
        from phoscrosstalk.config import load_config
        from phoscrosstalk.pinn.runner import run_pinn_pipeline

        K, M, N, dims = _make_small_dims()
        inps = _make_synthetic_inputs(K, M, N)

        cfg = load_config(None)
        cfg.pinn = _make_pinn_cfg(max_steps=2)
        cfg.solver = types.SimpleNamespace(
            ode_solver="tsit5", ode_adjoint="recursive",
            rtol=1e-3, atol=1e-5, max_steps=512, dt0=0.1,
            root_find_max_steps=5,
        )
        cfg.simulation = types.SimpleNamespace(dense_n_points=5)

        from phoscrosstalk import multistarts
        original_fn = multistarts.run_multi_start_optimization

        call_count = [0]
        def patched_multistart(*args, **kwargs):
            call_count[0] += 1
            return original_fn(*args, **kwargs)

        multistarts.run_multi_start_optimization = patched_multistart
        try:
            run_pinn_pipeline(
                cfg=cfg, dims=dims, t=inps["t"],
                P_scaled=inps["P_scaled"], A_scaled=inps["A_scaled"],
                prot_idx_for_A=inps["prot_idx_for_A"], W_data=inps["W_data"],
                W_data_prot=inps["W_data_prot"], Cg=inps["Cg"], Cl=inps["Cl"],
                site_prot_idx=inps["site_prot_idx"], K_site_kin=inps["K_site_kin"],
                R=inps["R"], L_alpha=inps["L_alpha"], kin_to_prot_idx=inps["kin_to_prot_idx"],
                receptor_mask_prot=inps["receptor_mask_prot"],
                receptor_mask_kin=inps["receptor_mask_kin"],
                mechanism="dist", outdir=str(tmp_path),
            )
        finally:
            multistarts.run_multi_start_optimization = original_fn

        assert call_count[0] == 0, "PINN runner must NOT call multistart optimization"


# ===========================================================================
# 6. Utils tests
# ===========================================================================


class TestPINNUtils:
    def test_state_labels_length(self):
        from phoscrosstalk.pinn.utils import state_labels
        K, M, N = 3, 2, 4
        labels = state_labels(K, M, N)
        assert len(labels) == 3 * K + M + N

    def test_state_labels_prefixes(self):
        from phoscrosstalk.pinn.utils import state_labels
        K, M, N = 2, 1, 2
        proteins = ["ERK", "AKT"]
        kinases  = ["EGFR"]
        sites    = ["ERK_T202", "AKT_S473"]
        labels = state_labels(K, M, N, proteins, kinases, sites)
        assert labels[0].startswith("R_rna:")
        assert labels[K].startswith("S:")
        assert labels[2 * K].startswith("A:")
        assert labels[3 * K].startswith("Kdyn:")
        assert labels[3 * K + M].startswith("p:")

    def test_pinn_param_count_positive(self):
        import jax
        from phoscrosstalk.pinn.utils import pinn_param_count
        from phoscrosstalk.pinn.model import PINNAugmentation
        key = jax.random.PRNGKey(0)
        model = PINNAugmentation(state_dim=6, width_size=8, depth=1, activation="tanh", key=key)
        n = pinn_param_count(model)
        assert n > 0, "PINN model must have trainable parameters"


# ===========================================================================
# 7. Regression: pinn.enabled = false preserves existing pipeline
# ===========================================================================


class TestPINNRegression:
    def test_pinn_disabled_cfg_defaults(self):
        """When pinn.enabled = false, config loads without touching mechanistic defaults."""
        from phoscrosstalk.config import load_config
        cfg = load_config(None)
        assert cfg.pinn.enabled is False
        # Mechanistic sections unchanged
        assert hasattr(cfg, "optimisation")
        assert hasattr(cfg, "solver")
        assert hasattr(cfg, "neural_ode")
        assert cfg.neural_ode.enabled is False

    def test_neuralode_config_unaffected_by_pinn(self, tmp_path):
        """[neural_ode] keys remain accessible and unchanged when [pinn] is added."""
        toml = tmp_path / "cfg.toml"
        toml.write_text(
            "[paths]\ndata = ''\nptm_intra = ''\nptm_inter = ''\n"
            "[neural_ode]\nenabled = false\nwidth = 16\ndepth = 2\nsteps = 1000\n"
            "learning_rate = 0.001\nprior_weight_k_act = 1.0\n"
            "prior_weight_s_prod = 1.0\ndata_weight_phospho = 1.0\n"
            "data_weight_abundance = 1.0\ndata_weight_mrna = 1.0\nseed = 0\n"
            "rtol = 1e-5\natol = 1e-7\ndt0 = 0.01\nmax_steps = 65536\n"
            "save_dense = false\ndense_n_points = 200\n"
            "[pinn]\nenabled = false\n"
        )
        from phoscrosstalk.config import load_config
        cfg = load_config(str(toml))
        assert cfg.neural_ode.width == 16
        assert cfg.neural_ode.steps == 1000
        assert cfg.pinn.enabled is False


# ===========================================================================
# 9. Model bundle save / load (round-trip)
# ===========================================================================


class TestPINNModelBundle:
    """Tests for save_pinn_model_bundle / load_pinn_model_bundle round-trip."""

    @pytest.fixture(autouse=True)
    def _require_eqx(self):
        pytest.importorskip("equinox", reason="equinox required for PINN bundle")
        pytest.importorskip("jax",     reason="jax required for PINN bundle")

    def _make_tiny_model(self, K=2, M=1, N=3):
        import jax
        from phoscrosstalk.pinn.model import PINNAugmentation

        state_dim = 3 * K + M + N
        return PINNAugmentation(
            state_dim=state_dim,
            width_size=4,
            depth=1,
            activation="tanh",
            key=jax.random.PRNGKey(0),
        )

    def test_save_bundle_creates_required_files(self, tmp_path):
        """save_pinn_model_bundle must create pinn_model.eqx, meta JSON, theta_opt.npy."""
        from phoscrosstalk.pinn.outputs import save_pinn_model_bundle

        K, M, N = 2, 1, 3
        model = self._make_tiny_model(K, M, N)
        theta = np.ones(10, dtype=np.float64)
        cfg = _make_pinn_cfg(width_size=4, depth=1, activation="tanh")

        bundle_dir = save_pinn_model_bundle(
            str(tmp_path),
            pinn_model=model,
            theta_opt=theta,
            pinn_cfg=cfg,
            K=K, M=M, N=N,
        )

        assert os.path.isdir(bundle_dir)
        assert os.path.isfile(os.path.join(bundle_dir, "pinn_model.eqx"))
        assert os.path.isfile(os.path.join(bundle_dir, "pinn_bundle_meta.json"))
        assert os.path.isfile(os.path.join(bundle_dir, "theta_opt.npy"))

    def test_bundle_meta_contains_structural_fields(self, tmp_path):
        """pinn_bundle_meta.json must contain state_dim, width_size, depth, activation, K, M, N."""
        from phoscrosstalk.pinn.outputs import save_pinn_model_bundle

        K, M, N = 2, 1, 3
        model = self._make_tiny_model(K, M, N)
        cfg = _make_pinn_cfg(width_size=4, depth=1, activation="relu")

        bundle_dir = save_pinn_model_bundle(
            str(tmp_path),
            pinn_model=model,
            theta_opt=np.zeros(5),
            pinn_cfg=cfg,
            K=K, M=M, N=N,
        )

        with open(os.path.join(bundle_dir, "pinn_bundle_meta.json")) as fh:
            meta = json.load(fh)

        assert meta["K"] == K
        assert meta["M"] == M
        assert meta["N"] == N
        assert meta["state_dim"] == 3 * K + M + N
        assert meta["width_size"] == 4
        assert meta["depth"] == 1
        assert meta["activation"] == "relu"
        assert "output_clamp" in meta
        assert meta["bundle_format_version"] == 1

    def test_load_bundle_recovers_model_and_theta(self, tmp_path):
        """load_pinn_model_bundle must return a working PINNAugmentation and theta_opt."""
        import jax
        import jax.numpy as jnp
        from phoscrosstalk.pinn.outputs import save_pinn_model_bundle, load_pinn_model_bundle
        from phoscrosstalk.pinn.model import PINNAugmentation

        K, M, N = 2, 1, 3
        state_dim = 3 * K + M + N
        model_orig = self._make_tiny_model(K, M, N)
        theta_orig = np.arange(10, dtype=np.float64)
        cfg = _make_pinn_cfg(width_size=4, depth=1, activation="tanh")

        bundle_dir = save_pinn_model_bundle(
            str(tmp_path),
            pinn_model=model_orig,
            theta_opt=theta_orig,
            pinn_cfg=cfg,
            K=K, M=M, N=N,
        )

        pinn_loaded, theta_loaded, meta = load_pinn_model_bundle(bundle_dir)

        # theta_opt round-trip
        assert theta_loaded is not None
        np.testing.assert_array_equal(theta_loaded, theta_orig)

        # model should be a PINNAugmentation
        assert isinstance(pinn_loaded, PINNAugmentation)

        # Forward pass should produce the same output as the original model
        x = jnp.ones(state_dim, dtype=jnp.float64)
        t = jnp.asarray(5.0, dtype=jnp.float64)
        out_orig   = model_orig(x, t)
        out_loaded = pinn_loaded(x, t)
        np.testing.assert_allclose(
            np.asarray(out_orig), np.asarray(out_loaded), rtol=1e-6,
            err_msg="Loaded model output differs from original",
        )

    def test_load_bundle_raises_on_missing_meta(self, tmp_path):
        """load_pinn_model_bundle raises FileNotFoundError when meta JSON is absent."""
        from phoscrosstalk.pinn.outputs import load_pinn_model_bundle

        with pytest.raises(FileNotFoundError, match="pinn_bundle_meta.json"):
            load_pinn_model_bundle(str(tmp_path))

    def test_save_bundle_with_dims_object(self, tmp_path):
        """save_pinn_model_bundle accepts a ModelDims object instead of K/M/N."""
        from phoscrosstalk.pinn.outputs import save_pinn_model_bundle, load_pinn_model_bundle
        from phoscrosstalk.config import ModelDims

        K, M, N = 3, 2, 5
        dims = ModelDims(K=K, M=M, N=N)
        model = self._make_tiny_model(K, M, N)
        cfg = _make_pinn_cfg(width_size=4, depth=1)

        bundle_dir = save_pinn_model_bundle(
            str(tmp_path),
            pinn_model=model,
            theta_opt=np.ones(15),
            pinn_cfg=cfg,
            dims=dims,
        )

        _, _, meta = load_pinn_model_bundle(bundle_dir)
        assert meta["K"] == K
        assert meta["M"] == M
        assert meta["N"] == N
        assert meta["state_dim"] == 3 * K + M + N

    def test_save_bundle_integrated_via_save_pinn_outputs(self, tmp_path):
        """save_pinn_outputs must call save_pinn_model_bundle internally."""
        from phoscrosstalk.pinn.outputs import save_pinn_outputs

        K, M, N, T = 2, 1, 3, 4
        model = self._make_tiny_model(K, M, N)
        theta = np.zeros(8, dtype=np.float64)
        cfg = _make_pinn_cfg(width_size=4, depth=1)

        save_pinn_outputs(
            str(tmp_path),
            theta_opt=theta,
            pinn_model=model,
            loss_components={"f1": 0.1, "f2": 0.05, "f3": 0.01, "f4": 0.0, "f_pinn_reg": 0.0},
            ts=np.linspace(0, 10, T),
            ys=np.zeros((T, 3 * K + M + N)),
            K=K, M=M, N=N,
            sites=[f"S{i}" for i in range(N)],
            proteins=[f"P{i}" for i in range(K)],
            kinases=[f"K{i}" for i in range(1)],
            P_data=np.zeros((N, T)),
            A_scaled=np.zeros((K, T)),
            prot_idx_for_A=np.arange(K),
            t=np.linspace(0, 10, T),
            pinn_cfg=cfg,
        )

        bundle_dir = os.path.join(str(tmp_path), "pinn_bundle")
        assert os.path.isdir(bundle_dir), "pinn_bundle/ should be created by save_pinn_outputs"
        assert os.path.isfile(os.path.join(bundle_dir, "pinn_model.eqx"))
        assert os.path.isfile(os.path.join(bundle_dir, "pinn_bundle_meta.json"))
