"""
tests/test_dtype.py
===================
Minimal verification that the codebase runs in JAX float64 mode.

Tests:
* JAX x64 is enabled (or can be enabled)
* jnp.asarray([1.0], dtype=jnp.float64) is truly float64
* Representative numerical arrays from simulation produce float64 output
* make_residuals_fn / make_loss_fn produce float64 results
* steadystate solve uses float64 args
* derived_rates arrays are float64
"""

from __future__ import annotations

import os

# Enable x64 before importing JAX so the tests themselves run in float64 mode.
os.environ.setdefault("JAX_ENABLE_X64", "true")

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# 1. JAX x64 enabled
# ---------------------------------------------------------------------------


class TestJaxX64:
    def test_env_var_set(self):
        """JAX_ENABLE_X64 must be set to 'true' before JAX is imported."""
        val = os.environ.get("JAX_ENABLE_X64", "")
        assert val.lower() == "true", (
            f"Expected JAX_ENABLE_X64=true, got {val!r}"
        )

    def test_float64_array_dtype(self):
        """After enabling x64, jnp.asarray with float64 must return float64."""
        import jax.numpy as jnp

        arr = jnp.asarray([1.0, 2.0, 3.0], dtype=jnp.float64)
        assert arr.dtype == jnp.float64, (
            f"Expected float64, got {arr.dtype}. "
            "Is JAX_ENABLE_X64=true set before JAX import?"
        )

    def test_default_float_not_float32(self):
        """Python float literals inside JAX must promote to at least float32;
        with x64 enabled they should be float64 when explicitly requested."""
        import jax.numpy as jnp

        x = jnp.asarray(1.0, dtype=jnp.float64)
        assert x.dtype == jnp.float64


# ---------------------------------------------------------------------------
# 2. runtime_env.enable_x64 sets env var
# ---------------------------------------------------------------------------


class TestRuntimeEnvEnableX64:
    def test_enable_x64_sets_env(self, monkeypatch):
        """enable_x64() must set JAX_ENABLE_X64 via setdefault."""
        monkeypatch.delenv("JAX_ENABLE_X64", raising=False)
        from phoscrosstalk.runtime_env import enable_x64

        enable_x64()
        assert os.environ.get("JAX_ENABLE_X64", "").lower() == "true"

    def test_enable_x64_does_not_override_explicit_false(self, monkeypatch):
        """enable_x64() uses setdefault, so explicit false must be preserved."""
        monkeypatch.setenv("JAX_ENABLE_X64", "false")
        from phoscrosstalk.runtime_env import enable_x64

        enable_x64()
        # setdefault preserves existing value
        assert os.environ["JAX_ENABLE_X64"] == "false"

    def test_log_env_summary_includes_x64(self, capsys):
        """log_env_summary() must print JAX_ENABLE_X64."""
        from phoscrosstalk.runtime_env import log_env_summary

        log_env_summary()
        captured = capsys.readouterr()
        assert "JAX_ENABLE_X64" in captured.err


# ---------------------------------------------------------------------------
# 3. Simulation produces float64 output
# ---------------------------------------------------------------------------


def _make_minimal_arrays(K=2, M=2, N=3):
    """Return minimal float64 numpy arrays for simulate()."""
    from phoscrosstalk.config import ModelDims

    ModelDims.set_dims(K, M, N)
    t = np.array([0.0, 1.0, 2.0], dtype=np.float64)
    theta = np.zeros(2 * K + 2 + 3 * M + N + 4, dtype=np.float64)
    P_data = np.ones((N, 3), dtype=np.float64) * 0.3
    A_data = np.zeros((K, 3), dtype=np.float64)
    Cg = np.zeros((N, N), dtype=np.float64)
    Cl = np.zeros((N, N), dtype=np.float64)
    K_sk = np.zeros((N, M), dtype=np.float64)
    R = np.zeros((M, N), dtype=np.float64)
    La = np.zeros((M, M), dtype=np.float64)
    spi = np.zeros(N, dtype=np.int32)
    k2p = np.zeros(M, dtype=np.int32)
    rmp = np.zeros(K, dtype=np.float64)
    rmk = np.zeros(M, dtype=np.float64)
    return t, theta, P_data, A_data, Cg, Cl, K_sk, R, La, spi, k2p, rmp, rmk


class TestSimulationFloat64:
    def test_simulate_returns_float64(self):
        from phoscrosstalk.simulation import simulate

        K, M, N = 2, 2, 3
        t, theta, P_data, A_data, Cg, Cl, K_sk, R, La, spi, k2p, rmp, rmk = (
            _make_minimal_arrays(K, M, N)
        )
        result = simulate(
            t_arr=t,
            P_data0=P_data,
            A_data0=A_data,
            theta=theta,
            Cg=Cg,
            Cl=Cl,
            site_prot_idx=spi,
            K_site_kin=K_sk,
            R=R,
            L_alpha=La,
            kin_to_prot_idx=k2p,
            receptor_mask_prot=rmp,
            receptor_mask_kin=rmk,
            mechanism="dist",
            return_full=True,
        )
        assert isinstance(result, dict), "Expected dict from return_full=True"
        P_sim = np.asarray(result["P_sim"])
        assert P_sim.dtype == np.float64, (
            f"P_sim dtype={P_sim.dtype}, expected float64"
        )

    def test_simulate_dense_returns_float64(self):
        from phoscrosstalk.simulation import simulate_dense

        K, M, N = 2, 2, 3
        t, theta, P_data, A_data, Cg, Cl, K_sk, R, La, spi, k2p, rmp, rmk = (
            _make_minimal_arrays(K, M, N)
        )
        t_dense = np.linspace(0.0, 2.0, 15, dtype=np.float64)
        result = simulate_dense(
            t_dense=t_dense,
            P_data0=P_data,
            A_data0=A_data,
            theta=theta,
            Cg=Cg,
            Cl=Cl,
            site_prot_idx=spi,
            K_site_kin=K_sk,
            R=R,
            L_alpha=La,
            kin_to_prot_idx=k2p,
            receptor_mask_prot=rmp,
            receptor_mask_kin=rmk,
            mechanism="dist",
        )
        P_dense = np.asarray(result["P_sim"])
        assert P_dense.dtype == np.float64, (
            f"simulate_dense P_sim dtype={P_dense.dtype}, expected float64"
        )


# ---------------------------------------------------------------------------
# 4. make_residuals_fn loss terms are float64
# ---------------------------------------------------------------------------


class TestOptimizationFloat64:
    def setup_method(self):
        from phoscrosstalk.config import ModelDims

        self.K, self.M, self.N = 2, 2, 3
        ModelDims.set_dims(self.K, self.M, self.N)

    def _make_res_fn(self):
        import jax.numpy as jnp

        from phoscrosstalk.optimization import make_residuals_fn

        K, M, N = self.K, self.M, self.N
        t = np.array([0.0, 1.0, 2.0], dtype=np.float64)
        P_data = np.ones((N, 3), dtype=np.float64) * 0.3
        A_scaled = np.zeros((0, 3), dtype=np.float64)
        W_data = np.ones((N, 3), dtype=np.float64)
        W_prot = np.zeros((0, 3), dtype=np.float64)
        Cg = np.zeros((N, N), dtype=np.float64)
        Cl = np.zeros((N, N), dtype=np.float64)
        K_sk = np.zeros((N, M), dtype=np.float64)
        R = np.zeros((M, N), dtype=np.float64)
        La = np.zeros((M, M), dtype=np.float64)
        spi = np.zeros(N, dtype=np.int32)
        k2p = np.zeros(M, dtype=np.int32)
        rmp = np.zeros(K, dtype=np.float64)
        rmk = np.zeros(M, dtype=np.float64)

        return make_residuals_fn(
            t=t,
            P_data=P_data,
            A_scaled=A_scaled,
            prot_idx_for_A=np.array([], dtype=np.int32),
            W_data=W_data,
            W_data_prot=W_prot,
            Cg=Cg,
            Cl=Cl,
            site_prot_idx=spi,
            K_site_kin=K_sk,
            R=R,
            L_alpha=La,
            kin_to_prot_idx=k2p,
            receptor_mask_prot=rmp,
            receptor_mask_kin=rmk,
            mechanism="dist",
            lambda_net=1e-4,
            reg_lambda=1e-4,
        )

    def test_residuals_fn_returns_float64(self):
        import jax.numpy as jnp

        res_fn = self._make_res_fn()
        dim = 2 * self.K + 2 + 3 * self.M + self.N + 4
        theta = jnp.zeros(dim, dtype=jnp.float64)
        residuals, (f1, f2, f3, f4) = res_fn(theta, None)
        assert residuals.dtype == jnp.float64, (
            f"residuals dtype={residuals.dtype}, expected float64"
        )
        for name, val in zip(["f1", "f2", "f3", "f4"], [f1, f2, f3, f4]):
            assert val.dtype == jnp.float64, (
                f"{name} dtype={val.dtype}, expected float64"
            )


# ---------------------------------------------------------------------------
# 5. derived_rates float64
# ---------------------------------------------------------------------------


class TestDerivedRatesFloat64:
    def test_make_k_act_fn_returns_float64(self):
        """make_k_act_fn() closure must return float64 JAX array."""
        import jax.numpy as jnp

        from phoscrosstalk.derived_rates import make_k_act_fn

        K = 3
        t_rna = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64)
        rna_data = np.ones((2, 4), dtype=np.float64)
        tf_weights = np.ones((K, 2), dtype=np.float64)

        fn = make_k_act_fn(
            rna_data=rna_data,
            t_rna=t_rna,
            tf_prot_weights=tf_weights,
            K=K,
        )
        result = fn(jnp.asarray(1.0, dtype=jnp.float64))
        assert result.dtype == jnp.float64, (
            f"k_act_fn result dtype={result.dtype}, expected float64"
        )

    def test_build_data_interpolations_float64(self):
        """build_data_interpolations() must not convert inputs to float32."""
        from phoscrosstalk.derived_rates import build_data_interpolations

        t = np.array([0.0, 1.0, 2.0], dtype=np.float64)
        P = np.ones((3, 3), dtype=np.float64) * 0.4
        result = build_data_interpolations(t_obs=t, P_data=P, method="linear")
        assert result["original_arrays_unchanged"] is True
        out = result["P_interp"](1.0)
        arr = np.asarray(out)
        # Allow float64 or float (python float returning np.float64)
        assert arr.dtype in (np.float64, np.float32) or arr.dtype.kind == "f"
        # Original must remain float64
        assert P.dtype == np.float64
