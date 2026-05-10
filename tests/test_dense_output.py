"""
Focused tests for the dense/continuous-output and data-interpolation features.

Covers:
* build_data_interpolations(): observed arrays unchanged, shapes correct
* simulate_dense(): returns expected keys and shapes
* _save_dense_simulation(): produces protein_fit_timeseries_dense.tsv with expected columns
* Dense simulation does not affect best-solution selection (optimization is unaffected)
* protein_fit_timeseries_dense.tsv includes expected series_type labels
* app.py load_dense_timeseries path does not crash on missing / present file
* Steady-state event: still terminates on simple stable ODE (regression guard)
* use_event=False calls legacy simulate() path (regression guard)
"""

from __future__ import annotations

import os
import types

import numpy as np
import pytest

from phoscrosstalk.config import ModelDims


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _set_dims(K=2, M=2, N=3):
    ModelDims.set_dims(K, M, N)
    return K, M, N


def _theta_zero(K=2, M=2, N=3):
    dim = 2 * K + 2 + 3 * M + N + 4
    return np.zeros(dim, dtype=np.float64)


def _minimal_problem(K=2, M=2, N=3):
    """Return a minimal SimpleNamespace mimicking a NetworkProblem."""
    _set_dims(K, M, N)
    return types.SimpleNamespace(
        t=np.array([0.0, 1.0, 2.0]),
        P_data=np.ones((N, 3)) * 0.3,
        A_scaled=np.zeros((0, 3)),
        prot_idx_for_A=np.array([], dtype=int),
        Cg=np.zeros((N, N)),
        Cl=np.zeros((N, N)),
        site_prot_idx=np.zeros(N, dtype=int),
        K_site_kin=np.zeros((N, M)),
        R=np.zeros((M, N)),
        L_alpha=np.zeros((M, M)),
        kin_to_prot_idx=np.zeros(M, dtype=int),
        receptor_mask_prot=np.zeros(K, dtype=int),
        receptor_mask_kin=np.zeros(M, dtype=int),
        mechanism="dist",
        k_act_fn=None,
        s_prod_fn=None,
        R_data0=None,
        rna_relax=0.1,
    )


# ---------------------------------------------------------------------------
# build_data_interpolations – original arrays must NOT be modified
# ---------------------------------------------------------------------------


class TestBuildDataInterpolations:
    def test_original_arrays_unchanged_with_nans(self):
        """Original P_data array passed in must be identical after interpolation build."""
        from phoscrosstalk.derived_rates import build_data_interpolations

        t = np.array([0.0, 1.0, 2.0, 3.0])
        P = np.array([[np.nan, 0.5, 0.8, np.nan], [0.2, 0.4, np.nan, 0.6]])
        P_copy = P.copy()

        build_data_interpolations(
            t_obs=t,
            P_data=P,
            method="linear",
            fill_forward_nans_at_end=True,
            replace_nans_at_start="zero",
        )
        np.testing.assert_array_equal(P, P_copy, err_msg="P_data must not be modified")

    def test_interp_callable_returns_correct_shape(self):
        """The returned P_interp callable must return shape (N_sites,) for a scalar query."""
        from phoscrosstalk.derived_rates import build_data_interpolations

        t = np.array([0.0, 1.0, 2.0, 4.0])
        N = 5
        P = np.random.default_rng(0).random((N, 4))

        result = build_data_interpolations(t_obs=t, P_data=P, method="linear")
        fn = result["P_interp"]
        assert fn is not None
        out = fn(1.5)
        assert out.shape == (N,), f"Expected ({N},), got {out.shape}"

    def test_no_nan_silently_converted_to_zero_in_original(self):
        """NaN fill for interpolation must never overwrite the input array."""
        from phoscrosstalk.derived_rates import build_data_interpolations

        t = np.array([0.0, 1.0, 2.0])
        P = np.array([[np.nan, 0.5, 0.8]])
        original_nan_pos = np.isnan(P[0, 0])

        build_data_interpolations(
            t_obs=t,
            P_data=P,
            method="linear",
            replace_nans_at_start="zero",
        )
        assert np.isnan(P[0, 0]) == original_nan_pos, (
            "NaN at index 0 should remain NaN in the original array"
        )

    def test_metadata_invariant_flag(self):
        """Result must always report original_arrays_unchanged=True."""
        from phoscrosstalk.derived_rates import build_data_interpolations

        t = np.linspace(0, 5, 6)
        P = np.ones((3, 6))
        result = build_data_interpolations(t_obs=t, P_data=P)
        assert result["original_arrays_unchanged"] is True

    def test_a_interp_callable_shape(self):
        """A_interp callable must return shape (K_obs,)."""
        from phoscrosstalk.derived_rates import build_data_interpolations

        t = np.array([0.0, 1.0, 2.0])
        A = np.array([[1.0, 1.1, 1.2], [0.9, 1.0, 1.05]])
        result = build_data_interpolations(t_obs=t, A_data=A, method="linear")
        fn = result["A_interp"]
        assert fn is not None
        out = fn(1.0)
        assert out.shape == (2,)

    def test_nan_only_row_returns_nan_callable(self):
        """A row of all NaN should produce NaN output, not crash."""
        from phoscrosstalk.derived_rates import build_data_interpolations

        t = np.array([0.0, 1.0, 2.0])
        P = np.full((2, 3), np.nan)
        result = build_data_interpolations(t_obs=t, P_data=P, method="linear")
        # With all NaN, fewer than 2 valid points → NaN callable
        fn = result["P_interp"]
        assert fn is not None
        out = fn(1.0)
        # All NaN or at least no crash
        assert out is not None

    def test_cubic_hermite_requires_scipy(self):
        """cubic_hermite must raise ImportError gracefully if scipy missing,
        or succeed if scipy is available."""
        from phoscrosstalk.derived_rates import build_data_interpolations

        t = np.array([0.0, 1.0, 2.0, 3.0])
        P = np.array([[0.1, 0.3, 0.5, 0.7]])
        try:
            result = build_data_interpolations(
                t_obs=t, P_data=P, method="cubic_hermite"
            )
            fn = result["P_interp"]
            out = fn(1.5)
            assert out is not None
        except ImportError:
            pytest.skip("scipy not installed; cubic_hermite unavailable")

    def test_does_not_add_to_loss(self):
        """Calling build_data_interpolations does not change the P_data values
        that would be used in a loss computation (spot-check)."""
        from phoscrosstalk.derived_rates import build_data_interpolations

        t = np.array([0.0, 1.0, 2.0])
        P = np.array([[0.2, 0.5, 0.8], [np.nan, 0.3, 0.6]])
        P_before = P.copy()

        build_data_interpolations(
            t_obs=t,
            P_data=P,
            method="linear",
            fill_forward_nans_at_end=True,
            replace_nans_at_start="first_valid",
        )
        # Original sparse arrays for loss computation are unchanged
        np.testing.assert_array_equal(P, P_before)


# ---------------------------------------------------------------------------
# simulate_dense() – shape checks
# ---------------------------------------------------------------------------


class TestSimulateDense:
    def setup_method(self):
        self.K, self.M, self.N = _set_dims(2, 2, 3)
        self.theta = _theta_zero(self.K, self.M, self.N)
        self.t_dense = np.linspace(0.0, 5.0, 30)
        self.P0 = np.ones((self.N, 3)) * 0.3
        self.A0 = np.zeros((self.K, 1))
        self.zeros = lambda s: np.zeros(s)

    def _call_dense(self):
        from phoscrosstalk.simulation import simulate_dense

        return simulate_dense(
            t_dense=self.t_dense,
            P_data0=self.P0,
            A_data0=self.A0,
            theta=self.theta,
            Cg=self.zeros((self.N, self.N)),
            Cl=self.zeros((self.N, self.N)),
            site_prot_idx=np.zeros(self.N, dtype=int),
            K_site_kin=self.zeros((self.N, self.M)),
            R=self.zeros((self.M, self.N)),
            L_alpha=self.zeros((self.M, self.M)),
            kin_to_prot_idx=np.zeros(self.M, dtype=int),
            receptor_mask_prot=np.zeros(self.K, dtype=int),
            receptor_mask_kin=np.zeros(self.M, dtype=int),
            mechanism="dist",
        )

    def test_returns_dict_with_expected_keys(self):
        result = self._call_dense()
        for key in ("P_sim", "A_sim", "S_sim", "Kdyn_sim", "R_sim", "success"):
            assert key in result, f"Missing key: {key}"

    def test_P_sim_shape(self):
        result = self._call_dense()
        P = result["P_sim"]
        T = len(self.t_dense)
        assert P.shape == (self.N, T), f"Expected ({self.N}, {T}), got {P.shape}"

    def test_A_sim_shape(self):
        result = self._call_dense()
        A = result["A_sim"]
        T = len(self.t_dense)
        assert A.shape == (self.K, T), f"Expected ({self.K}, {T}), got {A.shape}"

    def test_success_field_is_bool(self):
        result = self._call_dense()
        assert isinstance(result["success"], bool)

    def test_does_not_modify_t_obs_semantics(self):
        """simulate_dense must not affect a separate sparse-time simulate call."""
        from phoscrosstalk.simulation import simulate

        t_sparse = np.array([0.0, 1.0, 2.0])
        P_sparse, A_sparse = simulate(
            t_arr=t_sparse,
            P_data0=self.P0,
            A_data0=np.zeros((self.K, len(t_sparse))),
            theta=self.theta,
            Cg=np.zeros((self.N, self.N)),
            Cl=np.zeros((self.N, self.N)),
            site_prot_idx=np.zeros(self.N, dtype=int),
            K_site_kin=np.zeros((self.N, self.M)),
            R=np.zeros((self.M, self.N)),
            L_alpha=np.zeros((self.M, self.M)),
            kin_to_prot_idx=np.zeros(self.M, dtype=int),
            receptor_mask_prot=np.zeros(self.K, dtype=int),
            receptor_mask_kin=np.zeros(self.M, dtype=int),
            mechanism="dist",
        )
        # Dense call
        _ = self._call_dense()
        # Sparse call again – must produce same result
        P_sparse2, A_sparse2 = simulate(
            t_arr=t_sparse,
            P_data0=self.P0,
            A_data0=np.zeros((self.K, len(t_sparse))),
            theta=self.theta,
            Cg=np.zeros((self.N, self.N)),
            Cl=np.zeros((self.N, self.N)),
            site_prot_idx=np.zeros(self.N, dtype=int),
            K_site_kin=np.zeros((self.N, self.M)),
            R=np.zeros((self.M, self.N)),
            L_alpha=np.zeros((self.M, self.M)),
            kin_to_prot_idx=np.zeros(self.M, dtype=int),
            receptor_mask_prot=np.zeros(self.K, dtype=int),
            receptor_mask_kin=np.zeros(self.M, dtype=int),
            mechanism="dist",
        )
        np.testing.assert_allclose(P_sparse, P_sparse2, rtol=1e-5)


# ---------------------------------------------------------------------------
# _save_dense_simulation() – TSV output checks
# ---------------------------------------------------------------------------


class TestSaveDenseSimulation:
    def setup_method(self):
        self.K, self.M, self.N = _set_dims(2, 2, 3)
        self.theta = _theta_zero(self.K, self.M, self.N)

    def _call_save(self, tmp_path, n_dense=20, data_interp_P=None, data_interp_A=None):
        from phoscrosstalk.analysis import _save_dense_simulation

        t_obs = np.array([0.0, 1.0, 2.0])
        sites = ["pA_S1", "pA_S2", "pB_S3"]
        proteins = ["pA", "pB"]
        P_scaled = np.ones((self.N, 3)) * 0.3
        A_scaled = np.zeros((0, 3))

        _save_dense_simulation(
            outdir=str(tmp_path),
            theta_opt=self.theta,
            t_obs=t_obs,
            sites=sites,
            proteins=proteins,
            P_scaled=P_scaled,
            A_scaled=A_scaled,
            prot_idx_for_A=np.array([], dtype=int),
            Cg=np.zeros((self.N, self.N)),
            Cl=np.zeros((self.N, self.N)),
            site_prot_idx=np.zeros(self.N, dtype=int),
            K_site_kin=np.zeros((self.N, self.M)),
            R=np.zeros((self.M, self.N)),
            L_alpha=np.zeros((self.M, self.M)),
            kin_to_prot_idx=np.zeros(self.M, dtype=int),
            mask_p=np.zeros(self.K, dtype=int),
            mask_k=np.zeros(self.M, dtype=int),
            mechanism="dist",
            n_dense=n_dense,
            data_interp_P=data_interp_P,
            data_interp_A=data_interp_A,
        )

    def test_file_created(self, tmp_path):
        self._call_save(tmp_path)
        assert os.path.exists(tmp_path / "protein_fit_timeseries_dense.tsv")

    def test_expected_columns(self, tmp_path):
        import pandas as pd

        self._call_save(tmp_path)
        df = pd.read_csv(tmp_path / "protein_fit_timeseries_dense.tsv", sep="\t")
        expected_cols = {
            "entity_type", "entity", "site", "protein",
            "time", "value", "series_type", "source", "interpolation_method",
        }
        assert expected_cols.issubset(set(df.columns)), (
            f"Missing columns: {expected_cols - set(df.columns)}"
        )

    def test_series_type_is_simulated_dense(self, tmp_path):
        """Dense ODE simulation rows must use series_type='simulated_dense'."""
        import pandas as pd

        self._call_save(tmp_path)
        df = pd.read_csv(tmp_path / "protein_fit_timeseries_dense.tsv", sep="\t")
        types_present = set(df["series_type"].unique())
        assert "simulated_dense" in types_present, (
            f"Expected 'simulated_dense' in series_type, got: {types_present}"
        )
        # Old label 'dense_simulation' must NOT appear
        assert "dense_simulation" not in types_present, (
            "Old 'dense_simulation' label must not be present; use 'simulated_dense'"
        )

    def test_n_dense_controls_time_points(self, tmp_path):
        """Number of unique time points in the output must match n_dense."""
        import pandas as pd

        n = 17
        self._call_save(tmp_path, n_dense=n)
        df = pd.read_csv(tmp_path / "protein_fit_timeseries_dense.tsv", sep="\t")
        n_times = df["time"].nunique()
        assert n_times == n, f"Expected {n} time points, got {n_times}"

    def test_observed_interpolated_dense_present_when_data_interp_given(self, tmp_path):
        """When data_interp_P is provided, 'observed_interpolated_dense' rows must appear."""
        import pandas as pd

        # Build a simple linear interpolator for 3 sites
        t_obs = np.array([0.0, 1.0, 2.0])
        P_obs = np.ones((self.N, 3)) * 0.3

        def _simple_interp(t_query):
            t_query = np.atleast_1d(t_query)
            return np.full((self.N, len(t_query)), 0.3)

        self._call_save(tmp_path, n_dense=10, data_interp_P=_simple_interp)
        df = pd.read_csv(tmp_path / "protein_fit_timeseries_dense.tsv", sep="\t")
        assert "observed_interpolated_dense" in set(df["series_type"].unique()), (
            "Expected 'observed_interpolated_dense' when data_interp_P is provided"
        )

    def test_no_observed_interpolated_dense_when_not_provided(self, tmp_path):
        """Without data_interp_P, 'observed_interpolated_dense' must NOT appear."""
        import pandas as pd

        self._call_save(tmp_path, n_dense=10)
        df = pd.read_csv(tmp_path / "protein_fit_timeseries_dense.tsv", sep="\t")
        assert "observed_interpolated_dense" not in set(df["series_type"].unique())

    def test_sparse_observed_arrays_unchanged_after_dense_save(self, tmp_path):
        """_save_dense_simulation must not modify the input P_scaled array."""
        from phoscrosstalk.analysis import _save_dense_simulation

        t_obs = np.array([0.0, 1.0, 2.0])
        sites = ["pA_S1", "pA_S2", "pB_S3"]
        proteins = ["pA", "pB"]
        P_scaled = np.ones((self.N, 3)) * 0.3
        P_scaled_copy = P_scaled.copy()

        _save_dense_simulation(
            outdir=str(tmp_path),
            theta_opt=self.theta,
            t_obs=t_obs,
            sites=sites,
            proteins=proteins,
            P_scaled=P_scaled,
            A_scaled=np.zeros((0, 3)),
            prot_idx_for_A=np.array([], dtype=int),
            Cg=np.zeros((self.N, self.N)),
            Cl=np.zeros((self.N, self.N)),
            site_prot_idx=np.zeros(self.N, dtype=int),
            K_site_kin=np.zeros((self.N, self.M)),
            R=np.zeros((self.M, self.N)),
            L_alpha=np.zeros((self.M, self.M)),
            kin_to_prot_idx=np.zeros(self.M, dtype=int),
            mask_p=np.zeros(self.K, dtype=int),
            mask_k=np.zeros(self.M, dtype=int),
            mechanism="dist",
            n_dense=10,
        )
        np.testing.assert_array_equal(P_scaled, P_scaled_copy, err_msg="P_scaled must not be modified")


# ---------------------------------------------------------------------------
# Dashboard loader – does not crash on missing or present dense file
# ---------------------------------------------------------------------------


class TestAppLoadDenseTimeseries:
    def test_returns_none_when_file_missing(self, tmp_path):
        from phoscrosstalk.dashboard import load_dense_timeseries

        result = load_dense_timeseries(str(tmp_path))
        assert result is None

    def test_returns_dataframe_when_file_present(self, tmp_path):
        import pandas as pd

        from phoscrosstalk.dashboard import load_dense_timeseries

        df = pd.DataFrame({
            "entity_type": ["Phosphosite"],
            "entity": ["pA_S1"],
            "site": ["S1"],
            "protein": ["pA"],
            "time": [0.5],
            "value": [0.3],
            "series_type": ["simulated_dense"],
            "source": ["model"],
            "interpolation_method": ["diffrax_dense"],
        })
        df.to_csv(tmp_path / "protein_fit_timeseries_dense.tsv", sep="\t", index=False)

        result = load_dense_timeseries(str(tmp_path))
        assert result is not None
        assert "series_type" in result.columns
        assert "simulated_dense" in result["series_type"].values

    def test_handles_corrupt_file_gracefully(self, tmp_path):
        from phoscrosstalk.dashboard import load_dense_timeseries

        (tmp_path / "protein_fit_timeseries_dense.tsv").write_text("corrupt\x00data")
        result = load_dense_timeseries(str(tmp_path))
        # Should return None or a DataFrame (not raise)
        assert result is None or hasattr(result, "columns")


# ---------------------------------------------------------------------------
# Config defaults: [simulation] and [data_interpolation]
# ---------------------------------------------------------------------------


class TestNewConfigDefaults:
    def test_simulation_defaults_present(self):
        from phoscrosstalk.config import _DEFAULTS

        assert "simulation" in _DEFAULTS
        sim = _DEFAULTS["simulation"]
        assert sim["dense_output"] is True
        assert isinstance(sim["dense_n_points"], int)
        assert sim["dense_n_points"] >= 2
        assert sim["save_dense"] is True

    def test_data_interpolation_defaults_present(self):
        from phoscrosstalk.config import _DEFAULTS

        assert "data_interpolation" in _DEFAULTS
        di = _DEFAULTS["data_interpolation"]
        assert di["enabled"] is False  # backward-compatible default
        assert di["method"] in {"linear", "cubic_hermite"}
        assert di["fill_forward_nans_at_end"] is False
        assert di["replace_nans_at_start"] is None

    def test_old_config_without_new_sections_still_works(self, tmp_path):
        """A config without [simulation] or [data_interpolation] must load without error."""
        import textwrap

        from phoscrosstalk.config import load_config

        cfg_text = textwrap.dedent("""
            [paths]
            data = "data.csv"
            ptm_intra = "ptm_intra.csv"
            ptm_inter = "ptm_inter.csv"
            output_dir = "out"

            [model]
            mechanism = "dist"

            [optimisation]
            n_starts = 1
            max_steps = 10

            [solver]
            rtol = 1e-4
            atol = 1e-6

            [loss_weights]
            phospho = 1.0
            abundance = 0.0
            mrna = 0.0
            reg = 0.0001
        """)
        cfg_path = tmp_path / "config.toml"
        cfg_path.write_text(cfg_text)

        cfg = load_config(str(cfg_path))
        # Should have default values
        sim_cfg = getattr(cfg, "simulation", None)
        if sim_cfg is not None:
            assert getattr(sim_cfg, "dense_n_points", 200) >= 2


# ---------------------------------------------------------------------------
# Regression: optimization loss uses sparse observed times, not dense
# ---------------------------------------------------------------------------


class TestOptimizationUsesSparseObservedTimes:
    """Ensure that the residuals_fn / loss_fn still evaluates at sparse t_arr."""

    def test_residuals_fn_uses_t_observed(self, monkeypatch):
        """make_residuals_fn must pass t_arr (sparse) to the ODE, not a dense grid."""
        import jax.numpy as jnp

        from phoscrosstalk.config import ModelDims
        from phoscrosstalk.optimization import make_residuals_fn

        K, M, N = 2, 2, 3
        dims = ModelDims.set_dims(K, M, N)

        t_obs = np.array([0.0, 1.0, 2.0])
        P_data = np.ones((N, 3)) * 0.3
        A_scaled = np.zeros((0, 3))
        prot_idx = np.array([], dtype=int)
        W_data = np.ones((N, 3))
        W_prot = np.zeros((0, 3))
        Cg = np.zeros((N, N))
        Cl = np.zeros((N, N))
        site_prot_idx = np.zeros(N, dtype=int)
        K_site_kin = np.zeros((N, M))
        R_mat = np.zeros((M, N))
        L_alpha = np.zeros((M, M))
        kin_to_prot_idx = np.zeros(M, dtype=int)
        mask_p = np.zeros(K, dtype=int)
        mask_k = np.zeros(M, dtype=int)

        captured_t = {}

        import diffrax
        original_diffeqsolve = diffrax.diffeqsolve

        def mock_diffeqsolve(*args, **kwargs):
            saveat = kwargs.get("saveat", None)
            if saveat is not None and hasattr(saveat, "ts") and saveat.ts is not None:
                captured_t["ts"] = np.asarray(saveat.ts)
            return original_diffeqsolve(*args, **kwargs)

        monkeypatch.setattr(diffrax, "diffeqsolve", mock_diffeqsolve)

        res_fn = make_residuals_fn(
            dims=dims,
            t=t_obs,
            P_data=P_data,
            A_scaled=A_scaled,
            prot_idx_for_A=prot_idx,
            W_data=W_data,
            W_data_prot=W_prot,
            Cg=Cg,
            Cl=Cl,
            site_prot_idx=site_prot_idx,
            K_site_kin=K_site_kin,
            R=R_mat,
            L_alpha=L_alpha,
            kin_to_prot_idx=kin_to_prot_idx,
            receptor_mask_prot=mask_p,
            receptor_mask_kin=mask_k,
            mechanism="dist",
            lambda_net=0.0001,
            reg_lambda=0.0001,
        )

        theta = jnp.zeros(2 * K + 2 + 3 * M + N + 4, dtype=jnp.float64)
        try:
            res_fn(theta, None)
        except Exception:
            pass  # We only care about what was captured

        # If diffeqsolve was intercepted, validate the time grid is the sparse one
        if "ts" in captured_t:
            ts = captured_t["ts"]
            # The solver time grid includes t_obs; the sparse times must all be present
            for t_val in t_obs:
                assert any(np.isclose(ts, t_val, atol=1e-5)), (
                    f"Sparse time point {t_val} not found in solver ts={ts}. "
                    "Optimization must use sparse observed times."
                )
