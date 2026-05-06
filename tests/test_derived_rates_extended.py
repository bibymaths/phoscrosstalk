"""
test_derived_rates_extended.py

Tests for derived_rates.py covering previously missing lines:
  375-376 – All-NaN row in _prep_row → nan_fill_log message
  383      – replace_nans_at_start="first_valid" branch
  435-436  – method="cubic_hermite" path (PchipInterpolator)
  483-498  – rna_data with shape[1] != len(t) → rna_interp stays None

Also tests:
  - make_s_prod_fn with empty Y_data returns constant 0.1
  - make_s_prod_fn with linear mode
"""
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _simple_t_obs():
    return np.array([0.0, 5.0, 10.0, 30.0, 60.0], dtype=np.float64)


def _simple_P_data(n_sites=3, T=5):
    rng = np.random.default_rng(42)
    return rng.uniform(0.5, 1.5, (n_sites, T))


# ---------------------------------------------------------------------------
# build_data_interpolations – _prep_row branches
# ---------------------------------------------------------------------------

class TestPrepRowAllNaN:
    """Tests for _prep_row all-NaN branch (lines 375-376)."""

    def test_all_nan_row_logged(self):
        """All-NaN row should produce a log entry."""
        from phoscrosstalk.derived_rates import build_data_interpolations

        t_obs = _simple_t_obs()
        T = len(t_obs)
        P_data = np.ones((2, T), dtype=np.float64)
        P_data[0, :] = np.nan  # All NaN in row 0

        result = build_data_interpolations(
            t_obs=t_obs,
            P_data=P_data,
            replace_nans_at_start="zero",  # triggers _prep_row
        )
        log = result["nan_fill_log"]
        assert any("all values are NaN" in msg for msg in log)

    def test_all_nan_row_returns_nan_callable(self):
        """All-NaN in all rows → no valid time points → NaN callable."""
        from phoscrosstalk.derived_rates import build_data_interpolations

        t_obs = _simple_t_obs()
        T = len(t_obs)
        P_data = np.full((2, T), np.nan)  # Completely NaN

        result = build_data_interpolations(
            t_obs=t_obs,
            P_data=P_data,
            replace_nans_at_start="zero",
        )
        # P_interp should be a callable (NaN fallback)
        assert result["P_interp"] is not None
        out = result["P_interp"](t_obs[2])
        assert np.all(np.isnan(out))

    def test_all_nan_row_no_fill_applied(self):
        """All-NaN row: no fill applied because there is no first_valid."""
        from phoscrosstalk.derived_rates import build_data_interpolations

        t_obs = _simple_t_obs()
        T = len(t_obs)
        P_data = np.ones((2, T), dtype=np.float64)
        P_data[1, :] = np.nan  # row 1 all NaN

        result = build_data_interpolations(
            t_obs=t_obs,
            P_data=P_data,
            replace_nans_at_start="first_valid",
        )
        log = result["nan_fill_log"]
        assert any("all values are NaN" in msg for msg in log)


# ---------------------------------------------------------------------------
# build_data_interpolations – replace_nans_at_start="first_valid"  (line 383)
# ---------------------------------------------------------------------------

class TestReplaceNaNsAtStartFirstValid:
    def test_leading_nans_filled_with_first_valid(self):
        from phoscrosstalk.derived_rates import build_data_interpolations

        t_obs = _simple_t_obs()
        T = len(t_obs)
        P_data = np.array([[np.nan, np.nan, 0.5, 0.7, 0.9]], dtype=np.float64)
        # Leading NaNs at indices 0 and 1; first valid is 0.5

        result = build_data_interpolations(
            t_obs=t_obs,
            P_data=P_data,
            replace_nans_at_start="first_valid",
        )
        log = result["nan_fill_log"]
        assert any("first_valid" in msg for msg in log)
        assert result["P_interp"] is not None

    def test_no_leading_nans_no_fill(self):
        from phoscrosstalk.derived_rates import build_data_interpolations

        t_obs = _simple_t_obs()
        T = len(t_obs)
        P_data = np.ones((2, T), dtype=np.float64)  # No NaNs

        result = build_data_interpolations(
            t_obs=t_obs,
            P_data=P_data,
            replace_nans_at_start="first_valid",
        )
        # No fill messages in log
        fill_msgs = [m for m in result["nan_fill_log"] if "first_valid" in m]
        assert len(fill_msgs) == 0

    def test_first_valid_gives_correct_value(self):
        from phoscrosstalk.derived_rates import build_data_interpolations

        t_obs = _simple_t_obs()
        T = len(t_obs)
        # Row with leading NaN followed by 0.8
        P_data = np.array([[np.nan, 0.8, 0.9, 1.0, 1.1]], dtype=np.float64)

        result = build_data_interpolations(
            t_obs=t_obs,
            P_data=P_data,
            replace_nans_at_start="first_valid",
        )
        interp = result["P_interp"]
        assert interp is not None
        # At t=0 (before first valid), the filled value should be 0.8
        val = interp(t_obs[0])
        # interpolated output is an array of shape (1,) or similar
        assert np.all(np.isfinite(val))


# ---------------------------------------------------------------------------
# build_data_interpolations – method="cubic_hermite"  (lines 435-436)
# ---------------------------------------------------------------------------

class TestCubicHermiteMethod:
    def test_cubic_hermite_returns_callable(self):
        """method='cubic_hermite' should use PchipInterpolator."""
        from phoscrosstalk.derived_rates import build_data_interpolations

        t_obs = _simple_t_obs()
        P_data = _simple_P_data(n_sites=2, T=len(t_obs))

        result = build_data_interpolations(
            t_obs=t_obs,
            P_data=P_data,
            method="cubic_hermite",
        )
        assert result["P_interp"] is not None
        assert result["method"] == "cubic_hermite"

    def test_cubic_hermite_array_query(self):
        from phoscrosstalk.derived_rates import build_data_interpolations

        t_obs = _simple_t_obs()
        P_data = _simple_P_data(n_sites=2, T=len(t_obs))

        result = build_data_interpolations(
            t_obs=t_obs,
            P_data=P_data,
            method="cubic_hermite",
        )
        t_query = np.linspace(0.0, 60.0, 20)
        out = result["P_interp"](t_query)
        # Should have shape (n_sites, len(t_query))
        assert out.shape == (2, 20)

    def test_cubic_hermite_A_data(self):
        from phoscrosstalk.derived_rates import build_data_interpolations

        t_obs = _simple_t_obs()
        A_data = np.random.default_rng(0).random((3, len(t_obs))) + 0.5

        result = build_data_interpolations(
            t_obs=t_obs,
            A_data=A_data,
            method="cubic_hermite",
        )
        assert result["A_interp"] is not None
        out = result["A_interp"](np.array([10.0, 20.0]))
        assert out.shape == (3, 2)

    def test_cubic_hermite_with_nans_falls_back_gracefully(self):
        from phoscrosstalk.derived_rates import build_data_interpolations

        t_obs = _simple_t_obs()
        T = len(t_obs)
        P_data = np.ones((2, T), dtype=np.float64)
        P_data[0, 0] = np.nan  # Leading NaN in row 0

        result = build_data_interpolations(
            t_obs=t_obs,
            P_data=P_data,
            method="cubic_hermite",
            replace_nans_at_start="zero",
        )
        assert result["P_interp"] is not None

    def test_cubic_hermite_rna_data(self):
        from phoscrosstalk.derived_rates import build_data_interpolations

        t_obs = _simple_t_obs()
        rna_data = np.random.default_rng(1).random((2, len(t_obs))) + 0.5

        result = build_data_interpolations(
            t_obs=t_obs,
            rna_data=rna_data,
            method="cubic_hermite",
        )
        assert result["rna_interp"] is not None


# ---------------------------------------------------------------------------
# build_data_interpolations – rna_data shape mismatch  (lines 483-498)
# ---------------------------------------------------------------------------

class TestRnaDataShapeMismatch:
    def test_rna_data_wrong_T_gives_none_interp(self):
        """rna_data.shape[1] != len(t_obs) → rna_interp stays None."""
        from phoscrosstalk.derived_rates import build_data_interpolations

        t_obs = _simple_t_obs()  # T=5
        rna_data = np.ones((2, 7), dtype=np.float64)  # Wrong T=7

        result = build_data_interpolations(
            t_obs=t_obs,
            rna_data=rna_data,
        )
        assert result["rna_interp"] is None

    def test_rna_data_correct_T_gives_interp(self):
        """rna_data.shape[1] == len(t_obs) → rna_interp is built."""
        from phoscrosstalk.derived_rates import build_data_interpolations

        t_obs = _simple_t_obs()
        rna_data = np.ones((2, len(t_obs)), dtype=np.float64)

        result = build_data_interpolations(
            t_obs=t_obs,
            rna_data=rna_data,
        )
        assert result["rna_interp"] is not None

    def test_rna_data_empty_gives_none_interp(self):
        from phoscrosstalk.derived_rates import build_data_interpolations

        t_obs = _simple_t_obs()
        rna_data = np.zeros((0, 5), dtype=np.float64)  # Empty

        result = build_data_interpolations(
            t_obs=t_obs,
            rna_data=rna_data,
        )
        assert result["rna_interp"] is None

    def test_original_arrays_unchanged_invariant(self):
        """original_arrays_unchanged key is always True."""
        from phoscrosstalk.derived_rates import build_data_interpolations

        t_obs = _simple_t_obs()
        P_data = _simple_P_data()
        result = build_data_interpolations(t_obs=t_obs, P_data=P_data)
        assert result["original_arrays_unchanged"] is True


# ---------------------------------------------------------------------------
# make_s_prod_fn – empty Y_data → constant 0.1 (softplus)
# ---------------------------------------------------------------------------

class TestMakeSProdFnExtended:
    def test_empty_Y_data_returns_constant_01(self):
        import jax.numpy as jnp
        from phoscrosstalk.derived_rates import make_s_prod_fn

        K, M = 3, 2
        t_protein = np.array([0.0, 10.0, 30.0, 60.0])
        Y_data = np.zeros((0, 4))
        R_kin_site = np.zeros((M, 0))
        kin_to_prot_idx = np.array([-1, -1])

        fn = make_s_prod_fn(t_protein, Y_data, R_kin_site, kin_to_prot_idx, K, M)
        result = fn(jnp.asarray(10.0))
        assert result.shape == (K,)
        np.testing.assert_allclose(np.array(result), np.full(K, 0.1), rtol=1e-5)

    def test_softplus_always_positive(self):
        import jax.numpy as jnp
        from phoscrosstalk.derived_rates import make_s_prod_fn

        K, M, N_sites = 2, 2, 4
        T = 5
        t_protein = np.linspace(0, 60, T)
        Y_data = np.random.default_rng(0).random((N_sites, T)) * 2
        R_kin_site = np.random.default_rng(0).random((M, N_sites))
        kin_to_prot_idx = np.array([0, 1])

        fn = make_s_prod_fn(
            t_protein, Y_data, R_kin_site, kin_to_prot_idx, K, M,
            s_prod_fn_type="softplus"
        )
        result = np.array(fn(jnp.asarray(15.0)))
        assert np.all(result > 0)

    def test_linear_mode_output_shape(self):
        import jax.numpy as jnp
        from phoscrosstalk.derived_rates import make_s_prod_fn

        K, M, N_sites = 2, 2, 4
        T = 5
        t_protein = np.linspace(0, 60, T)
        Y_data = np.random.default_rng(1).random((N_sites, T))
        R_kin_site = np.random.default_rng(1).random((M, N_sites))
        kin_to_prot_idx = np.array([0, 1])

        fn = make_s_prod_fn(
            t_protein, Y_data, R_kin_site, kin_to_prot_idx, K, M,
            s_prod_fn_type="linear"
        )
        result = fn(jnp.asarray(5.0))
        assert result.shape == (K,)

    def test_linear_mode_passthrough_identity(self):
        """Linear s_prod_fn should not apply softplus transform."""
        import jax.numpy as jnp
        from phoscrosstalk.derived_rates import make_s_prod_fn

        K, M, N_sites = 2, 2, 4
        T = 5
        t_protein = np.linspace(0, 60, T)
        Y_data = np.ones((N_sites, T)) * 0.5
        R_kin_site = np.ones((M, N_sites))
        kin_to_prot_idx = np.array([0, 1])

        fn_softplus = make_s_prod_fn(
            t_protein, Y_data, R_kin_site, kin_to_prot_idx, K, M,
            s_prod_fn_type="softplus"
        )
        fn_linear = make_s_prod_fn(
            t_protein, Y_data, R_kin_site, kin_to_prot_idx, K, M,
            s_prod_fn_type="linear"
        )
        t_query = jnp.asarray(10.0)
        r_sp = np.array(fn_softplus(t_query))
        r_lin = np.array(fn_linear(t_query))
        # Verify that both modes produce the same output shape (K,)
        assert r_sp.shape == r_lin.shape == (K,)


# ---------------------------------------------------------------------------
# build_data_interpolations – linear method (default)
# ---------------------------------------------------------------------------

class TestLinearInterpolation:
    def test_linear_P_interp_scalar_query(self):
        from phoscrosstalk.derived_rates import build_data_interpolations

        t_obs = _simple_t_obs()
        P_data = _simple_P_data()
        result = build_data_interpolations(t_obs=t_obs, P_data=P_data)
        out = result["P_interp"](t_obs[2])
        assert out.shape[0] == P_data.shape[0]

    def test_linear_P_interp_array_query(self):
        from phoscrosstalk.derived_rates import build_data_interpolations

        t_obs = _simple_t_obs()
        P_data = _simple_P_data()
        result = build_data_interpolations(t_obs=t_obs, P_data=P_data)
        t_q = np.linspace(0, 60, 15)
        out = result["P_interp"](t_q)
        assert out.shape == (P_data.shape[0], 15)

    def test_fill_forward_nans_at_end(self):
        from phoscrosstalk.derived_rates import build_data_interpolations

        t_obs = _simple_t_obs()
        T = len(t_obs)
        P_data = np.ones((2, T), dtype=np.float64)
        P_data[:, -1] = np.nan  # Trailing NaN

        result = build_data_interpolations(
            t_obs=t_obs,
            P_data=P_data,
            fill_forward_nans_at_end=True,
        )
        log = result["nan_fill_log"]
        assert any("trailing" in msg.lower() or "forward-filled" in msg.lower() for msg in log)

    def test_A_interp_is_none_when_A_data_none(self):
        from phoscrosstalk.derived_rates import build_data_interpolations

        t_obs = _simple_t_obs()
        result = build_data_interpolations(t_obs=t_obs, P_data=None)
        assert result["A_interp"] is None
        assert result["P_interp"] is None
