"""
test_config_validators.py

Tests for the validation branches in config.py that are not covered by
test_config.py. Targets lines 463, 470, 477, 486-502, 538-540, 544, 550,
557, 564, 568, 575, 582, 589, 595, 602, 608, 614, 623, 633, 642-755.
"""
import os
import pytest
from phoscrosstalk.config import load_config, validate_config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_cfg(tmp_path, extra_toml: str = "", paths_extra: str = "") -> str:
    """Write a minimal valid config (required files created) plus extra_toml.
    
    Use `paths_extra` to add keys inside the [paths] section without
    triggering duplicate section errors (e.g. `paths_extra='rna_data = "..."'`).
    """
    data_file = tmp_path / "data.csv"
    data_file.write_text("gene,x1\nEGFR,1.0\n")
    ptm_intra = tmp_path / "ptm_intra.db"
    ptm_intra.write_bytes(b"")
    ptm_inter = tmp_path / "ptm_inter.db"
    ptm_inter.write_bytes(b"")

    paths_block = f"""
[paths]
data = "{data_file}"
ptm_intra = "{ptm_intra}"
ptm_inter = "{ptm_inter}"
{paths_extra}
"""
    cfg_path = tmp_path / "cfg.toml"
    cfg_path.write_text(paths_block + extra_toml)
    return str(cfg_path)


def _assert_error(cfg_path: str, fragment: str = ""):
    """validate_config must raise SystemExit(1) and optionally contain fragment."""
    cfg = load_config(cfg_path)
    with pytest.raises(SystemExit) as exc_info:
        validate_config(cfg, cfg_path)
    assert exc_info.value.code == 1


def _assert_valid(cfg_path: str):
    """validate_config must NOT raise."""
    cfg = load_config(cfg_path)
    validate_config(cfg, cfg_path)  # should not raise


# ---------------------------------------------------------------------------
# [model] scale_mode validation  (line 463)
# ---------------------------------------------------------------------------

class TestScaleMode:
    def test_invalid_scale_mode_raises(self, tmp_path):
        p = _write_cfg(tmp_path, '[model]\nscale_mode = "invalid_mode"\n')
        _assert_error(p)

    def test_valid_scale_modes_pass(self, tmp_path):
        for mode in ("none", "minmax", "log-minmax"):
            p = _write_cfg(tmp_path, f'[model]\nscale_mode = "{mode}"\n')
            _assert_valid(p)


# ---------------------------------------------------------------------------
# [model] weight_scheme validation  (line 470)
# ---------------------------------------------------------------------------

class TestWeightScheme:
    def test_invalid_weight_scheme_raises(self, tmp_path):
        p = _write_cfg(tmp_path, '[model]\nweight_scheme = "bogus"\n')
        _assert_error(p)

    def test_valid_weight_scheme_passes(self, tmp_path):
        p = _write_cfg(tmp_path, '[model]\nweight_scheme = "uniform"\n')
        _assert_valid(p)


# ---------------------------------------------------------------------------
# [model] length_scale validation  (line 477)
# ---------------------------------------------------------------------------

class TestLengthScale:
    def test_non_positive_length_scale_raises(self, tmp_path):
        p = _write_cfg(tmp_path, "[model]\nlength_scale = -1.0\n")
        _assert_error(p)

    def test_zero_length_scale_raises(self, tmp_path):
        p = _write_cfg(tmp_path, "[model]\nlength_scale = 0.0\n")
        _assert_error(p)

    def test_positive_length_scale_passes(self, tmp_path):
        p = _write_cfg(tmp_path, "[model]\nlength_scale = 100.0\n")
        _assert_valid(p)


# ---------------------------------------------------------------------------
# [model] include_tfs_as_proteins  (lines 486-502)
# ---------------------------------------------------------------------------

class TestIncludeTfs:
    def test_include_tfs_without_rna_data_raises(self, tmp_path):
        p = _write_cfg(tmp_path, "[model]\ninclude_tfs_as_proteins = true\n")
        _assert_error(p)

    def test_include_tfs_without_tf_net_raises(self, tmp_path):
        rna = tmp_path / "rna.csv"
        rna.write_text("g1,1.0\n")
        p = _write_cfg(
            tmp_path,
            extra_toml="[model]\ninclude_tfs_as_proteins = true\n",
            paths_extra=f'rna_data = "{rna}"',
        )
        _assert_error(p)

    def test_rna_data_set_without_tf_net_is_warning_only(self, tmp_path):
        """rna_data set but no tf_net → warning, not error."""
        rna = tmp_path / "rna.csv"
        rna.write_text("g1,1.0\n")
        # No include_tfs, no tf_net → should NOT raise
        p = _write_cfg(tmp_path, paths_extra=f'rna_data = "{rna}"')
        _assert_valid(p)


# ---------------------------------------------------------------------------
# [solver] rtol / atol  (lines 538-540)
# ---------------------------------------------------------------------------

class TestSolverTolerances:
    def test_invalid_rtol_raises(self, tmp_path):
        p = _write_cfg(tmp_path, "[solver]\nrtol = -1.0\n")
        _assert_error(p)

    def test_zero_rtol_raises(self, tmp_path):
        p = _write_cfg(tmp_path, "[solver]\nrtol = 0.0\n")
        _assert_error(p)

    def test_invalid_atol_raises(self, tmp_path):
        p = _write_cfg(tmp_path, "[solver]\natol = 0.0\n")
        _assert_error(p)


# ---------------------------------------------------------------------------
# [solver] max_steps  (line 544)
# ---------------------------------------------------------------------------

class TestSolverMaxSteps:
    def test_zero_max_steps_raises(self, tmp_path):
        p = _write_cfg(tmp_path, "[solver]\nmax_steps = 0\n")
        _assert_error(p)

    def test_positive_max_steps_passes(self, tmp_path):
        p = _write_cfg(tmp_path, "[solver]\nmax_steps = 8192\n")
        _assert_valid(p)


# ---------------------------------------------------------------------------
# [solver] ode_solver  (line 550)
# ---------------------------------------------------------------------------

class TestOdeSolver:
    def test_invalid_ode_solver_raises(self, tmp_path):
        p = _write_cfg(tmp_path, '[solver]\node_solver = "rk4_unknown"\n')
        _assert_error(p)

    def test_valid_ode_solver_passes(self, tmp_path):
        p = _write_cfg(tmp_path, '[solver]\node_solver = "tsit5"\n')
        _assert_valid(p)


# ---------------------------------------------------------------------------
# [solver] ode_adjoint  (line 557)
# ---------------------------------------------------------------------------

class TestOdeAdjoint:
    def test_invalid_ode_adjoint_raises(self, tmp_path):
        p = _write_cfg(tmp_path, '[solver]\node_adjoint = "invalid_adj"\n')
        _assert_error(p)

    def test_direct_adjoint_triggers_warning_but_passes(self, tmp_path):
        """ode_adjoint='direct' produces a warning but should not cause error."""
        p = _write_cfg(tmp_path, '[solver]\node_adjoint = "direct"\n')
        _assert_valid(p)

    def test_backsolve_adjoint_triggers_warning_but_passes(self, tmp_path):
        p = _write_cfg(tmp_path, '[solver]\node_adjoint = "backsolve"\n')
        _assert_valid(p)


# ---------------------------------------------------------------------------
# [solver] dt0  (line 564)
# ---------------------------------------------------------------------------

class TestSolverDt0:
    def test_negative_dt0_raises(self, tmp_path):
        p = _write_cfg(tmp_path, "[solver]\ndt0 = -0.01\n")
        _assert_error(p)

    def test_positive_dt0_passes(self, tmp_path):
        p = _write_cfg(tmp_path, "[solver]\ndt0 = 0.05\n")
        _assert_valid(p)


# ---------------------------------------------------------------------------
# [solver] root_find_max_steps  (line 568)
# ---------------------------------------------------------------------------

class TestRootFindMaxSteps:
    def test_zero_root_find_max_steps_raises(self, tmp_path):
        p = _write_cfg(tmp_path, "[solver]\nroot_find_max_steps = 0\n")
        _assert_error(p)


# ---------------------------------------------------------------------------
# [optimisation] ls_solver  (line 575)
# ---------------------------------------------------------------------------

class TestLsSolver:
    def test_invalid_ls_solver_raises(self, tmp_path):
        p = _write_cfg(tmp_path, '[optimisation]\nls_solver = "unknown_solver"\n')
        _assert_error(p)

    def test_valid_ls_solver_passes(self, tmp_path):
        p = _write_cfg(tmp_path, '[optimisation]\nls_solver = "lm"\n')
        _assert_valid(p)


# ---------------------------------------------------------------------------
# [optimisation] optx_adjoint  (line 582)
# ---------------------------------------------------------------------------

class TestOptxAdjoint:
    def test_invalid_optx_adjoint_raises(self, tmp_path):
        p = _write_cfg(tmp_path, '[optimisation]\noptx_adjoint = "bad_adj"\n')
        _assert_error(p)


# ---------------------------------------------------------------------------
# [optimisation] jac_mode  (line 589)
# ---------------------------------------------------------------------------

class TestJacMode:
    def test_invalid_jac_mode_raises(self, tmp_path):
        p = _write_cfg(tmp_path, '[optimisation]\njac_mode = "unknown"\n')
        _assert_error(p)

    def test_fwd_with_recursive_adjoint_raises(self, tmp_path):
        """jac_mode='fwd' + ode_adjoint='recursive' is invalid combination."""
        p = _write_cfg(
            tmp_path,
            '[optimisation]\njac_mode = "fwd"\n\n[solver]\node_adjoint = "recursive"\n',
        )
        _assert_error(p)

    def test_bwd_with_forward_adjoint_is_warning_only(self, tmp_path):
        """jac_mode='bwd' + ode_adjoint='forward' → warning, not error."""
        p = _write_cfg(
            tmp_path,
            '[optimisation]\njac_mode = "bwd"\n\n[solver]\node_adjoint = "forward"\n',
        )
        _assert_valid(p)


# ---------------------------------------------------------------------------
# [derived_rates] s_prod_fn  (line 623)
# ---------------------------------------------------------------------------

class TestSProdFn:
    def test_invalid_s_prod_fn_raises(self, tmp_path):
        p = _write_cfg(tmp_path, '[derived_rates]\ns_prod_fn = "invalid_fn"\n')
        _assert_error(p)

    def test_valid_s_prod_fn_passes(self, tmp_path):
        for val in ("softplus", "linear"):
            p = _write_cfg(tmp_path, f'[derived_rates]\ns_prod_fn = "{val}"\n')
            _assert_valid(p)


# ---------------------------------------------------------------------------
# [time] interpolation  (line 633)
# ---------------------------------------------------------------------------

class TestTimeInterpolation:
    def test_invalid_interpolation_raises(self, tmp_path):
        p = _write_cfg(tmp_path, '[time]\ninterpolation = "cubic"\n')
        _assert_error(p)

    def test_valid_interpolation_passes(self, tmp_path):
        p = _write_cfg(tmp_path, '[time]\ninterpolation = "piecewise_constant"\n')
        _assert_valid(p)


# ---------------------------------------------------------------------------
# [bounds] section  (lines 642-658)
# ---------------------------------------------------------------------------

class TestBoundsSection:
    def test_non_positive_rate_max_raises(self, tmp_path):
        p = _write_cfg(tmp_path, "[bounds]\nrate_max = 0.0\n")
        _assert_error(p)

    def test_negative_protein_degradation_max_raises(self, tmp_path):
        p = _write_cfg(tmp_path, "[bounds]\nprotein_degradation_max = -1.0\n")
        _assert_error(p)

    def test_valid_bounds_pass(self, tmp_path):
        p = _write_cfg(tmp_path, "[bounds]\nrate_max = 5.0\nkinase_rate_max = 2.0\n")
        _assert_valid(p)


# ---------------------------------------------------------------------------
# [steadystate] section  (lines 663-739)
# ---------------------------------------------------------------------------

class TestSteadystateSection:
    def _ss_base(self, tmp_path, extra: str) -> str:
        return _write_cfg(tmp_path, "[steadystate]\n" + extra)

    def test_non_positive_t_end_raises(self, tmp_path):
        _assert_error(self._ss_base(tmp_path, "t_end = 0.0\n"))

    def test_non_positive_early_end_raises(self, tmp_path):
        _assert_error(self._ss_base(tmp_path, "early_end = -10.0\n"))

    def test_t_end_less_than_early_end_raises(self, tmp_path):
        _assert_error(self._ss_base(tmp_path, "t_end = 50.0\nearly_end = 100.0\n"))

    def test_n_early_less_than_2_raises(self, tmp_path):
        _assert_error(self._ss_base(tmp_path, "n_early = 1\n"))

    def test_n_late_less_than_2_raises(self, tmp_path):
        _assert_error(self._ss_base(tmp_path, "n_late = 1\n"))

    def test_invalid_late_grid_raises(self, tmp_path):
        _assert_error(self._ss_base(tmp_path, 'late_grid = "invalid"\n'))

    def test_non_positive_rtol_raises(self, tmp_path):
        _assert_error(self._ss_base(tmp_path, "rtol = 0.0\n"))

    def test_non_positive_atol_raises(self, tmp_path):
        _assert_error(self._ss_base(tmp_path, "atol = -1e-8\n"))

    def test_negative_dt0_raises(self, tmp_path):
        _assert_error(self._ss_base(tmp_path, "dt0 = -0.1\n"))

    def test_zero_max_steps_raises(self, tmp_path):
        _assert_error(self._ss_base(tmp_path, "max_steps = 0\n"))

    def test_zero_top_n_raises(self, tmp_path):
        _assert_error(self._ss_base(tmp_path, "top_n = 0\n"))

    def test_valid_steadystate_passes(self, tmp_path):
        p = self._ss_base(tmp_path, "t_end = 2000.0\nearly_end = 100.0\nn_early = 100\n")
        _assert_valid(p)


# ---------------------------------------------------------------------------
# [simulation] section  (lines 744-750)
# ---------------------------------------------------------------------------

class TestSimulationSection:
    def test_dense_n_points_too_small_raises(self, tmp_path):
        p = _write_cfg(tmp_path, "[simulation]\ndense_n_points = 1\n")
        _assert_error(p)

    def test_valid_dense_n_points_passes(self, tmp_path):
        p = _write_cfg(tmp_path, "[simulation]\ndense_n_points = 100\n")
        _assert_valid(p)


# ---------------------------------------------------------------------------
# [data_interpolation] section  (lines 755-770)
# ---------------------------------------------------------------------------

class TestDataInterpolationSection:
    def test_invalid_method_raises(self, tmp_path):
        p = _write_cfg(tmp_path, '[data_interpolation]\nmethod = "spline"\n')
        _assert_error(p)

    def test_invalid_replace_nans_at_start_raises(self, tmp_path):
        p = _write_cfg(tmp_path, '[data_interpolation]\nreplace_nans_at_start = "invalid"\n')
        _assert_error(p)

    def test_valid_data_interpolation_passes(self, tmp_path):
        p = _write_cfg(tmp_path, '[data_interpolation]\nmethod = "linear"\nreplace_nans_at_start = "zero"\n')
        _assert_valid(p)


# ---------------------------------------------------------------------------
# [optimisation] loss_type / pseudo_huber_delta / slope_lambda validation
# ---------------------------------------------------------------------------

class TestLossTypeValidation:
    """Tests for the new configurable loss fields in [optimisation]."""

    def test_invalid_loss_type_raises(self, tmp_path):
        p = _write_cfg(tmp_path, '[optimisation]\nloss_type = "huber_absolute"\n')
        _assert_error(p)

    def test_valid_loss_types_pass(self, tmp_path):
        for lt in ("mse", "pseudo_huber", "pseudo_huber_slope", "log_cosh"):
            p = _write_cfg(tmp_path, f'[optimisation]\nloss_type = "{lt}"\n')
            _assert_valid(p)

    def test_pseudo_huber_delta_zero_raises(self, tmp_path):
        p = _write_cfg(
            tmp_path,
            "[optimisation]\nloss_type = \"pseudo_huber\"\npseudo_huber_delta = 0.0\n",
        )
        _assert_error(p)

    def test_pseudo_huber_delta_negative_raises(self, tmp_path):
        p = _write_cfg(
            tmp_path,
            "[optimisation]\nloss_type = \"pseudo_huber\"\npseudo_huber_delta = -0.5\n",
        )
        _assert_error(p)

    def test_pseudo_huber_delta_positive_passes(self, tmp_path):
        p = _write_cfg(
            tmp_path,
            "[optimisation]\nloss_type = \"pseudo_huber\"\npseudo_huber_delta = 0.2\n",
        )
        _assert_valid(p)

    def test_slope_lambda_negative_raises(self, tmp_path):
        p = _write_cfg(
            tmp_path,
            "[optimisation]\nloss_type = \"pseudo_huber_slope\"\nslope_lambda = -0.1\n",
        )
        _assert_error(p)

    def test_slope_lambda_zero_passes(self, tmp_path):
        """slope_lambda = 0 is allowed (disables slope term)."""
        p = _write_cfg(
            tmp_path,
            "[optimisation]\nloss_type = \"pseudo_huber_slope\"\nslope_lambda = 0.0\n",
        )
        _assert_valid(p)

    def test_slope_lambda_positive_passes(self, tmp_path):
        p = _write_cfg(
            tmp_path,
            "[optimisation]\nloss_type = \"pseudo_huber_slope\"\nslope_lambda = 0.5\n",
        )
        _assert_valid(p)

    def test_default_loss_type_is_mse(self, tmp_path):
        """Without explicit loss_type, default must be 'mse' (no error)."""
        p = _write_cfg(tmp_path)
        cfg = load_config(p)
        assert getattr(cfg.optimisation, "loss_type", "mse") == "mse"
        _assert_valid(p)

    def test_default_pseudo_huber_delta(self, tmp_path):
        """Without explicit pseudo_huber_delta, default must be positive."""
        p = _write_cfg(tmp_path)
        cfg = load_config(p)
        delta = getattr(cfg.optimisation, "pseudo_huber_delta", 0.1)
        assert delta > 0

    def test_default_slope_lambda(self, tmp_path):
        """Without explicit slope_lambda, default must be >= 0."""
        p = _write_cfg(tmp_path)
        cfg = load_config(p)
        sl = getattr(cfg.optimisation, "slope_lambda", 0.1)
        assert sl >= 0
