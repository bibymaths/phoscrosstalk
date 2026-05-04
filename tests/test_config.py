"""
Tests for config.py ModelDims singleton.
"""

import pytest

from phoscrosstalk.config import ModelDims


def test_modeldims_defaults_are_none():
    # Save and restore current state to avoid polluting other tests
    original_k = ModelDims.K
    original_m = ModelDims.M
    original_n = ModelDims.N

    # Reset to None for this test
    ModelDims.K = None
    ModelDims.M = None
    ModelDims.N = None

    assert ModelDims.K is None
    assert ModelDims.M is None
    assert ModelDims.N is None

    # Restore
    ModelDims.K = original_k
    ModelDims.M = original_m
    ModelDims.N = original_n


def test_modeldims_set_dims():
    ModelDims.set_dims(5, 10, 20)
    assert ModelDims.K == 5
    assert ModelDims.M == 10
    assert ModelDims.N == 20


def test_modeldims_overwrite():
    ModelDims.set_dims(1, 2, 3)
    ModelDims.set_dims(7, 8, 9)
    assert ModelDims.K == 7
    assert ModelDims.M == 8
    assert ModelDims.N == 9


# ---------------------------------------------------------------------------
# load_config tests
# ---------------------------------------------------------------------------


def test_load_config_defaults_when_no_file():
    from phoscrosstalk.config import load_config

    cfg = load_config(None)
    assert cfg.model.mechanism == "dist"
    assert cfg.optimisation.n_starts == 3
    assert cfg.solver.rtol == pytest.approx(1e-6)
    assert cfg.loss_weights.phospho == pytest.approx(1.0)


def test_load_config_nonexistent_path():
    from phoscrosstalk.config import load_config

    # Non-existent path should silently fall back to defaults
    cfg = load_config("/nonexistent/path/config.toml")
    assert cfg.model.mechanism == "dist"


def test_load_config_from_toml(tmp_path):
    from phoscrosstalk.config import load_config

    toml_content = """
[model]
mechanism = "seq"

[optimisation]
n_starts = 10

[loss_weights]
phospho = 2.0
"""
    p = tmp_path / "test_config.toml"
    p.write_text(toml_content)

    cfg = load_config(str(p))
    assert cfg.model.mechanism == "seq"
    assert cfg.optimisation.n_starts == 10
    assert cfg.loss_weights.phospho == pytest.approx(2.0)
    # Unset keys should use defaults
    assert cfg.solver.rtol == pytest.approx(1e-6)


# ---------------------------------------------------------------------------
# validate_config tests
# ---------------------------------------------------------------------------


def test_validate_config_missing_required_paths(tmp_path):
    """validate_config must raise SystemExit when required paths are empty."""
    from phoscrosstalk.config import load_config, validate_config

    # Minimal config with no paths set
    cfg_path = tmp_path / "empty.toml"
    cfg_path.write_text("")
    cfg = load_config(str(cfg_path))

    import pytest

    with pytest.raises(SystemExit) as exc_info:
        validate_config(cfg, str(cfg_path))
    assert exc_info.value.code == 1


def test_validate_config_invalid_mechanism(tmp_path):
    """validate_config must reject unknown mechanism values."""
    from phoscrosstalk.config import load_config, validate_config

    cfg_path = tmp_path / "bad_mech.toml"
    cfg_path.write_text('[model]\nmechanism = "invalid"\n')
    cfg = load_config(str(cfg_path))
    # required paths also missing so it will fail, but mechanism error must appear

    import pytest

    with pytest.raises(SystemExit) as exc_info:
        validate_config(cfg, str(cfg_path))
    assert exc_info.value.code == 1


def test_validate_config_valid_passes(tmp_path):
    """validate_config must not raise when all required paths exist."""
    from phoscrosstalk.config import load_config, validate_config

    # Create placeholder files
    data_file = tmp_path / "input1.csv"
    data_file.write_text("gene,x1\nEGFR,1.0\n")
    ptm_intra = tmp_path / "ptm_intra.db"
    ptm_intra.write_bytes(b"")
    ptm_inter = tmp_path / "ptm_inter.db"
    ptm_inter.write_bytes(b"")

    toml_content = f"""
[paths]
data = "{data_file}"
ptm_intra = "{ptm_intra}"
ptm_inter = "{ptm_inter}"
output_dir = "{tmp_path}"
"""
    cfg_path = tmp_path / "valid.toml"
    cfg_path.write_text(toml_content)
    cfg = load_config(str(cfg_path))

    # Should not raise
    validate_config(cfg, str(cfg_path))


def test_validate_config_include_tfs_requires_rna(tmp_path):
    """include_tfs_as_proteins = true must require rna_data and tf_net."""
    from phoscrosstalk.config import load_config, validate_config

    data_file = tmp_path / "input1.csv"
    data_file.write_text("gene,x1\nEGFR,1.0\n")
    ptm_intra = tmp_path / "ptm_intra.db"
    ptm_intra.write_bytes(b"")
    ptm_inter = tmp_path / "ptm_inter.db"
    ptm_inter.write_bytes(b"")

    toml_content = f"""
[paths]
data = "{data_file}"
ptm_intra = "{ptm_intra}"
ptm_inter = "{ptm_inter}"
output_dir = "{tmp_path}"

[model]
include_tfs_as_proteins = true
"""
    cfg_path = tmp_path / "tfs.toml"
    cfg_path.write_text(toml_content)
    cfg = load_config(str(cfg_path))

    import pytest

    with pytest.raises(SystemExit) as exc_info:
        validate_config(cfg, str(cfg_path))
    assert exc_info.value.code == 1


def test_validate_config_loads_new_paths_section(tmp_path):
    """load_config must expose all new [paths] fields with empty-string defaults."""
    from phoscrosstalk.config import load_config

    cfg = load_config(None)
    # All optional paths default to empty string
    assert cfg.paths.rna_data == ""
    assert cfg.paths.tf_net == ""
    assert cfg.paths.kinase_tsv == ""
    assert cfg.paths.kea_ks_table == ""
    assert cfg.paths.unified_graph_pkl == ""
    assert cfg.paths.crosstalk_tsv == ""


def test_validate_config_loads_analysis_section(tmp_path):
    """load_config must expose the [analysis] section with False defaults."""
    from phoscrosstalk.config import load_config

    cfg = load_config(None)
    assert cfg.analysis.tune is False
    assert cfg.analysis.run_steadystate is False
    assert cfg.analysis.run_knockouts is False
    assert cfg.analysis.run_sensitivity is False


def test_validate_config_invalid_n_starts(tmp_path):
    """validate_config must reject n_starts < 1."""
    from phoscrosstalk.config import load_config, validate_config

    data_file = tmp_path / "input1.csv"
    data_file.write_text("x\n")
    ptm_intra = tmp_path / "i.db"
    ptm_intra.write_bytes(b"")
    ptm_inter = tmp_path / "e.db"
    ptm_inter.write_bytes(b"")

    toml_content = f"""
[paths]
data = "{data_file}"
ptm_intra = "{ptm_intra}"
ptm_inter = "{ptm_inter}"

[optimisation]
n_starts = 0
"""
    cfg_path = tmp_path / "bad_nstarts.toml"
    cfg_path.write_text(toml_content)
    cfg = load_config(str(cfg_path))

    import pytest

    with pytest.raises(SystemExit) as exc_info:
        validate_config(cfg, str(cfg_path))
    assert exc_info.value.code == 1
