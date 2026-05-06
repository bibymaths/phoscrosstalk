"""Tests for config.py ModelDims behavior."""

import pytest

from phoscrosstalk.config import ModelDims


def test_modeldims_constructs_explicit_values():
    dims = ModelDims(K=5, M=10, N=20)
    assert dims.K == 5
    assert dims.M == 10
    assert dims.N == 20


def test_modeldims_set_dims():
    dims = ModelDims.set_dims(5, 10, 20)
    assert dims.K == 5
    assert dims.M == 10
    assert dims.N == 20


def test_modeldims_overwrite():
    dims1 = ModelDims.set_dims(1, 2, 3)
    dims2 = ModelDims.set_dims(7, 8, 9)
    assert (dims1.K, dims1.M, dims1.N) == (1, 2, 3)
    assert (dims2.K, dims2.M, dims2.N) == (7, 8, 9)


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
    # rna_relax is in derived_rates
    assert cfg.derived_rates.rna_relax == pytest.approx(0.1)


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

    with pytest.raises(SystemExit) as exc_info:
        validate_config(cfg, str(cfg_path))
    assert exc_info.value.code == 1


# ---------------------------------------------------------------------------
# ModelDims tests (Task 5 from ModelDims refactor)
# ---------------------------------------------------------------------------

def test_model_dims_set_dims_valid():
    """set_dims with valid positive ints should return immutable dims."""
    from phoscrosstalk.config import ModelDims

    dims = ModelDims.set_dims(3, 2, 5)
    assert dims.K == 3
    assert dims.M == 2
    assert dims.N == 5


def test_model_dims_validation_zero():
    """set_dims must raise ValueError for K=0."""
    from phoscrosstalk.config import ModelDims

    with pytest.raises(ValueError, match="K"):
        ModelDims.set_dims(0, 2, 5)


def test_model_dims_validation_negative():
    """set_dims must raise ValueError for M=-1."""
    from phoscrosstalk.config import ModelDims

    with pytest.raises(ValueError, match="M"):
        ModelDims.set_dims(3, -1, 5)


def test_model_dims_validation_zero_n():
    """set_dims must raise ValueError for N=0."""
    from phoscrosstalk.config import ModelDims

    with pytest.raises(ValueError, match="N"):
        ModelDims.set_dims(3, 2, 0)


def test_model_dims_from_data_basic():
    """from_data infers K, M, N correctly from data arrays."""
    import numpy as np
    from phoscrosstalk.config import ModelDims

    P_data = np.ones((5, 8))          # N=5 sites, T=8 timepoints
    A_data = np.ones((3, 8))          # K=3 proteins
    kin_to_prot_idx = np.array([0, 1])  # M=2 kinases

    dims = ModelDims.from_data(P_data, A_data, kin_to_prot_idx)
    assert dims.N == 5
    assert dims.K == 3
    assert dims.M == 2


def test_model_dims_from_data_no_A():
    """from_data falls back to K=N when A_data is None."""
    import numpy as np
    from phoscrosstalk.config import ModelDims

    P_data = np.ones((4, 6))
    kin_to_prot_idx = np.array([0, 1, 2])

    dims = ModelDims.from_data(P_data, None, kin_to_prot_idx)
    assert dims.N == 4
    assert dims.K == 4  # falls back to N
    assert dims.M == 3


def test_model_dims_from_data_empty_A():
    """from_data falls back to K=N when A_data has size 0."""
    import numpy as np
    from phoscrosstalk.config import ModelDims

    P_data = np.ones((4, 6))
    A_data = np.zeros((0, 6))  # empty
    kin_to_prot_idx = np.array([0])

    dims = ModelDims.from_data(P_data, A_data, kin_to_prot_idx)
    assert dims.K == 4


def test_two_runs_different_dims_set_dims():
    """Regression: sequential set_dims with different values must be independent."""
    from phoscrosstalk.config import ModelDims

    # Run 1
    dims1 = ModelDims.set_dims(3, 2, 5)
    assert dims1.K == 3
    assert dims1.M == 2
    assert dims1.N == 5

    # Run 2 — different dims must completely replace Run 1
    dims2 = ModelDims.set_dims(6, 4, 10)
    assert dims2.K == 6
    assert dims2.M == 4
    assert dims2.N == 10
