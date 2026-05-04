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
