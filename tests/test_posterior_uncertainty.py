import numpy as np
from types import SimpleNamespace

from phoscrosstalk import posterior


def test_cfg_get_supports_none_and_namespace():
    assert posterior._cfg_get(None, "x", 5) == 5
    cfg = SimpleNamespace(alpha=3)
    assert posterior._cfg_get(cfg, "alpha", 0) == 3
    assert posterior._cfg_get(cfg, "missing", 7) == 7


def test_clip_theta_open_moves_to_open_interval():
    theta = np.array([0.0, 0.5, 1.0])
    xl = np.zeros(3)
    xu = np.ones(3)
    clipped = posterior._clip_theta_open(theta, xl, xu)
    assert np.all(clipped > xl)
    assert np.all(clipped < xu)


def test_resample_residual_matrix_modes():
    rng = np.random.default_rng(0)
    resid = np.array([[1.0, 2.0], [3.0, 4.0]])
    out_case = posterior._resample_residual_matrix(rng, resid, mode="case")
    out_time = posterior._resample_residual_matrix(rng, resid, mode="time")
    out_el = posterior._resample_residual_matrix(rng, resid, mode="element")
    assert out_case.shape == resid.shape
    assert out_time.shape == resid.shape
    assert out_el.shape == resid.shape


def test_summarize_samples_has_expected_columns():
    samples = np.array([[1.0, 2.0], [1.5, 2.5], [2.0, 3.0]])
    df = posterior._summarize_samples(samples, ["a", "b"])
    assert list(df["parameter"]) == ["a", "b"]
    assert {"mean", "std", "q2.5", "q97.5", "n_success"}.issubset(df.columns)
