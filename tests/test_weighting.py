import numpy as np
import pytest

from phoscrosstalk import weighting


def test_as_2d_float_matrix_handles_none_empty_and_invalid():
    assert weighting._as_2d_float_matrix(None, "x").shape == (0, 0)
    assert weighting._as_2d_float_matrix(np.array([]), "x").shape == (0, 0)
    with pytest.raises(ValueError, match="2D matrix"):
        weighting._as_2d_float_matrix(np.array([1.0, 2.0]), "x")
    with pytest.raises(ValueError, match="non-finite"):
        weighting._as_2d_float_matrix(np.array([[1.0, np.nan]]), "x")


def test_as_time_vector_and_normalize_mean_one():
    t = weighting._as_time_vector([0, 1, 2], "t")
    np.testing.assert_allclose(t, np.array([0.0, 1.0, 2.0]))
    with pytest.raises(ValueError, match="1D time vector"):
        weighting._as_time_vector([[0, 1]], "t")
    with pytest.raises(ValueError, match="at least one time point"):
        weighting._as_time_vector([], "t")
    with pytest.raises(ValueError, match="non-finite"):
        weighting._as_time_vector([0, np.inf], "t")

    empty = weighting._normalize_mean_one(np.array([]))
    assert empty.size == 0
    zeros = weighting._normalize_mean_one(np.zeros(3))
    np.testing.assert_allclose(zeros, np.ones(3))
    w = weighting._normalize_mean_one(np.array([1.0, 2.0, 3.0]))
    assert np.mean(w) == pytest.approx(1.0)


def test_compute_noise_weights_and_time_schemes():
    X = np.array([[1.0, 1.0, 1.0], [1.0, 2.0, 4.0]])
    noise_w = weighting._compute_noise_weights(X, name="X")
    assert noise_w.shape == (2,)
    assert np.mean(noise_w) == pytest.approx(1.0)
    assert noise_w[0] > noise_w[1]

    one_tp = weighting._compute_noise_weights(np.array([[1.0], [2.0]]), name="X")
    np.testing.assert_allclose(one_tp, np.ones(2))
    empty = weighting._compute_noise_weights(np.zeros((0, 0)), name="X")
    assert empty.shape == (0,)

    t = np.array([0.0, 1.0, 2.0, 4.0])
    uniform = weighting._time_weights_uniform(t)
    np.testing.assert_allclose(uniform, np.ones_like(t))
    early = weighting._time_weights_early_emphasis(t, strength=2.0)
    late = weighting._time_weights_late_emphasis(t, strength=2.0)
    assert early[0] > early[-1]
    assert late[0] < late[-1]
    moderate = weighting._time_weights_early_emphasis_moderate(t)
    assert moderate[0] > moderate[-1]
    fallback_early = weighting._time_weights_early_emphasis(t, t_mid=0.0, strength=2.0)
    fallback_late = weighting._time_weights_late_emphasis(t, t_mid=np.nan, strength=2.0)
    assert np.isfinite(fallback_early).all()
    assert np.isfinite(fallback_late).all()
    with pytest.raises(ValueError, match="strength must be positive"):
        weighting._time_weights_early_emphasis(t, strength=0)
    with pytest.raises(ValueError, match="strength must be positive"):
        weighting._time_weights_late_emphasis(t, strength=0)
    with pytest.raises(ValueError, match="Unknown weighting scheme"):
        weighting._time_weights_by_scheme(t, "bad")


def test_build_weight_matrices_all_modalities_and_validations():
    t = np.array([0.0, 1.0, 2.0])
    Y = np.array([[1.0, 1.2, 1.1], [0.9, 1.0, 0.95]])
    A = np.array([[1.0, 1.0, 1.0]])
    t_rna = np.array([0.0, 2.0])
    RNA = np.array([[1.0, 1.3], [0.8, 0.9]])

    W_data, W_prot, W_mrna = weighting.build_weight_matrices(
        t, Y, A, t_mrna=t_rna, rna_data=RNA, scheme="early_emphasis"
    )
    assert W_data.shape == Y.shape
    assert W_prot.shape == A.shape
    assert W_mrna.shape == RNA.shape
    assert np.isfinite(W_data).all()
    assert np.isfinite(W_prot).all()
    assert np.isfinite(W_mrna).all()

    W_data2, W_prot2, W_mrna2 = weighting.build_weight_matrices(
        t, Y, None, t_mrna=t_rna, rna_data=RNA, scheme="flat_no_noise"
    )
    assert np.all(W_data2 > 0)
    assert W_prot2.shape == (0, len(t))
    assert np.all(W_mrna2 > 0)

    with pytest.raises(ValueError, match="Y has 2 columns"):
        weighting.build_weight_matrices(t, np.ones((1, 2)))
    with pytest.raises(ValueError, match="A_data has 2 columns"):
        weighting.build_weight_matrices(t, Y, np.ones((1, 2)))
    ok = weighting.build_weight_matrices(
        t, Y, None, t_mrna=np.array([0.0, 1.0, 2.0]), rna_data=np.ones((1, 3))
    )
    assert ok[2].shape == (1, 3)
    with pytest.raises(ValueError, match="rna_data was provided but t_mrna is None"):
        weighting.build_weight_matrices(t, Y, None, rna_data=RNA)
    with pytest.raises(ValueError, match="rna_data has 3 columns"):
        weighting.build_weight_matrices(t, Y, None, t_mrna=t_rna, rna_data=np.ones((1, 3)))
