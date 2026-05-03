"""
Tests for config.py ModelDims singleton.
"""

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
