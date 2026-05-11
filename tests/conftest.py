import json
from types import SimpleNamespace

import numpy as np
import pytest

from phoscrosstalk.config import ModelDims


@pytest.fixture
def tiny_time():
    return np.array([0.0, 1.0, 2.0], dtype=np.float64)


@pytest.fixture
def tiny_phospho():
    return np.array([[1.0, 1.1, 1.2], [0.9, 1.0, 1.1]], dtype=np.float64)


@pytest.fixture
def tiny_protein():
    return np.array([[1.0, 1.0, 1.0], [1.2, 1.1, 1.0]], dtype=np.float64)


@pytest.fixture
def tiny_rna():
    return np.array([[1.0, 1.1, 1.2], [0.8, 0.9, 1.0]], dtype=np.float64)


@pytest.fixture
def tiny_dims():
    return ModelDims(K=2, M=1, N=2)


@pytest.fixture
def tiny_bounds():
    xl = np.array([-2.0, -2.0, -2.0], dtype=np.float64)
    xu = np.array([2.0, 2.0, 2.0], dtype=np.float64)
    return xl, xu


@pytest.fixture
def time_axes_file(tmp_path):
    path = tmp_path / "time_axes.json"
    path.write_text(
        json.dumps({"phosphosite_time_points": [0.0, 1.0, 2.0]}),
        encoding="utf-8",
    )
    return path


@pytest.fixture
def minimal_runtime_cfg():
    return SimpleNamespace(
        cpu_threads="auto",
        parallel_starts="auto",
        threads_per_start="auto",
        reserve_cores=0,
        use_physical_cores=True,
    )
