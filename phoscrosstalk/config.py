"""
config.py
Global configuration and state management for the Phospho-Network Model.

Provides:
  * ModelDims  – static dimension holder (K proteins, M kinases, N sites).
  * load_config(path)  – parse a config.toml file and return a SimpleNamespace
                         with all tuneable parameters.
  * DEFAULT_TIMEPOINTS – legacy default time-point array.
  * EPS                – small constant for numerical stability.
"""

import os
from types import SimpleNamespace

import numpy as np

try:
    import tomllib  # stdlib Python ≥ 3.11
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # fallback for older environments


# ---------------------------------------------------------------------------
# Global dimensions container
# ---------------------------------------------------------------------------


class ModelDims:
    """
    Global container for storing model dimensions (Proteins, Kinases, Sites).

    Acts as a static state holder to avoid passing dimensions recursively
    through every function in the simulation pipeline.
    """

    K: int = None  # Number of Proteins
    M: int = None  # Number of Kinases
    N: int = None  # Number of Phosphosites

    @classmethod
    def set_dims(cls, k, m, n):
        """
        Set the global dimensions for the current model context.

        Args:
            k (int): Number of unique proteins (K).
            m (int): Number of kinases (M).
            n (int): Number of phosphorylation sites (N).

        Returns:
            None
        """
        cls.K = k
        cls.M = m
        cls.N = n


# ---------------------------------------------------------------------------
# TOML config loader
# ---------------------------------------------------------------------------

_DEFAULTS = {
    "paths": {
        "data_dir": "data_timeseries",
        "output_dir": "results",
    },
    "model": {
        "mechanism": "dist",
        "scale_mode": "none",
        "length_scale": 50.0,
        "weight_scheme": "uniform",
    },
    "optimisation": {
        "n_starts": 3,
        "max_steps": 500,
        "loss_type": "mse",
        "lambda_net": 0.0001,
        "reg_lambda": 0.0001,
    },
    "loss_weights": {
        "phospho": 1.0,
        "abundance": 1.0,
        "mrna": 1.0,
        "reg": 1.0,
    },
    "solver": {
        "rtol": 1e-6,
        "atol": 1e-9,
        "max_steps": 16384,
    },
    "time": {
        "mrna_time_points": [4, 8, 15, 30, 60, 120, 240, 480, 960],
        "interpolation": "piecewise_constant",
    },
    "derived_rates": {
        "s_prod_fn": "softplus",
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    """Merge *override* into *base* recursively; returns merged copy."""
    result = dict(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
    return result


def load_config(path: str | None = None) -> SimpleNamespace:
    """
    Parse *path* (a TOML file) and return a :class:`SimpleNamespace` containing
    all configuration sections as nested :class:`SimpleNamespace` objects.

    If *path* is ``None`` or the file does not exist the built-in defaults are
    used instead – no error is raised so the CLI can run without a config file.

    Args:
        path: Path to a ``config.toml`` file.  Defaults to ``None``.

    Returns:
        SimpleNamespace with sections: paths, model, optimisation, loss_weights,
        solver, time, derived_rates.
    """
    merged = dict(_DEFAULTS)
    if path is not None and os.path.exists(path):
        with open(path, "rb") as fh:
            toml_data = tomllib.load(fh)
        merged = _deep_merge(_DEFAULTS, toml_data)

    def _to_ns(d):
        if isinstance(d, dict):
            return SimpleNamespace(**{k: _to_ns(v) for k, v in d.items()})
        return d

    return _to_ns(merged)


# ---------------------------------------------------------------------------
# Legacy constants
# ---------------------------------------------------------------------------

DEFAULT_TIMEPOINTS = np.array(
    [0.0, 0.5, 0.75, 1.0, 2.0, 4.0, 8.0, 16.0, 30.0, 60.0, 120.0, 240.0, 480.0, 960.0]
)

EPS = 1e-8
