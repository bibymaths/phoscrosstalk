"""
PhosCrosstalk.

Optimization-driven modeling of phosphorylation crosstalk networks using
transcriptomics, proteomics, phosphoproteomics, kinase-site priors, and
mechanistic ODE-based simulations.

This package intentionally avoids importing JAX-dependent modules at package
import time. Runtime environment variables such as ``JAX_PLATFORMS`` and
``XLA_FLAGS`` must be configured before JAX is imported. Public heavy symbols
are therefore exposed through lazy imports using module-level ``__getattr__``.
"""

from __future__ import annotations

from importlib import import_module
from importlib.metadata import PackageNotFoundError, metadata, version
from typing import Any

_PACKAGE_NAME = "phoscrosstalk"

try:
    __version__ = version(_PACKAGE_NAME)
except PackageNotFoundError:
    __version__ = "0.1.0"

try:
    _meta = metadata(_PACKAGE_NAME)
except PackageNotFoundError:
    _meta = {}

__author__ = _meta.get("Author", "Abhinav Mishra")
__email__ = _meta.get("Author-email", "mishraabhinav36@gmail.com")
__license__ = _meta.get("License-Expression", "BSD-3-Clause")
__description__ = _meta.get(
    "Summary",
    (
        "PhosCrosstalk is a Python package for optimization-driven modeling of "
        "phosphorylation crosstalk networks using transcriptomics, proteomics, "
        "phosphoproteomics, kinase-site priors, and mechanistic ODE-based "
        "simulations."
    ),
)
__url__ = "https://github.com/bibymaths/phoscrosstalk"
__docs__ = "https://bibymaths.github.io/phoscrosstalk"

# ---------------------------------------------------------------------------
# Lazy public API
# ---------------------------------------------------------------------------
# Do not eagerly import these modules here. Several of them import JAX, Diffrax,
# Optimistix, or Equinox either directly or indirectly.
#
# Keep this list limited to stable public API objects. Do not expose every
# internal helper from every module.
# ---------------------------------------------------------------------------

_LAZY_ATTRS: dict[str, str] = {
    # Configuration
    "ModelDims": "phoscrosstalk.config",
    "load_config": "phoscrosstalk.config",
    "validate_config": "phoscrosstalk.config",

    # Mechanisms / RHS construction
    "decode_theta": "phoscrosstalk.mechanisms",
    "make_rhs": "phoscrosstalk.mechanisms",
    "compute_prev_site_idx": "phoscrosstalk.mechanisms",

    # Optimization
    "NetworkProblem": "phoscrosstalk.optimization",
    "create_bounds": "phoscrosstalk.optimization",
    "make_residuals_fn": "phoscrosstalk.optimization",
    "validate_problem_shapes": "phoscrosstalk.optimization",
    "build_parameter_labels": "phoscrosstalk.optimization",
    "compute_second_order_sensitivities": "phoscrosstalk.optimization",

    # Multistart fitting
    "run_multi_start_optimization": "phoscrosstalk.multistarts",

    # Derived rates
    "make_k_act_fn": "phoscrosstalk.derived_rates",
    "make_s_prod_fn": "phoscrosstalk.derived_rates",

    # Weighting
    "build_weight_matrices": "phoscrosstalk.weighting",

    # Runtime environment
    "enable_x64": "phoscrosstalk.runtime_env",
    "setup_cpu_env": "phoscrosstalk.runtime_env",
    "plan_cpu_runtime": "phoscrosstalk.runtime_env",
    "log_env_summary": "phoscrosstalk.runtime_env",
}

__all__ = [
    "__version__",
    "__author__",
    "__email__",
    "__license__",
    "__description__",
    "__url__",
    "__docs__",
    *_LAZY_ATTRS.keys(),
]


def __getattr__(name: str) -> Any:
    """
    Lazily import selected public symbols.

    This prevents JAX-dependent modules from being imported when the package is
    first imported, which is required so the CLI can configure JAX/XLA runtime
    settings before JAX is loaded.
    """
    if name not in _LAZY_ATTRS:
        raise AttributeError(f"module {_PACKAGE_NAME!r} has no attribute {name!r}")

    module = import_module(_LAZY_ATTRS[name])
    obj = getattr(module, name)

    # Cache the object so repeated access does not re-import/re-resolve it.
    globals()[name] = obj
    return obj


def __dir__() -> list[str]:
    """Return a stable list of public package attributes for autocomplete."""
    return sorted(set(globals()) | set(__all__))