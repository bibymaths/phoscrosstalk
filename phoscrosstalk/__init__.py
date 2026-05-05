__version__ = "0.1.0"
__author__ = "Abhinav Mishra"
__email__ = "mishraabhinav36@gmail.com"
__license__ = "BSD-3-Clause"
__description__ = (
    "A tool to analyze phosphoproteomic crosstalk using graph-based regularization."
)

# NOTE: Do NOT import JAX-dependent modules here.
# phoscrosstalk/__init__.py is executed when the installed console-script
# entry point resolves ``phoscrosstalk.main:cli``.  Any JAX import at this
# level would occur BEFORE main.py can set ``JAX_PLATFORMS`` and ``XLA_FLAGS``
# via runtime_env.setup_cpu_env(), causing those settings to be ignored.
#
# Public symbols that were previously imported eagerly are now provided via
# __getattr__ (PEP 562 lazy module attributes) so that callers who do
#   from phoscrosstalk import build_parameter_labels
# still work correctly, but only after JAX has already been configured.

__all__ = [
    "build_parameter_labels",
    "compute_second_order_sensitivities",
]

_LAZY_ATTRS = {
    "build_parameter_labels": "phoscrosstalk.optimization",
    "compute_second_order_sensitivities": "phoscrosstalk.optimization",
}


def __getattr__(name: str):
    if name in _LAZY_ATTRS:
        import importlib

        mod = importlib.import_module(_LAZY_ATTRS[name])
        obj = getattr(mod, name)
        # Cache on the package so repeated access is fast
        globals()[name] = obj
        return obj
    raise AttributeError(f"module 'phoscrosstalk' has no attribute {name!r}")
