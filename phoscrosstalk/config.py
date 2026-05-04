"""
config.py
Global configuration and state management for the Phospho-Network Model.

Provides:
  * ModelDims  – static dimension holder (K proteins, M kinases, N sites).
  * load_config(path)  – parse a config.toml file and return a SimpleNamespace
                         with all tuneable parameters.
  * validate_config(cfg, config_path) – validate all required fields and file
                         paths; raises SystemExit(1) on any error.
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
        "data": "",
        "ptm_intra": "",
        "ptm_inter": "",
        "output_dir": "results",
        "crosstalk_tsv": "",
        "rna_data": "",
        "tf_net": "",
        "kinase_tsv": "",
        "kea_ks_table": "",
        "unified_graph_pkl": "",
    },
    "model": {
        "mechanism": "dist",
        "scale_mode": "none",
        "length_scale": 50.0,
        "weight_scheme": "uniform",
        "receptors": [],
        "receptor_kinases": [],
        "include_tfs_as_proteins": False,
    },
    "optimisation": {
        "n_starts": 3,
        "max_steps": 500,
        "solver": "hybrid",
        "verbose": False,
        "rtol": 1e-8,
        "atol": 1e-8,
        "loss_type": "mse",
        "lambda_net": 0.0001,
        "reg_lambda": 0.0001,
    },
    "hybrid": {
        "lhs_n_samples": 512,
        "lhs_top_p": 16,
        "skip_lhs": False,
        "es_algo": "sep_cma_es",
        "es_popsize": 64,
        "es_n_generations": 200,
        "es_sigma_init": 0.3,
        "es_top_k": 5,
        "lm_max_steps": 500,
        "lm_rtol": 1e-8,
        "lm_atol": 1e-8,
        "seed": 0,
        "verbose": False,
    },
    "qdax": {
        "n_centroids": 1024,
        "batch_size": 256,
        "n_iterations": 2000,
        "iso_sigma": 0.005,
        "line_sigma": 0.05,
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
        "rna_relax": 0.1,
    },
    "bounds": {
        "rate_min": 1e-5,
        "rate_max": 10.0,
        "protein_degradation_max": 0.5,
        "kinase_rate_max": 3.0,
        "phosphatase_rate_max": 5.0,
        "gamma_abs_max": 2.0,
        "rna_max": 10.0,
        "abundance_max": 5.0,
    },
    "analysis": {
        "tune": False,
        "run_steadystate": False,
        "run_knockouts": False,
        "run_sensitivity": False,
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
    used instead – no error is raised so the caller can detect and report the
    missing file.

    Args:
        path: Path to a ``config.toml`` file.  Defaults to ``None``.

    Returns:
        SimpleNamespace with sections: paths, model, optimisation, loss_weights,
        solver, time, derived_rates, analysis.
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
# Config validator
# ---------------------------------------------------------------------------

_VALID_MECHANISMS = {"dist", "seq", "rand"}
_VALID_INTERP = {"piecewise_constant", "linear"}
_VALID_SCALE = {"none", "minmax", "zscore"}
_VALID_WEIGHT = {"uniform", "inverse_variance", "time_weighted"}
_VALID_SPROD = {"softplus", "linear"}


def _opt(val: str) -> str | None:
    """Return *val* stripped, or ``None`` if it is empty/absent."""
    v = (val or "").strip()
    return v if v else None


def validate_config(cfg: SimpleNamespace, config_path: str | None = None) -> None:
    """
    Validate all required fields in *cfg* and fail fast with clear error messages.

    Checks performed
    ----------------
    * Required input paths are non-empty and the files exist.
    * ``mechanism`` is one of ``{"dist", "seq", "rand"}``.
    * At least one kinase-prior source is present (``kinase_tsv`` or ``kea_ks_table``).
    * Numeric settings are valid (``n_starts > 0``, ``max_steps > 0``, tolerances > 0).
    * ``include_tfs_as_proteins`` requires ``rna_data`` and ``tf_net``.
    * Warns when filenames contain ``filtered`` while ``include_tfs_as_proteins`` is true.
    * Optional paths that are non-empty must point to existing files.

    Raises
    ------
    SystemExit(1) on the first batch of detected errors.

    Args:
        cfg:         Loaded config namespace from :func:`load_config`.
        config_path: Path that was passed to :func:`load_config` (for error messages).
    """
    errors: list[str] = []
    warnings: list[str] = []

    cfg_label = config_path or "config.toml"

    # -------------------------------------------------------------------
    # Required paths
    # -------------------------------------------------------------------
    data = _opt(cfg.paths.data)
    ptm_intra = _opt(cfg.paths.ptm_intra)
    ptm_inter = _opt(cfg.paths.ptm_inter)

    if not data:
        errors.append(
            f"  [paths] data is required – set it in {cfg_label}\n"
            '  e.g.  data = "data_timeseries/input1.csv"'
        )
    elif not os.path.exists(data):
        errors.append(f"  [paths] data file not found: {data!r}")

    if not ptm_intra:
        errors.append(
            f"  [paths] ptm_intra is required – set it in {cfg_label}\n"
            '  e.g.  ptm_intra = "data_curated/processed/ptm_intra.db"'
        )
    elif not os.path.exists(ptm_intra):
        errors.append(f"  [paths] ptm_intra file not found: {ptm_intra!r}")

    if not ptm_inter:
        errors.append(
            f"  [paths] ptm_inter is required – set it in {cfg_label}\n"
            '  e.g.  ptm_inter = "data_curated/processed/ptm_inter.db"'
        )
    elif not os.path.exists(ptm_inter):
        errors.append(f"  [paths] ptm_inter file not found: {ptm_inter!r}")

    # -------------------------------------------------------------------
    # Optional paths that, when set, must exist
    # -------------------------------------------------------------------
    optional_paths = {
        "crosstalk_tsv": _opt(cfg.paths.crosstalk_tsv),
        "rna_data": _opt(cfg.paths.rna_data),
        "tf_net": _opt(cfg.paths.tf_net),
        "kinase_tsv": _opt(cfg.paths.kinase_tsv),
        "kea_ks_table": _opt(cfg.paths.kea_ks_table),
        "unified_graph_pkl": _opt(cfg.paths.unified_graph_pkl),
    }
    for field, path in optional_paths.items():
        if path and not os.path.exists(path):
            errors.append(f"  [paths] {field} file not found: {path!r}")

    rna_data = optional_paths["rna_data"]
    tf_net = optional_paths["tf_net"]
    kinase_tsv = optional_paths["kinase_tsv"]
    kea_ks_table = optional_paths["kea_ks_table"]

    # -------------------------------------------------------------------
    # Kinase-prior requirement
    # -------------------------------------------------------------------
    if not kinase_tsv and not kea_ks_table:
        warnings.append(
            "  [paths] Neither kinase_tsv nor kea_ks_table is set. "
            "The model will fall back to an identity kinase mapping (no biological priors). "
            "Set [paths] kinase_tsv or [paths] kea_ks_table in config.toml."
        )

    # -------------------------------------------------------------------
    # Model settings
    # -------------------------------------------------------------------
    mechanism = getattr(cfg.model, "mechanism", "dist")
    if mechanism not in _VALID_MECHANISMS:
        errors.append(
            f"  [model] mechanism = {mechanism!r} is invalid. "
            f"Must be one of: {sorted(_VALID_MECHANISMS)}"
        )

    scale_mode = getattr(cfg.model, "scale_mode", "none")
    if scale_mode not in _VALID_SCALE:
        errors.append(
            f"  [model] scale_mode = {scale_mode!r} is invalid. "
            f"Must be one of: {sorted(_VALID_SCALE)}"
        )

    weight_scheme = getattr(cfg.model, "weight_scheme", "uniform")
    if weight_scheme not in _VALID_WEIGHT:
        errors.append(
            f"  [model] weight_scheme = {weight_scheme!r} is invalid. "
            f"Must be one of: {sorted(_VALID_WEIGHT)}"
        )

    length_scale = getattr(cfg.model, "length_scale", 50.0)
    if not isinstance(length_scale, (int, float)) or length_scale <= 0:
        errors.append(
            f"  [model] length_scale = {length_scale!r} must be a positive number."
        )

    # -------------------------------------------------------------------
    # RNA / TF requirements
    # -------------------------------------------------------------------
    include_tfs = getattr(cfg.model, "include_tfs_as_proteins", False)
    if include_tfs:
        if not rna_data:
            errors.append(
                "  [model] include_tfs_as_proteins = true requires [paths] rna_data to be set."
            )
        if not tf_net:
            errors.append(
                "  [model] include_tfs_as_proteins = true requires [paths] tf_net to be set."
            )
        # Warn if filenames look like filtered data
        for field, path in [("data", data), ("rna_data", rna_data)]:
            if path and "filtered" in os.path.basename(path).lower():
                warnings.append(
                    f"  [model] include_tfs_as_proteins = true expects full (unfiltered) "
                    f"time-series files. [paths] {field} appears to be filtered: {path!r}. "
                    "Filtered files may exclude TF proteins before modeling."
                )

    if rna_data and not tf_net:
        warnings.append(
            "  [paths] rna_data is set but tf_net is absent. "
            "k_act will default to constant 1.0 (no TF-driven activation)."
        )

    # -------------------------------------------------------------------
    # Optimisation settings
    # -------------------------------------------------------------------
    n_starts = getattr(cfg.optimisation, "n_starts", 3)
    if not isinstance(n_starts, int) or n_starts < 1:
        errors.append(
            f"  [optimisation] n_starts = {n_starts!r} must be a positive integer."
        )

    max_steps = getattr(cfg.optimisation, "max_steps", 500)
    if not isinstance(max_steps, int) or max_steps < 1:
        errors.append(
            f"  [optimisation] max_steps = {max_steps!r} must be a positive integer."
        )

    lambda_net = getattr(cfg.optimisation, "lambda_net", 0.0001)
    if not isinstance(lambda_net, (int, float)) or lambda_net < 0:
        errors.append(f"  [optimisation] lambda_net = {lambda_net!r} must be >= 0.")

    reg_lambda = getattr(cfg.optimisation, "reg_lambda", 0.0001)
    if not isinstance(reg_lambda, (int, float)) or reg_lambda < 0:
        errors.append(f"  [optimisation] reg_lambda = {reg_lambda!r} must be >= 0.")

    # -------------------------------------------------------------------
    # Solver tolerances
    # -------------------------------------------------------------------
    rtol = getattr(cfg.solver, "rtol", 1e-6)
    atol = getattr(cfg.solver, "atol", 1e-9)
    if not isinstance(rtol, (int, float)) or rtol <= 0:
        errors.append(f"  [solver] rtol = {rtol!r} must be a positive number.")
    if not isinstance(atol, (int, float)) or atol <= 0:
        errors.append(f"  [solver] atol = {atol!r} must be a positive number.")

    solver_max = getattr(cfg.solver, "max_steps", 16384)
    if not isinstance(solver_max, int) or solver_max < 1:
        errors.append(
            f"  [solver] max_steps = {solver_max!r} must be a positive integer."
        )

    # -------------------------------------------------------------------
    # Derived rates
    # -------------------------------------------------------------------
    s_prod_fn = getattr(cfg.derived_rates, "s_prod_fn", "softplus")
    if s_prod_fn not in _VALID_SPROD:
        errors.append(
            f"  [derived_rates] s_prod_fn = {s_prod_fn!r} is invalid. "
            f"Must be one of: {sorted(_VALID_SPROD)}"
        )

    # -------------------------------------------------------------------
    # Time interpolation
    # -------------------------------------------------------------------
    interp = getattr(cfg.time, "interpolation", "piecewise_constant")
    if interp not in _VALID_INTERP:
        errors.append(
            f"  [time] interpolation = {interp!r} is invalid. "
            f"Must be one of: {sorted(_VALID_INTERP)}"
        )

    # -------------------------------------------------------------------
    # Bounds (optional section – all values must be positive when present)
    # -------------------------------------------------------------------
    bounds_cfg = getattr(cfg, "bounds", None)
    if bounds_cfg is not None:
        _positive_bound_fields = [
            "rate_min",
            "rate_max",
            "protein_degradation_max",
            "kinase_rate_max",
            "phosphatase_rate_max",
            "gamma_abs_max",
            "rna_max",
            "abundance_max",
        ]
        for field in _positive_bound_fields:
            val = getattr(bounds_cfg, field, None)
            if val is not None and (not isinstance(val, (int, float)) or val <= 0):
                errors.append(
                    f"  [bounds] {field} = {val!r} must be a positive number."
                )

    # -------------------------------------------------------------------
    # Report
    # -------------------------------------------------------------------
    if warnings:
        for w in warnings:
            print(f"WARNING:{w}", flush=True)

    if errors:
        print(
            f"\nERROR: Configuration validation failed ({len(errors)} error(s)):\n"
            + "\n".join(errors)
            + f"\n\nFix the above errors in {cfg_label} and re-run.\n",
            flush=True,
        )
        raise SystemExit(1)


# ---------------------------------------------------------------------------
# Legacy constants
# ---------------------------------------------------------------------------

DEFAULT_TIMEPOINTS = np.array(
    [0.0, 0.5, 0.75, 1.0, 2.0, 4.0, 8.0, 16.0, 30.0, 60.0, 120.0, 240.0, 480.0, 960.0]
)

EPS = 1e-8
