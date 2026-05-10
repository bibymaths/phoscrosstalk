# SPDX-License-Identifier: MIT
"""
PINN-specific configuration helpers.

Provides:
  * PINN_DEFAULTS    – default values for the [pinn] TOML section.
  * validate_pinn_config(pinn_cfg) – raise ValueError on invalid PINN config.
  * get_pinn_cfg(cfg) – extract and validate PINNConfig from a loaded config.
"""

from __future__ import annotations

from types import SimpleNamespace

# ---------------------------------------------------------------------------
# Defaults (mirrored in config._DEFAULTS["pinn"])
# ---------------------------------------------------------------------------

PINN_DEFAULTS: dict = {
    "enabled": False,
    # Architecture
    "width_size": 64,
    "depth": 2,
    "activation": "tanh",
    # Regularisation
    "lambda_pinn": 0.1,
    "regularize": "residual_l2",
    # Optimisation
    "max_steps": 500,
    "learning_rate": 1e-3,
    "rtol": 1e-8,
    "atol": 1e-8,
    "seed": 0,
    "verbose": True,
    "print_every": 50,
    "grad_clip": 1.0,
    "optimizer": "adam",
    # Runtime
    "use_max_machine_threads": True,
}

_VALID_ACTIVATIONS = {"tanh", "relu", "gelu", "silu", "softplus", "elu"}
_VALID_REGULARIZE = {"residual_l2", "param_l2"}


def validate_pinn_config(pinn_cfg: SimpleNamespace) -> None:
    """
    Validate the [pinn] config section.

    Raises
    ------
    ValueError  on the first detected violation.
    """
    errors: list[str] = []

    width = int(getattr(pinn_cfg, "width_size", 64))
    if width <= 0:
        errors.append(f"[pinn] width_size = {width!r} must be > 0.")

    depth = int(getattr(pinn_cfg, "depth", 2))
    if depth < 1:
        errors.append(f"[pinn] depth = {depth!r} must be >= 1.")

    activation = str(getattr(pinn_cfg, "activation", "tanh"))
    if activation not in _VALID_ACTIVATIONS:
        errors.append(
            f"[pinn] activation = {activation!r} is invalid. "
            f"Must be one of: {sorted(_VALID_ACTIVATIONS)}"
        )

    lambda_pinn = float(getattr(pinn_cfg, "lambda_pinn", 0.1))
    if lambda_pinn < 0:
        errors.append(f"[pinn] lambda_pinn = {lambda_pinn!r} must be >= 0.")

    max_steps = int(getattr(pinn_cfg, "max_steps", 500))
    if max_steps < 1:
        errors.append(f"[pinn] max_steps = {max_steps!r} must be > 0.")

    regularize = str(getattr(pinn_cfg, "regularize", "residual_l2"))
    if regularize not in _VALID_REGULARIZE:
        errors.append(
            f"[pinn] regularize = {regularize!r} is invalid. "
            f"Must be one of: {sorted(_VALID_REGULARIZE)}"
        )

    rtol = float(getattr(pinn_cfg, "rtol", 1e-8))
    atol = float(getattr(pinn_cfg, "atol", 1e-8))
    if rtol <= 0:
        errors.append(f"[pinn] rtol = {rtol!r} must be > 0.")
    if atol <= 0:
        errors.append(f"[pinn] atol = {atol!r} must be > 0.")

    if errors:
        raise ValueError("PINN config validation failed:\n  " + "\n  ".join(errors))


def get_pinn_cfg(cfg: SimpleNamespace) -> SimpleNamespace:
    """
    Extract and return the pinn config namespace from a loaded config.

    If the [pinn] section is absent, returns defaults with enabled = False.
    Validates the extracted config and raises ValueError on invalid values.
    """
    pinn_raw = getattr(cfg, "pinn", None)
    if pinn_raw is None:
        return SimpleNamespace(**PINN_DEFAULTS)
    # Fill any missing keys with defaults
    merged = dict(PINN_DEFAULTS)
    for k, v in vars(pinn_raw).items():
        merged[k] = v
    ns = SimpleNamespace(**merged)
    if getattr(ns, "enabled", False):
        validate_pinn_config(ns)
    return ns
