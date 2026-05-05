# SPDX-License-Identifier: MIT
"""
solver_config.py

Centralised Diffrax / Optimistix solver factories.

Purpose
-------
Keep solver, adjoint, and autodiff choices configurable from config.toml instead
of hard-coding them inside optimization.py and simulation.py.
"""

from __future__ import annotations

import diffrax
import optimistix as optx


_VALID_ODE_SOLVERS = {
    "tsit5",
    "dopri5",
    "dopri8",
    "bosh3",
    "kvaerno3",
    "kvaerno4",
    "kvaerno5",
}

_VALID_ODE_ADJOINTS = {
    "forward",
    "recursive",
    "checkpoint",
    "direct",
    "backsolve",
    "none",
}

_VALID_LS_SOLVERS = {
    "lm",
    "indirect_lm",
    "dogleg",
    "gauss_newton",
}

_VALID_OPTX_ADJOINTS = {
    "implicit",
    "checkpoint"
}


def make_diffrax_solver(
    kind: str,
    *,
    root_find_max_steps: int = 10,
    scan_kind: str | None = None,
):
    """
    Build a Diffrax ODE solver.

    Parameters
    ----------
    kind
        "tsit5", "dopri5", "dopri8", "bosh3", "kvaerno3", "kvaerno4", "kvaerno5".
    root_find_max_steps
        Used only by implicit Kvaerno solvers.
    scan_kind
        Optional scan kind for Tsit5. Use "bounded" only where higher-order AD
        needs it, e.g. Hessian paths.
    """
    kind = kind.lower()

    if kind not in _VALID_ODE_SOLVERS:
        raise ValueError(
            f"Unknown Diffrax ODE solver {kind!r}. "
            f"Valid choices: {sorted(_VALID_ODE_SOLVERS)}"
        )

    if kind == "tsit5":
        if scan_kind is None:
            return diffrax.Tsit5()
        return diffrax.Tsit5(scan_kind=scan_kind)

    if kind == "dopri5":
        return diffrax.Dopri5()

    if kind == "dopri8":
        return diffrax.Dopri8()

    if kind == "bosh3":
        return diffrax.Bosh3()

    if kind == "kvaerno3":
        return diffrax.Kvaerno3(root_find_max_steps=root_find_max_steps)

    if kind == "kvaerno4":
        return diffrax.Kvaerno4(root_find_max_steps=root_find_max_steps)

    if kind == "kvaerno5":
        return diffrax.Kvaerno5(root_find_max_steps=root_find_max_steps)

    raise AssertionError("unreachable")


def make_diffrax_adjoint(kind: str | None):
    """
    Build a Diffrax adjoint.

    For Optimistix least_squares with jac="fwd", use "forward".
    For scalar reverse-mode training/minimisation, use "recursive".
    Use "direct" only when mixed forward/reverse AD is unavoidable.
    """
    if kind is None:
        return None

    kind = kind.lower()

    if kind not in _VALID_ODE_ADJOINTS:
        raise ValueError(
            f"Unknown Diffrax adjoint {kind!r}. "
            f"Valid choices: {sorted(_VALID_ODE_ADJOINTS)}"
        )

    if kind == "none":
        return None

    if kind == "forward":
        return diffrax.ForwardMode()

    if kind in {"recursive", "checkpoint"}:
        return diffrax.RecursiveCheckpointAdjoint()

    if kind == "direct":
        return diffrax.DirectAdjoint()

    if kind == "backsolve":
        return diffrax.BacksolveAdjoint()

    raise AssertionError("unreachable")


def make_stepsize_controller(rtol: float, atol: float):
    """Build the Diffrax adaptive step-size controller."""
    return diffrax.PIDController(rtol=rtol, atol=atol)


def make_ls_solver(kind: str, rtol: float, atol: float, *, verbose: bool = False):
    """
    Build an Optimistix least-squares solver.
    """
    kind = kind.lower()

    if kind not in _VALID_LS_SOLVERS:
        raise ValueError(
            f"Unknown least-squares solver {kind!r}. "
            f"Valid choices: {sorted(_VALID_LS_SOLVERS)}"
        )

    if kind == "lm":
        return optx.LevenbergMarquardt(rtol=rtol, atol=atol, verbose=verbose)

    if kind == "indirect_lm":
        return optx.IndirectLevenbergMarquardt(
            rtol=rtol,
            atol=atol,
            verbose=verbose,
        )

    if kind == "dogleg":
        return optx.Dogleg(rtol=rtol, atol=atol, verbose=verbose)

    if kind == "gauss_newton":
        return optx.GaussNewton(rtol=rtol, atol=atol, verbose=verbose)

    raise AssertionError("unreachable")


def make_optx_adjoint(kind: str):
    """
    Build an Optimistix adjoint.

    This controls differentiation through the Optimistix solve itself.
    For normal fitting where you are not differentiating through the optimiser,
    ImplicitAdjoint is still the clean default.
    """
    kind = kind.lower()

    if kind not in _VALID_OPTX_ADJOINTS:
        raise ValueError(
            f"Unknown Optimistix adjoint {kind!r}. "
            f"Valid choices: {sorted(_VALID_OPTX_ADJOINTS)}"
        )

    if kind == "implicit":
        return optx.ImplicitAdjoint()

    if kind == "checkpoint":
        return optx.RecursiveCheckpointAdjoint()

    raise AssertionError("unreachable")