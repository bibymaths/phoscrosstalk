"""
steadystate.py
Long-horizon relaxation analysis for the phospho-network model.

Simulates the network far beyond the observed time window under terminal/derived
input to study convergence of the relative phosphosite signal and other bounded
states (S, A, Kdyn).  This is NOT a strict biological steady state unless derived
rate closures extrapolate/hold inputs beyond observed data.

Public interface:
    build_long_horizon_time_grid(...)  -- construct a piecewise time grid
    run_steadystate_analysis(...)      -- main entry point called from main.py
"""

import json
import os

import diffrax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from phoscrosstalk.config import ModelDims
from phoscrosstalk.logger import get_logger
from phoscrosstalk.mechanisms import compute_prev_site_idx, make_rhs
from phoscrosstalk.simulation import build_full_A0, simulate
from phoscrosstalk.solver_config import make_diffrax_solver, make_stepsize_controller

logger = get_logger()

# Small epsilon for relative convergence metrics
_EPS = 1e-12


def build_long_horizon_time_grid(
    t_end: float,
    early_end: float,
    n_early: int,
    n_late: int,
    late_grid: str = "geomspace",
) -> np.ndarray:
    """Build a piecewise time grid for long-horizon relaxation analysis.

    The grid has two segments:
    * ``[0, early_end]`` sampled with ``n_early`` uniformly spaced points.
    * ``(early_end, t_end]`` sampled with ``n_late`` points, either linearly
      or geometrically (``late_grid``).

    Args:
        t_end:      Final time (must be > ``early_end > 0``).
        early_end:  End of the fine early grid segment.
        n_early:    Points in ``[0, early_end]`` (must be >= 2).
        n_late:     Points in ``(early_end, t_end]`` (must be >= 2).
        late_grid:  ``"geomspace"`` (default) or ``"linear"``.

    Returns:
        np.ndarray: Sorted, unique, float64 time vector.
    """
    if early_end <= 0:
        raise ValueError(f"early_end must be > 0, got {early_end}")
    if t_end <= early_end:
        raise ValueError(f"t_end ({t_end}) must be > early_end ({early_end})")
    if n_early < 2:
        raise ValueError(f"n_early must be >= 2, got {n_early}")
    if n_late < 2:
        raise ValueError(f"n_late must be >= 2, got {n_late}")
    if late_grid not in {"linear", "geomspace"}:
        raise ValueError(
            f"late_grid must be 'linear' or 'geomspace', got {late_grid!r}"
        )

    t_early = np.linspace(0.0, early_end, n_early, dtype=np.float64)
    late_start = np.nextafter(early_end, np.inf)
    if late_grid == "geomspace":
        t_late = np.geomspace(late_start, t_end, n_late, dtype=np.float64)
    else:
        t_late = np.linspace(late_start, t_end, n_late, dtype=np.float64)

    combined = np.concatenate([t_early, t_late])
    return np.unique(combined).astype(np.float64)


def _log_matrix_diagnostics(name: str, arr: np.ndarray) -> None:
    total = arr.size
    n_finite = int(np.sum(np.isfinite(arr)))
    n_nan = int(np.sum(np.isnan(arr)))
    n_inf = int(np.sum(np.isinf(arr)))
    finite_vals = arr[np.isfinite(arr)]
    if finite_vals.size > 0:
        range_str = (
            f"min={finite_vals.min():.4g}  "
            f"mean={finite_vals.mean():.4g}  "
            f"max={finite_vals.max():.4g}"
        )
    else:
        range_str = "(no finite values)"
    logger.info(
        f"   [diag] {name}: shape={arr.shape}  "
        f"finite={n_finite}/{total}  NaN={n_nan}  Inf={n_inf}  {range_str}"
    )


def _save_diagnostics_tsv(ss_dir: str, matrices: dict) -> None:
    rows = []
    for name, arr in matrices.items():
        total = arr.size
        n_finite = int(np.sum(np.isfinite(arr)))
        n_nan = int(np.sum(np.isnan(arr)))
        n_inf = int(np.sum(np.isinf(arr)))
        finite_vals = arr[np.isfinite(arr)]
        rows.append(
            {
                "state": name,
                "shape": str(arr.shape),
                "total_elements": total,
                "finite_count": n_finite,
                "nan_count": n_nan,
                "inf_count": n_inf,
                "min": float(finite_vals.min())
                if finite_vals.size > 0
                else float("nan"),
                "mean": float(finite_vals.mean())
                if finite_vals.size > 0
                else float("nan"),
                "max": float(finite_vals.max())
                if finite_vals.size > 0
                else float("nan"),
            }
        )
    pd.DataFrame(rows).to_csv(
        os.path.join(ss_dir, "steadystate_diagnostics.tsv"), sep="\t", index=False
    )


def _save_convergence_tsv(ss_dir: str, matrices: dict) -> None:
    """Save absolute and relative last-step deltas to steadystate_convergence.tsv."""
    rows = []
    for name, X in matrices.items():
        if X.shape[1] < 2:
            rows.append(
                {
                    "state": name,
                    "mean_abs_delta": float("nan"),
                    "max_abs_delta": float("nan"),
                    "mean_rel_delta": float("nan"),
                    "max_rel_delta": float("nan"),
                }
            )
            continue
        delta_abs = np.abs(X[:, -1] - X[:, -2])
        delta_rel = delta_abs / (np.abs(X[:, -2]) + _EPS)
        fa = delta_abs[np.isfinite(delta_abs)]
        fr = delta_rel[np.isfinite(delta_rel)]
        rows.append(
            {
                "state": name,
                "mean_abs_delta": float(fa.mean()) if fa.size > 0 else float("nan"),
                "max_abs_delta": float(fa.max()) if fa.size > 0 else float("nan"),
                "mean_rel_delta": float(fr.mean()) if fr.size > 0 else float("nan"),
                "max_rel_delta": float(fr.max()) if fr.size > 0 else float("nan"),
            }
        )
    pd.DataFrame(rows).to_csv(
        os.path.join(ss_dir, "steadystate_convergence.tsv"), sep="\t", index=False
    )


def _save_output_tsvs(
    ss_dir, P_ss, A_ss, S_ss, Kdyn_ss, t_long, sites, proteins, kinases
):
    """Save the four primary output TSVs (backward-compatible filenames).

    Note: steadystate_sites.tsv stores relative phosphosite signal, not occupancy.
    """
    cols = [f"t_{t:.1f}" for t in t_long]
    pd.DataFrame(P_ss, index=sites, columns=cols).to_csv(
        os.path.join(ss_dir, "steadystate_sites.tsv"), sep="\t"
    )
    pd.DataFrame(A_ss, index=proteins, columns=cols).to_csv(
        os.path.join(ss_dir, "steadystate_proteins.tsv"), sep="\t"
    )
    pd.DataFrame(S_ss, index=proteins, columns=cols).to_csv(
        os.path.join(ss_dir, "steadystate_S.tsv"), sep="\t"
    )
    pd.DataFrame(Kdyn_ss, index=kinases, columns=cols).to_csv(
        os.path.join(ss_dir, "steadystate_Kdyn.tsv"), sep="\t"
    )


def _save_failure_diagnostics(ss_dir: str, message: str) -> None:
    path = os.path.join(ss_dir, "steadystate_diagnostics.tsv")
    with open(path, "w") as fh:
        fh.write(f"# Long-horizon simulation failed\n# Error: {message}\n")
    logger.info(f"   -> Failure note written to {path}")


def _save_metadata_json(
    ss_dir,
    t_end,
    early_end,
    n_early,
    n_late,
    late_grid,
    rtol,
    atol,
    dt0,
    max_steps,
    mechanism,
    has_k_act_fn,
    has_s_prod_fn,
    has_R_data0,
    use_event=False,
    event_rtol=None,
    event_atol=None,
    solve_status="unknown",
    t_final=None,
    final_deriv_norm=None,
    final_state_norm=None,
):
    meta = {
        "horizon": {
            "t_end": t_end,
            "early_end": early_end,
            "n_early": n_early,
            "n_late": n_late,
            "late_grid": late_grid,
        },
        "solver": {
            "rtol": rtol,
            "atol": atol,
            "dt0": dt0,
            "max_steps": max_steps,
        },
        "event": {
            "use_event": use_event,
            "event_rtol": event_rtol,
            "event_atol": event_atol,
        },
        "solve_result": {
            "status": solve_status,
            "t_final": t_final,
            "final_deriv_norm": final_deriv_norm,
            "final_state_norm": final_state_norm,
            "terminated_by_steady_state_event": solve_status == "event_occurred",
        },
        "model": {
            "mechanism": mechanism,
            "k_act_fn_provided": has_k_act_fn,
            "s_prod_fn_provided": has_s_prod_fn,
            "R_data0_provided": has_R_data0,
        },
        "note": (
            "Long-horizon relaxation analysis.  Derived-rate closures "
            "(k_act_fn, s_prod_fn) extrapolate beyond observed data using "
            "their configured interpolation scheme."
        ),
    }
    with open(os.path.join(ss_dir, "steadystate_metadata.json"), "w") as fh:
        json.dump(meta, fh, indent=2)


def _run_steadystate_solve(
    t_long: np.ndarray,
    P_data: np.ndarray,
    A0_initial: np.ndarray,
    theta_opt: np.ndarray,
    Cg,
    Cl,
    site_prot_idx,
    K_site_kin,
    R,
    L_alpha,
    kin_to_prot_idx,
    receptor_mask_prot,
    receptor_mask_kin,
    mechanism: str,
    rtol: float,
    atol: float,
    dt0: float,
    max_steps: int,
    k_act_fn=None,
    s_prod_fn=None,
    R_data0=None,
    rna_relax: float = 0.1,
    use_event: bool = True,
    event_rtol: float | None = None,
    event_atol: float | None = None,
    dims: ModelDims | None = None,
):
    """Run the steady-state ODE solve with optional Diffrax steady_state_event.

    When ``use_event=True`` (default), terminates early once the vector field
    satisfies ``norm(f) < atol_ev + rtol_ev * norm(y)``.  After the event fires,
    Diffrax fills remaining ``SaveAt`` time-points with ``inf``; those are
    converted to ``NaN`` here so downstream code (which already masks non-finite
    values) handles them correctly.

    Returns
    -------
    P_ss, A_ss, S_ss, Kdyn_ss : np.ndarray  (entities × time)
        Simulation outputs; NaN where the solve did not reach.
    t_trimmed : np.ndarray
        Time vector trimmed to the last valid (finite) column.
    solve_status : str
        One of "successful", "event_occurred", "max_steps_reached", "failed".
    t_final_actual : float
        Last time point with at least one finite value.
    """
    if dims is None:
        dims = ModelDims.set_dims(A0_initial.shape[0], len(kin_to_prot_idx), P_data.shape[0])
    K, M, N = dims.K, dims.M, dims.N
    T = len(t_long)
    N_sites = P_data.shape[0]

    # Build initial conditions (mirrors simulation.py)
    x0 = np.zeros(3 * K + M + N, dtype=np.float64)

    if R_data0 is not None:
        r_data = np.asarray(R_data0, dtype=np.float64)
        r0 = r_data[:, 0].copy() if r_data.ndim > 1 else r_data.copy()
        r0 = np.nan_to_num(r0, nan=1.0, posinf=5.0, neginf=0.0)
        r0 = np.clip(r0, 0.0, 10.0)
    else:
        r0 = np.ones(K, dtype=np.float64)
    x0[:K] = r0

    a0 = np.nan_to_num(A0_initial[:, 0].astype(np.float64), nan=1.0, posinf=5.0, neginf=0.0)
    a0 = np.clip(a0, 0.0, 5.0)
    x0[2 * K : 3 * K] = a0

    p0 = np.nan_to_num(P_data[:, 0].astype(np.float64), nan=0.0, posinf=10.0, neginf=0.0)
    p0 = np.clip(p0, 0.0, None)
    x0[3 * K + M :] = p0

    _nan_P = np.full((N_sites, T), np.nan)
    _nan_A = np.full((K, T), np.nan)
    _nan_S = np.full((K, T), np.nan)
    _nan_K = np.full((M, T), np.nan)

    if not np.all(np.isfinite(x0)):
        return _nan_P, _nan_A, _nan_S, _nan_K, t_long, "failed", float(t_long[0])

    prev_site_idx = compute_prev_site_idx(
        np.asarray(site_prot_idx, dtype=np.int32), N
    )

    args = (
        jnp.asarray(theta_opt, dtype=jnp.float64),
        jnp.asarray(Cg, dtype=jnp.float64),
        jnp.asarray(Cl, dtype=jnp.float64),
        jnp.asarray(site_prot_idx, dtype=jnp.int32),
        jnp.asarray(K_site_kin, dtype=jnp.float64),
        jnp.asarray(R, dtype=jnp.float64),
        jnp.asarray(L_alpha, dtype=jnp.float64),
        jnp.asarray(kin_to_prot_idx, dtype=jnp.int32),
        jnp.asarray(receptor_mask_prot, dtype=jnp.float64),
        jnp.asarray(receptor_mask_kin, dtype=jnp.float64),
        jnp.asarray(prev_site_idx, dtype=jnp.int32),
    )

    rhs_fn = make_rhs(
        K, M, N, mechanism,
        k_act_fn=k_act_fn, s_prod_fn=s_prod_fn, rna_relax=rna_relax,
    )
    term = diffrax.ODETerm(rhs_fn)
    t_eval = jnp.asarray(t_long, dtype=jnp.float64)
    y0_jax = jnp.asarray(x0, dtype=jnp.float64)
    saveat = diffrax.SaveAt(ts=t_eval)
    stepsize_ctrl = make_stepsize_controller(rtol=rtol, atol=atol)
    solver = make_diffrax_solver("tsit5")

    # Build steady-state event if requested
    event = None
    if use_event:
        _ertol = event_rtol if event_rtol is not None else rtol
        _eatol = event_atol if event_atol is not None else atol
        cond_fn = diffrax.steady_state_event(rtol=_ertol, atol=_eatol)
        event = diffrax.Event(cond_fn=cond_fn)

    try:
        sol = diffrax.diffeqsolve(
            term,
            solver,
            t0=float(t_long[0]),
            t1=float(t_long[-1]),
            dt0=dt0,
            y0=y0_jax,
            args=args,
            saveat=saveat,
            stepsize_controller=stepsize_ctrl,
            max_steps=max_steps,
            event=event,
            throw=False,
            adjoint=diffrax.DirectAdjoint(),
        )
    except Exception as exc:
        logger.warning(f"[!] Steady-state solve raised exception: {exc}")
        return _nan_P, _nan_A, _nan_S, _nan_K, t_long, "failed", float(t_long[0])

    # Determine termination status
    result = sol.result
    if result == diffrax.RESULTS.successful:
        solve_status = "successful"
    elif result == diffrax.RESULTS.event_occurred:
        solve_status = "event_occurred"
    elif result == diffrax.RESULTS.max_steps_reached:
        solve_status = "max_steps_reached"
    else:
        solve_status = "failed"

    xs_all = np.asarray(sol.ys, dtype=np.float64)  # (T, state_dim)

    # After a steady-state event, Diffrax fills remaining save-points with inf.
    # Convert those to NaN so downstream code (which masks non-finite values) is
    # consistent and unambiguous about missing / not-yet-reached time-points.
    if solve_status == "event_occurred":
        xs_all = np.where(np.isinf(xs_all), np.nan, xs_all)

    # Find the last time-point that has at least one finite state value.
    row_finite = np.any(np.isfinite(xs_all), axis=1)  # (T,)
    if np.any(row_finite):
        last_valid_idx = int(np.max(np.where(row_finite)[0]))
        t_final_actual = float(t_long[last_valid_idx])
    else:
        last_valid_idx = -1
        t_final_actual = float(t_long[0])

    # Slice outputs at the save-time indices (all T, clipping to state dims)
    def _safe_slice(start, stop):
        out = xs_all[:, start:stop]  # (T, dim)
        return np.where(np.isfinite(out), out, np.nan)

    R_rna_out = _safe_slice(0, K)
    S_out = _safe_slice(K, 2 * K)
    A_out = _safe_slice(2 * K, 3 * K)
    Kdyn_out = _safe_slice(3 * K, 3 * K + M)
    P_out = _safe_slice(3 * K + M, 3 * K + M + N)

    # Clip bounded states (mirrors simulation.py)
    np.clip(S_out, 0.0, 1.0, out=S_out)
    np.clip(Kdyn_out, 0.0, 1.0, out=Kdyn_out)
    np.clip(A_out, 0.0, 5.0, out=A_out)
    np.clip(P_out, 0.0, None, out=P_out)

    # Trim to last valid index if event fired early
    if last_valid_idx >= 0 and last_valid_idx < T - 1:
        t_trimmed = t_long[: last_valid_idx + 1]
        P_trim = P_out[: last_valid_idx + 1, :].T  # (N, T_trim)
        A_trim = A_out[: last_valid_idx + 1, :].T
        S_trim = S_out[: last_valid_idx + 1, :].T
        K_trim = Kdyn_out[: last_valid_idx + 1, :].T
    else:
        t_trimmed = t_long
        P_trim = P_out.T  # (N, T)
        A_trim = A_out.T
        S_trim = S_out.T
        K_trim = Kdyn_out.T

    return P_trim, A_trim, S_trim, K_trim, t_trimmed, solve_status, t_final_actual


def run_steadystate_analysis(
    outdir: str,
    problem,
    theta_opt: np.ndarray,
    sites: list,
    proteins: list,
    kinases: list,
    t_end: float = 2000.0,
    early_end: float = 100.0,
    n_early: int = 100,
    n_late: int = 80,
    late_grid: str = "geomspace",
    rtol: float = 1e-6,
    atol: float = 1e-8,
    dt0: float | None = 0.1,
    max_steps: int = 131072,
    top_n: int = 10,
    skip_plots_on_nonfinite: bool = True,
    strict: bool = False,
    use_event: bool = True,
    event_rtol: float | None = None,
    event_atol: float | None = None,
    dims: ModelDims | None = None,
) -> None:
    """Simulate the network over a long time horizon (terminal-input relaxation).

    Uses the same fitted derived-rate closures (``k_act_fn``, ``s_prod_fn``)
    and RNA initial conditions (``R_data0``, ``rna_relax``) as the main
    optimisation/simulation path.  The closures extrapolate beyond the observed
    time window using their configured interpolation scheme.

    This is a *long-horizon relaxation* analysis, not a strict biological
    steady state, because derived inputs are not recomputed from a converged
    mRNA level.

    When ``use_event=True`` (default), a ``diffrax.steady_state_event`` terminates
    the solve early once the vector field norm falls below the convergence threshold
    ``norm(f) < atol_ev + rtol_ev * norm(y)``, saving compute time for quickly
    converging systems.  ``event_rtol``/``event_atol`` default to ``rtol``/``atol``
    when not set.  The fallback (long-horizon grid + ``t_end``) still acts as the
    maximum integration limit.

    Args:
        outdir:                  Root output directory; results go to
                                 ``{outdir}/steadystate/``.
        problem (NetworkProblem): Fitted :class:`NetworkProblem` instance.
        theta_opt:               Optimised parameter vector.
        sites:                   Phosphosite labels (length N).
        proteins:                Protein labels (length K).
        kinases:                 Kinase labels (length M).
        t_end:                   Final simulation time in minutes.
        early_end:               End of the fine early time segment.
        n_early:                 Points in the early segment ``[0, early_end]``.
        n_late:                  Points in the late segment ``(early_end, t_end]``.
        late_grid:               ``"geomspace"`` (default) or ``"linear"``.
        rtol:                    ODE solver relative tolerance.
        atol:                    ODE solver absolute tolerance.
        dt0:                     Initial ODE step size (``None`` = auto).
        max_steps:               Maximum ODE solver steps.
        top_n:                   Number of top-dynamic-range trajectories to plot.
        skip_plots_on_nonfinite: Skip seaborn/matplotlib plots when all-NaN.
        strict:                  Raise RuntimeError on all-NaN output.
        use_event:               Use ``diffrax.steady_state_event`` for early
                                 termination when the ODE converges (default True).
        event_rtol:              Relative tolerance for the steady-state event
                                 (defaults to ``rtol`` when ``None``).
        event_atol:              Absolute tolerance for the steady-state event
                                 (defaults to ``atol`` when ``None``).

    Output files (written to ``{outdir}/steadystate/``)
    ----------------------------------------------------
    Backward-compatible:
      steadystate_sites.tsv      -- relative phosphosite signal (not occupancy)
      steadystate_proteins.tsv   -- protein abundance state
      steadystate_S.tsv          -- protein signalling/activation fraction
      steadystate_Kdyn.tsv       -- kinase activity fraction

    New:
      steadystate_diagnostics.tsv  -- per-state finite/NaN/Inf counts and ranges
      steadystate_convergence.tsv  -- per-state absolute and relative last-step deltas
      steadystate_metadata.json    -- horizon, tolerances, event info, solve status
    """
    logger.info("[*] Running Steady-state analysis")
    ss_dir = os.path.join(outdir, "steadystate")
    os.makedirs(ss_dir, exist_ok=True)

    # 1. Build long time grid
    t_long = build_long_horizon_time_grid(
        t_end=t_end,
        early_end=early_end,
        n_early=n_early,
        n_late=n_late,
        late_grid=late_grid,
    )
    logger.info("   -> Time grid:")
    logger.info("      Points = %d", len(t_long))
    logger.info("      Start = %.1f min", t_long[0])
    logger.info("      End = %.1f min", t_long[-1])
    logger.info("      Early end = %s", early_end)
    logger.info("      Late grid = %r", late_grid)

    # 2. Build initial conditions
    if dims is None:
        dims = ModelDims.set_dims(len(proteins), len(kinases), len(sites))
    K = dims.K
    A_scaled = problem.A_scaled
    prot_idx_for_A = problem.prot_idx_for_A
    if A_scaled.size > 0:
        A0_initial = build_full_A0(K, 1, A_scaled[:, 0:1], prot_idx_for_A)
    else:
        A0_initial = np.zeros((K, 1), dtype=float)

    # p initial condition comes from problem.P_data[:, 0], clipped only to [0, +inf).
    # Do NOT clip to [0, 1]: p is a non-negative relative phosphosite signal.

    # Derived-rate closures and RNA settings from the fitted problem
    k_act_fn = getattr(problem, "k_act_fn", None)
    s_prod_fn = getattr(problem, "s_prod_fn", None)
    R_data0 = getattr(problem, "R_data0", None)
    rna_relax = getattr(problem, "rna_relax", 0.1)

    logger.info("   -> Derived/input rate functions:")
    logger.info(
        "      k_act_fn = %s",
        "set" if k_act_fn is not None else "None (constant 1.0)",
    )
    logger.info(
        "      s_prod_fn = %s",
        "set" if s_prod_fn is not None else "None (constant 0.1)",
    )
    logger.info(
        "      R_data0 = %s",
        "set" if R_data0 is not None else "None (default 1.0)",
    )
    logger.info("      rna_relax = %s", rna_relax)

    # 3. Run simulation using Diffrax-native steady-state event (or plain simulate)
    _solve_dt0 = dt0 if dt0 is not None else 0.01

    if use_event:
        logger.info("   -> Using diffrax.steady_state_event:")
        logger.info("      event_rtol = %.1e", event_rtol or rtol)
        logger.info("      event_atol = %.1e", event_atol or atol)
        try:
            P_ss, A_ss, S_ss, Kdyn_ss, t_out, solve_status, t_final_actual = (
                _run_steadystate_solve(
                    t_long,
                    problem.P_data,
                    A0_initial,
                    theta_opt,
                    problem.Cg,
                    problem.Cl,
                    problem.site_prot_idx,
                    problem.K_site_kin,
                    problem.R,
                    problem.L_alpha,
                    problem.kin_to_prot_idx,
                    problem.receptor_mask_prot,
                    problem.receptor_mask_kin,
                    problem.mechanism,
                    rtol=rtol,
                    atol=atol,
                    dt0=_solve_dt0,
                    max_steps=max_steps,
                    k_act_fn=k_act_fn,
                    s_prod_fn=s_prod_fn,
                    R_data0=R_data0,
                    rna_relax=rna_relax,
                    use_event=True,
                    event_rtol=event_rtol,
                    event_atol=event_atol,
                    dims=dims,
                )
            )
        except Exception as exc:
            logger.warning(f"[!] Long-horizon simulation failed: {exc}")
            _save_failure_diagnostics(ss_dir, str(exc))
            if strict:
                raise RuntimeError(
                    "Long-horizon relaxation simulation failed (strict=True)."
                ) from exc
            return
    else:
        # Legacy path: plain simulate() over full long-horizon grid
        try:
            result = simulate(
                t_long,
                problem.P_data,
                A0_initial,
                theta_opt,
                problem.Cg,
                problem.Cl,
                problem.site_prot_idx,
                problem.K_site_kin,
                problem.R,
                problem.L_alpha,
                problem.kin_to_prot_idx,
                problem.receptor_mask_prot,
                problem.receptor_mask_kin,
                problem.mechanism,
                full_output=True,
                rtol=rtol,
                atol=atol,
                max_steps=max_steps,
                dt0=_solve_dt0,
                k_act_fn=k_act_fn,
                s_prod_fn=s_prod_fn,
                R_data0=R_data0,
                rna_relax=rna_relax,
                dims=dims,
            )
        except RuntimeError as exc:
            logger.warning(f"[!] Long-horizon simulation failed with RuntimeError: {exc}")
            _save_failure_diagnostics(ss_dir, str(exc))
            if strict:
                raise RuntimeError(
                    "Long-horizon relaxation simulation failed (strict=True)."
                ) from exc
            return

        P_ss, A_ss, S_ss, Kdyn_ss = result
        t_out = t_long
        solve_status = "successful" if np.any(np.isfinite(P_ss)) else "failed"
        t_final_actual = float(t_long[-1])

    # Log termination status
    _status_msg = {
        "successful": "Solve reached t_end normally.",
        "event_occurred": (
            f"Steady-state event fired; solution converged at "
            f"t≈{t_final_actual:.1f} min (< t_end={t_end:.1f})."
        ),
        "max_steps_reached": "Max steps reached before t_end; output may be incomplete.",
        "failed": "Solve failed; check diagnostics.",
        "unknown": "Status unknown.",
    }.get(solve_status, f"Status: {solve_status}")

    logger.info("   -> Solve status:")
    logger.info("      Status key = %s", solve_status)
    logger.info("      Message = %s", _status_msg)

    # Compute final derivative norm and state norm for logging/metadata
    final_deriv_norm = None
    final_state_norm = None
    if np.any(np.isfinite(P_ss)):
        # Use last finite column as a proxy for the final state norm
        last_col = P_ss[:, -1]
        finite_last = last_col[np.isfinite(last_col)]
        if finite_last.size > 0:
            final_state_norm = float(np.sqrt(np.mean(finite_last**2)))
        # Use second-to-last finite column for derivative norm proxy
        if P_ss.shape[1] >= 2:
            prev_col = P_ss[:, -2]
            finite_prev = prev_col[np.isfinite(prev_col)]
            if finite_last.size > 0 and finite_prev.size > 0 and finite_last.shape == finite_prev.shape:
                deriv_proxy = np.abs(finite_last - finite_prev)
                final_deriv_norm = float(np.sqrt(np.mean(deriv_proxy**2)))

    logger.info("   -> Final solve diagnostics:")
    logger.info("      t_final = %.2f", t_final_actual)
    logger.info("      final_state_norm(P) = %s", final_state_norm)
    logger.info("      final_deriv_norm_proxy(P) = %s", final_deriv_norm)

    # 4. Non-finite diagnostics
    matrices = {
        "P (rel. phosphosite signal)": P_ss,
        "A (abundance)": A_ss,
        "S (activation)": S_ss,
        "Kdyn (kinase activity)": Kdyn_ss,
    }
    for name, arr in matrices.items():
        _log_matrix_diagnostics(name, arr)
    _save_diagnostics_tsv(ss_dir, matrices)

    p_all_nonfinite = not np.any(np.isfinite(P_ss))
    if p_all_nonfinite:
        logger.warning(
            "[!] P_ss is entirely non-finite after long-horizon simulation. "
            "Solver may have failed or diverged.  Check steadystate_diagnostics.tsv."
        )
        if strict:
            raise RuntimeError(
                "P_ss is all non-finite in long-horizon relaxation (strict=True)."
            )
        _save_output_tsvs(
            ss_dir, P_ss, A_ss, S_ss, Kdyn_ss, t_out, sites, proteins, kinases
        )
        _save_convergence_tsv(ss_dir, matrices)
        _save_metadata_json(
            ss_dir,
            t_end,
            early_end,
            n_early,
            n_late,
            late_grid,
            rtol,
            atol,
            dt0,
            max_steps,
            getattr(problem, "mechanism", "unknown"),
            k_act_fn is not None,
            s_prod_fn is not None,
            R_data0 is not None,
            use_event=use_event,
            event_rtol=event_rtol,
            event_atol=event_atol,
            solve_status=solve_status,
            t_final=t_final_actual,
            final_deriv_norm=final_deriv_norm,
            final_state_norm=final_state_norm,
        )
        if skip_plots_on_nonfinite:
            logger.warning("[!] Skipping plots due to all-NaN P_ss output.")
            return

    # 5. Convergence metrics
    _save_convergence_tsv(ss_dir, matrices)
    if P_ss.shape[1] >= 2:
        delta_abs_p = np.abs(P_ss[:, -1] - P_ss[:, -2])
        finite_delta = delta_abs_p[np.isfinite(delta_abs_p)]
        if finite_delta.size > 0:
            logger.info(
                f"   -> Convergence metric (mean |delta P| at last step): {finite_delta.mean():.6e}"  # noqa: E501
            )

    # 6. Save output TSVs
    _save_output_tsvs(
        ss_dir, P_ss, A_ss, S_ss, Kdyn_ss, t_out, sites, proteins, kinases
    )

    # 7. Plots
    _plot_convergence_heatmap(
        ss_dir, P_ss, t_out, "Phosphosites", skip_plots_on_nonfinite
    )
    _plot_convergence_heatmap(ss_dir, A_ss, t_out, "Proteins", skip_plots_on_nonfinite)
    _plot_convergence_heatmap(
        ss_dir, S_ss, t_out, "Protein_Activity_S", skip_plots_on_nonfinite
    )
    _plot_convergence_heatmap(
        ss_dir, Kdyn_ss, t_out, "Kinase_Activity_Kdyn", skip_plots_on_nonfinite
    )

    _plot_trajectories(
        ss_dir,
        P_ss,
        t_out,
        sites,
        "Top_Changing_Sites",
        ylabel="Relative phosphosite signal",
        top_n=top_n,
        skip_on_nonfinite=skip_plots_on_nonfinite,
    )
    _plot_trajectories(
        ss_dir,
        A_ss,
        t_out,
        proteins,
        "Protein_Abundance_A",
        ylabel="Protein abundance state",
        top_n=top_n,
        skip_on_nonfinite=skip_plots_on_nonfinite,
    )
    _plot_trajectories(
        ss_dir,
        S_ss,
        t_out,
        proteins,
        "S_Dynamics",
        ylabel="Fraction active",
        top_n=top_n,
        skip_on_nonfinite=skip_plots_on_nonfinite,
    )
    _plot_trajectories(
        ss_dir,
        Kdyn_ss,
        t_out,
        kinases,
        "Kdyn_Dynamics",
        ylabel="Fraction active",
        top_n=top_n,
        skip_on_nonfinite=skip_plots_on_nonfinite,
    )

    # 8. Metadata JSON
    _save_metadata_json(
        ss_dir,
        t_end,
        early_end,
        n_early,
        n_late,
        late_grid,
        rtol,
        atol,
        dt0,
        max_steps,
        getattr(problem, "mechanism", "unknown"),
        k_act_fn is not None,
        s_prod_fn is not None,
        R_data0 is not None,
        use_event=use_event,
        event_rtol=event_rtol,
        event_atol=event_atol,
        solve_status=solve_status,
        t_final=t_final_actual,
        final_deriv_norm=final_deriv_norm,
        final_state_norm=final_state_norm,
    )

    logger.success("[*] Steady state analysis complete.")


def _plot_convergence_heatmap(
    outdir: str,
    data: np.ndarray,
    t: np.ndarray,
    label: str,
    skip_on_nonfinite: bool = True,
) -> None:
    """Heatmap of trajectories sorted by their final long-horizon value.

    Args:
        outdir:            Output directory.
        data:              Matrix (rows=entities, cols=time).
        t:                 Time vector.
        label:             Entity label for title and filename.
        skip_on_nonfinite: Skip when all values are non-finite.
    """
    if data.size == 0:
        logger.warning(f"   [plot] Skipping heatmap for {label}: empty data.")
        return

    if not np.any(np.isfinite(data)):
        logger.warning(
            f"   [plot] Skipping heatmap for {label}: all values are non-finite."
        )
        if skip_on_nonfinite:
            return

    sort_key = np.nan_to_num(data[:, -1], nan=-np.inf)
    idx = np.argsort(sort_key)[::-1]
    sorted_data = data[idx, :]
    plot_data = np.where(np.isfinite(sorted_data), sorted_data, np.nan)

    finite_vals = plot_data[np.isfinite(plot_data)]
    if finite_vals.size > 0:
        vmin, vmax = float(finite_vals.min()), float(finite_vals.max())
        if np.isclose(vmin, vmax):
            vmin, vmax = vmin - 1e-6, vmax + 1e-6
    else:
        vmin, vmax = 0.0, 1.0

    fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)
    sns.heatmap(
        plot_data,
        cmap="viridis",
        xticklabels=False,
        yticklabels=False,
        vmin=vmin,
        vmax=vmax,
        ax=ax,
    )
    ax.set_xlabel(f"Time (0 to {t[-1]:.0f} min)")
    ax.set_ylabel(f"{label} (sorted by final value)")
    ax.set_title(f"{label} – long-horizon relaxation")
    fig.savefig(os.path.join(outdir, f"heatmap_convergence_{label}.png"), dpi=300)
    plt.close(fig)


def _plot_trajectories(
    outdir: str,
    data: np.ndarray,
    t: np.ndarray,
    names: list,
    filename_suffix: str,
    ylabel: str = "State value",
    top_n: int = 10,
    skip_on_nonfinite: bool = True,
) -> None:
    """Line plots for the top-N entities with the largest dynamic range.

    Args:
        outdir:            Output directory.
        data:              Matrix (rows=entities, cols=time).
        t:                 Time vector.
        names:             Entity names for rows.
        filename_suffix:   Output filename suffix.
        ylabel:            Y-axis label.
        top_n:             Number of top-dynamic-range entities to plot.
        skip_on_nonfinite: Skip when no finite dynamic range exists.
    """
    if data.size == 0:
        logger.warning(
            f"   [plot] Skipping trajectories for {filename_suffix}: empty data."
        )
        return

    # Check for any finite values before calling nanmax/nanmin to avoid
    # "All-NaN slice encountered" RuntimeWarning from NumPy.
    if not np.any(np.isfinite(data)):
        logger.warning(
            f"   [plot] Skipping trajectories for {filename_suffix}: no finite values."
        )
        if skip_on_nonfinite:
            return
        # Nothing to plot; return even if skip_on_nonfinite is False
        return

    finite_data = np.where(np.isfinite(data), data, np.nan)
    # Compute per-row range; rows that are entirely NaN produce NaN (not a warning
    # because we already confirmed at least one finite value exists globally).
    with np.errstate(all="ignore"):
        dynamic_range = np.nanmax(finite_data, axis=1) - np.nanmin(finite_data, axis=1)

    if not np.any(np.isfinite(dynamic_range)):
        logger.warning(
            f"   [plot] Skipping trajectories for {filename_suffix}: no finite dynamic range."  # noqa: E501
        )
        if skip_on_nonfinite:
            return

    valid = np.where(np.isfinite(dynamic_range))[0]
    if valid.size == 0:
        return
    top_n_actual = min(top_n, len(valid))
    top_indices = valid[np.argsort(dynamic_range[valid])[-top_n_actual:]]

    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
    for idx in top_indices:
        ax.plot(t, finite_data[idx], label=str(names[idx]), linewidth=2, alpha=0.8)

    ax.set_xscale("symlog", linthresh=10)
    ax.set_xlabel("Time (SymLog scale, min)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"Long-horizon relaxation – top {top_n_actual} by dynamic range")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize="small")
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(outdir, f"trajectories_{filename_suffix}.png"), dpi=300)
    plt.close(fig)
