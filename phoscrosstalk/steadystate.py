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

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from phoscrosstalk.config import ModelDims
from phoscrosstalk.logger import get_logger
from phoscrosstalk.simulation import build_full_A0, simulate_ode

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
) -> None:
    """Simulate the network over a long time horizon (terminal-input relaxation).

    Uses the same fitted derived-rate closures (``k_act_fn``, ``s_prod_fn``)
    and RNA initial conditions (``R_data0``, ``rna_relax``) as the main
    optimisation/simulation path.  The closures extrapolate beyond the observed
    time window using their configured interpolation scheme.

    This is a *long-horizon relaxation* analysis, not a strict biological
    steady state, because derived inputs are not recomputed from a converged
    mRNA level.

    Args:
        outdir:                  Root output directory; results go to
                                 ``{outdir}/steadystate/``.
        problem:                 Fitted :class:`NetworkProblem` instance.
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
      steadystate_metadata.json    -- horizon, tolerances, mechanism, closure info
    """
    logger.info("\n[*] Running long-horizon relaxation analysis...")
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
    logger.info(
        f"   -> Time grid: {len(t_long)} points  "
        f"[{t_long[0]:.1f}, {t_long[-1]:.1f}] min  "
        f"(early_end={early_end}, late_grid={late_grid!r})"
    )

    # 2. Build initial conditions
    K = ModelDims.K
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

    logger.info(
        f"   -> k_act_fn={'set' if k_act_fn is not None else 'None (constant 1.0)'}  "
        f"s_prod_fn={'set' if s_prod_fn is not None else 'None (constant 0.1)'}  "
        f"R_data0={'set' if R_data0 is not None else 'None (default 1.0)'}  "
        f"rna_relax={rna_relax}"
    )

    # 3. JAX/Diffrax simulation over the long horizon
    try:
        result = simulate_ode(
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
            dt0=dt0 if dt0 is not None else 0.01,
            k_act_fn=k_act_fn,
            s_prod_fn=s_prod_fn,
            R_data0=R_data0,
            rna_relax=rna_relax,
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
            ss_dir, P_ss, A_ss, S_ss, Kdyn_ss, t_long, sites, proteins, kinases
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
        )
        if skip_plots_on_nonfinite:
            logger.warning("[!] Skipping plots due to all-NaN P_ss output.")
            return

    # 5. Convergence metrics
    _save_convergence_tsv(ss_dir, matrices)
    delta_abs_p = np.abs(P_ss[:, -1] - P_ss[:, -2])
    finite_delta = delta_abs_p[np.isfinite(delta_abs_p)]
    if finite_delta.size > 0:
        logger.info(
            f"   -> Convergence metric (mean |delta P| at last step): {finite_delta.mean():.6e}"  # noqa: E501
        )

    # 6. Save output TSVs
    _save_output_tsvs(
        ss_dir, P_ss, A_ss, S_ss, Kdyn_ss, t_long, sites, proteins, kinases
    )

    # 7. Plots
    _plot_convergence_heatmap(
        ss_dir, P_ss, t_long, "Phosphosites", skip_plots_on_nonfinite
    )
    _plot_convergence_heatmap(ss_dir, A_ss, t_long, "Proteins", skip_plots_on_nonfinite)
    _plot_convergence_heatmap(
        ss_dir, S_ss, t_long, "Protein_Activity_S", skip_plots_on_nonfinite
    )
    _plot_convergence_heatmap(
        ss_dir, Kdyn_ss, t_long, "Kinase_Activity_Kdyn", skip_plots_on_nonfinite
    )

    _plot_trajectories(
        ss_dir,
        P_ss,
        t_long,
        sites,
        "Top_Changing_Sites",
        ylabel="Relative phosphosite signal",
        top_n=top_n,
        skip_on_nonfinite=skip_plots_on_nonfinite,
    )
    _plot_trajectories(
        ss_dir,
        A_ss,
        t_long,
        proteins,
        "Protein_Abundance_A",
        ylabel="Protein abundance state",
        top_n=top_n,
        skip_on_nonfinite=skip_plots_on_nonfinite,
    )
    _plot_trajectories(
        ss_dir,
        S_ss,
        t_long,
        proteins,
        "S_Dynamics",
        ylabel="Fraction active",
        top_n=top_n,
        skip_on_nonfinite=skip_plots_on_nonfinite,
    )
    _plot_trajectories(
        ss_dir,
        Kdyn_ss,
        t_long,
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
    )

    logger.info("[*] Long-horizon relaxation analysis complete.")


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
