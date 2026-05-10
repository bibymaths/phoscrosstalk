# SPDX-License-Identifier: MIT
"""
Plotting and inspection utilities for saved PINN and neuralODE model bundles.

Both model families produce self-contained bundle directories on disk.  This
module lets you inspect and compare them **without** re-running the full
pipeline.

Public API
----------
``plot_pinn_bundle_analysis(model_dir, output_dir=None)``
    Comprehensive plots for a PINN bundle.

``plot_neuralode_bundle_analysis(model_dir, output_dir=None)``
    Comprehensive plots for a neuralODE bundle.

``plot_model_comparison(pinn_dir, neuralode_dir, output_dir)``
    Side-by-side comparison of the two model families.

Bundle directory layouts
------------------------
PINN bundle (written by ``save_pinn_model_bundle`` / ``save_pinn_outputs``)::

    {outdir}/pinn_bundle/
        pinn_model.eqx
        pinn_bundle_meta.json
        theta_opt.npy
    {outdir}/
        pinn_metadata.json          (optional)
        pinn_loss_components.tsv    (optional)
        pinn_fit_timeseries.tsv     (optional)
        pinn_residuals.tsv          (optional)

neuralODE bundle (written by ``save_neural_ode_bundle`` /
``run_neural_latent_rate_refinement``)::

    {neural_outdir}/neural_ode_bundle/
        neural_ode_model.eqx
        neural_ode_bundle_meta.json
        theta_refined.npy
    {neural_outdir}/
        neural_metadata.json        (optional)
        neural_training_losses.tsv  (optional)
        neural_fit_timeseries.tsv   (optional)
        neural_latent_rates.tsv     (optional)
        neural_latent_rates.npz     (optional)

Notes
-----
* All output plots are saved as both ``.png`` and ``.pdf``.
* If a required input file is missing, the corresponding plot is skipped with
  a warning; the function never raises on missing optional files.
* Only matplotlib is used — seaborn is never imported.
* All paths are handled as :class:`pathlib.Path` objects internally.
"""

from __future__ import annotations

import csv
import json
import pathlib
from typing import Any, Sequence

import matplotlib

from phoscrosstalk.logger import get_logger

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_logger = get_logger()


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _read_json(path: pathlib.Path) -> dict | None:
    """Return parsed JSON or *None* if the file is missing/invalid."""
    try:
        with open(path) as fh:
            return json.load(fh)
    except FileNotFoundError:
        return None
    except Exception as exc:
        _logger.warning("Could not read %s: %s", path, exc)
        return None


def _read_csv_if_exists(path: pathlib.Path) -> list[dict] | None:
    """Return a list-of-dicts from a CSV/TSV or *None* if absent."""
    if not path.is_file():
        return None
    try:
        sep = "\t" if path.suffix in (".tsv", ".tab") else ","
        rows: list[dict] = []
        with open(path, newline="") as fh:
            reader = csv.DictReader(fh, delimiter=sep)
            for row in reader:
                rows.append({k: _coerce_numeric(v) for k, v in row.items()})
        return rows or None
    except Exception as exc:
        _logger.warning("Could not read %s: %s", path, exc)
        return None


def _coerce_numeric(value: str) -> Any:
    """Try to convert a string to int → float → str."""
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _load_npz_if_exists(path: pathlib.Path) -> dict | None:
    """Return dict of arrays from an .npz file or *None* if absent."""
    if not path.is_file():
        return None
    try:
        data = np.load(path, allow_pickle=False)
        return dict(data)
    except Exception as exc:
        _logger.warning("Could not load %s: %s", path, exc)
        return None


def _ensure_dir(path: pathlib.Path) -> pathlib.Path:
    """Create *path* and all parents; return *path*."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def _savefig(fig: plt.Figure, output_dir: pathlib.Path, stem: str) -> None:
    """Save *fig* as ``{output_dir}/{stem}.png`` and ``.pdf``."""
    for ext in ("png", "pdf"):
        dest = output_dir / f"{stem}.{ext}"
        try:
            fig.savefig(dest, bbox_inches="tight", dpi=150)
            _logger.info("Saved %s", dest)
        except Exception as exc:
            _logger.warning("Could not save %s: %s", dest, exc)
    plt.close(fig)


def _get_labels(
        labels_json: dict | None,
        key: str,
        fallback_count: int,
) -> list[str]:
    """Return label list from *labels_json[key]* or generic fallbacks."""
    if labels_json is not None and key in labels_json:
        vals = labels_json[key]
        if isinstance(vals, list) and all(isinstance(v, str) for v in vals):
            return vals
    return [f"{key}_{i}" for i in range(fallback_count)]


def _safe_array(x: Any) -> np.ndarray | None:
    """Convert *x* to a NumPy array; return *None* on failure."""
    if x is None:
        return None
    try:
        arr = np.asarray(x, dtype=float)
        if arr.size == 0:
            return None
        return arr
    except Exception:
        return None


def _plot_matrix_heatmap(
        ax: plt.Axes,
        matrix: np.ndarray,
        row_labels: Sequence[str],
        col_labels: Sequence[str],
        title: str = "",
        cmap: str = "RdBu_r",
        fmt: str = ".2f",
) -> None:
    """Draw a labelled heatmap on *ax*.

    Args:
        ax:         Target Axes.
        matrix:     2-D array of shape ``(n_rows, n_cols)``.
        row_labels: Row tick labels.
        col_labels: Column tick labels.
        title:      Axes title.
        cmap:       Matplotlib colormap name.
        fmt:        Python format spec used to annotate each cell (e.g.
                    ``".2f"``).  Annotation is only applied for small matrices
                    (≤ 15 rows, ≤ 20 columns).
    """
    im = ax.imshow(matrix, cmap=cmap, aspect="auto")
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=7)
    ax.figure.colorbar(im, ax=ax, shrink=0.8)
    if title:
        ax.set_title(title, fontsize=9)
    # Annotate cells for small matrices
    if matrix.shape[0] <= 15 and matrix.shape[1] <= 20:
        for ri in range(matrix.shape[0]):
            for ci in range(matrix.shape[1]):
                val = matrix[ri, ci]
                ax.text(
                    ci,
                    ri,
                    format(val, fmt),
                    ha="center",
                    va="center",
                    fontsize=6,
                    color="white" if abs(val) > 0.5 * np.nanmax(np.abs(matrix)) else "black",
                )


def _plot_timeseries_grid(
        axes: Sequence[plt.Axes],
        t: np.ndarray,
        trajectories: Sequence[tuple[str, np.ndarray, str]],
        scatter_data: Sequence[tuple[str, np.ndarray, np.ndarray]] | None = None,
        title: str = "",
) -> None:
    """
    Plot multiple time series into *axes* (one entity per axis).

    Args:
        axes: Pre-created Axes sequence (len ≥ len(trajectories)).
        t: Common time vector.
        trajectories: Sequence of ``(label, values_1d, color)`` tuples.
        scatter_data: Optional sequence of ``(label, t_obs, y_obs)`` tuples.
        title: Suptitle for the whole grid (not applied here; caller sets it).
    """
    for ax, (label, values, color) in zip(axes, trajectories):
        ax.plot(t, values, color=color, lw=1.5, label="fit")
        if scatter_data is not None:
            for sc_label, t_obs, y_obs in scatter_data:
                mask = np.isfinite(y_obs)
                ax.scatter(
                    t_obs[mask],
                    y_obs[mask],
                    s=18,
                    marker="o",
                    label="obs",
                    zorder=5,
                )
        ax.set_title(label, fontsize=7)
        ax.tick_params(labelsize=6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    for ax in axes[len(trajectories):]:
        ax.set_visible(False)


def _infer_state_slices(metadata: dict) -> dict[str, slice]:
    """
    Infer state-vector slices for each biological compartment.

    The ODE state order is ``[R(K), S(M), A(K), Kdyn(K), P(N)]``
    (total length = ``3K + M + N``).

    Args:
        metadata: Parsed JSON metadata dict with ``K``, ``M``, ``N`` keys.

    Returns:
        dict mapping compartment name to a ``slice`` into the state vector.
        Returns an empty dict if dimensions cannot be determined.
    """
    try:
        K = int(metadata.get("K", 0))
        M = int(metadata.get("M", 0))
        N = int(metadata.get("N", 0))
        if K <= 0:
            return {}
        slices: dict[str, slice] = {
            "mRNA (R)": slice(0, K),
            "Kinase (S)": slice(K, K + M),
            "Abundance (A)": slice(K + M, 2 * K + M),
            "Kinase dyn (Kd)": slice(2 * K + M, 3 * K + M),
            "Phosphosite (P)": slice(3 * K + M, 3 * K + M + N),
        }
        return slices
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# PINN bundle analysis
# ---------------------------------------------------------------------------

def plot_pinn_bundle_analysis(
        model_dir: str | pathlib.Path,
        output_dir: str | pathlib.Path | None = None,
) -> None:
    """
    Load a PINN model bundle and produce a comprehensive set of diagnostic plots.

    The function looks for the following files relative to *model_dir* and its
    **parent** directory (the canonical PINN run output directory):

    In ``model_dir`` (the bundle subdirectory):
      * ``pinn_bundle_meta.json``   — structural metadata
      * ``theta_opt.npy``           — mechanistic rate vector
      * ``pinn_model.eqx``          — Equinox model weights (presence checked only)

    In ``model_dir/..`` (the PINN run output directory):
      * ``pinn_metadata.json``      — run-level metadata
      * ``pinn_loss_components.tsv``— final loss values per term
      * ``pinn_fit_timeseries.tsv`` — fitted vs observed timeseries (long format)
      * ``pinn_residuals.tsv``      — residual data (optional)
      * ``bounds.npz``              — parameter bounds (optional)
      * ``labels.json``             — entity labels (optional)

    Missing files are skipped with a ``logging.warning``; the function never
    raises because of a missing optional file.

    Args:
        model_dir:  Path to the bundle directory (e.g. ``…/pinn_bundle/``).
        output_dir: Directory where plots are written.  Defaults to
                    ``model_dir/plots/``.
    """
    model_dir = pathlib.Path(model_dir)
    run_dir = model_dir.parent  # canonical PINN run output directory
    if output_dir is None:
        output_dir = model_dir / "plots"
    output_dir = pathlib.Path(output_dir)
    _ensure_dir(output_dir)

    _logger.info("[bundle_analysis] PINN analysis: model_dir=%s", model_dir)

    # ------------------------------------------------------------------ #
    # Load data                                                           #
    # ------------------------------------------------------------------ #
    bundle_meta = _read_json(model_dir / "pinn_bundle_meta.json")
    run_meta = _read_json(run_dir / "pinn_metadata.json")
    labels_json = _read_json(run_dir / "labels.json")
    meta = bundle_meta or run_meta or {}

    theta = _safe_array(
        np.load(model_dir / "theta_opt.npy")
        if (model_dir / "theta_opt.npy").is_file()
        else (np.load(run_dir / "theta_opt.npy") if (run_dir / "theta_opt.npy").is_file() else None)
    )
    loss_rows = _read_csv_if_exists(run_dir / "pinn_loss_components.tsv")
    ts_rows = _read_csv_if_exists(run_dir / "pinn_fit_timeseries.tsv")
    resid_rows = _read_csv_if_exists(run_dir / "pinn_residuals.tsv")
    bounds = _load_npz_if_exists(run_dir / "bounds.npz")

    # ------------------------------------------------------------------ #
    # Plot 1: Bundle metadata summary (text table)                       #
    # ------------------------------------------------------------------ #
    _pinn_plot_metadata(meta, output_dir, labels_json)

    # ------------------------------------------------------------------ #
    # Plot 2: Loss components bar chart                                   #
    # ------------------------------------------------------------------ #
    _pinn_plot_loss_components(loss_rows, run_meta, output_dir)

    # ------------------------------------------------------------------ #
    # Plot 3: Mechanistic parameter vector                                #
    # ------------------------------------------------------------------ #
    _pinn_plot_theta(theta, bounds, labels_json, output_dir)

    # ------------------------------------------------------------------ #
    # Plot 4: Fitted vs observed timeseries                               #
    # ------------------------------------------------------------------ #
    _pinn_plot_timeseries(ts_rows, labels_json, output_dir)

    # ------------------------------------------------------------------ #
    # Plot 5: Residuals (optional)                                        #
    # ------------------------------------------------------------------ #
    _pinn_plot_residuals(resid_rows, labels_json, output_dir)

    _logger.info("[bundle_analysis] PINN plots saved to %s", output_dir)


def _pinn_plot_metadata(
        meta: dict,
        output_dir: pathlib.Path,
        labels_json: dict | None,
) -> None:
    """Text-table summary of PINN bundle metadata.

    Also annotates which slice of the state vector corresponds to each
    biological compartment (derived from ``K``, ``M``, ``N`` in *meta*).
    """
    if not meta:
        _logger.warning("[bundle_analysis] PINN: no metadata found; skipping metadata plot.")
        return
    try:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.axis("off")
        interesting = {
            k: v for k, v in meta.items()
            if isinstance(v, (bool, int, float, str)) and not k.startswith("_")
        }
        # Augment with state-slice annotations derived from K/M/N
        state_slices = _infer_state_slices(meta)
        for compartment, sl in state_slices.items():
            interesting[f"slice: {compartment}"] = f"[{sl.start}:{sl.stop}]"

        rows_table = [(k, str(v)) for k, v in sorted(interesting.items())]
        table = ax.table(
            cellText=rows_table,
            colLabels=["Parameter", "Value"],
            cellLoc="left",
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1.0, 1.5)
        ax.set_title("PINN Bundle Metadata", fontsize=11, pad=12)
        _savefig(fig, output_dir, "pinn_bundle_metadata")
    except Exception as exc:
        _logger.warning("[bundle_analysis] PINN metadata plot failed: %s", exc)


def _pinn_plot_loss_components(
        loss_rows: list[dict] | None,
        run_meta: dict | None,
        output_dir: pathlib.Path,
) -> None:
    """Bar chart of PINN loss components."""
    # Prefer explicit loss_components TSV; fall back to meta scalars.
    loss: dict[str, float] = {}
    if loss_rows:
        loss = {k: float(v) for k, v in loss_rows[0].items() if isinstance(v, (int, float))}
    elif run_meta:
        for key in ("f1", "f2", "f3", "f4", "f_pinn_reg"):
            if key in run_meta:
                loss[key] = float(run_meta[key])
    if not loss:
        _logger.warning("[bundle_analysis] PINN: no loss data found; skipping loss plot.")
        return
    try:
        _PRETTY = {
            "f1": "Phosphosite (f₁)",
            "f2": "Abundance (f₂)",
            "f3": "Regularisation (f₃)",
            "f4": "mRNA (f₄)",
            "f_pinn_reg": "PINN reg",
        }
        keys = list(loss.keys())
        vals = [loss[k] for k in keys]
        xlabs = [_PRETTY.get(k, k) for k in keys]
        colors = plt.cm.tab10(np.linspace(0, 0.9, len(keys)))

        fig, ax = plt.subplots(figsize=(7, 3.5))
        bars = ax.bar(range(len(keys)), vals, color=colors, edgecolor="k", linewidth=0.5)
        ax.bar_label(bars, fmt="%.3g", fontsize=7, padding=2)
        ax.set_xticks(range(len(keys)))
        ax.set_xticklabels(xlabs, rotation=20, ha="right", fontsize=8)
        ax.set_ylabel("Loss value")
        ax.set_title("PINN: Final Loss Components", fontsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        _savefig(fig, output_dir, "pinn_loss_components")
    except Exception as exc:
        _logger.warning("[bundle_analysis] PINN loss plot failed: %s", exc)


def _pinn_plot_theta(
        theta: np.ndarray | None,
        bounds: dict | None,
        labels_json: dict | None,
        output_dir: pathlib.Path,
) -> None:
    """Bar chart of mechanistic parameter vector with optional bounds."""
    if theta is None or theta.ndim != 1:
        _logger.warning("[bundle_analysis] PINN: theta_opt not available; skipping theta plot.")
        return
    try:
        n = len(theta)
        xlabs = _get_labels(labels_json, "theta", n)

        xl = _safe_array(bounds.get("xl") if bounds else None)
        xu = _safe_array(bounds.get("xu") if bounds else None)

        fig, ax = plt.subplots(figsize=(max(6, n * 0.35), 3.5))
        ax.bar(range(n), theta, color="#4C72B0", edgecolor="k", linewidth=0.4, alpha=0.85, label="θ_opt")
        if xl is not None and xu is not None and xl.shape == theta.shape:
            ax.vlines(range(n), xl, xu, color="red", linewidth=1.5, alpha=0.6, label="bounds")
        ax.set_xticks(range(n))
        ax.set_xticklabels(xlabs, rotation=70, ha="right", fontsize=7)
        ax.set_ylabel("Parameter value")
        ax.set_title("PINN: Mechanistic Parameter Vector (θ_opt)", fontsize=10)
        ax.legend(fontsize=7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        _savefig(fig, output_dir, "pinn_theta_vector")
    except Exception as exc:
        _logger.warning("[bundle_analysis] PINN theta plot failed: %s", exc)


def _pinn_plot_timeseries(
        ts_rows: list[dict] | None,
        labels_json: dict | None,
        output_dir: pathlib.Path,
) -> None:
    """Per-entity fitted vs observed timeseries plots."""
    if not ts_rows:
        _logger.warning("[bundle_analysis] PINN: no timeseries data; skipping timeseries plot.")
        return
    try:
        # Group rows by entity_type
        by_type: dict[str, dict[str, list[dict]]] = {}
        for row in ts_rows:
            etype = str(row.get("entity_type", "unknown"))
            entity = str(row.get("entity", "unknown"))
            by_type.setdefault(etype, {}).setdefault(entity, []).append(row)

        for etype, entities in by_type.items():
            enames = sorted(entities.keys())
            ncols = min(4, len(enames))
            nrows = (len(enames) + ncols - 1) // ncols
            fig, axes = plt.subplots(
                nrows, ncols,
                figsize=(3.5 * ncols, 2.5 * nrows),
                squeeze=False,
            )
            axes_flat = axes.flatten()
            for ax_idx, ename in enumerate(enames):
                rows_e = sorted(entities[ename], key=lambda r: r.get("time", 0))
                t_vals = np.array([r.get("time", 0) for r in rows_e], dtype=float)
                sim = np.array([r.get("value_sim", np.nan) for r in rows_e], dtype=float)
                obs = np.array([r.get("value_obs", np.nan) for r in rows_e], dtype=float)
                ax = axes_flat[ax_idx]
                ax.plot(t_vals, sim, color="#2196F3", lw=1.5, label="PINN fit")
                mask = np.isfinite(obs)
                if mask.any():
                    ax.scatter(t_vals[mask], obs[mask], s=18, color="#E91E63",
                               zorder=5, label="observed")
                ax.set_title(ename, fontsize=7)
                ax.tick_params(labelsize=6)
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                if ax_idx == 0:
                    ax.legend(fontsize=6)
            for ax in axes_flat[len(enames):]:
                ax.set_visible(False)
            fig.suptitle(f"PINN: Fitted vs Observed — {etype}", fontsize=10, y=1.01)
            fig.tight_layout()
            safe_etype = etype.replace(" ", "_").replace("/", "_")
            _savefig(fig, output_dir, f"pinn_timeseries_{safe_etype}")
    except Exception as exc:
        _logger.warning("[bundle_analysis] PINN timeseries plot failed: %s", exc)


def _pinn_plot_residuals(
        resid_rows: list[dict] | None,
        labels_json: dict | None,
        output_dir: pathlib.Path,
) -> None:
    """Residual scatter plot (sim - obs) per entity type."""
    if not resid_rows:
        return  # silently skip — residuals are truly optional
    try:
        by_type: dict[str, list[float]] = {}
        for row in resid_rows:
            etype = str(row.get("entity_type", "unknown"))
            resid = row.get("residual", row.get("residual_neural", np.nan))
            try:
                by_type.setdefault(etype, []).append(float(resid))
            except (TypeError, ValueError):
                pass

        if not by_type:
            return
        fig, axes = plt.subplots(1, len(by_type), figsize=(4 * len(by_type), 3.5))
        if len(by_type) == 1:
            axes = [axes]
        for ax, (etype, vals) in zip(axes, by_type.items()):
            arr = np.array(vals, dtype=float)
            arr = arr[np.isfinite(arr)]
            if len(arr) == 0:
                ax.set_visible(False)
                continue
            ax.hist(arr, bins=30, color="#4CAF50", edgecolor="k", linewidth=0.4, alpha=0.85)
            ax.axvline(0, color="red", lw=1.2, ls="--")
            ax.set_title(f"Residuals — {etype}", fontsize=8)
            ax.set_xlabel("residual (sim − obs)")
            ax.set_ylabel("count")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
        fig.tight_layout()
        _savefig(fig, output_dir, "pinn_residuals_histogram")
    except Exception as exc:
        _logger.warning("[bundle_analysis] PINN residuals plot failed: %s", exc)


# ---------------------------------------------------------------------------
# neuralODE bundle analysis
# ---------------------------------------------------------------------------

def plot_neuralode_bundle_analysis(
        model_dir: str | pathlib.Path,
        output_dir: str | pathlib.Path | None = None,
) -> None:
    """
    Load a neuralODE model bundle and produce a comprehensive set of diagnostic
    plots.

    The function looks for files in *model_dir* (the bundle subdirectory) and in
    *model_dir/..* (the neural ODE run output directory):

    In ``model_dir`` (the bundle subdirectory):
      * ``neural_ode_bundle_meta.json`` — structural metadata
      * ``theta_refined.npy``           — mechanistic rate vector
      * ``neural_ode_model.eqx``        — weights (presence checked only)

    In ``model_dir/..`` (the neuralODE run output directory):
      * ``neural_metadata.json``        — run-level metadata (optional)
      * ``neural_training_losses.tsv``  — per-step training loss (optional)
      * ``neural_fit_timeseries.tsv``   — fitted vs observed (optional)
      * ``neural_latent_rates.tsv``     — mechanistic vs neural rates (optional)
      * ``neural_latent_rates.npz``     — array form of latent rates (optional)
      * ``labels.json``                 — entity labels (optional)

    Args:
        model_dir:  Path to the bundle directory (e.g. ``…/neural_ode_bundle/``).
        output_dir: Directory where plots are written.  Defaults to
                    ``model_dir/plots/``.
    """
    model_dir = pathlib.Path(model_dir)
    run_dir = model_dir.parent  # neural_ode/ output directory
    if output_dir is None:
        output_dir = model_dir / "plots"
    output_dir = pathlib.Path(output_dir)
    _ensure_dir(output_dir)

    _logger.info("[bundle_analysis] neuralODE analysis: model_dir=%s", model_dir)

    # ------------------------------------------------------------------ #
    # Load data                                                           #
    # ------------------------------------------------------------------ #
    bundle_meta = _read_json(model_dir / "neural_ode_bundle_meta.json")
    run_meta = _read_json(run_dir / "neural_metadata.json")
    labels_json = _read_json(run_dir / "labels.json")
    meta = bundle_meta or run_meta or {}

    theta_path = model_dir / "theta_refined.npy"
    theta = _safe_array(np.load(theta_path) if theta_path.is_file() else None)

    loss_rows = _read_csv_if_exists(run_dir / "neural_training_losses.tsv")
    ts_rows = _read_csv_if_exists(run_dir / "neural_fit_timeseries.tsv")
    rates_rows = _read_csv_if_exists(run_dir / "neural_latent_rates.tsv")
    rates_npz = _load_npz_if_exists(run_dir / "neural_latent_rates.npz")

    # ------------------------------------------------------------------ #
    # Plot 1: Metadata summary                                            #
    # ------------------------------------------------------------------ #
    _neuralode_plot_metadata(meta, output_dir)

    # ------------------------------------------------------------------ #
    # Plot 2: Training loss history                                       #
    # ------------------------------------------------------------------ #
    _neuralode_plot_loss_history(loss_rows, output_dir)

    # ------------------------------------------------------------------ #
    # Plot 3: Mechanistic parameter vector                                #
    # ------------------------------------------------------------------ #
    _neuralode_plot_theta(theta, labels_json, output_dir)

    # ------------------------------------------------------------------ #
    # Plot 4: Learned latent rates vs mechanistic priors                  #
    # ------------------------------------------------------------------ #
    _neuralode_plot_latent_rates(rates_rows, rates_npz, output_dir)

    # ------------------------------------------------------------------ #
    # Plot 5: Fitted vs observed timeseries                               #
    # ------------------------------------------------------------------ #
    _neuralode_plot_timeseries(ts_rows, labels_json, output_dir)

    _logger.info("[bundle_analysis] neuralODE plots saved to %s", output_dir)


def _neuralode_plot_metadata(meta: dict, output_dir: pathlib.Path) -> None:
    """Text-table summary of neuralODE bundle metadata."""
    if not meta:
        _logger.warning("[bundle_analysis] neuralODE: no metadata; skipping metadata plot.")
        return
    try:
        interesting = {
            k: v for k, v in meta.items()
            if isinstance(v, (bool, int, float, str)) and not k.startswith("_")
        }
        rows_table = [(k, str(v)) for k, v in sorted(interesting.items())]
        fig, ax = plt.subplots(figsize=(8, max(3, 0.35 * len(rows_table))))
        ax.axis("off")
        table = ax.table(
            cellText=rows_table,
            colLabels=["Parameter", "Value"],
            cellLoc="left",
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1.0, 1.4)
        ax.set_title("neuralODE Bundle Metadata", fontsize=11, pad=12)
        _savefig(fig, output_dir, "neuralode_bundle_metadata")
    except Exception as exc:
        _logger.warning("[bundle_analysis] neuralODE metadata plot failed: %s", exc)


def _neuralode_plot_loss_history(
        loss_rows: list[dict] | None,
        output_dir: pathlib.Path,
) -> None:
    """Line plot of neuralODE training loss over steps."""
    if not loss_rows:
        _logger.warning("[bundle_analysis] neuralODE: no training loss data; skipping loss plot.")
        return
    try:
        steps = np.array([r.get("step", i) for i, r in enumerate(loss_rows)], dtype=float)
        total_col = "neural_loss_total"
        component_cols = [
            c for c in loss_rows[0].keys()
            if c.startswith("neural_loss_") and c != total_col
        ]

        fig, axes = plt.subplots(
            1 + bool(component_cols), 1,
            figsize=(8, 4 + 2.5 * bool(component_cols)),
            sharex=True,
        )
        if not isinstance(axes, np.ndarray):
            axes = np.array([axes])

        # Total loss
        total = np.array([r.get(total_col, np.nan) for r in loss_rows], dtype=float)
        axes[0].plot(steps, total, color="#1565C0", lw=1.8, label="total loss")
        axes[0].set_ylabel("Loss")
        axes[0].set_title("neuralODE: Training Loss", fontsize=10)
        axes[0].legend(fontsize=7)
        axes[0].spines["top"].set_visible(False)
        axes[0].spines["right"].set_visible(False)

        # Loss components
        if component_cols and len(axes) > 1:
            colors = plt.cm.tab10(np.linspace(0, 0.9, len(component_cols)))
            for col, color in zip(component_cols, colors):
                vals = np.array([r.get(col, np.nan) for r in loss_rows], dtype=float)
                label = col.replace("neural_loss_", "")
                axes[1].plot(steps, vals, color=color, lw=1.2, label=label, alpha=0.9)
            axes[1].set_ylabel("Component loss")
            axes[1].set_xlabel("Step")
            axes[1].legend(fontsize=6, ncol=2)
            axes[1].spines["top"].set_visible(False)
            axes[1].spines["right"].set_visible(False)
        else:
            axes[0].set_xlabel("Step")

        fig.tight_layout()
        _savefig(fig, output_dir, "neuralode_training_loss")
    except Exception as exc:
        _logger.warning("[bundle_analysis] neuralODE loss history plot failed: %s", exc)


def _neuralode_plot_theta(
        theta: np.ndarray | None,
        labels_json: dict | None,
        output_dir: pathlib.Path,
) -> None:
    """Bar chart of refined mechanistic parameter vector."""
    if theta is None or theta.ndim != 1:
        _logger.warning("[bundle_analysis] neuralODE: theta_refined not available; skipping.")
        return
    try:
        n = len(theta)
        xlabs = _get_labels(labels_json, "theta", n)
        fig, ax = plt.subplots(figsize=(max(6, n * 0.35), 3.5))
        ax.bar(range(n), theta, color="#7B1FA2", edgecolor="k", linewidth=0.4, alpha=0.85)
        ax.set_xticks(range(n))
        ax.set_xticklabels(xlabs, rotation=70, ha="right", fontsize=7)
        ax.set_ylabel("Parameter value")
        ax.set_title("neuralODE: Refined Mechanistic Parameter Vector (θ_refined)", fontsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        _savefig(fig, output_dir, "neuralode_theta_refined")
    except Exception as exc:
        _logger.warning("[bundle_analysis] neuralODE theta plot failed: %s", exc)


def _neuralode_plot_latent_rates(
        rates_rows: list[dict] | None,
        rates_npz: dict | None,
        output_dir: pathlib.Path,
) -> None:
    """
    Line plots of mechanistic prior vs neural learned rates (k_act, s_prod).
    """
    if not rates_rows and rates_npz is None:
        _logger.warning("[bundle_analysis] neuralODE: no latent rate data; skipping rates plot.")
        return
    try:
        if rates_rows:
            # From TSV (long format): rate_type, entity, time, mechanistic_prior, neural_learned
            by_rate: dict[str, dict[str, list]] = {}
            for row in rates_rows:
                rt = str(row.get("rate_type", "unknown"))
                ent = str(row.get("entity", "unknown"))
                t_val = float(row.get("time", 0.0))
                mech = float(row.get("mechanistic_prior", np.nan))
                neu = float(row.get("neural_learned", np.nan))
                by_rate.setdefault(rt, {}).setdefault(ent, {"t": [], "mech": [], "neural": []})
                by_rate[rt][ent]["t"].append(t_val)
                by_rate[rt][ent]["mech"].append(mech)
                by_rate[rt][ent]["neural"].append(neu)

            for rate_type, entities in by_rate.items():
                enames = sorted(entities.keys())
                ncols = min(4, len(enames))
                nrows = (len(enames) + ncols - 1) // ncols
                fig, axes = plt.subplots(nrows, ncols,
                                         figsize=(3.5 * ncols, 2.5 * nrows),
                                         squeeze=False)
                axes_flat = axes.flatten()
                for ax_idx, ename in enumerate(enames):
                    d = entities[ename]
                    sort_idx = np.argsort(d["t"])
                    t_a = np.array(d["t"])[sort_idx]
                    m_a = np.array(d["mech"])[sort_idx]
                    n_a = np.array(d["neural"])[sort_idx]
                    ax = axes_flat[ax_idx]
                    ax.plot(t_a, m_a, color="#F57C00", lw=1.2, ls="--", label="prior")
                    ax.plot(t_a, n_a, color="#1565C0", lw=1.5, label="neural")
                    ax.set_title(ename, fontsize=7)
                    ax.tick_params(labelsize=6)
                    ax.spines["top"].set_visible(False)
                    ax.spines["right"].set_visible(False)
                    if ax_idx == 0:
                        ax.legend(fontsize=6)
                for ax in axes_flat[len(enames):]:
                    ax.set_visible(False)
                fig.suptitle(f"neuralODE: {rate_type} — Prior vs Learned", fontsize=10)
                fig.tight_layout()
                _savefig(fig, output_dir, f"neuralode_latent_{rate_type}")

        elif rates_npz is not None:
            # Use array form from .npz
            _neuralode_plot_latent_rates_npz(rates_npz, output_dir)

    except Exception as exc:
        _logger.warning("[bundle_analysis] neuralODE latent rates plot failed: %s", exc)


def _neuralode_plot_latent_rates_npz(
        rates_npz: dict,
        output_dir: pathlib.Path,
) -> None:
    """Heatmap of learned latent rates from npz arrays."""
    try:
        t_obs = _safe_array(rates_npz.get("t_obs"))
        t_kact = _safe_array(rates_npz.get("t_kact"))
        k_act_mech = _safe_array(rates_npz.get("k_act_mechanistic"))  # (K, T)
        k_act_neural = _safe_array(rates_npz.get("k_act_neural"))  # (K, T)
        s_prod_mech = _safe_array(rates_npz.get("s_prod_mechanistic"))  # (K, T)
        s_prod_neural = _safe_array(rates_npz.get("s_prod_neural"))  # (K, T)
        proteins_arr = rates_npz.get("proteins")
        proteins = list(proteins_arr) if proteins_arr is not None else None

        for rate_name, mech_arr, neural_arr, t_arr in [
            ("k_act", k_act_mech, k_act_neural, t_kact),
            ("s_prod", s_prod_mech, s_prod_neural, t_obs),
        ]:
            if mech_arr is None or neural_arr is None or t_arr is None:
                continue
            K = mech_arr.shape[0]
            row_labels = proteins[:K] if proteins else [f"prot_{i}" for i in range(K)]
            col_labels = [f"t={t:.1f}" for t in t_arr]
            delta = neural_arr - mech_arr

            fig, axes = plt.subplots(1, 3, figsize=(14, max(3, 0.6 * K + 1)))
            _plot_matrix_heatmap(axes[0], mech_arr, row_labels, col_labels,
                                 title=f"{rate_name}: Prior", cmap="YlOrRd")
            _plot_matrix_heatmap(axes[1], neural_arr, row_labels, col_labels,
                                 title=f"{rate_name}: Neural", cmap="YlOrRd")
            _plot_matrix_heatmap(axes[2], delta, row_labels, col_labels,
                                 title=f"{rate_name}: Δ (Neural − Prior)", cmap="RdBu_r")
            fig.suptitle(f"neuralODE: {rate_name} Rate Heatmaps", fontsize=11)
            fig.tight_layout()
            _savefig(fig, output_dir, f"neuralode_{rate_name}_heatmap")
    except Exception as exc:
        _logger.warning("[bundle_analysis] neuralODE npz rate heatmap failed: %s", exc)


def _neuralode_plot_timeseries(
        ts_rows: list[dict] | None,
        labels_json: dict | None,
        output_dir: pathlib.Path,
) -> None:
    """Per-entity fitted vs observed timeseries plots."""
    if not ts_rows:
        _logger.warning("[bundle_analysis] neuralODE: no timeseries data; skipping timeseries plot.")
        return
    try:
        by_type: dict[str, dict[str, list[dict]]] = {}
        for row in ts_rows:
            etype = str(row.get("entity_type", "unknown"))
            entity = str(row.get("entity", "unknown"))
            by_type.setdefault(etype, {}).setdefault(entity, []).append(row)

        for etype, entities in by_type.items():
            enames = sorted(entities.keys())
            ncols = min(4, len(enames))
            nrows = (len(enames) + ncols - 1) // ncols
            fig, axes = plt.subplots(nrows, ncols,
                                     figsize=(3.5 * ncols, 2.5 * nrows),
                                     squeeze=False)
            axes_flat = axes.flatten()
            for ax_idx, ename in enumerate(enames):
                rows_e = sorted(entities[ename], key=lambda r: r.get("time", 0))
                t_vals = np.array([r.get("time", 0) for r in rows_e], dtype=float)
                neural = np.array([r.get("value_neural", np.nan) for r in rows_e], dtype=float)
                obs = np.array([r.get("value_observed", np.nan) for r in rows_e], dtype=float)
                ax = axes_flat[ax_idx]
                ax.plot(t_vals, neural, color="#7B1FA2", lw=1.5, label="neural")
                mask = np.isfinite(obs)
                if mask.any():
                    ax.scatter(t_vals[mask], obs[mask], s=18, color="#E91E63",
                               zorder=5, label="observed")
                ax.set_title(ename, fontsize=7)
                ax.tick_params(labelsize=6)
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                if ax_idx == 0:
                    ax.legend(fontsize=6)
            for ax in axes_flat[len(enames):]:
                ax.set_visible(False)
            fig.suptitle(f"neuralODE: Fitted vs Observed — {etype}", fontsize=10, y=1.01)
            fig.tight_layout()
            safe_etype = etype.replace(" ", "_").replace("/", "_")
            _savefig(fig, output_dir, f"neuralode_timeseries_{safe_etype}")
    except Exception as exc:
        _logger.warning("[bundle_analysis] neuralODE timeseries plot failed: %s", exc)


# ---------------------------------------------------------------------------
# Cross-model comparison
# ---------------------------------------------------------------------------

def plot_model_comparison(
        pinn_dir: str | pathlib.Path,
        neuralode_dir: str | pathlib.Path,
        output_dir: str | pathlib.Path,
) -> None:
    """
    Produce side-by-side comparison plots for a PINN bundle and a neuralODE
    bundle.

    Generates the following plots in *output_dir*:

    * ``comparison_theta.{png,pdf}`` — mechanistic parameter vectors
      side-by-side (PINN ``theta_opt`` vs neuralODE ``theta_refined``).
    * ``comparison_timeseries_{etype}.{png,pdf}`` — fitted vs observed
      trajectories for each entity type, with one column for PINN and one for
      neuralODE.
    * ``comparison_loss.{png,pdf}`` — loss component overview for both models.

    Files that cannot be loaded are skipped with a warning.

    Args:
        pinn_dir:      Path to the PINN bundle directory
                       (e.g. ``…/pinn_bundle/``).
        neuralode_dir: Path to the neuralODE bundle directory
                       (e.g. ``…/neural_ode_bundle/``).
        output_dir:    Directory where comparison plots are written.
    """
    pinn_dir = pathlib.Path(pinn_dir)
    neuralode_dir = pathlib.Path(neuralode_dir)
    output_dir = pathlib.Path(output_dir)
    _ensure_dir(output_dir)

    _logger.info(
        "[bundle_analysis] comparison: pinn=%s  neuralode=%s  out=%s",
        pinn_dir, neuralode_dir, output_dir,
    )

    pinn_run_dir = pinn_dir.parent
    node_run_dir = neuralode_dir.parent

    # ------------------------------------------------------------------ #
    # Load theta vectors                                                  #
    # ------------------------------------------------------------------ #
    pinn_theta = _safe_array(
        np.load(pinn_dir / "theta_opt.npy")
        if (pinn_dir / "theta_opt.npy").is_file()
        else (np.load(pinn_run_dir / "theta_opt.npy") if (pinn_run_dir / "theta_opt.npy").is_file() else None)
    )
    node_theta = _safe_array(
        np.load(neuralode_dir / "theta_refined.npy")
        if (neuralode_dir / "theta_refined.npy").is_file()
        else None
    )

    # ------------------------------------------------------------------ #
    # Plot: theta comparison                                              #
    # ------------------------------------------------------------------ #
    _comparison_plot_theta(pinn_theta, node_theta, output_dir)

    # ------------------------------------------------------------------ #
    # Load loss data for comparison bar chart                             #
    # ------------------------------------------------------------------ #
    pinn_loss_rows = _read_csv_if_exists(pinn_run_dir / "pinn_loss_components.tsv")
    pinn_meta = _read_json(pinn_dir / "pinn_bundle_meta.json") or _read_json(pinn_run_dir / "pinn_metadata.json") or {}
    node_loss_rows = _read_csv_if_exists(node_run_dir / "neural_training_losses.tsv")
    node_meta = _read_json(neuralode_dir / "neural_ode_bundle_meta.json") or _read_json(
        node_run_dir / "neural_metadata.json") or {}

    _comparison_plot_loss(pinn_loss_rows, pinn_meta, node_loss_rows, node_meta, output_dir)

    # ------------------------------------------------------------------ #
    # Load timeseries data for comparison                                 #
    # ------------------------------------------------------------------ #
    pinn_ts = _read_csv_if_exists(pinn_run_dir / "pinn_fit_timeseries.tsv")
    node_ts = _read_csv_if_exists(node_run_dir / "neural_fit_timeseries.tsv")
    labels = _read_json(pinn_run_dir / "labels.json") or _read_json(node_run_dir / "labels.json")

    _comparison_plot_timeseries(pinn_ts, node_ts, labels, output_dir)

    _logger.info("[bundle_analysis] Comparison plots saved to %s", output_dir)


def _comparison_plot_theta(
        pinn_theta: np.ndarray | None,
        node_theta: np.ndarray | None,
        output_dir: pathlib.Path,
) -> None:
    """Bar chart comparing PINN theta_opt and neuralODE theta_refined."""
    if pinn_theta is None and node_theta is None:
        _logger.warning("[bundle_analysis] comparison: no theta data available.")
        return
    try:
        n_pinn = len(pinn_theta) if pinn_theta is not None else 0
        n_node = len(node_theta) if node_theta is not None else 0
        n_max = max(n_pinn, n_node)
        if n_max == 0:
            return

        x = np.arange(n_max)
        fig, ax = plt.subplots(figsize=(max(6, n_max * 0.4), 4))
        width = 0.38
        if pinn_theta is not None:
            ax.bar(x[:n_pinn] - width / 2, pinn_theta, width=width,
                   color="#4C72B0", alpha=0.85, edgecolor="k", lw=0.4, label="PINN θ_opt")
        if node_theta is not None:
            ax.bar(x[:n_node] + width / 2, node_theta, width=width,
                   color="#7B1FA2", alpha=0.85, edgecolor="k", lw=0.4, label="neuralODE θ_refined")
        ax.set_xticks(x)
        ax.set_xticklabels([f"θ{i}" for i in range(n_max)], rotation=70, ha="right", fontsize=7)
        ax.set_ylabel("Parameter value")
        ax.set_title("Model Comparison: Mechanistic Parameter Vectors", fontsize=10)
        ax.legend(fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        _savefig(fig, output_dir, "comparison_theta")
    except Exception as exc:
        _logger.warning("[bundle_analysis] comparison theta plot failed: %s", exc)


def _comparison_plot_loss(
        pinn_loss_rows: list[dict] | None,
        pinn_meta: dict,
        node_loss_rows: list[dict] | None,
        node_meta: dict,
        output_dir: pathlib.Path,
) -> None:
    """Grouped bar chart comparing key loss components."""
    _COMMON_KEYS = ["f1", "f2", "f3", "f4"]
    pinn_loss: dict[str, float] = {}
    node_loss: dict[str, float] = {}

    if pinn_loss_rows:
        pinn_loss = {k: float(v) for k, v in pinn_loss_rows[0].items() if k in _COMMON_KEYS}
    elif pinn_meta:
        pinn_loss = {k: float(pinn_meta[k]) for k in _COMMON_KEYS if k in pinn_meta}

    if node_loss_rows:
        last = node_loss_rows[-1]
        for k, mapping in [("f1", "neural_loss_phospho"), ("f2", "neural_loss_abundance"),
                           ("f3", "neural_loss_k_act_prior"), ("f4", "neural_loss_mrna")]:
            if mapping in last:
                node_loss[k] = float(last[mapping])
    elif node_meta:
        for k in _COMMON_KEYS:
            if k in node_meta:
                node_loss[k] = float(node_meta[k])

    if not pinn_loss and not node_loss:
        _logger.warning("[bundle_analysis] comparison: no loss data for either model.")
        return
    try:
        all_keys = sorted(set(list(pinn_loss.keys()) + list(node_loss.keys())))
        _PRETTY = {"f1": "Phospho (f₁)", "f2": "Abund. (f₂)",
                   "f3": "Reg. (f₃)", "f4": "mRNA (f₄)"}
        x = np.arange(len(all_keys))
        width = 0.38
        fig, ax = plt.subplots(figsize=(max(5, len(all_keys) * 1.2), 4))
        if pinn_loss:
            pinn_vals = [pinn_loss.get(k, 0.0) for k in all_keys]
            ax.bar(x - width / 2, pinn_vals, width=width,
                   color="#4C72B0", alpha=0.85, edgecolor="k", lw=0.4, label="PINN")
        if node_loss:
            node_vals = [node_loss.get(k, 0.0) for k in all_keys]
            ax.bar(x + width / 2, node_vals, width=width,
                   color="#7B1FA2", alpha=0.85, edgecolor="k", lw=0.4, label="neuralODE")
        ax.set_xticks(x)
        ax.set_xticklabels([_PRETTY.get(k, k) for k in all_keys], fontsize=9)
        ax.set_ylabel("Loss value")
        ax.set_title("Model Comparison: Loss Components", fontsize=10)
        ax.legend(fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        _savefig(fig, output_dir, "comparison_loss")
    except Exception as exc:
        _logger.warning("[bundle_analysis] comparison loss plot failed: %s", exc)


def _comparison_plot_timeseries(
        pinn_ts: list[dict] | None,
        node_ts: list[dict] | None,
        labels_json: dict | None,
        output_dir: pathlib.Path,
) -> None:
    """Side-by-side timeseries comparison for each entity type."""
    if not pinn_ts and not node_ts:
        _logger.warning("[bundle_analysis] comparison: no timeseries data for either model.")
        return
    try:
        def _group(rows):
            g: dict[str, dict[str, list]] = {}
            if not rows:
                return g
            for row in rows:
                etype = str(row.get("entity_type", "unknown"))
                entity = str(row.get("entity", "unknown"))
                g.setdefault(etype, {}).setdefault(entity, []).append(row)
            return g

        pinn_grouped = _group(pinn_ts)
        node_grouped = _group(node_ts)
        all_etypes = sorted(set(list(pinn_grouped.keys()) + list(node_grouped.keys())))

        for etype in all_etypes:
            pe = pinn_grouped.get(etype, {})
            ne = node_grouped.get(etype, {})
            all_enames = sorted(set(list(pe.keys()) + list(ne.keys())))
            if not all_enames:
                continue

            ncols = min(4, len(all_enames))
            nrows = (len(all_enames) + ncols - 1) // ncols
            fig, axes = plt.subplots(
                nrows, ncols * 2,
                figsize=(3.0 * ncols * 2, 2.5 * nrows),
                squeeze=False,
            )

            for en_idx, ename in enumerate(all_enames):
                row_i = en_idx // ncols
                col_i = (en_idx % ncols) * 2  # left = PINN, right = neuralODE

                for col_offset, model_name, rows_d, sim_key, color in [
                    (0, "PINN", pe.get(ename, []), "value_sim", "#4C72B0"),
                    (1, "neuralODE", ne.get(ename, []), "value_neural", "#7B1FA2"),
                ]:
                    ax = axes[row_i, col_i + col_offset]
                    if rows_d:
                        rows_sorted = sorted(rows_d, key=lambda r: r.get("time", 0))
                        t_vals = np.array([r.get("time", 0) for r in rows_sorted], dtype=float)
                        sim = np.array([r.get(sim_key, np.nan) for r in rows_sorted], dtype=float)
                        obs_k = "value_obs" if model_name == "PINN" else "value_observed"
                        obs = np.array([r.get(obs_k, np.nan) for r in rows_sorted], dtype=float)
                        ax.plot(t_vals, sim, color=color, lw=1.5, label=model_name)
                        mask = np.isfinite(obs)
                        if mask.any():
                            ax.scatter(t_vals[mask], obs[mask], s=16, color="#E91E63",
                                       zorder=5, label="obs")
                        if en_idx == 0 and col_offset == 0:
                            ax.legend(fontsize=5)
                    else:
                        ax.text(0.5, 0.5, "n/a", transform=ax.transAxes,
                                ha="center", va="center", fontsize=8, color="gray")
                    ax.set_title(f"{ename}\n({model_name})", fontsize=6)
                    ax.tick_params(labelsize=5)
                    ax.spines["top"].set_visible(False)
                    ax.spines["right"].set_visible(False)

            for ax in axes.flatten()[len(all_enames) * 2:]:
                ax.set_visible(False)

            fig.suptitle(f"Comparison: {etype} — PINN vs neuralODE", fontsize=10)
            fig.tight_layout()
            safe_etype = etype.replace(" ", "_").replace("/", "_")
            _savefig(fig, output_dir, f"comparison_timeseries_{safe_etype}")
    except Exception as exc:
        _logger.warning("[bundle_analysis] comparison timeseries plot failed: %s", exc)
