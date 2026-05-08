# SPDX-License-Identifier: MIT
"""
pinn/plotting.py
PINN-specific plots.

Provides:
  * plot_pinn_residual_heatmap – neural correction heatmap (state × time).
  * plot_pinn_loss_trajectory  – training loss per step.
  * plot_pinn_fit_comparison   – fitted vs observed trajectories.
  * save_pinn_plots            – convenience wrapper that calls all plots.
"""

from __future__ import annotations

import logging
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from phoscrosstalk.pinn.utils import state_labels

_logger = logging.getLogger(__name__)


def plot_pinn_residual_heatmap(
    outdir: str,
    pinn_residuals: np.ndarray,
    ts: np.ndarray,
    K: int,
    M: int,
    N: int,
    proteins: list[str],
    kinases: list[str],
    sites: list[str],
    filename: str = "pinn_residuals.png",
) -> None:
    """
    Save a heatmap of |f_pinn(x, t)| with rows = state labels, columns = time.

    Large absolute residuals indicate candidate missing biological mechanisms.
    Small residuals indicate that the mechanistic RHS explains the state well.
    """
    if pinn_residuals is None or len(pinn_residuals) == 0:
        _logger.warning("[pinn] No PINN residuals available for heatmap.")
        return

    labels = state_labels(K, M, N, proteins, kinases, sites)
    abs_res = np.abs(pinn_residuals)  # (T, state_dim)

    state_dim = abs_res.shape[1]
    n_labels  = min(state_dim, len(labels))

    try:
        fig, ax = plt.subplots(
            figsize=(max(8, len(ts) // 10), max(6, n_labels // 4))
        )
        im = ax.imshow(
            abs_res[:, :n_labels].T,
            aspect="auto",
            cmap="YlOrRd",
            interpolation="nearest",
        )
        plt.colorbar(im, ax=ax, label="|f_pinn|")

        ax.set_xlabel("Time step")
        ax.set_ylabel("State")
        ax.set_title(
            "PINN neural residual magnitude  |f_pinn(x, t)|\n"
            "Large values → candidate missing biological mechanisms"
        )
        ax.set_yticks(range(n_labels))
        ax.set_yticklabels(labels[:n_labels], fontsize=max(4, 8 - n_labels // 20))

        # Mark time axis with approximate time values
        n_xticks = min(10, len(ts))
        xtick_pos   = np.linspace(0, len(ts) - 1, n_xticks, dtype=int)
        xtick_labels = [f"{float(ts[i]):.1f}" for i in xtick_pos]
        ax.set_xticks(xtick_pos)
        ax.set_xticklabels(xtick_labels, rotation=45, ha="right")

        fig.tight_layout()
        path = os.path.join(outdir, filename)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        _logger.info("[pinn] Saved %s", path)
    except Exception as exc:
        _logger.warning("[pinn] Could not save residual heatmap: %s", exc)
        plt.close("all")


def plot_pinn_loss_trajectory(
    outdir: str,
    loss_history: list[dict],
    filename: str = "pinn_loss_trajectory.png",
) -> None:
    """
    Plot the PINN training loss components over steps.
    """
    if not loss_history:
        return
    try:
        steps     = [d.get("step", i) for i, d in enumerate(loss_history)]
        total     = [d.get("total_loss",   0.0) for d in loss_history]
        f1_hist   = [d.get("f1",           0.0) for d in loss_history]
        f2_hist   = [d.get("f2",           0.0) for d in loss_history]
        f3_hist   = [d.get("f3",           0.0) for d in loss_history]
        f4_hist   = [d.get("f4",           0.0) for d in loss_history]
        fp_hist   = [d.get("f_pinn_reg",   0.0) for d in loss_history]

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.semilogy(steps, total,   label="total",       color="black", lw=2)
        ax.semilogy(steps, f1_hist, label="f1 phospho",  linestyle="--")
        ax.semilogy(steps, f2_hist, label="f2 abundance",linestyle="--")
        ax.semilogy(steps, f3_hist, label="f3 reg",      linestyle="--")
        ax.semilogy(steps, f4_hist, label="f4 mRNA",     linestyle="--")
        ax.semilogy(steps, fp_hist, label="f_pinn_reg",  linestyle="-.")
        ax.set_xlabel("Training step")
        ax.set_ylabel("Loss (log scale)")
        ax.set_title("PINN training loss trajectory")
        ax.legend(fontsize=8, ncol=2)
        fig.tight_layout()
        path = os.path.join(outdir, filename)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        _logger.info("[pinn] Saved %s", path)
    except Exception as exc:
        _logger.warning("[pinn] Could not save loss trajectory: %s", exc)
        plt.close("all")


def plot_pinn_fit_comparison(
    outdir: str,
    ts: np.ndarray,
    ys: np.ndarray,
    t_obs: np.ndarray,
    P_data: np.ndarray,
    sites: list[str],
    K: int,
    M: int,
    N: int,
    filename: str = "pinn_fit_comparison.png",
    max_panels: int = 20,
) -> None:
    """
    Plot fitted vs observed phosphosite trajectories (first max_panels sites).
    """
    if ys is None or P_data is None:
        return
    try:
        ts_np  = np.asarray(ts)
        ys_np  = np.asarray(ys)
        n_plot = min(max_panels, N)
        ncols  = min(4, n_plot)
        nrows  = (n_plot + ncols - 1) // ncols

        obs_idx = np.searchsorted(ts_np, t_obs, side="left")
        obs_idx = np.clip(obs_idx, 0, len(ts_np) - 1)

        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3), squeeze=False)
        for i in range(n_plot):
            r, c = divmod(i, ncols)
            ax = axes[r][c]
            p_sim = np.clip(ys_np[:, 3 * K + M + i], 0.0, None)
            ax.plot(ts_np, p_sim, label="PINN fit", color="C0")
            ax.scatter(t_obs, P_data[i], color="C1", s=20, label="observed", zorder=3)
            ax.set_title(sites[i] if i < len(sites) else str(i), fontsize=8)
            ax.set_xlabel("t", fontsize=7)
            ax.tick_params(labelsize=6)
            if r == 0 and c == 0:
                ax.legend(fontsize=6)

        # Hide unused panels
        for i in range(n_plot, nrows * ncols):
            r, c = divmod(i, ncols)
            axes[r][c].set_visible(False)

        fig.suptitle("PINN fitted vs observed phosphosite trajectories", fontsize=10)
        fig.tight_layout()
        path = os.path.join(outdir, filename)
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        _logger.info("[pinn] Saved %s", path)
    except Exception as exc:
        _logger.warning("[pinn] Could not save fit comparison: %s", exc)
        plt.close("all")


def save_pinn_plots(
    outdir: str,
    *,
    pinn_residuals: np.ndarray | None,
    ts: np.ndarray | None,
    ys: np.ndarray | None,
    t_obs: np.ndarray | None,
    P_data: np.ndarray | None,
    loss_history: list[dict],
    K: int,
    M: int,
    N: int,
    proteins: list[str],
    kinases: list[str],
    sites: list[str],
) -> None:
    """Convenience wrapper that saves all PINN plots."""
    os.makedirs(outdir, exist_ok=True)

    if pinn_residuals is not None and ts is not None:
        plot_pinn_residual_heatmap(
            outdir, pinn_residuals, ts, K, M, N, proteins, kinases, sites
        )

    if loss_history:
        plot_pinn_loss_trajectory(outdir, loss_history)

    if ys is not None and P_data is not None and t_obs is not None and ts is not None:
        plot_pinn_fit_comparison(
            outdir, ts, ys, t_obs, P_data, sites, K, M, N
        )
