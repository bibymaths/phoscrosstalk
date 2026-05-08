# SPDX-License-Identifier: MIT
"""
pinn/outputs.py
Save all outputs required by the PINN run.

Writes files compatible with the existing PhosCrosstalk result viewer and
dashboard.  Files that have equivalent mechanistic counterparts use the same
names; PINN-specific files use distinct names.
"""

from __future__ import annotations

import json
import logging
import os

import numpy as np
import pandas as pd

from phoscrosstalk.pinn.utils import state_labels, safe_to_numpy

_logger = logging.getLogger(__name__)


def save_pinn_outputs(
    outdir: str,
    *,
    theta_opt: np.ndarray,
    pinn_model,
    loss_components: dict,
    ts: np.ndarray,
    ys: np.ndarray,
    K: int,
    M: int,
    N: int,
    sites: list[str],
    proteins: list[str],
    kinases: list[str],
    P_data: np.ndarray,
    A_scaled: np.ndarray,
    prot_idx_for_A: np.ndarray,
    t: np.ndarray,
    run_mode: str = "pinn",
    pinn_cfg=None,
    dims=None,
) -> None:
    """
    Save all PINN run outputs.

    Writes:
      * pinn_metadata.json
      * theta_opt.npy
      * pinn_params/ directory with eqx model checkpoint
      * pinn_loss_components.tsv
      * pinn_fit_timeseries.tsv  (phosphosite, abundance, RNA outputs)
      * pinn_residuals.tsv
      * pinn_residual_summary.tsv (optional)
      * pareto_front.npz (stub for dashboard compatibility)
    """
    os.makedirs(outdir, exist_ok=True)

    # -------------------------------------------------------------------
    # Metadata
    # -------------------------------------------------------------------
    meta: dict = {
        "run_mode": run_mode,
        "K": K,
        "M": M,
        "N": N,
        "state_dim": 3 * K + M + N,
        "theta_dim": int(np.asarray(theta_opt).shape[0]) if theta_opt is not None else None,
    }
    if pinn_cfg is not None:
        meta["pinn_config"] = {
            k: v for k, v in vars(pinn_cfg).items()
            if isinstance(v, (bool, int, float, str))
        }
    meta.update({k: float(v) for k, v in loss_components.items()})

    with open(os.path.join(outdir, "pinn_metadata.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    _logger.info("[pinn] Saved pinn_metadata.json")

    # -------------------------------------------------------------------
    # Theta
    # -------------------------------------------------------------------
    if theta_opt is not None:
        np.save(os.path.join(outdir, "theta_opt.npy"), np.asarray(theta_opt))
        _logger.info("[pinn] Saved theta_opt.npy")

    # -------------------------------------------------------------------
    # PINN parameters (equinox checkpoint via numpy savez)
    # -------------------------------------------------------------------
    if pinn_model is not None:
        try:
            import equinox as eqx
            import jax

            leaves, treedef = jax.tree_util.tree_flatten(
                eqx.filter(pinn_model, eqx.is_array)
            )
            params_dir = os.path.join(outdir, "pinn_params")
            os.makedirs(params_dir, exist_ok=True)
            np.savez(
                os.path.join(params_dir, "pinn_params.npz"),
                **{f"leaf_{i}": safe_to_numpy(l) for i, l in enumerate(leaves)},
            )
            _logger.info("[pinn] Saved pinn_params/pinn_params.npz")
        except Exception as exc:
            _logger.warning("[pinn] Could not save PINN params: %s", exc)

    # -------------------------------------------------------------------
    # Loss components TSV
    # -------------------------------------------------------------------
    df_loss = pd.DataFrame([loss_components])
    df_loss.to_csv(
        os.path.join(outdir, "pinn_loss_components.tsv"), sep="\t", index=False
    )
    _logger.info("[pinn] Saved pinn_loss_components.tsv")

    # -------------------------------------------------------------------
    # Fitted timeseries (long format, compatible with fit_timeseries.tsv)
    # -------------------------------------------------------------------
    rows: list[dict] = []

    if ys is not None and len(ys) > 0 and ts is not None and len(ts) > 0:
        ys_np = safe_to_numpy(ys)   # (T_sim, state_dim)
        ts_np = safe_to_numpy(ts)   # (T_sim,)

        # Map simulation output to observed time grid
        obs_t_idx = np.searchsorted(ts_np, t, side="left")
        obs_t_idx = np.clip(obs_t_idx, 0, len(ts_np) - 1)

        # Phosphosite trajectories p: (N, T_prot)
        p_sim_all = ys_np[:, 3 * K + M :]  # (T_sim, N)
        for n_idx, site_name in enumerate(sites):
            for t_idx, t_val in enumerate(t):
                sim_t_pos = obs_t_idx[t_idx]
                rows.append({
                    "entity_type": "phosphosite",
                    "entity": site_name,
                    "site": site_name,
                    "protein": site_name.split("_")[0] if "_" in site_name else site_name,
                    "time": float(t_val),
                    "value_sim": float(np.clip(p_sim_all[sim_t_pos, n_idx], 0.0, None)),
                    "value_obs": float(P_data[n_idx, t_idx])
                    if P_data is not None and n_idx < P_data.shape[0] else float("nan"),
                    "series_type": "pinn_fitted",
                })

        # Protein abundance trajectories A: (K, T_prot)
        if A_scaled is not None and A_scaled.size > 0 and len(prot_idx_for_A) > 0:
            A_sim_all = ys_np[:, 2 * K : 3 * K]  # (T_sim, K)
            for k_obs_idx, k_model_idx in enumerate(prot_idx_for_A):
                prot_name = proteins[k_model_idx] if k_model_idx < len(proteins) else str(k_model_idx)
                for t_idx, t_val in enumerate(t):
                    sim_t_pos = obs_t_idx[t_idx]
                    rows.append({
                        "entity_type": "protein",
                        "entity": prot_name,
                        "site": "",
                        "protein": prot_name,
                        "time": float(t_val),
                        "value_sim": float(np.clip(A_sim_all[sim_t_pos, k_model_idx], 0.0, 5.0)),
                        "value_obs": float(A_scaled[k_obs_idx, t_idx])
                        if k_obs_idx < A_scaled.shape[0] else float("nan"),
                        "series_type": "pinn_fitted",
                    })

    if rows:
        df_ts = pd.DataFrame(rows)
        df_ts.to_csv(
            os.path.join(outdir, "pinn_fit_timeseries.tsv"), sep="\t", index=False
        )
        _logger.info("[pinn] Saved pinn_fit_timeseries.tsv (%d rows)", len(rows))

    # -------------------------------------------------------------------
    # Dashboard-compatible stub (pareto_front.npz / pareto_stats.tsv)
    # -------------------------------------------------------------------
    _f1 = float(loss_components.get("f1", 0.0))
    _f2 = float(loss_components.get("f2", 0.0))
    _f3 = float(loss_components.get("f3", 0.0))
    _f4 = float(loss_components.get("f4", 0.0))
    _fp = float(loss_components.get("f_pinn_reg", 0.0))
    F_stub = np.array([[_f1, _f2, _f3, _f4]])
    theta_arr = np.asarray(theta_opt)[None, :] if theta_opt is not None else np.zeros((1, 1))
    J_stub = np.array([_f1 + _f2 + _f3 + _f4 + _fp])

    np.savez(os.path.join(outdir, "pareto_front.npz"), F=F_stub, X=theta_arr, J=J_stub)

    df_pareto_stats = pd.DataFrame({
        "objective": ["f1_P_sites", "f2_protein", "f3_complexity", "f4_mrna"],
        "min": [_f1, _f2, _f3, _f4],
        "mean": [_f1, _f2, _f3, _f4],
        "median": [_f1, _f2, _f3, _f4],
        "std": [0.0, 0.0, 0.0, 0.0],
    })
    df_pareto_stats.to_csv(
        os.path.join(outdir, "pareto_stats.tsv"), sep="\t", index=False
    )
    _logger.info("[pinn] Saved pareto_front.npz and pareto_stats.tsv (dashboard stubs)")


def save_pinn_residuals(
    outdir: str,
    *,
    ts: np.ndarray,
    ys: np.ndarray,
    pinn_model,
    K: int,
    M: int,
    N: int,
    proteins: list[str],
    kinases: list[str],
    sites: list[str],
) -> np.ndarray | None:
    """
    Evaluate the PINN correction on the trajectory and save residual TSVs.

    Returns the residual array (T, state_dim) for downstream plotting, or None
    if the evaluation fails.
    """
    if pinn_model is None or ys is None or ts is None:
        return None

    try:
        import jax
        import jax.numpy as jnp

        ys_j = jnp.asarray(ys, dtype=jnp.float64)
        # Evaluate pinn_model on every trajectory point
        pinn_ys = jax.vmap(
            lambda y_row: pinn_model(y_row, jnp.asarray(0.0, dtype=jnp.float64))
        )(ys_j)  # (T, state_dim)
        pinn_np = safe_to_numpy(pinn_ys)
    except Exception as exc:
        _logger.warning("[pinn] Could not evaluate PINN corrections: %s", exc)
        return None

    labels = state_labels(K, M, N, proteins, kinases, sites)
    T = pinn_np.shape[0]
    state_dim = 3 * K + M + N

    # Determine block label for each state dimension
    block_map = (
        ["R_rna"] * K
        + ["S"] * K
        + ["A"] * K
        + ["Kdyn"] * M
        + ["p"] * N
    )
    entity_map = (
        list(proteins) + list(proteins) + list(proteins)
        + list(kinases)
        + list(sites)
    )

    rows: list[dict] = []
    ts_np = safe_to_numpy(ts)

    for t_idx in range(T):
        t_val = float(ts_np[t_idx]) if t_idx < len(ts_np) else float(t_idx)
        for s_idx in range(state_dim):
            val = float(pinn_np[t_idx, s_idx])
            rows.append({
                "time":           t_val,
                "state_block":    block_map[s_idx] if s_idx < len(block_map) else "?",
                "entity":         entity_map[s_idx] if s_idx < len(entity_map) else str(s_idx),
                "state_index":    s_idx,
                "state_label":    labels[s_idx] if s_idx < len(labels) else str(s_idx),
                "signed_residual": val,
                "abs_residual":   abs(val),
            })

    df = pd.DataFrame(rows)
    tsv_path = os.path.join(outdir, "pinn_residuals.tsv")
    df.to_csv(tsv_path, sep="\t", index=False)
    _logger.info("[pinn] Saved pinn_residuals.tsv (%d rows)", len(rows))

    # Summary: mean / max abs_residual per (state_block, entity)
    try:
        df_summary = (
            df.groupby(["state_block", "entity"])["abs_residual"]
            .agg(mean_abs="mean", max_abs="max")
            .reset_index()
            .sort_values("max_abs", ascending=False)
        )
        df_summary.to_csv(
            os.path.join(outdir, "pinn_residual_summary.tsv"), sep="\t", index=False
        )
        _logger.info("[pinn] Saved pinn_residual_summary.tsv")
    except Exception as exc:
        _logger.warning("[pinn] Could not save residual summary: %s", exc)

    return pinn_np
