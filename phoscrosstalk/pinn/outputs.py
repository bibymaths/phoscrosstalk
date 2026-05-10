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
import os

import numpy as np
import pandas as pd

from phoscrosstalk.logger import get_logger
from phoscrosstalk.pinn.utils import state_labels, safe_to_numpy

_logger = get_logger().logger

# Name of the bundle sub-directory written by save_pinn_model_bundle().
_BUNDLE_SUBDIR = "pinn_bundle"


def save_pinn_model_bundle(
    outdir: str,
    *,
    pinn_model,
    theta_opt: np.ndarray | None,
    pinn_cfg=None,
    dims=None,
    K: int | None = None,
    M: int | None = None,
    N: int | None = None,
) -> str:
    """Save the trained PINN model as a self-contained, reproducible bundle.

    Writes three files to ``{outdir}/pinn_bundle/``:

    * ``pinn_model.eqx`` — Equinox-serialised model weights
      (``eqx.tree_serialise_leaves``).  Load with
      :func:`load_pinn_model_bundle`.
    * ``pinn_bundle_meta.json`` — structural metadata required to re-create
      the :class:`~phoscrosstalk.pinn.model.PINNAugmentation` skeleton before
      deserialising weights.  Includes ``state_dim``, ``width_size``,
      ``depth``, ``activation``, ``output_clamp``, ``K``, ``M``, ``N``.
    * ``theta_opt.npy`` — mechanistic rate parameter vector (copy; the
      canonical copy is written by :func:`save_pinn_outputs`).

    Args:
        outdir:     Top-level run output directory.  The bundle lands in
                    ``{outdir}/pinn_bundle/``.
        pinn_model: Trained :class:`~phoscrosstalk.pinn.model.PINNAugmentation`
                    Equinox module.
        theta_opt:  Mechanistic parameter vector ``(theta_dim,)``.
        pinn_cfg:   PINN config ``SimpleNamespace`` (provides ``width_size``,
                    ``depth``, ``activation``).
        dims:       :class:`~phoscrosstalk.config.ModelDims` with ``K``, ``M``,
                    ``N`` attributes.  If *None*, supply ``K``/``M``/``N``
                    explicitly.
        K, M, N:    Model dimensions (used only when *dims* is ``None``).

    Returns:
        str: Path to the bundle directory.

    Raises:
        RuntimeError: If neither *pinn_model* nor *theta_opt* is provided.
    """
    import equinox as eqx

    if pinn_model is None and theta_opt is None:
        raise RuntimeError("save_pinn_model_bundle: both pinn_model and theta_opt are None.")

    # Resolve dimensions from dims or explicit kwargs.
    _K = int(dims.K) if dims is not None else int(K or 0)
    _M = int(dims.M) if dims is not None else int(M or 0)
    _N = int(dims.N) if dims is not None else int(N or 0)

    # Validate that all dimensions are positive so the saved metadata is valid.
    if dims is None and (_K <= 0 or _M <= 0 or _N <= 0):
        raise ValueError(
            "save_pinn_model_bundle: when dims=None you must supply explicit positive "
            f"values for K, M, and N.  Got K={_K}, M={_M}, N={_N}."
        )
    state_dim = 3 * _K + _M + _N

    # Resolve architecture from config or model attributes.
    _width   = int(getattr(pinn_cfg, "width_size", 64)) if pinn_cfg is not None else 64
    _depth   = int(getattr(pinn_cfg, "depth", 2))       if pinn_cfg is not None else 2
    _act     = str(getattr(pinn_cfg, "activation", "tanh")) if pinn_cfg is not None else "tanh"

    # Inspect model for output_clamp (stored as a class constant).
    from phoscrosstalk.pinn.model import _PINN_OUTPUT_CLAMP
    _clamp   = float(_PINN_OUTPUT_CLAMP)

    bundle_dir = os.path.join(outdir, _BUNDLE_SUBDIR)
    os.makedirs(bundle_dir, exist_ok=True)

    # ------------------------------------------------------------------ #
    # 1. Structural metadata (needed to rebuild the skeleton)             #
    # ------------------------------------------------------------------ #
    meta: dict = {
        "state_dim": state_dim,
        "K": _K,
        "M": _M,
        "N": _N,
        "width_size": _width,
        "depth": _depth,
        "activation": _act,
        "output_clamp": _clamp,
        "theta_dim": int(np.asarray(theta_opt).shape[0]) if theta_opt is not None else None,
        "bundle_format_version": 1,
    }
    meta_path = os.path.join(bundle_dir, "pinn_bundle_meta.json")
    with open(meta_path, "w") as fh:
        json.dump(meta, fh, indent=2)
    _logger.info("[pinn] Saved %s", meta_path)

    # ------------------------------------------------------------------ #
    # 2. Equinox model weights                                            #
    # ------------------------------------------------------------------ #
    if pinn_model is not None:
        model_path = os.path.join(bundle_dir, "pinn_model.eqx")
        try:
            eqx.tree_serialise_leaves(model_path, pinn_model)
            _logger.info("[pinn] Saved %s", model_path)
        except Exception as exc:
            _logger.warning("[pinn] Could not serialise PINN model with eqx: %s", exc)

    # ------------------------------------------------------------------ #
    # 3. Mechanistic parameter vector (copy)                              #
    # ------------------------------------------------------------------ #
    if theta_opt is not None:
        theta_path = os.path.join(bundle_dir, "theta_opt.npy")
        np.save(theta_path, np.asarray(theta_opt, dtype=np.float64))
        _logger.info("[pinn] Saved %s", theta_path)

    _logger.info("[pinn] Model bundle written to %s", bundle_dir)
    return bundle_dir


def load_pinn_model_bundle(
    bundle_dir: str,
) -> tuple:
    """Load a PINN model bundle saved by :func:`save_pinn_model_bundle`.

    Reconstructs the :class:`~phoscrosstalk.pinn.model.PINNAugmentation`
    skeleton from ``pinn_bundle_meta.json``, then deserialises the weights
    from ``pinn_model.eqx`` using :func:`equinox.tree_deserialise_leaves`.

    Args:
        bundle_dir: Path to the ``pinn_bundle/`` directory created by
                    :func:`save_pinn_model_bundle`.

    Returns:
        tuple: ``(pinn_model, theta_opt, meta)`` where

        * ``pinn_model`` – loaded :class:`~phoscrosstalk.pinn.model.PINNAugmentation`
          (or ``None`` if ``pinn_model.eqx`` is absent).
        * ``theta_opt`` – ``np.ndarray`` mechanistic parameters
          (or ``None`` if ``theta_opt.npy`` is absent).
        * ``meta`` – ``dict`` with the structural metadata.

    Raises:
        FileNotFoundError: If ``pinn_bundle_meta.json`` is not found in
                           *bundle_dir*.
    """
    import equinox as eqx
    import jax

    meta_path = os.path.join(bundle_dir, "pinn_bundle_meta.json")
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(
            f"load_pinn_model_bundle: {meta_path!r} not found.  "
            "Was the bundle created with save_pinn_model_bundle()?"
        )

    with open(meta_path) as fh:
        meta = json.load(fh)

    from phoscrosstalk.pinn.model import PINNAugmentation

    state_dim  = int(meta["state_dim"])
    width_size = int(meta.get("width_size", 64))
    depth      = int(meta.get("depth", 2))
    activation = str(meta.get("activation", "tanh"))
    output_clamp = float(meta.get("output_clamp", 10.0))

    # Build the skeleton with a fixed key (weights will be overwritten).
    skeleton = PINNAugmentation(
        state_dim=state_dim,
        width_size=width_size,
        depth=depth,
        activation=activation,
        key=jax.random.PRNGKey(0),
        output_clamp=output_clamp,
    )

    pinn_model = None
    model_path = os.path.join(bundle_dir, "pinn_model.eqx")
    if os.path.isfile(model_path):
        try:
            pinn_model = eqx.tree_deserialise_leaves(model_path, skeleton)
            _logger.info("[pinn] Loaded pinn_model from %s", model_path)
        except Exception as exc:
            _logger.warning("[pinn] Could not deserialise PINN model: %s", exc)
    else:
        _logger.warning("[pinn] pinn_model.eqx not found in %s", bundle_dir)

    theta_opt = None
    theta_path = os.path.join(bundle_dir, "theta_opt.npy")
    if os.path.isfile(theta_path):
        theta_opt = np.load(theta_path)
        _logger.info("[pinn] Loaded theta_opt from %s (shape=%s)", theta_path, theta_opt.shape)
    else:
        _logger.warning("[pinn] theta_opt.npy not found in %s", bundle_dir)

    return pinn_model, theta_opt, meta


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

    # -------------------------------------------------------------------
    # Model bundle (reproducible Equinox checkpoint)
    # -------------------------------------------------------------------
    try:
        save_pinn_model_bundle(
            outdir,
            pinn_model=pinn_model,
            theta_opt=theta_opt,
            pinn_cfg=pinn_cfg,
            dims=dims,
            K=K, M=M, N=N,
        )
    except Exception as exc:
        _logger.warning("[pinn] Could not save model bundle: %s", exc)


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
        ts_j = jnp.asarray(ts, dtype=jnp.float64)
        # Evaluate pinn_model on every trajectory point, including the correct time.
        pinn_ys = jax.vmap(
            lambda y_row, t_val: pinn_model(y_row, jnp.asarray(t_val, dtype=jnp.float64))
        )(ys_j, ts_j)  # (T, state_dim)
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
