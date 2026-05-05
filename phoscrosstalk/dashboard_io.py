"""
dashboard_io.py
Helper module for the PhosCrosstalk Streamlit dashboard.

Provides:
- Artifact loaders (st.cache_data-compatible) for all run outputs.
- Dashboard cache builder (results_dir/dashboard_cache/).
- Run directory validator.
- Dashboard manifest writer.

Cache policy:
  - All loaded artifacts are st.cache_data-wrapped.
  - Derived / reconstructed files go to results_dir/dashboard_cache/.
  - Existing fitted results are never overwritten.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

_SCHEMA_VERSION = "1.0"

# Required files / dirs for a valid run directory
_REQUIRED = {
    "run_config.json": "Run configuration provenance",
    "fitted_params.npz": "Fitted parameter vector",
    "preopt_snapshot": "Pre-optimisation snapshot directory",
}

# Optional but expected
_OPTIONAL_FILES = [
    "fit_timeseries.tsv",
    "internal_states.tsv",
    "derived_rates.npz",
    "derived_rates_long.tsv",
    "mrna_fit_timeseries.tsv",
    "mrna_diagnostics.tsv",
    "network_cytoscape_edges.csv",
    "biological_scores.tsv",
    "pareto_front.npz",
    "pareto_front_with_J.tsv",
    "pareto_points.tsv",
    "pareto_stats.tsv",
    "parameter_summary_global.txt",
    "parameter_summary_kinases.tsv",
    "parameter_summary_proteins.tsv",
    "parameter_summary_sites.tsv",
]

_OPTIONAL_DIRS = [
    "knockouts",
    "sensitivity",
    "steadystate",
]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_run_directory(results_dir: str) -> dict[str, Any]:
    """
    Check a results directory for required and optional artifacts.

    Returns:
        dict with keys:
          'valid' (bool), 'missing_required' (list), 'present_optional' (list),
          'missing_optional' (list), 'has_knockouts' (bool), 'has_sensitivity' (bool),
          'has_steadystate' (bool), 'has_mrna' (bool), 'has_network' (bool)
    """
    missing_req = [
        label
        for name, label in _REQUIRED.items()
        if not os.path.exists(os.path.join(results_dir, name))
    ]

    present_opt = []
    missing_opt = []
    for name in _OPTIONAL_FILES:
        p = os.path.join(results_dir, name)
        (present_opt if os.path.exists(p) else missing_opt).append(name)

    for dname in _OPTIONAL_DIRS:
        p = os.path.join(results_dir, dname)
        (present_opt if os.path.isdir(p) else missing_opt).append(dname + "/")

    return {
        "valid": len(missing_req) == 0,
        "missing_required": missing_req,
        "present_optional": present_opt,
        "missing_optional": missing_opt,
        "has_knockouts": os.path.isdir(os.path.join(results_dir, "knockouts")),
        "has_sensitivity": os.path.isdir(os.path.join(results_dir, "sensitivity")),
        "has_steadystate": os.path.isdir(os.path.join(results_dir, "steadystate")),
        "has_mrna": os.path.exists(
            os.path.join(results_dir, "mrna_fit_timeseries.tsv")
        ),
        "has_network": os.path.exists(
            os.path.join(results_dir, "network_cytoscape_edges.csv")
        ),
    }


# ---------------------------------------------------------------------------
# Provenance helpers
# ---------------------------------------------------------------------------


def _mtime(path: str) -> float:
    try:
        return os.path.getmtime(path)
    except OSError:
        return 0.0


def _md5(path: str, chunk: int = 65536) -> str:
    h = hashlib.md5()
    try:
        with open(path, "rb") as fh:
            while True:
                buf = fh.read(chunk)
                if not buf:
                    break
                h.update(buf)
        return h.hexdigest()
    except OSError:
        return ""


# ---------------------------------------------------------------------------
# Core loaders
# ---------------------------------------------------------------------------


def load_run_config(results_dir: str) -> dict[str, Any]:
    """
    Load run_config.json from a results directory.

    Returns an empty dict if the file is missing or malformed.
    """
    path = os.path.join(results_dir, "run_config.json")
    if not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return {}


def load_entity_labels(results_dir: str) -> dict[str, list[str]]:
    """
    Load sites, proteins, kinases, and A_proteins label lists.

    Reads from preopt_snapshot/*.txt or entity_labels.npz (if present).
    Returns dict with keys: 'sites', 'proteins', 'kinases', 'A_proteins'.
    """
    snap_dir = os.path.join(results_dir, "preopt_snapshot")

    # Prefer consolidated npz
    npz_path = os.path.join(snap_dir, "entity_labels.npz")
    if os.path.exists(npz_path):
        try:
            d = np.load(npz_path, allow_pickle=True)
            return {
                "sites": list(d["sites"]),
                "proteins": list(d["proteins"]),
                "kinases": list(d["kinases"]),
                "A_proteins": list(d["A_proteins"]) if "A_proteins" in d else [],
            }
        except Exception:
            pass

    def _read_txt(name: str) -> list[str]:
        p = os.path.join(snap_dir, f"{name}.txt")
        if not os.path.exists(p):
            return []
        with open(p, encoding="utf-8") as fh:
            return [ln.strip() for ln in fh if ln.strip()]

    return {
        "sites": _read_txt("sites"),
        "proteins": _read_txt("proteins"),
        "kinases": _read_txt("kinases"),
        "A_proteins": _read_txt("A_proteins"),
    }


def load_preopt_snapshot(results_dir: str) -> dict[str, Any] | None:
    """
    Load all numeric arrays from preopt_snapshot/.

    Reads from preopt_snapshot/preopt_snapshot.npz first; falls back to
    individual TSV files.

    Returns None if the snapshot directory is missing or contains no data.
    """
    snap_dir = os.path.join(results_dir, "preopt_snapshot")
    if not os.path.isdir(snap_dir):
        return None

    # Try the consolidated npz
    npz_path = os.path.join(snap_dir, "preopt_snapshot.npz")
    if os.path.exists(npz_path):
        try:
            raw = np.load(npz_path, allow_pickle=True)
            return {k: raw[k] for k in raw.files}
        except Exception:
            pass

    # Fall back to individual TSVs / TXTs
    def _vec(name, dtype=float, required=True):
        p = os.path.join(snap_dir, f"{name}.tsv")
        if not os.path.exists(p):
            if required:
                raise FileNotFoundError(p)
            return np.array([], dtype=dtype)
        return np.atleast_1d(np.loadtxt(p, delimiter="\t", dtype=dtype))

    def _mat(name, dtype=float, required=True):
        p = os.path.join(snap_dir, f"{name}.tsv")
        if not os.path.exists(p):
            if required:
                raise FileNotFoundError(p)
            return np.empty((0, 0), dtype=dtype)
        if os.path.getsize(p) == 0:
            return np.empty((0, 0), dtype=dtype)
        return np.loadtxt(p, delimiter="\t", dtype=dtype, ndmin=2)

    def _meta() -> dict[str, str]:
        p = os.path.join(snap_dir, "meta.txt")
        if not os.path.exists(p):
            return {}
        meta = {}
        with open(p, encoding="utf-8") as fh:
            for ln in fh:
                ln = ln.strip()
                if "\t" in ln:
                    k, v = ln.split("\t", 1)
                    meta[k] = v
        return meta

    try:
        snap: dict[str, Any] = {
            "t": _vec("t"),
            "Y": _mat("Y"),
            "P_scaled": _mat("P_scaled"),
            "A_scaled": _mat("A_scaled", required=False),
            "A_data": _mat("A_data", required=False),
            "W_data": _mat("W_data", required=False),
            "W_data_prot": _mat("W_data_prot", required=False),
            "Cg": _mat("Cg"),
            "Cl": _mat("Cl"),
            "K_site_kin": _mat("K_site_kin"),
            "R": _mat("R"),
            "L_alpha": _mat("L_alpha"),
            "site_prot_idx": _vec("site_prot_idx", dtype=int),
            "kin_to_prot_idx": _vec("kin_to_prot_idx", dtype=int),
            "receptor_mask_prot": _vec("receptor_mask_prot", dtype=int),
            "receptor_mask_kin": _vec("receptor_mask_kin", dtype=int),
            "positions": _vec("positions", required=False),
            "xl": _vec("xl", required=False),
            "xu": _vec("xu", required=False),
            "meta": _meta(),
        }
        return snap
    except FileNotFoundError:
        return None


def load_fitted_params(results_dir: str) -> dict[str, Any] | None:
    """
    Load fitted_params.npz.

    Returns dict with at least {'theta': np.ndarray} or None if missing.
    """
    path = os.path.join(results_dir, "fitted_params.npz")
    if not os.path.exists(path):
        return None
    try:
        raw = np.load(path, allow_pickle=True)
        return {k: raw[k] for k in raw.files}
    except Exception:
        return None


def load_fit_timeseries(results_dir: str) -> pd.DataFrame | None:
    """
    Load fit_timeseries.tsv.

    Columns: Type, Protein, Residue, sim_t<j>, data_t<j>
    Returns None if missing.
    """
    path = os.path.join(results_dir, "fit_timeseries.tsv")
    if not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path, sep="\t")
    except Exception:
        return None


def load_internal_states(results_dir: str) -> pd.DataFrame | None:
    """
    Load internal_states.tsv.

    Columns: Type, ID, t<j>
    Returns None if missing.
    """
    path = os.path.join(results_dir, "internal_states.tsv")
    if not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path, sep="\t")
    except Exception:
        return None


def load_derived_rates(results_dir: str) -> dict[str, Any] | None:
    """
    Load derived rates from derived_rates.npz or reconstruct from derived_rates_long.tsv.

    Returns dict with possible keys:
      'proteins', 't_k_act', 't_s_prod', 'k_act', 's_prod'
    Returns None if neither source is available.
    """
    # Try NPZ first
    npz_path = os.path.join(results_dir, "derived_rates.npz")
    if os.path.exists(npz_path):
        try:
            raw = np.load(npz_path, allow_pickle=True)
            return {k: raw[k] for k in raw.files}
        except Exception:
            pass

    # Reconstruct from long TSV
    tsv_path = os.path.join(results_dir, "derived_rates_long.tsv")
    if not os.path.exists(tsv_path):
        return None
    try:
        df = pd.read_csv(tsv_path, sep="\t")
        result: dict[str, Any] = {}
        for rate_type in df["rate_type"].unique():
            sub = df[df["rate_type"] == rate_type]
            proteins = sorted(sub["entity"].unique())
            times = sorted(sub["time"].unique())
            mat = np.zeros((len(proteins), len(times)), dtype=float)
            t_idx = {t: i for i, t in enumerate(times)}
            p_idx = {p: i for i, p in enumerate(proteins)}
            for _, row in sub.iterrows():
                mat[p_idx[row["entity"]], t_idx[row["time"]]] = row["value"]
            result[rate_type] = mat
            result[f"t_{rate_type}"] = np.array(times)
        result["proteins"] = np.array(proteins, dtype=object)
        return result
    except Exception:
        return None


def load_knockout_outputs(results_dir: str) -> dict[str, pd.DataFrame] | None:
    """
    Load knockout TSV files from knockouts/ subdirectory.

    Returns dict with possible keys:
      'knockout_fc' (phosphosites), 'knockout_fc_Kdyn_sim', plus any others found.
    Returns None if the directory does not exist.
    """
    ko_dir = os.path.join(results_dir, "knockouts")
    if not os.path.isdir(ko_dir):
        return None
    result = {}
    for fname in os.listdir(ko_dir):
        if fname.endswith(".tsv"):
            key = fname.replace(".tsv", "")
            try:
                result[key] = pd.read_csv(
                    os.path.join(ko_dir, fname), sep="\t", index_col=0
                )
            except Exception:
                pass
    return result if result else None


def load_sensitivity_outputs(results_dir: str) -> dict[str, Any] | None:
    """
    Load sensitivity analysis outputs.

    Returns dict with possible keys:
      'sobol' (DataFrame), 'perturbation_data' (DataFrame|None),
      'perturbation_params' (DataFrame|None)
    Returns None if the sensitivity directory does not exist.
    """
    sens_dir = os.path.join(results_dir, "sensitivity")
    if not os.path.isdir(sens_dir):
        return None

    def _load_tsv(name):
        p = os.path.join(sens_dir, name)
        if not os.path.exists(p):
            return None
        try:
            return pd.read_csv(p, sep="\t")
        except Exception:
            return None

    sobol = _load_tsv("sobol_indices_labeled.tsv")
    if sobol is None:
        return None

    return {
        "sobol": sobol,
        "perturbation_data": _load_tsv("perturbation_data.tsv"),
        "perturbation_params": _load_tsv("perturbation_params.tsv"),
    }


def load_steadystate_outputs(results_dir: str) -> dict[str, Any] | None:
    """
    Load steady-state TSV files from steadystate/ subdirectory.

    Returns dict with possible keys:
      'Kdyn' (DataFrame), 'S' (DataFrame), 'proteins' (DataFrame),
      'sites' (DataFrame)
    Returns None if the directory does not exist.
    """
    ss_dir = os.path.join(results_dir, "steadystate")
    if not os.path.isdir(ss_dir):
        return None

    name_map = {
        "steadystate_Kdyn.tsv": "Kdyn",
        "steadystate_S.tsv": "S",
        "steadystate_proteins.tsv": "proteins",
        "steadystate_sites.tsv": "sites",
    }

    result = {}
    for fname, key in name_map.items():
        p = os.path.join(ss_dir, fname)
        if not os.path.exists(p):
            continue
        try:
            result[key] = pd.read_csv(p, sep="\t", index_col=0)
        except Exception:
            pass

    return result if result else None


# ---------------------------------------------------------------------------
# Dashboard cache builder
# ---------------------------------------------------------------------------


def build_dashboard_cache(results_dir: str, force: bool = False) -> dict[str, Any]:
    """
    Build or refresh machine-readable NPZ artefacts under results_dir/dashboard_cache/.

    Only creates files that are missing or outdated (unless force=True).
    Writes a manifest at dashboard_cache/dashboard_manifest.json.

    Returns a summary dict with:
      'cache_dir', 'created', 'skipped', 'errors', 'manifest_path'
    """
    cache_dir = os.path.join(results_dir, "dashboard_cache")
    os.makedirs(cache_dir, exist_ok=True)

    created: list[str] = []
    skipped: list[str] = []
    errors: list[str] = []
    source_info: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # 1. preopt_snapshot/preopt_snapshot.npz
    # ------------------------------------------------------------------
    snap_dir = os.path.join(results_dir, "preopt_snapshot")
    snap_npz = os.path.join(snap_dir, "preopt_snapshot.npz")
    if not os.path.exists(snap_npz) or force:
        snap = load_preopt_snapshot(results_dir)
        if snap is not None:
            try:
                save_dict = {
                    k: v
                    for k, v in snap.items()
                    if k != "meta" and isinstance(v, np.ndarray)
                }
                np.savez(snap_npz, **save_dict)
                created.append(snap_npz)
                source_info[snap_npz] = {
                    "sources": [snap_dir],
                    "mtime": _mtime(snap_npz),
                }
            except Exception as exc:
                errors.append(f"preopt_snapshot.npz: {exc}")
        else:
            skipped.append(snap_npz)
    else:
        skipped.append(snap_npz)

    # ------------------------------------------------------------------
    # 2. preopt_snapshot/entity_labels.npz
    # ------------------------------------------------------------------
    labels_npz = os.path.join(snap_dir, "entity_labels.npz")
    if not os.path.exists(labels_npz) or force:
        labels = load_entity_labels(results_dir)
        non_empty = any(labels.get(k) for k in ("sites", "proteins", "kinases"))
        if non_empty:
            try:
                np.savez(
                    labels_npz,
                    sites=np.array(labels["sites"], dtype=object),
                    proteins=np.array(labels["proteins"], dtype=object),
                    kinases=np.array(labels["kinases"], dtype=object),
                    A_proteins=np.array(labels["A_proteins"], dtype=object),
                )
                created.append(labels_npz)
            except Exception as exc:
                errors.append(f"entity_labels.npz: {exc}")
        else:
            skipped.append(labels_npz)
    else:
        skipped.append(labels_npz)

    # ------------------------------------------------------------------
    # 3. dashboard_cache/fitted_timeseries.npz
    # ------------------------------------------------------------------
    ft_npz = os.path.join(cache_dir, "fitted_timeseries.npz")
    if not os.path.exists(ft_npz) or force:
        df_ft = load_fit_timeseries(results_dir)
        df_is = load_internal_states(results_dir)
        if df_ft is not None:
            try:
                _write_fitted_timeseries_npz(ft_npz, df_ft, df_is)
                created.append(ft_npz)
            except Exception as exc:
                errors.append(f"fitted_timeseries.npz: {exc}")
        else:
            skipped.append(ft_npz)
    else:
        skipped.append(ft_npz)

    # ------------------------------------------------------------------
    # 4. dashboard_cache/derived_rates.npz
    # ------------------------------------------------------------------
    dr_npz = os.path.join(cache_dir, "derived_rates.npz")
    src_dr_npz = os.path.join(results_dir, "derived_rates.npz")
    if os.path.exists(src_dr_npz):
        if not os.path.exists(dr_npz) or force:
            try:
                import shutil

                shutil.copy2(src_dr_npz, dr_npz)
                created.append(dr_npz)
            except Exception as exc:
                errors.append(f"derived_rates.npz copy: {exc}")
        else:
            skipped.append(dr_npz)
    else:
        dr = load_derived_rates(results_dir)
        if dr is not None and (not os.path.exists(dr_npz) or force):
            try:
                save_dict = {k: v for k, v in dr.items() if isinstance(v, np.ndarray)}
                np.savez(dr_npz, **save_dict)
                created.append(dr_npz)
            except Exception as exc:
                errors.append(f"derived_rates.npz reconstruct: {exc}")
        else:
            skipped.append(dr_npz)

    # ------------------------------------------------------------------
    # 5. dashboard_cache/knockout_outputs.npz
    # ------------------------------------------------------------------
    ko_npz = os.path.join(cache_dir, "knockout_outputs.npz")
    if not os.path.exists(ko_npz) or force:
        ko = load_knockout_outputs(results_dir)
        if ko:
            try:
                save_dict = {}
                for key, df in ko.items():
                    save_dict[key + "_values"] = df.values.astype(float)
                    save_dict[key + "_rows"] = np.array(list(df.index), dtype=object)
                    save_dict[key + "_cols"] = np.array(list(df.columns), dtype=object)
                np.savez(ko_npz, **save_dict)
                created.append(ko_npz)
            except Exception as exc:
                errors.append(f"knockout_outputs.npz: {exc}")
        else:
            skipped.append(ko_npz)
    else:
        skipped.append(ko_npz)

    # ------------------------------------------------------------------
    # 6. dashboard_cache/sensitivity_outputs.npz
    # ------------------------------------------------------------------
    sens_npz = os.path.join(cache_dir, "sensitivity_outputs.npz")
    if not os.path.exists(sens_npz) or force:
        sens = load_sensitivity_outputs(results_dir)
        if sens:
            try:
                sobol = sens["sobol"]
                save_dict = {
                    "sobol_values": sobol.values,
                    "sobol_columns": np.array(list(sobol.columns), dtype=object),
                }
                if "Parameter" in sobol.columns:
                    save_dict["sobol_params"] = np.array(
                        list(sobol["Parameter"]), dtype=object
                    )
                np.savez(sens_npz, **save_dict)
                created.append(sens_npz)
            except Exception as exc:
                errors.append(f"sensitivity_outputs.npz: {exc}")
        else:
            skipped.append(sens_npz)
    else:
        skipped.append(sens_npz)

    # ------------------------------------------------------------------
    # 7. dashboard_cache/steadystate_outputs.npz
    # ------------------------------------------------------------------
    ss_npz = os.path.join(cache_dir, "steadystate_outputs.npz")
    if not os.path.exists(ss_npz) or force:
        ss = load_steadystate_outputs(results_dir)
        if ss:
            try:
                save_dict = {}
                for key, df in ss.items():
                    arr = df.values.astype(float)
                    finite_arr = np.where(np.isfinite(arr), arr, np.nan)
                    save_dict[key + "_values"] = finite_arr
                    save_dict[key + "_rows"] = np.array(list(df.index), dtype=object)
                    save_dict[key + "_cols"] = np.array(list(df.columns), dtype=object)
                np.savez(ss_npz, **save_dict)
                created.append(ss_npz)
            except Exception as exc:
                errors.append(f"steadystate_outputs.npz: {exc}")
        else:
            skipped.append(ss_npz)
    else:
        skipped.append(ss_npz)

    # ------------------------------------------------------------------
    # Write manifest
    # ------------------------------------------------------------------
    manifest = {
        "schema_version": _SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "results_dir": os.path.abspath(results_dir),
        "created_files": created,
        "skipped_files": skipped,
        "errors": errors,
        "source_info": source_info,
    }
    manifest_path = os.path.join(cache_dir, "dashboard_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    return {
        "cache_dir": cache_dir,
        "created": created,
        "skipped": skipped,
        "errors": errors,
        "manifest_path": manifest_path,
    }


def load_dashboard_manifest(results_dir: str) -> dict[str, Any] | None:
    """Load the dashboard manifest JSON, returning None if absent."""
    path = os.path.join(results_dir, "dashboard_cache", "dashboard_manifest.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _write_fitted_timeseries_npz(
    out_path: str, df_ft: pd.DataFrame, df_is: pd.DataFrame | None
) -> None:
    """Convert fit_timeseries.tsv (and internal_states.tsv) to NPZ."""
    sim_cols = [c for c in df_ft.columns if c.startswith("sim_t")]
    data_cols = [c for c in df_ft.columns if c.startswith("data_t")]

    df_sites = df_ft[df_ft["Type"] == "Phosphosite"]
    df_prots = df_ft[df_ft["Type"] == "ProteinAbundance"]

    save_dict: dict[str, Any] = {}

    if not df_sites.empty:
        save_dict["P_sim"] = df_sites[sim_cols].values.astype(float)
        save_dict["P_data"] = df_sites[data_cols].values.astype(float)
        save_dict["site_labels"] = np.array(
            [f"{r['Protein']}_{r['Residue']}" for _, r in df_sites.iterrows()],
            dtype=object,
        )

    if not df_prots.empty:
        save_dict["A_sim"] = df_prots[sim_cols].values.astype(float)
        save_dict["A_data"] = df_prots[data_cols].values.astype(float)
        save_dict["protein_labels_A"] = np.array(
            list(df_prots["Protein"]), dtype=object
        )

    # time axis from column count
    save_dict["T"] = np.array([len(sim_cols)])

    if df_is is not None:
        df_S = df_is[df_is["Type"] == "S_sim"]
        df_K = df_is[df_is["Type"] == "Kdyn_sim"]
        t_cols = [c for c in df_is.columns if c.startswith("t")]
        if not df_S.empty:
            save_dict["S_sim"] = df_S[t_cols].values.astype(float)
            save_dict["S_labels"] = np.array(list(df_S["ID"]), dtype=object)
        if not df_K.empty:
            save_dict["Kdyn_sim"] = df_K[t_cols].values.astype(float)
            save_dict["Kdyn_labels"] = np.array(list(df_K["ID"]), dtype=object)

    np.savez(out_path, **save_dict)


def extract_time_axis(df: pd.DataFrame, prefix: str = "sim_t") -> np.ndarray:
    """
    Return a numeric time axis from column names like 'sim_t0', 'sim_t1', ...

    Falls back to 0-based integer indices if no numeric suffix is found.
    """
    cols = [c for c in df.columns if c.startswith(prefix)]
    try:
        return np.array([float(c.replace(prefix, "")) for c in cols])
    except ValueError:
        return np.arange(len(cols), dtype=float)


def get_time_vals(snap: dict[str, Any]) -> np.ndarray:
    """Return the original time vector from a preopt snapshot dict."""
    if snap is None:
        return np.array([])
    t = snap.get("t", np.array([]))
    return np.asarray(t, dtype=float).ravel()
