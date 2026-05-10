"""
Lightweight run-management helpers for the PhosCrosstalk Streamlit dashboard.

Each run is tracked via a small JSON file:
    runs/<run_id>/run_metadata.json

Run IDs are ISO-8601 timestamps (``YYYY-MM-DD_HH-MM-SS``).

Schema
------
{
    "run_id":       "2026-05-08_19-30-00",
    "command":      ["phoscrosstalk", "--config", "..."],
    "config_path":  "<absolute path to config.toml>",
    "output_dir":   "<absolute path to results dir>",
    "log_path":     "<absolute path to phoscrosstalk.log>",
    "status":       "running" | "completed" | "failed" | "stopped",
    "started_at":   "2026-05-08T19:30:00",
    "ended_at":     null | "2026-05-08T19:45:00",
    "return_code":  null | 0 | 1 | ...
}
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_METADATA_FILENAME = "run_metadata.json"
_RUNS_BASE_DEFAULT = "runs"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _make_run_id() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def _metadata_path(run_dir: Path) -> Path:
    return Path(run_dir) / _METADATA_FILENAME


# ---------------------------------------------------------------------------
# Create / update
# ---------------------------------------------------------------------------


def create_run(
        runs_base_dir: str | Path,
        config_path: Path,
        output_dir: Path,
        command: list[str],
        log_filename: str = "phoscrosstalk.log",
) -> tuple[str, Path]:
    """
    Create a new run directory and write initial metadata.

    Args:
        runs_base_dir: Parent directory that holds all runs (e.g. ``runs/``).
        config_path:   Path to the ``config.toml`` that will be used.
        output_dir:    Directory where pipeline results will be written.
        command:       Full command list (from :func:`pipeline_runner.build_command`).
        log_filename:  Log file name inside the run directory.

    Returns:
        Tuple of (run_id, run_dir).
    """
    runs_base_dir = Path(runs_base_dir)
    run_id = _make_run_id()
    run_dir = runs_base_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    log_path = run_dir / log_filename

    metadata: dict[str, Any] = {
        "run_id": run_id,
        "command": [str(c) for c in command],
        "config_path": str(config_path.resolve()),
        "output_dir": str(Path(output_dir).resolve()),
        "log_path": str(log_path.resolve()),
        "status": "running",
        "started_at": _now_iso(),
        "ended_at": None,
        "return_code": None,
    }

    _write_metadata(run_dir, metadata)
    return run_id, run_dir


def update_run(
        run_dir: Path,
        **fields: Any,
) -> dict[str, Any]:
    """
    Update one or more fields in the run metadata JSON.

    Args:
        run_dir: Run directory containing ``run_metadata.json``.
        **fields: Key-value pairs to update.

    Returns:
        The updated metadata dict.
    """
    meta = load_run_metadata(run_dir) or {}
    meta.update(fields)
    _write_metadata(run_dir, meta)
    return meta


def mark_run_ended(run_dir: Path, return_code: int | None, status: str) -> None:
    """
    Convenience wrapper: set ``ended_at``, ``return_code``, and ``status``.

    Args:
        run_dir:     Run directory.
        return_code: Process exit code (or None if killed).
        status:      ``"completed"``, ``"failed"``, or ``"stopped"``.
    """
    update_run(
        run_dir,
        ended_at=_now_iso(),
        return_code=return_code,
        status=status,
    )


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------


def load_run_metadata(run_dir: Path) -> dict[str, Any] | None:
    """
    Load ``run_metadata.json`` from *run_dir*.

    Returns None if the file is missing or unreadable.
    """
    path = _metadata_path(Path(run_dir))
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None


def list_runs(runs_base_dir: str | Path = _RUNS_BASE_DEFAULT) -> list[dict[str, Any]]:
    """
    Return a list of run metadata dicts, sorted newest-first.

    Silently skips run directories that are missing or have corrupt metadata.

    Args:
        runs_base_dir: Parent directory holding all run subdirectories.

    Returns:
        List of metadata dicts, most recent first.
    """
    base = Path(runs_base_dir)
    if not base.exists():
        return []

    runs: list[dict[str, Any]] = []
    for entry in sorted(base.iterdir(), reverse=True):
        if not entry.is_dir():
            continue
        meta = load_run_metadata(entry)
        if meta is not None:
            runs.append(meta)

    return runs


def get_latest_run(runs_base_dir: str | Path = _RUNS_BASE_DEFAULT) -> dict[str, Any] | None:
    """Return the most recent run metadata, or None."""
    runs = list_runs(runs_base_dir)
    return runs[0] if runs else None


# ---------------------------------------------------------------------------
# Internal write helper
# ---------------------------------------------------------------------------


def _write_metadata(run_dir: Path, metadata: dict[str, Any]) -> None:
    path = _metadata_path(Path(run_dir))
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)
