"""
pipeline_runner.py
Utilities for launching, monitoring, and stopping the PhosCrosstalk pipeline
from inside the Streamlit dashboard.

Design principles:
- Never use shell=True; always pass a command list.
- Store process PID in a file so it survives Streamlit reruns.
- Stream stdout+stderr to a persistent log file.
- Support graceful terminate → kill on stop.
- Cross-platform (os.killpg used only on POSIX).
"""

from __future__ import annotations

import os
import platform
import shutil
import signal
import subprocess
import time
from pathlib import Path


# ---------------------------------------------------------------------------
# Process helpers
# ---------------------------------------------------------------------------


def find_phoscrosstalk_executable() -> str | None:
    """Return the absolute path to the *phoscrosstalk* executable, or None."""
    return shutil.which("phoscrosstalk")


def _validate_user_path(path: Path, label: str = "path") -> None:
    """
    Raise ``ValueError`` if *path* points to a suspicious system location.

    This is a lightweight guard for the local dashboard use-case.  It does
    NOT try to be a comprehensive sandbox; it just prevents accidental
    misuse of obvious system paths.
    """
    path_str = str(path)
    _BLOCKED_PREFIXES = ("/etc/", "/bin/", "/sbin/", "/usr/bin/", "/usr/sbin/",
                         "/proc/", "/sys/", "/dev/", "/boot/")
    for prefix in _BLOCKED_PREFIXES:
        if path_str.startswith(prefix):
            raise ValueError(
                f"{label} resolves to a restricted system path: {path_str}"
            )


def build_command(config_path: Path) -> list[str]:
    """
    Build the command list for running the pipeline.

    The command is always passed as a list (``shell=False``), so the config
    path is never interpreted by a shell.  Only the known ``phoscrosstalk``
    executable is invoked — no arbitrary user commands are accepted here.

    Args:
        config_path: Path to ``config.toml``.

    Returns:
        A list suitable for :func:`subprocess.Popen` with ``shell=False``.

    Raises:
        FileNotFoundError: if the ``phoscrosstalk`` executable is not on PATH.
        ValueError: if *config_path* resolves to a suspicious system path.
    """
    exe = find_phoscrosstalk_executable()
    if exe is None:
        raise FileNotFoundError(
            "Could not find `phoscrosstalk` on PATH. "
            "Install the package or activate the correct Python environment."
        )

    # Resolve and do a basic sanity check on the config path.
    resolved = Path(config_path).resolve()
    _validate_user_path(resolved, label="config_path")

    # Return a strict list so subprocess never invokes a shell.
    return [exe, "--config", str(resolved)]


# ---------------------------------------------------------------------------
# Process launch / stop
# ---------------------------------------------------------------------------


def start_pipeline(
    cmd: list[str],
    run_dir: Path,
    log_filename: str = "phoscrosstalk.log",
) -> subprocess.Popen:
    """
    Start the pipeline as a subprocess.

    Stdout and stderr are merged and written to *run_dir/log_filename*.
    A PID file is written to *run_dir/pipeline.pid*.

    Args:
        cmd:          Command list (from :func:`build_command`).
        run_dir:      Directory where the run artefacts live.
        log_filename: Name of the log file inside *run_dir*.

    Returns:
        The running :class:`subprocess.Popen` object.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    log_path = run_dir / log_filename
    log_fh = open(log_path, "w", encoding="utf-8", buffering=1)  # line-buffered

    kwargs: dict = {
        "stdout": log_fh,
        "stderr": subprocess.STDOUT,
        "text": True,
        "bufsize": 1,
        # Never use shell=True — we always pass a validated command list.
        "shell": False,
    }

    # On POSIX create a new process group so we can kill the whole tree.
    if platform.system() != "Windows":
        kwargs["preexec_fn"] = os.setsid

    try:
        proc = subprocess.Popen(cmd, **kwargs)
    except Exception:
        log_fh.close()
        raise

    # Attach the file handle so stop_pipeline can close it on cleanup.
    proc._dashboard_log_fh = log_fh  # type: ignore[attr-defined]

    # Write PID file for rerun recovery
    pid_path = run_dir / "pipeline.pid"
    pid_path.write_text(str(proc.pid), encoding="utf-8")

    return proc


def stop_pipeline(
    process: subprocess.Popen | None,
    run_dir: Path | None = None,
    timeout: float = 10.0,
) -> None:
    """
    Gracefully stop a running pipeline process.

    1. Try SIGTERM (or terminate() on Windows).
    2. Wait up to *timeout* seconds.
    3. If still alive, SIGKILL (or kill() on Windows).

    Cleans up the PID file if *run_dir* is provided.

    Args:
        process:  The :class:`subprocess.Popen` object to stop.
        run_dir:  Run directory; used to clean up the PID file.
        timeout:  Seconds to wait after SIGTERM before escalating.
    """
    if process is None:
        return

    try:
        if process.poll() is None:
            if platform.system() != "Windows":
                try:
                    pgid = os.getpgid(process.pid)
                    os.killpg(pgid, signal.SIGTERM)
                except (ProcessLookupError, PermissionError):
                    process.terminate()
            else:
                process.terminate()

            try:
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                if platform.system() != "Windows":
                    try:
                        pgid = os.getpgid(process.pid)
                        os.killpg(pgid, signal.SIGKILL)
                    except (ProcessLookupError, PermissionError):
                        process.kill()
                else:
                    process.kill()
                process.wait()
    except (ProcessLookupError, PermissionError, OSError):
        pass  # Process already gone

    # Close the log file handle attached by start_pipeline, if present.
    log_fh = getattr(process, "_dashboard_log_fh", None)
    if log_fh is not None:
        try:
            log_fh.close()
        except OSError:
            pass

    if run_dir is not None:
        pid_path = Path(run_dir) / "pipeline.pid"
        pid_path.unlink(missing_ok=True)


def is_process_alive(pid: int) -> bool:
    """
    Return True if a process with *pid* is currently running.

    Uses ``os.kill(pid, 0)`` which sends no signal but checks existence.
    """
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def read_pid_file(run_dir: Path) -> int | None:
    """Read the PID stored in *run_dir/pipeline.pid*, or return None."""
    pid_path = Path(run_dir) / "pipeline.pid"
    if not pid_path.exists():
        return None
    try:
        return int(pid_path.read_text(encoding="utf-8").strip())
    except (ValueError, OSError):
        return None


# ---------------------------------------------------------------------------
# Log helpers
# ---------------------------------------------------------------------------


def read_log_tail(log_path: Path, n_lines: int = 500) -> str:
    """
    Return the last *n_lines* lines from *log_path*.

    Returns an empty string if the file does not exist.
    """
    log_path = Path(log_path)
    if not log_path.exists():
        return ""
    try:
        with open(log_path, encoding="utf-8", errors="replace") as fh:
            lines = fh.readlines()
        return "".join(lines[-n_lines:])
    except OSError:
        return ""


def get_log_path(run_dir: Path, log_filename: str = "phoscrosstalk.log") -> Path:
    """Return the expected log file path for a run directory."""
    return Path(run_dir) / log_filename


def append_log(run_dir: Path, message: str, log_filename: str = "phoscrosstalk.log") -> None:
    """Append a plain-text *message* to the run log (e.g. stop events)."""
    log_path = Path(run_dir) / log_filename
    try:
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(f"\n[dashboard] {time.strftime('%Y-%m-%dT%H:%M:%S')} {message}\n")
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Console rendering helper
# ---------------------------------------------------------------------------


def render_console(log_text: str, max_lines: int = 500) -> str:
    """
    Escape *log_text* for safe embedding in an HTML block.

    Returns the HTML string.  Callers should use ``st.markdown(..., unsafe_allow_html=True)``.
    """
    import html as _html

    lines = log_text.splitlines()
    if len(lines) > max_lines:
        lines = lines[-max_lines:]
    escaped = _html.escape("\n".join(lines))
    return (
        '<div style="'
        "background-color:#0d1117;"
        "color:#c9d1d9;"
        "padding:1rem;"
        "border-radius:0.5rem;"
        "font-family:ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,monospace;"
        "font-size:0.82rem;"
        "max-height:520px;"
        "overflow-y:auto;"
        "white-space:pre-wrap;"
        'word-break:break-all;">'
        f"{escaped}"
        "</div>"
    )
