"""
runtime_env.py
Configure XLA/JAX CPU threading environment variables.

This module MUST be imported (and ``setup_cpu_env`` called) before any JAX,
jaxlib, Diffrax, Optimistix, Equinox, Lineax, or any project module that
imports them.  Once JAX/jaxlib is imported the values of ``JAX_PLATFORMS`` and
``XLA_FLAGS`` are frozen by the XLA runtime and cannot be changed.

Priority order for the CPU-thread count:
  1. Explicit ``n_threads`` argument (usually from ``config.toml``
     ``[runtime] cpu_threads``).
  2. ``SLURM_CPUS_PER_TASK`` environment variable.
  3. ``OMP_NUM_THREADS`` environment variable.
  4. ``os.cpu_count()`` (physical + logical threads visible to the process).

BLAS/OpenMP libraries (OpenBLAS, MKL, Accelerate, Numba) are capped to the
same value via ``*_NUM_THREADS`` variables so they do not over-subscribe the
CPU affinity mask allocated by SLURM.
"""

import os
import sys


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _detect_n_threads(cfg_threads=None) -> int:
    """
    Return the number of CPU threads to configure.

    Parameters
    ----------
    cfg_threads : int | str | None
        Value from ``config.toml [runtime] cpu_threads``.
        ``None``, ``"auto"``, or ``0`` triggers environment-variable detection.
    """
    # Config override (must be a positive integer)
    if cfg_threads is not None and str(cfg_threads).lower() not in ("auto", "", "0"):
        try:
            n = int(cfg_threads)
            if n > 0:
                return n
        except (ValueError, TypeError):
            pass

    # SLURM allocation
    slurm = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm is not None:
        try:
            n = int(slurm)
            if n > 0:
                return n
        except ValueError:
            pass

    # Existing OMP setting (respect user-set parallelism)
    omp = os.environ.get("OMP_NUM_THREADS")
    if omp is not None:
        try:
            n = int(omp)
            if n > 0:
                return n
        except ValueError:
            pass

    # Fallback: all CPUs visible to this process
    return max(1, os.cpu_count() or 1)


def _merge_xla_flags(existing: str, additions: dict) -> str:
    """
    Merge *additions* into the *existing* ``XLA_FLAGS`` string.

    Each key in *additions* is a flag name that may carry a leading ``--``
    prefix (e.g. ``"--xla_cpu_multi_thread_eigen"``) or no prefix (e.g.
    ``"intra_op_parallelism_threads"``).  The corresponding value is a string.

    An existing token whose normalised name matches is replaced in-place; new
    tokens are appended.  The leading ``--`` (or its absence) of the *existing*
    token is preserved when replacing, and the format of *additions* is used
    when appending.

    Parameters
    ----------
    existing : str
        Current value of ``XLA_FLAGS`` (may be empty).
    additions : dict[str, str]
        Flags to add/update.  Keys are flag names; values are string values.

    Returns
    -------
    str
        Updated ``XLA_FLAGS`` string.
    """
    tokens = existing.split() if existing.strip() else []

    # Map normalised name → index in tokens list
    idx_map: dict[str, int] = {}
    for i, tok in enumerate(tokens):
        norm = tok.lstrip("-").split("=")[0]
        idx_map[norm] = i

    for flag_key, flag_val in additions.items():
        # Determine the canonical form the flag should take
        stripped = flag_key.lstrip("-")
        norm_name = stripped.split("=")[0]

        # Preserve leading "--" if the original flag_key carries it; otherwise none
        prefix = "--" if flag_key.startswith("-") else ""
        token = f"{prefix}{norm_name}={flag_val}"

        if norm_name in idx_map:
            # Replace existing token, preserving its own "--" prefix
            old_tok = tokens[idx_map[norm_name]]
            old_prefix = "--" if old_tok.startswith("-") else ""
            tokens[idx_map[norm_name]] = f"{old_prefix}{norm_name}={flag_val}"
        else:
            tokens.append(token)

    return " ".join(tokens)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def setup_cpu_env(n_threads=None) -> int:
    """
    Set XLA/JAX CPU threading environment variables.

    Call this function **before** importing JAX, jaxlib, Diffrax, Optimistix,
    Equinox, Lineax, or any project module that imports them.

    Parameters
    ----------
    n_threads : int | str | None
        Number of CPU threads.  ``None`` or ``"auto"`` triggers auto-detection
        (SLURM_CPUS_PER_TASK → OMP_NUM_THREADS → os.cpu_count).

    Returns
    -------
    int
        The thread count that was configured.

    Side effects
    ------------
    Sets (via ``os.environ``):
      - ``JAX_PLATFORMS``            → ``"cpu"``  (only if not already set)
      - ``XLA_FLAGS``                → adds/updates threading flags
      - ``OMP_NUM_THREADS``          → ``str(n_threads)``  (only if not set)
      - ``OPENBLAS_NUM_THREADS``     → ``str(n_threads)``  (only if not set)
      - ``MKL_NUM_THREADS``          → ``str(n_threads)``  (only if not set)
      - ``VECLIB_MAXIMUM_THREADS``   → ``str(n_threads)``  (only if not set)
      - ``NUMBA_NUM_THREADS``        → ``str(n_threads)``  (only if not set)
    """
    n = _detect_n_threads(n_threads)

    # --- JAX/XLA platform ---
    os.environ.setdefault("JAX_PLATFORMS", "cpu")

    # --- XLA CPU threading flags ---
    # --xla_cpu_multi_thread_eigen=true  enables multi-threaded Eigen kernels
    # intra_op_parallelism_threads=N     caps XLA intra-op thread pool size
    xla_additions = {
        "--xla_cpu_multi_thread_eigen": "true",
        "intra_op_parallelism_threads": str(n),
    }
    existing_xla = os.environ.get("XLA_FLAGS", "")
    os.environ["XLA_FLAGS"] = _merge_xla_flags(existing_xla, xla_additions)

    # --- BLAS / OpenMP / Numba caps (avoid CPU oversubscription) ---
    n_str = str(n)
    os.environ.setdefault("OMP_NUM_THREADS", n_str)
    os.environ.setdefault("OPENBLAS_NUM_THREADS", n_str)
    os.environ.setdefault("MKL_NUM_THREADS", n_str)
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", n_str)
    os.environ.setdefault("NUMBA_NUM_THREADS", n_str)

    return n


def log_env_summary(logger=None) -> None:
    """
    Log a summary of the active CPU/XLA environment variables.

    Parameters
    ----------
    logger : logging.Logger | None
        A logger instance.  If ``None``, output is written to ``sys.stderr``.
    """
    lines = [
        "[runtime_env] Active CPU/XLA environment:",
        f"  JAX_PLATFORMS            = {os.environ.get('JAX_PLATFORMS', '(not set)')}",
        f"  XLA_FLAGS                = {os.environ.get('XLA_FLAGS', '(not set)')}",
        f"  OMP_NUM_THREADS          = {os.environ.get('OMP_NUM_THREADS', '(not set)')}",
        f"  OPENBLAS_NUM_THREADS     = {os.environ.get('OPENBLAS_NUM_THREADS', '(not set)')}",
        f"  MKL_NUM_THREADS          = {os.environ.get('MKL_NUM_THREADS', '(not set)')}",
        f"  NUMBA_NUM_THREADS        = {os.environ.get('NUMBA_NUM_THREADS', '(not set)')}",
        f"  SLURM_CPUS_PER_TASK      = {os.environ.get('SLURM_CPUS_PER_TASK', '(not set)')}",
        f"  SLURM_JOB_ID             = {os.environ.get('SLURM_JOB_ID', '(not set)')}",
    ]
    message = "\n".join(lines)
    if logger is not None:
        logger.info(message)
    else:
        print(message, file=sys.stderr, flush=True)
