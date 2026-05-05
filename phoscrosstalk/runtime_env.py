"""
runtime_env.py
Configure XLA/JAX CPU threading environment variables.

This module MUST be imported (and ``setup_cpu_env`` called) before any JAX,
jaxlib, Diffrax, Optimistix, Equinox, Lineax, or any project module that
imports them.  Once JAX/jaxlib is imported the values of ``JAX_PLATFORMS`` and
``XLA_FLAGS`` are frozen by the XLA runtime and cannot be changed.

Priority order for the CPU-thread count (legacy ``_detect_n_threads``):
  1. Explicit ``n_threads`` argument (usually from ``config.toml``
     ``[runtime] cpu_threads``).
  2. ``SLURM_CPUS_PER_TASK`` environment variable.
  3. ``OMP_NUM_THREADS`` environment variable.
  4. ``os.cpu_count()`` (physical + logical threads visible to the process).

For richer topology-aware planning use ``detect_cpu_topology()`` and
``plan_cpu_runtime()``.

BLAS/OpenMP libraries (OpenBLAS, MKL, Accelerate, Numba) are capped to the
same value via ``*_NUM_THREADS`` variables so they do not over-subscribe the
CPU affinity mask allocated by SLURM.
"""

import os
import sys
from types import SimpleNamespace


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
# CPU topology detection and parallelism planning
# ---------------------------------------------------------------------------


def detect_cpu_topology() -> SimpleNamespace:
    """
    Detect available CPU resources using multiple detection methods.

    Detection priority for ``total_available``:
      1. ``SLURM_CPUS_PER_TASK``  – scheduler allocation (most authoritative)
      2. ``os.sched_getaffinity(0)`` – Linux affinity mask (cgroups/container aware)
      3. ``psutil.cpu_count(logical=False)`` – physical cores (if psutil installed)
      4. ``os.cpu_count()``  – OS reported logical CPUs

    Returns
    -------
    SimpleNamespace with fields:
      logical_cpus    : int        – total logical CPUs visible to the OS
      physical_cores  : int|None   – physical cores (None if not detectable)
      affinity_cpus   : int|None   – CPUs in process affinity mask (Linux only)
      slurm_cpus      : int|None   – SLURM_CPUS_PER_TASK if set and valid
      total_available : int        – recommended usable CPU count
      source          : str        – which detection method determined total_available
    """
    logical_cpus = os.cpu_count() or 1
    physical_cores = None
    affinity_cpus = None

    # Try CPU affinity mask (most accurate on Linux/cgroups/SLURM without CPUS_PER_TASK)
    try:
        aff = os.sched_getaffinity(0)  # type: ignore[attr-defined]
        affinity_cpus = len(aff)
    except (AttributeError, OSError):
        pass

    # Try psutil for physical core count
    try:
        import psutil  # optional dependency

        pc = psutil.cpu_count(logical=False)
        if pc is not None and pc > 0:
            physical_cores = pc
        lc = psutil.cpu_count(logical=True)
        if lc is not None and lc > 0:
            logical_cpus = lc
    except ImportError:
        pass

    # SLURM allocation
    slurm_cpus = None
    slurm_str = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm_str is not None:
        try:
            v = int(slurm_str)
            if v > 0:
                slurm_cpus = v
        except ValueError:
            pass

    # Determine total_available and source
    if slurm_cpus is not None:
        total_available = slurm_cpus
        source = "SLURM_CPUS_PER_TASK"
    elif affinity_cpus is not None:
        total_available = affinity_cpus
        source = "os.sched_getaffinity"
    elif physical_cores is not None:
        # If logical CPUs ≥ 2× physical cores, hyperthreading is likely active.
        # Use physical cores as the conservative default for compute-heavy workloads.
        if logical_cpus >= 2 * physical_cores:
            total_available = physical_cores
            source = "psutil.cpu_count(logical=False) [HT detected]"
        else:
            total_available = logical_cpus
            source = "psutil.cpu_count(logical=True)"
    else:
        total_available = logical_cpus
        source = "os.cpu_count()"

    return SimpleNamespace(
        logical_cpus=logical_cpus,
        physical_cores=physical_cores,
        affinity_cpus=affinity_cpus,
        slurm_cpus=slurm_cpus,
        total_available=total_available,
        source=source,
    )


def plan_cpu_runtime(
    cpu_threads="auto",
    n_starts=1,
    parallel_starts="auto",
    threads_per_start="auto",
    reserve_cores=0,
    use_physical_cores=True,
    default_threads_per_run=4,
) -> SimpleNamespace:
    """
    Calculate a safe CPU parallelism plan for multi-start optimisation.

    Avoids oversubscription by dividing the total CPU budget among parallel
    workers.  When ``n_starts == 1`` the entire budget is allocated to that
    single run.  When ``n_starts > 1`` the budget is divided conservatively.

    Parameters
    ----------
    cpu_threads : int | str
        Total CPU budget override.  ``"auto"`` → detected from environment.
    n_starts : int
        Number of optimisation starts (used to decide parallelism).
    parallel_starts : int | str
        Number of starts to run in parallel.  ``"auto"`` → calculated.
    threads_per_start : int | str
        Threads allocated to each parallel start.  ``"auto"`` → calculated.
    reserve_cores : int
        CPUs to reserve for the OS/parent process.  Default 0.
    use_physical_cores : bool
        When True and SLURM is not active, prefer physical cores over logical
        CPUs as the usable budget (avoids hyperthreading oversubscription).
    default_threads_per_run : int
        Conservative default per-run thread count for JAX/ODE workloads.
        Used when neither ``parallel_starts`` nor ``threads_per_start`` is set.

    Returns
    -------
    SimpleNamespace with fields:
      topo                 : SimpleNamespace  – from detect_cpu_topology()
      total_available_cpus : int
      n_parallel_runs      : int
      threads_per_run      : int
      xla_threads          : int
      blas_threads         : int
      omp_threads          : int
    """
    topo = detect_cpu_topology()
    n_starts = max(1, int(n_starts))

    # --- Determine total CPU budget ---
    if cpu_threads is not None and str(cpu_threads).lower() not in ("auto", "", "0"):
        try:
            budget = max(1, int(cpu_threads))
        except (ValueError, TypeError):
            budget = topo.total_available
    else:
        budget = topo.total_available

    # Apply reservation
    budget = max(1, budget - max(0, int(reserve_cores)))

    # Usable cores for parallelism decisions
    # When SLURM is active its CPU count is already a hard limit; use it as-is.
    if use_physical_cores and topo.physical_cores is not None and topo.slurm_cpus is None:
        usable = min(budget, topo.physical_cores)
    else:
        usable = budget
    usable = max(1, usable)

    # --- Parse explicit overrides ---
    def _parse_int(val, name):
        if val is not None and str(val).lower() not in ("auto", "", "0"):
            try:
                return max(1, int(val))
            except (ValueError, TypeError):
                pass
        return None

    n_parallel_explicit = _parse_int(parallel_starts, "parallel_starts")
    tpr_explicit = _parse_int(threads_per_start, "threads_per_start")

    # --- Single-start case: give everything to the one run ---
    if n_starts == 1:
        n_parallel = n_parallel_explicit or 1
        tpr = tpr_explicit or usable
    else:
        # Multi-start: balance parallelism vs threads per run
        if n_parallel_explicit is None and tpr_explicit is None:
            # Neither specified → conservative default
            n_parallel = min(n_starts, max(1, usable // default_threads_per_run))
            tpr = max(1, usable // n_parallel)
        elif n_parallel_explicit is None:
            # threads_per_start given → derive n_parallel
            n_parallel = min(n_starts, max(1, usable // tpr_explicit))
            tpr = tpr_explicit
        elif tpr_explicit is None:
            # parallel_starts given → derive threads_per_run
            n_parallel = n_parallel_explicit
            tpr = max(1, usable // n_parallel)
        else:
            # Both explicitly set → use as-is
            n_parallel = n_parallel_explicit
            tpr = tpr_explicit

    n_parallel = max(1, min(n_parallel, n_starts))
    tpr = max(1, tpr)

    return SimpleNamespace(
        topo=topo,
        total_available_cpus=budget,
        n_parallel_runs=n_parallel,
        threads_per_run=tpr,
        xla_threads=tpr,
        blas_threads=tpr,
        omp_threads=tpr,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def apply_cpu_env(n_threads: int, overwrite: bool = True) -> None:
    """
    Apply CPU thread-count settings to environment variables.

    Unlike ``setup_cpu_env``, this function can be called in worker processes
    to enforce per-run thread caps after the parent process has already
    configured its own environment.

    Parameters
    ----------
    n_threads : int
        Number of threads to configure.
    overwrite : bool
        When ``True`` (default) overwrite any existing values.
        When ``False`` use ``os.environ.setdefault`` (preserve existing).

    Note
    ----
    XLA/JAX intra-op thread settings are **always merged** (not blindly
    replaced) so that other XLA flags set externally are preserved.
    If JAX has already been imported, the ``XLA_FLAGS`` / ``JAX_PLATFORMS``
    changes will have no effect on the running XLA runtime.
    """
    n_str = str(int(n_threads))

    # JAX platform – always setdefault to avoid overriding GPU configurations
    os.environ.setdefault("JAX_PLATFORMS", "cpu")

    # XLA threading flags – always merge so existing flags are preserved
    xla_additions = {
        "--xla_cpu_multi_thread_eigen": "true",
        "intra_op_parallelism_threads": n_str,
    }
    existing_xla = os.environ.get("XLA_FLAGS", "")
    os.environ["XLA_FLAGS"] = _merge_xla_flags(existing_xla, xla_additions)

    # BLAS / OpenMP / Numba caps
    _thread_vars = [
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMBA_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ]
    if overwrite:
        for var in _thread_vars:
            os.environ[var] = n_str
    else:
        for var in _thread_vars:
            os.environ.setdefault(var, n_str)


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
    Sets (via ``os.environ.setdefault``, preserving any externally set values):
      - ``JAX_PLATFORMS``            → ``"cpu"``
      - ``XLA_FLAGS``                → adds/updates threading flags (always merged)
      - ``OMP_NUM_THREADS``          → ``str(n_threads)``
      - ``OPENBLAS_NUM_THREADS``     → ``str(n_threads)``
      - ``MKL_NUM_THREADS``          → ``str(n_threads)``
      - ``VECLIB_MAXIMUM_THREADS``   → ``str(n_threads)``
      - ``NUMBA_NUM_THREADS``        → ``str(n_threads)``
      - ``NUMEXPR_NUM_THREADS``      → ``str(n_threads)``
    """
    n = _detect_n_threads(n_threads)
    apply_cpu_env(n, overwrite=False)
    return n


def log_env_summary(logger=None, plan=None) -> None:
    """
    Log a summary of the active CPU/XLA environment variables.

    Parameters
    ----------
    logger : logging.Logger | None
        A logger instance.  If ``None``, output is written to ``sys.stderr``.
    plan : SimpleNamespace | None
        Optional result from ``plan_cpu_runtime()``.  When provided, the
        runtime CPU plan is included in the summary.
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

    if plan is not None:
        topo = plan.topo
        lines += [
            "[runtime_env] Runtime CPU plan:",
            f"  SLURM active             = {topo.slurm_cpus is not None}",
            f"  logical CPUs             = {topo.logical_cpus}",
            f"  physical cores           = {topo.physical_cores if topo.physical_cores is not None else 'unknown'}",
            f"  affinity CPUs            = {topo.affinity_cpus if topo.affinity_cpus is not None else 'n/a'}",
            f"  usable CPUs (budget)     = {plan.total_available_cpus}",
            f"  CPU source               = {topo.source}",
            f"  n_starts                 = (see optimisation config)",
            f"  parallel starts          = {plan.n_parallel_runs}",
            f"  threads per start        = {plan.threads_per_run}",
            f"  XLA intra-op threads     = {plan.xla_threads}",
            f"  BLAS/OpenMP threads      = {plan.blas_threads}",
        ]

    message = "\n".join(lines)
    if logger is not None:
        logger.info(message)
    else:
        print(message, file=sys.stderr, flush=True)
