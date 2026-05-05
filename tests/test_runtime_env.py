"""
Tests for phoscrosstalk.runtime_env.

Verifies that:
- _detect_n_threads honours explicit values, SLURM_CPUS_PER_TASK,
  OMP_NUM_THREADS, and falls back to os.cpu_count().
- _merge_xla_flags correctly appends and replaces XLA flag tokens.
- setup_cpu_env sets JAX_PLATFORMS, XLA_FLAGS, and BLAS thread caps
  without overwriting values already present in the environment.
- The module is importable without triggering any JAX import.
"""

import os

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _clear_env(*names, monkeypatch):
    """Remove env vars so each test starts from a clean state."""
    for name in names:
        monkeypatch.delenv(name, raising=False)


_THREAD_ENV_VARS = [
    "SLURM_CPUS_PER_TASK",
    "OMP_NUM_THREADS",
    "JAX_PLATFORMS",
    "XLA_FLAGS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMBA_NUM_THREADS",
]


# ---------------------------------------------------------------------------
# _detect_n_threads
# ---------------------------------------------------------------------------


class TestDetectNThreads:
    def test_explicit_integer_wins(self, monkeypatch):
        from phoscrosstalk.runtime_env import _detect_n_threads

        monkeypatch.setenv("SLURM_CPUS_PER_TASK", "16")
        assert _detect_n_threads(4) == 4

    def test_slurm_used_when_no_explicit(self, monkeypatch):
        from phoscrosstalk.runtime_env import _detect_n_threads

        _clear_env("OMP_NUM_THREADS", monkeypatch=monkeypatch)
        monkeypatch.setenv("SLURM_CPUS_PER_TASK", "8")
        assert _detect_n_threads() == 8

    def test_omp_used_when_no_slurm(self, monkeypatch):
        from phoscrosstalk.runtime_env import _detect_n_threads

        _clear_env("SLURM_CPUS_PER_TASK", monkeypatch=monkeypatch)
        monkeypatch.setenv("OMP_NUM_THREADS", "6")
        assert _detect_n_threads() == 6

    def test_auto_string_triggers_detection(self, monkeypatch):
        from phoscrosstalk.runtime_env import _detect_n_threads

        _clear_env("SLURM_CPUS_PER_TASK", "OMP_NUM_THREADS", monkeypatch=monkeypatch)
        result = _detect_n_threads("auto")
        assert result >= 1

    def test_fallback_to_cpu_count(self, monkeypatch):
        from phoscrosstalk.runtime_env import _detect_n_threads

        _clear_env("SLURM_CPUS_PER_TASK", "OMP_NUM_THREADS", monkeypatch=monkeypatch)
        result = _detect_n_threads()
        expected = max(1, os.cpu_count() or 1)
        assert result == expected

    def test_invalid_slurm_falls_through(self, monkeypatch):
        from phoscrosstalk.runtime_env import _detect_n_threads

        _clear_env("OMP_NUM_THREADS", monkeypatch=monkeypatch)
        monkeypatch.setenv("SLURM_CPUS_PER_TASK", "not_a_number")
        monkeypatch.setenv("OMP_NUM_THREADS", "3")
        assert _detect_n_threads() == 3

    def test_slurm_priority_over_omp(self, monkeypatch):
        from phoscrosstalk.runtime_env import _detect_n_threads

        monkeypatch.setenv("SLURM_CPUS_PER_TASK", "12")
        monkeypatch.setenv("OMP_NUM_THREADS", "4")
        assert _detect_n_threads() == 12


# ---------------------------------------------------------------------------
# _merge_xla_flags
# ---------------------------------------------------------------------------


class TestMergeXlaFlags:
    def test_empty_existing_appends_all(self):
        from phoscrosstalk.runtime_env import _merge_xla_flags

        result = _merge_xla_flags(
            "",
            {
                "--xla_cpu_multi_thread_eigen": "true",
                "intra_op_parallelism_threads": "8",
            },
        )
        assert "--xla_cpu_multi_thread_eigen=true" in result
        assert "intra_op_parallelism_threads=8" in result

    def test_existing_flag_is_replaced(self):
        from phoscrosstalk.runtime_env import _merge_xla_flags

        existing = "--xla_cpu_multi_thread_eigen=false --some_other_flag=1"
        result = _merge_xla_flags(
            existing, {"--xla_cpu_multi_thread_eigen": "true"}
        )
        assert "--xla_cpu_multi_thread_eigen=true" in result
        assert "--xla_cpu_multi_thread_eigen=false" not in result
        assert "--some_other_flag=1" in result

    def test_intra_op_threads_updated(self):
        from phoscrosstalk.runtime_env import _merge_xla_flags

        existing = "intra_op_parallelism_threads=4"
        result = _merge_xla_flags(
            existing, {"intra_op_parallelism_threads": "16"}
        )
        assert "intra_op_parallelism_threads=16" in result
        assert "intra_op_parallelism_threads=4" not in result

    def test_no_duplicate_flags(self):
        from phoscrosstalk.runtime_env import _merge_xla_flags

        result = _merge_xla_flags(
            "--xla_cpu_multi_thread_eigen=true",
            {"--xla_cpu_multi_thread_eigen": "true"},
        )
        assert result.count("xla_cpu_multi_thread_eigen") == 1

    def test_whitespace_only_existing(self):
        from phoscrosstalk.runtime_env import _merge_xla_flags

        result = _merge_xla_flags("   ", {"--xla_cpu_multi_thread_eigen": "true"})
        assert "--xla_cpu_multi_thread_eigen=true" in result


# ---------------------------------------------------------------------------
# setup_cpu_env
# ---------------------------------------------------------------------------


class TestSetupCpuEnv:
    def test_sets_jax_platforms_default(self, monkeypatch):
        from phoscrosstalk.runtime_env import setup_cpu_env

        _clear_env(*_THREAD_ENV_VARS, monkeypatch=monkeypatch)
        setup_cpu_env(n_threads=2)
        assert os.environ["JAX_PLATFORMS"] == "cpu"

    def test_does_not_overwrite_jax_platforms(self, monkeypatch):
        from phoscrosstalk.runtime_env import setup_cpu_env

        _clear_env(*_THREAD_ENV_VARS, monkeypatch=monkeypatch)
        monkeypatch.setenv("JAX_PLATFORMS", "gpu")
        setup_cpu_env(n_threads=2)
        assert os.environ["JAX_PLATFORMS"] == "gpu"

    def test_sets_xla_flags(self, monkeypatch):
        from phoscrosstalk.runtime_env import setup_cpu_env

        _clear_env(*_THREAD_ENV_VARS, monkeypatch=monkeypatch)
        setup_cpu_env(n_threads=4)
        xla = os.environ.get("XLA_FLAGS", "")
        assert "--xla_cpu_multi_thread_eigen=true" in xla
        assert "intra_op_parallelism_threads=4" in xla

    def test_merges_into_existing_xla_flags(self, monkeypatch):
        from phoscrosstalk.runtime_env import setup_cpu_env

        _clear_env(*_THREAD_ENV_VARS, monkeypatch=monkeypatch)
        monkeypatch.setenv("XLA_FLAGS", "--some_debug_flag=1")
        setup_cpu_env(n_threads=2)
        xla = os.environ["XLA_FLAGS"]
        assert "--some_debug_flag=1" in xla
        assert "--xla_cpu_multi_thread_eigen=true" in xla

    def test_sets_blas_thread_caps(self, monkeypatch):
        from phoscrosstalk.runtime_env import setup_cpu_env

        _clear_env(*_THREAD_ENV_VARS, monkeypatch=monkeypatch)
        setup_cpu_env(n_threads=3)
        assert os.environ.get("OMP_NUM_THREADS") == "3"
        assert os.environ.get("OPENBLAS_NUM_THREADS") == "3"
        assert os.environ.get("MKL_NUM_THREADS") == "3"
        assert os.environ.get("NUMBA_NUM_THREADS") == "3"

    def test_does_not_overwrite_existing_blas_caps(self, monkeypatch):
        from phoscrosstalk.runtime_env import setup_cpu_env

        _clear_env(*_THREAD_ENV_VARS, monkeypatch=monkeypatch)
        monkeypatch.setenv("OMP_NUM_THREADS", "99")
        setup_cpu_env(n_threads=3)
        assert os.environ["OMP_NUM_THREADS"] == "99"

    def test_returns_thread_count(self, monkeypatch):
        from phoscrosstalk.runtime_env import setup_cpu_env

        _clear_env(*_THREAD_ENV_VARS, monkeypatch=monkeypatch)
        n = setup_cpu_env(n_threads=7)
        assert n == 7

    def test_auto_returns_positive_int(self, monkeypatch):
        from phoscrosstalk.runtime_env import setup_cpu_env

        _clear_env(*_THREAD_ENV_VARS, monkeypatch=monkeypatch)
        n = setup_cpu_env(n_threads="auto")
        assert isinstance(n, int) and n >= 1

    def test_slurm_env_used_on_auto(self, monkeypatch):
        from phoscrosstalk.runtime_env import setup_cpu_env

        _clear_env(*_THREAD_ENV_VARS, monkeypatch=monkeypatch)
        monkeypatch.setenv("SLURM_CPUS_PER_TASK", "16")
        n = setup_cpu_env()
        assert n == 16
        assert "intra_op_parallelism_threads=16" in os.environ.get("XLA_FLAGS", "")


# ---------------------------------------------------------------------------
# Module-level importability without JAX
# ---------------------------------------------------------------------------


def test_runtime_env_importable_without_jax(monkeypatch):
    """runtime_env must be importable using only the standard library."""
    import importlib
    import sys

    # Remove cached module so it is freshly imported
    for mod in list(sys.modules):
        if mod == "phoscrosstalk.runtime_env":
            del sys.modules[mod]

    mod = importlib.import_module("phoscrosstalk.runtime_env")
    assert hasattr(mod, "setup_cpu_env")
    assert hasattr(mod, "log_env_summary")
    assert hasattr(mod, "_detect_n_threads")
    assert hasattr(mod, "_merge_xla_flags")
