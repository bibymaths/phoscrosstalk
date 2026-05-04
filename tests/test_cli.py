"""
Tests for the CLI entry-point and argument parsing.

These tests verify that:
- The `phoscrosstalk` entry-point is importable and callable.
- Running without a valid config (missing required paths) exits non-zero.
- `--help` exits 0.
- `--version` exits 0.
- Unknown CLI flags are rejected.
- A valid config file is accepted.
"""

import sys

import pytest

from phoscrosstalk.main import cli, main


def test_cli_is_callable():
    assert callable(cli)


def test_main_missing_required_args(monkeypatch):
    """Running main() with a config that lacks required fields must exit non-zero."""
    monkeypatch.setattr(sys, "argv", ["phoscrosstalk"])
    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code != 0


def test_main_help_exits_zero(monkeypatch):
    """--help should print usage and exit 0."""
    monkeypatch.setattr(sys, "argv", ["phoscrosstalk", "--help"])
    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 0


def test_main_version_exits_zero(monkeypatch):
    """--version should print version and exit 0."""
    monkeypatch.setattr(sys, "argv", ["phoscrosstalk", "--version"])
    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 0


def test_main_unknown_flag_rejected(monkeypatch):
    """Unknown CLI flags should be rejected (argparse exit 2)."""
    monkeypatch.setattr(sys, "argv", ["phoscrosstalk", "--data", "fake.csv"])
    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 2, (
        "Unknown flag --data should be rejected with exit code 2"
    )


def test_main_config_flag_accepted(monkeypatch, tmp_path):
    """--config with an existing file should be accepted (proceed past argparse)."""
    # Write a minimal config that will fail validation (missing required paths)
    # but must not fail at argparse level (exit 2)
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text("")  # empty config → validation fails with exit 1

    monkeypatch.setattr(sys, "argv", ["phoscrosstalk", "--config", str(cfg_path)])
    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code != 2, (
        "--config flag should be accepted by argparse (exit 2 means it was rejected)"
    )
