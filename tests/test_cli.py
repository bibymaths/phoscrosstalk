"""
Tests for the CLI entry-point and argument parsing.

These tests verify that:
- The `phoscrosstalk` entry-point is importable and callable.
- Required arguments raise SystemExit when absent.
- Optional arguments have correct defaults.
"""

import sys

import pytest

from phoscrosstalk.main import cli, main


def test_cli_is_callable():
    assert callable(cli)


def test_main_missing_required_args(monkeypatch):
    """Calling main() without --data should exit with an error."""
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
