#!/usr/bin/env python3
"""
compare_pinn_neuralode_bundles.py

Standalone PINN vs neuralODE bundle comparison script.

Expected layout:

    <outdir>/
        pinn_bundle/
        neural_ode/
            neural_ode_bundle/

Outputs:

    <outdir>/bundle_analysis/comparison/
"""

from __future__ import annotations

import argparse
import pathlib

from phoscrosstalk.bundle_analysis import plot_model_comparison
from phoscrosstalk.logger import configure_logger


def compare_pinn_neuralode_bundles(
    outdir: str | pathlib.Path,
    *,
    output_dir: str | pathlib.Path | None = None,
    strict: bool = False,
) -> int:
    """
    Compare saved PINN and neuralODE model bundles.

    Args:
        outdir:
            Top-level PhosCrosstalk output directory.
        output_dir:
            Optional output directory for comparison plots.
            Defaults to <outdir>/bundle_analysis/comparison.
        strict:
            If True, missing bundle directories raise FileNotFoundError.
            If False, missing bundles are logged and the script exits cleanly.

    Returns:
        Process-style exit code:
            0 = success or skipped non-strict
            1 = failed
    """
    outdir = pathlib.Path(outdir).resolve()

    pinn_bundle_dir = outdir / "pinn_bundle"
    neural_bundle_dir = outdir / "neural_ode" / "neural_ode_bundle"
    comparison_outdir = (
        pathlib.Path(output_dir).resolve()
        if output_dir is not None
        else outdir / "bundle_analysis" / "comparison"
    )

    logger.info("[bundle_analysis] Output directory: %s", outdir)
    logger.info("[bundle_analysis] PINN bundle: %s", pinn_bundle_dir)
    logger.info("[bundle_analysis] neuralODE bundle: %s", neural_bundle_dir)
    logger.info("[bundle_analysis] Comparison output: %s", comparison_outdir)

    pinn_exists = pinn_bundle_dir.is_dir()
    neural_exists = neural_bundle_dir.is_dir()

    if not pinn_exists or not neural_exists:
        msg = (
            "[bundle_analysis] Skipping PINN-vs-neuralODE comparison. "
            f"PINN bundle exists={pinn_exists}; neuralODE bundle exists={neural_exists}"
        )

        if strict:
            raise FileNotFoundError(msg)

        logger.info(msg)
        return 0

    comparison_outdir.mkdir(parents=True, exist_ok=True)

    try:
        plot_model_comparison(
            pinn_dir=pinn_bundle_dir,
            neuralode_dir=neural_bundle_dir,
            output_dir=comparison_outdir,
        )

        logger.success(
            "[bundle_analysis] PINN-vs-neuralODE comparison saved to %s",
            comparison_outdir,
        )
        return 0

    except Exception as exc:
        logger.warning(
            "[bundle_analysis] Final model comparison failed: %s",
            exc,
        )
        return 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="compare-pinn-neuralode-bundles",
        description=(
            "Compare saved PINN and neuralODE bundles from a PhosCrosstalk output directory."
        ),
    )

    parser.add_argument(
        "outdir",
        help="Top-level PhosCrosstalk output directory.",
    )

    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Directory for comparison plots. "
            "Default: <outdir>/bundle_analysis/comparison"
        ),
    )

    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail with an error if either bundle directory is missing.",
    )

    parser.add_argument(
        "--log-file",
        default=None,
        help="Optional log file path. Default: <outdir>/bundle_analysis/comparison.log",
    )

    return parser.parse_args()


if __name__ == "__main__":
    cli_args = parse_args()

    _outdir = pathlib.Path(cli_args.outdir).resolve()
    _default_log = _outdir / "bundle_analysis" / "comparison.log"
    _log_file = cli_args.log_file or str(_default_log)

    _default_log.parent.mkdir(parents=True, exist_ok=True)

    logger = configure_logger(_log_file, timestamp=True)

    try:
        raise SystemExit(
            compare_pinn_neuralode_bundles(
                cli_args.outdir,
                output_dir=cli_args.output_dir,
                strict=cli_args.strict,
            )
        )
    except Exception as exc:
        logger.error("[bundle_analysis] Comparison script failed: %s", exc)
        raise SystemExit(1) from exc