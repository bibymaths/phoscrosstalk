"""
tests/test_bundle_analysis.py
Tests for phoscrosstalk.bundle_analysis — plotting/inspection utilities for
PINN and neuralODE model bundles.

All tests are self-contained: they build minimal fake bundle directories in
``tmp_path`` without requiring the full pipeline, equinox, or JAX.
"""

from __future__ import annotations

import csv
import json
import pathlib

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers to build minimal fake bundle directories
# ---------------------------------------------------------------------------

def _write_pinn_bundle(root: pathlib.Path, K: int = 3, M: int = 2, N: int = 4) -> None:
    """Create a minimal PINN bundle inside *root*."""
    bundle = root / "pinn_bundle"
    bundle.mkdir(parents=True, exist_ok=True)

    # Structural metadata
    meta = {
        "K": K, "M": M, "N": N,
        "state_dim": 3 * K + M + N,
        "width_size": 32, "depth": 2,
        "activation": "tanh", "output_clamp": 0.1,
        "theta_dim": 10, "bundle_format_version": 1,
    }
    (bundle / "pinn_bundle_meta.json").write_text(json.dumps(meta, indent=2))

    # theta_opt
    np.save(str(bundle / "theta_opt.npy"), np.random.randn(10))

    # pinn_metadata.json (in parent = run_dir)
    run_meta = {"K": K, "M": M, "N": N, "f1": 0.12, "f2": 0.08, "f3": 0.05, "f4": 0.02}
    (root / "pinn_metadata.json").write_text(json.dumps(run_meta, indent=2))

    # pinn_loss_components.tsv
    loss_tsv = root / "pinn_loss_components.tsv"
    with open(loss_tsv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["f1","f2","f3","f4","f_pinn_reg"], delimiter="\t")
        writer.writeheader()
        writer.writerow({"f1": 0.12, "f2": 0.08, "f3": 0.05, "f4": 0.02, "f_pinn_reg": 0.001})

    # pinn_fit_timeseries.tsv
    ts_path = root / "pinn_fit_timeseries.tsv"
    t_vals = [0.0, 1.0, 2.0, 3.0]
    sites = ["pA_S100", "pB_S200"]
    with open(ts_path, "w", newline="") as fh:
        writer = csv.DictWriter(
            fh, delimiter="\t",
            fieldnames=["entity_type","entity","site","protein","time","value_sim","value_obs","series_type"],
        )
        writer.writeheader()
        for site in sites:
            for t in t_vals:
                writer.writerow({
                    "entity_type": "phosphosite", "entity": site, "site": site,
                    "protein": site.split("_")[0], "time": t,
                    "value_sim": float(np.random.rand()),
                    "value_obs": float(np.random.rand()),
                    "series_type": "pinn_fitted",
                })


def _write_neuralode_bundle(
    root: pathlib.Path, K: int = 3, M: int = 2, N: int = 4
) -> None:
    """Create a minimal neuralODE bundle inside *root*."""
    bundle = root / "neural_ode_bundle"
    bundle.mkdir(parents=True, exist_ok=True)

    meta = {
        "K": K, "width": 8, "depth": 1,
        "in_size": 1 + 2 * K, "learn_theta": False,
        "theta_dim": 10, "bundle_format_version": 1,
    }
    (bundle / "neural_ode_bundle_meta.json").write_text(json.dumps(meta, indent=2))

    np.save(str(bundle / "theta_refined.npy"), np.random.randn(10))

    # Run-level metadata
    (root / "neural_metadata.json").write_text(json.dumps({
        "K": K, "M": M, "N": N,
        "initial_loss": 1.23, "final_loss": 0.45,
        "width": 8, "depth": 1,
    }))

    # Training losses TSV
    loss_path = root / "neural_training_losses.tsv"
    cols = ["step","neural_loss_total","neural_loss_phospho","neural_loss_abundance",
            "neural_loss_mrna","neural_loss_k_act_prior","neural_loss_s_prod_prior"]
    with open(loss_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=cols, delimiter="\t")
        writer.writeheader()
        for step in range(0, 30, 10):
            writer.writerow({
                "step": step,
                "neural_loss_total": 1.0 - step * 0.01,
                "neural_loss_phospho": 0.5,
                "neural_loss_abundance": 0.3,
                "neural_loss_mrna": 0.1,
                "neural_loss_k_act_prior": 0.05,
                "neural_loss_s_prod_prior": 0.05,
            })

    # Fit timeseries TSV
    ts_path = root / "neural_fit_timeseries.tsv"
    sites = ["pA_S100", "pB_S200"]
    t_vals = [0.0, 1.0, 2.0]
    with open(ts_path, "w", newline="") as fh:
        writer = csv.DictWriter(
            fh, delimiter="\t",
            fieldnames=["entity_type","entity","time","value_neural","value_observed"],
        )
        writer.writeheader()
        for site in sites:
            for t in t_vals:
                writer.writerow({
                    "entity_type": "phosphosite", "entity": site, "time": t,
                    "value_neural": float(np.random.rand()),
                    "value_observed": float(np.random.rand()),
                })

    # Latent rates TSV
    rates_path = root / "neural_latent_rates.tsv"
    proteins = [f"PROT{i}" for i in range(K)]
    t_vals_r = [0.0, 1.0]
    with open(rates_path, "w", newline="") as fh:
        writer = csv.DictWriter(
            fh, delimiter="\t",
            fieldnames=["rate_type","entity","time","mechanistic_prior","neural_learned"],
        )
        writer.writeheader()
        for rt in ("k_act", "s_prod"):
            for prot in proteins:
                for t in t_vals_r:
                    writer.writerow({
                        "rate_type": rt, "entity": prot, "time": t,
                        "mechanistic_prior": float(np.random.rand()),
                        "neural_learned": float(np.random.rand()),
                    })


# ===========================================================================
# Tests for private helpers
# ===========================================================================

class TestPrivateHelpers:
    """Unit tests for _read_json, _read_csv_if_exists, etc."""

    def test_read_json_existing(self, tmp_path):
        from phoscrosstalk.bundle_analysis import _read_json
        p = tmp_path / "meta.json"
        p.write_text('{"key": 42}')
        assert _read_json(p) == {"key": 42}

    def test_read_json_missing(self, tmp_path):
        from phoscrosstalk.bundle_analysis import _read_json
        assert _read_json(tmp_path / "nonexistent.json") is None

    def test_read_csv_if_exists_tsv(self, tmp_path):
        from phoscrosstalk.bundle_analysis import _read_csv_if_exists
        p = tmp_path / "data.tsv"
        p.write_text("a\tb\n1\t2\n3\t4\n")
        rows = _read_csv_if_exists(p)
        assert rows is not None
        assert len(rows) == 2
        # _coerce_numeric converts "1" → int(1)
        assert rows[0]["a"] == 1 and isinstance(rows[0]["a"], int)

    def test_read_csv_if_exists_missing(self, tmp_path):
        from phoscrosstalk.bundle_analysis import _read_csv_if_exists
        assert _read_csv_if_exists(tmp_path / "missing.tsv") is None

    def test_load_npz_if_exists(self, tmp_path):
        from phoscrosstalk.bundle_analysis import _load_npz_if_exists
        arr = np.array([1.0, 2.0, 3.0])
        p = tmp_path / "data.npz"
        np.savez(str(p), arr=arr)
        data = _load_npz_if_exists(p)
        assert data is not None
        np.testing.assert_array_equal(data["arr"], arr)

    def test_load_npz_if_exists_missing(self, tmp_path):
        from phoscrosstalk.bundle_analysis import _load_npz_if_exists
        assert _load_npz_if_exists(tmp_path / "missing.npz") is None

    def test_ensure_dir(self, tmp_path):
        from phoscrosstalk.bundle_analysis import _ensure_dir
        d = tmp_path / "a" / "b" / "c"
        result = _ensure_dir(d)
        assert result.is_dir()
        assert result == d

    def test_get_labels_from_json(self):
        from phoscrosstalk.bundle_analysis import _get_labels
        lj = {"proteins": ["ProtA", "ProtB", "ProtC"]}
        assert _get_labels(lj, "proteins", 5) == ["ProtA", "ProtB", "ProtC"]

    def test_get_labels_fallback(self):
        from phoscrosstalk.bundle_analysis import _get_labels
        assert _get_labels(None, "proteins", 3) == ["proteins_0", "proteins_1", "proteins_2"]

    def test_safe_array_valid(self):
        from phoscrosstalk.bundle_analysis import _safe_array
        arr = _safe_array([1.0, 2.0, 3.0])
        assert arr is not None
        assert arr.dtype == float

    def test_safe_array_none(self):
        from phoscrosstalk.bundle_analysis import _safe_array
        assert _safe_array(None) is None

    def test_safe_array_empty(self):
        from phoscrosstalk.bundle_analysis import _safe_array
        assert _safe_array([]) is None

    def test_infer_state_slices(self):
        from phoscrosstalk.bundle_analysis import _infer_state_slices
        meta = {"K": 3, "M": 2, "N": 4}
        slices = _infer_state_slices(meta)
        assert "mRNA (R)"        in slices
        assert "Phosphosite (P)" in slices
        assert slices["mRNA (R)"].start == 0
        assert slices["mRNA (R)"].stop  == 3

    def test_infer_state_slices_no_K(self):
        from phoscrosstalk.bundle_analysis import _infer_state_slices
        assert _infer_state_slices({}) == {}

    def test_savefig_creates_both_formats(self, tmp_path):
        from phoscrosstalk.bundle_analysis import _savefig
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3])
        _savefig(fig, tmp_path, "test_fig")
        assert (tmp_path / "test_fig.png").is_file()
        assert (tmp_path / "test_fig.pdf").is_file()


# ===========================================================================
# Tests for plot_pinn_bundle_analysis
# ===========================================================================

class TestPlotPinnBundleAnalysis:
    """Tests for plot_pinn_bundle_analysis."""

    def test_creates_output_dir(self, tmp_path):
        """Output directory is created if it does not exist."""
        from phoscrosstalk.bundle_analysis import plot_pinn_bundle_analysis
        _write_pinn_bundle(tmp_path)
        out = tmp_path / "custom_plots"
        plot_pinn_bundle_analysis(tmp_path / "pinn_bundle", out)
        assert out.is_dir()

    def test_default_output_dir_is_plots_subdir(self, tmp_path):
        """When output_dir is None, plots go into model_dir/plots/."""
        from phoscrosstalk.bundle_analysis import plot_pinn_bundle_analysis
        _write_pinn_bundle(tmp_path)
        plot_pinn_bundle_analysis(tmp_path / "pinn_bundle")
        assert (tmp_path / "pinn_bundle" / "plots").is_dir()

    def test_produces_metadata_plot(self, tmp_path):
        """pinn_bundle_metadata.png is created."""
        from phoscrosstalk.bundle_analysis import plot_pinn_bundle_analysis
        _write_pinn_bundle(tmp_path)
        out = tmp_path / "plots"
        plot_pinn_bundle_analysis(tmp_path / "pinn_bundle", out)
        assert (out / "pinn_bundle_metadata.png").is_file()

    def test_produces_loss_plot(self, tmp_path):
        """pinn_loss_components.png is created."""
        from phoscrosstalk.bundle_analysis import plot_pinn_bundle_analysis
        _write_pinn_bundle(tmp_path)
        out = tmp_path / "plots"
        plot_pinn_bundle_analysis(tmp_path / "pinn_bundle", out)
        assert (out / "pinn_loss_components.png").is_file()

    def test_produces_theta_plot(self, tmp_path):
        """pinn_theta_vector.png is created."""
        from phoscrosstalk.bundle_analysis import plot_pinn_bundle_analysis
        _write_pinn_bundle(tmp_path)
        out = tmp_path / "plots"
        plot_pinn_bundle_analysis(tmp_path / "pinn_bundle", out)
        assert (out / "pinn_theta_vector.png").is_file()

    def test_produces_timeseries_plot(self, tmp_path):
        """pinn_timeseries_<etype>.png is created for each entity type."""
        from phoscrosstalk.bundle_analysis import plot_pinn_bundle_analysis
        _write_pinn_bundle(tmp_path)
        out = tmp_path / "plots"
        plot_pinn_bundle_analysis(tmp_path / "pinn_bundle", out)
        png_files = list(out.glob("pinn_timeseries_*.png"))
        assert len(png_files) >= 1

    def test_does_not_raise_with_minimal_bundle(self, tmp_path):
        """Function succeeds even with only the meta JSON (no other files)."""
        from phoscrosstalk.bundle_analysis import plot_pinn_bundle_analysis
        bundle = tmp_path / "pinn_bundle"
        bundle.mkdir()
        meta = {"K": 2, "M": 1, "N": 2, "bundle_format_version": 1}
        (bundle / "pinn_bundle_meta.json").write_text(json.dumps(meta))
        # Must not raise even though most optional files are absent.
        plot_pinn_bundle_analysis(bundle)

    def test_all_plots_also_saved_as_pdf(self, tmp_path):
        """Every .png produced also has a matching .pdf."""
        from phoscrosstalk.bundle_analysis import plot_pinn_bundle_analysis
        _write_pinn_bundle(tmp_path)
        out = tmp_path / "plots"
        plot_pinn_bundle_analysis(tmp_path / "pinn_bundle", out)
        pngs = list(out.glob("*.png"))
        assert len(pngs) > 0
        for png in pngs:
            assert (out / png.stem).with_suffix(".pdf").is_file(), \
                f"No PDF for {png.name}"


# ===========================================================================
# Tests for plot_neuralode_bundle_analysis
# ===========================================================================

class TestPlotNeuralodeBundleAnalysis:
    """Tests for plot_neuralode_bundle_analysis."""

    def test_creates_output_dir(self, tmp_path):
        from phoscrosstalk.bundle_analysis import plot_neuralode_bundle_analysis
        _write_neuralode_bundle(tmp_path)
        out = tmp_path / "node_plots"
        plot_neuralode_bundle_analysis(tmp_path / "neural_ode_bundle", out)
        assert out.is_dir()

    def test_default_output_dir_is_plots_subdir(self, tmp_path):
        from phoscrosstalk.bundle_analysis import plot_neuralode_bundle_analysis
        _write_neuralode_bundle(tmp_path)
        plot_neuralode_bundle_analysis(tmp_path / "neural_ode_bundle")
        assert (tmp_path / "neural_ode_bundle" / "plots").is_dir()

    def test_produces_metadata_plot(self, tmp_path):
        from phoscrosstalk.bundle_analysis import plot_neuralode_bundle_analysis
        _write_neuralode_bundle(tmp_path)
        out = tmp_path / "plots"
        plot_neuralode_bundle_analysis(tmp_path / "neural_ode_bundle", out)
        assert (out / "neuralode_bundle_metadata.png").is_file()

    def test_produces_loss_history_plot(self, tmp_path):
        from phoscrosstalk.bundle_analysis import plot_neuralode_bundle_analysis
        _write_neuralode_bundle(tmp_path)
        out = tmp_path / "plots"
        plot_neuralode_bundle_analysis(tmp_path / "neural_ode_bundle", out)
        assert (out / "neuralode_training_loss.png").is_file()

    def test_produces_theta_plot(self, tmp_path):
        from phoscrosstalk.bundle_analysis import plot_neuralode_bundle_analysis
        _write_neuralode_bundle(tmp_path)
        out = tmp_path / "plots"
        plot_neuralode_bundle_analysis(tmp_path / "neural_ode_bundle", out)
        assert (out / "neuralode_theta_refined.png").is_file()

    def test_produces_latent_rates_plot(self, tmp_path):
        from phoscrosstalk.bundle_analysis import plot_neuralode_bundle_analysis
        _write_neuralode_bundle(tmp_path)
        out = tmp_path / "plots"
        plot_neuralode_bundle_analysis(tmp_path / "neural_ode_bundle", out)
        rate_pngs = list(out.glob("neuralode_latent_*.png"))
        assert len(rate_pngs) >= 1

    def test_produces_timeseries_plot(self, tmp_path):
        from phoscrosstalk.bundle_analysis import plot_neuralode_bundle_analysis
        _write_neuralode_bundle(tmp_path)
        out = tmp_path / "plots"
        plot_neuralode_bundle_analysis(tmp_path / "neural_ode_bundle", out)
        ts_pngs = list(out.glob("neuralode_timeseries_*.png"))
        assert len(ts_pngs) >= 1

    def test_does_not_raise_with_minimal_bundle(self, tmp_path):
        from phoscrosstalk.bundle_analysis import plot_neuralode_bundle_analysis
        bundle = tmp_path / "neural_ode_bundle"
        bundle.mkdir()
        meta = {"K": 2, "width": 4, "depth": 1, "in_size": 5, "bundle_format_version": 1}
        (bundle / "neural_ode_bundle_meta.json").write_text(json.dumps(meta))
        plot_neuralode_bundle_analysis(bundle)

    def test_all_plots_also_saved_as_pdf(self, tmp_path):
        from phoscrosstalk.bundle_analysis import plot_neuralode_bundle_analysis
        _write_neuralode_bundle(tmp_path)
        out = tmp_path / "plots"
        plot_neuralode_bundle_analysis(tmp_path / "neural_ode_bundle", out)
        pngs = list(out.glob("*.png"))
        assert len(pngs) > 0
        for png in pngs:
            assert (out / png.stem).with_suffix(".pdf").is_file(), \
                f"No PDF for {png.name}"


# ===========================================================================
# Tests for plot_model_comparison
# ===========================================================================

class TestPlotModelComparison:
    """Tests for plot_model_comparison."""

    def test_creates_output_dir(self, tmp_path):
        from phoscrosstalk.bundle_analysis import plot_model_comparison
        pinn_root = tmp_path / "pinn_run"
        node_root = tmp_path / "node_run"
        pinn_root.mkdir()
        node_root.mkdir()
        _write_pinn_bundle(pinn_root)
        _write_neuralode_bundle(node_root)
        out = tmp_path / "comparison"
        plot_model_comparison(
            pinn_root / "pinn_bundle",
            node_root / "neural_ode_bundle",
            out,
        )
        assert out.is_dir()

    def test_produces_theta_comparison_plot(self, tmp_path):
        from phoscrosstalk.bundle_analysis import plot_model_comparison
        pinn_root = tmp_path / "pinn_run"
        node_root = tmp_path / "node_run"
        pinn_root.mkdir()
        node_root.mkdir()
        _write_pinn_bundle(pinn_root)
        _write_neuralode_bundle(node_root)
        out = tmp_path / "comparison"
        plot_model_comparison(pinn_root / "pinn_bundle", node_root / "neural_ode_bundle", out)
        assert (out / "comparison_theta.png").is_file()

    def test_produces_loss_comparison_plot(self, tmp_path):
        from phoscrosstalk.bundle_analysis import plot_model_comparison
        pinn_root = tmp_path / "pinn_run"
        node_root = tmp_path / "node_run"
        pinn_root.mkdir()
        node_root.mkdir()
        _write_pinn_bundle(pinn_root)
        _write_neuralode_bundle(node_root)
        out = tmp_path / "comparison"
        plot_model_comparison(pinn_root / "pinn_bundle", node_root / "neural_ode_bundle", out)
        assert (out / "comparison_loss.png").is_file()

    def test_produces_timeseries_comparison_plot(self, tmp_path):
        from phoscrosstalk.bundle_analysis import plot_model_comparison
        pinn_root = tmp_path / "pinn_run"
        node_root = tmp_path / "node_run"
        pinn_root.mkdir()
        node_root.mkdir()
        _write_pinn_bundle(pinn_root)
        _write_neuralode_bundle(node_root)
        out = tmp_path / "comparison"
        plot_model_comparison(pinn_root / "pinn_bundle", node_root / "neural_ode_bundle", out)
        ts_pngs = list(out.glob("comparison_timeseries_*.png"))
        assert len(ts_pngs) >= 1

    def test_never_mixes_pinn_and_neuralode_files(self, tmp_path):
        """Comparison output must never overwrite PINN-specific or neuralODE-specific plots."""
        from phoscrosstalk.bundle_analysis import plot_model_comparison
        pinn_root = tmp_path / "pinn_run"
        node_root = tmp_path / "node_run"
        pinn_root.mkdir()
        node_root.mkdir()
        _write_pinn_bundle(pinn_root)
        _write_neuralode_bundle(node_root)
        out = tmp_path / "comparison"
        plot_model_comparison(pinn_root / "pinn_bundle", node_root / "neural_ode_bundle", out)
        # Output files must have "comparison_" prefix (not pinn_ or neuralode_ prefix)
        for f in out.glob("*.png"):
            assert f.name.startswith("comparison_"), \
                f"Unexpected non-comparison file: {f.name}"

    def test_does_not_raise_with_empty_bundles(self, tmp_path):
        """Function must not raise when both bundles have no optional files."""
        from phoscrosstalk.bundle_analysis import plot_model_comparison
        pinn_bundle = tmp_path / "pinn_run" / "pinn_bundle"
        node_bundle = tmp_path / "node_run" / "neural_ode_bundle"
        pinn_bundle.mkdir(parents=True)
        node_bundle.mkdir(parents=True)
        (pinn_bundle / "pinn_bundle_meta.json").write_text(
            json.dumps({"K": 2, "M": 1, "N": 2, "bundle_format_version": 1})
        )
        (node_bundle / "neural_ode_bundle_meta.json").write_text(
            json.dumps({"K": 2, "width": 4, "depth": 1, "in_size": 5, "bundle_format_version": 1})
        )
        out = tmp_path / "comparison"
        plot_model_comparison(pinn_bundle, node_bundle, out)

    def test_all_comparison_plots_also_saved_as_pdf(self, tmp_path):
        from phoscrosstalk.bundle_analysis import plot_model_comparison
        pinn_root = tmp_path / "pinn_run"
        node_root = tmp_path / "node_run"
        pinn_root.mkdir()
        node_root.mkdir()
        _write_pinn_bundle(pinn_root)
        _write_neuralode_bundle(node_root)
        out = tmp_path / "comparison"
        plot_model_comparison(pinn_root / "pinn_bundle", node_root / "neural_ode_bundle", out)
        pngs = list(out.glob("*.png"))
        assert len(pngs) > 0
        for png in pngs:
            assert (out / png.stem).with_suffix(".pdf").is_file(), \
                f"No PDF for {png.name}"


# ===========================================================================
# Isolation: PINN analysis must not touch neuralODE output names, and vice versa
# ===========================================================================

class TestBundleIsolation:
    """Verify that PINN and neuralODE analysis functions produce distinct outputs."""

    def test_pinn_analysis_does_not_produce_neuralode_files(self, tmp_path):
        from phoscrosstalk.bundle_analysis import plot_pinn_bundle_analysis
        _write_pinn_bundle(tmp_path)
        out = tmp_path / "plots"
        plot_pinn_bundle_analysis(tmp_path / "pinn_bundle", out)
        for f in out.glob("*.png"):
            assert not f.name.startswith("neuralode_"), \
                f"PINN analysis wrote neuralODE-named file: {f.name}"

    def test_neuralode_analysis_does_not_produce_pinn_files(self, tmp_path):
        from phoscrosstalk.bundle_analysis import plot_neuralode_bundle_analysis
        _write_neuralode_bundle(tmp_path)
        out = tmp_path / "plots"
        plot_neuralode_bundle_analysis(tmp_path / "neural_ode_bundle", out)
        for f in out.glob("*.png"):
            # pinn_* prefix must never appear
            assert not f.name.startswith("pinn_"), \
                f"neuralODE analysis wrote PINN-named file: {f.name}"
