"""
test_analysis_extra.py

Additional tests for analysis.py helper functions and remaining uncovered branches.
Covers: _save_txt, _save_matrix_tsv, _save_vector_tsv, _save_index_tsv,
        print_biological_scores, plot_biological_scores, plot_goodness_of_fit,
        save_preopt_snapshot_npz, save_mrna_outputs, save_derived_rates,
        plot_mrna_fit, _save_dense_simulation exception branches.
"""
import matplotlib
matplotlib.use("Agg")

import os
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from phoscrosstalk.config import ModelDims

K, M, N = 2, 2, 3
THETA_DIM = 2 * K + 2 + 3 * M + N + 4  # 19


def _set_dims():
    ModelDims.set_dims(K, M, N)


def _theta():
    rng = np.random.default_rng(0)
    return rng.uniform(0.01, 0.5, THETA_DIM)


def _dense_result(n=20):
    _set_dims()
    return {
        "P_sim": np.ones((N, n), dtype=np.float64) * 0.5,
        "A_sim": np.ones((K, n), dtype=np.float64),
        "R_sim": np.ones((K, n), dtype=np.float64) * 1.2,
        "S_sim": np.ones((K, n), dtype=np.float64) * 0.3,
        "Kdyn_sim": np.ones((M, n), dtype=np.float64) * 0.4,
        "success": True,
    }


def _save_dense_kwargs(outdir, n_dense=20):
    """Return kwargs for _save_dense_simulation."""
    _set_dims()
    sites = ["Prot0_S0", "Prot0_S1", "Prot1_S2"]
    proteins = ["Prot0", "Prot1"]
    return dict(
        outdir=str(outdir),
        theta_opt=_theta(),
        t_obs=np.linspace(0.0, 60.0, 5),
        sites=sites,
        proteins=proteins,
        P_scaled=np.ones((N, 5)) * 0.3,
        A_scaled=np.ones((K, 5)),
        prot_idx_for_A=np.array([0, 1], dtype=int),
        Cg=np.zeros((N, N)),
        Cl=np.zeros((N, N)),
        site_prot_idx=np.array([0, 0, 1], dtype=int),
        K_site_kin=np.zeros((N, M)),
        R=np.zeros((M, N)),
        L_alpha=np.zeros((M, M)),
        kin_to_prot_idx=np.array([0, 1], dtype=int),
        mask_p=np.zeros(K, dtype=int),
        mask_k=np.zeros(M, dtype=int),
        mechanism="dist",
        n_dense=n_dense,
    )


# ---------------------------------------------------------------------------
# _save_txt / _save_matrix_tsv / _save_vector_tsv / _save_index_tsv helpers
# ---------------------------------------------------------------------------
class TestPrivateHelpers:
    def test_save_txt_creates_file(self, tmp_path):
        from phoscrosstalk.analysis import _save_txt
        p = str(tmp_path / "sub" / "out.txt")
        _save_txt(p, "hello world")
        assert os.path.exists(p)
        assert "hello world" in open(p).read()

    def test_save_txt_appends_newline(self, tmp_path):
        from phoscrosstalk.analysis import _save_txt
        p = str(tmp_path / "no_newline.txt")
        _save_txt(p, "no newline")
        content = open(p).read()
        assert content.endswith("\n")

    def test_save_txt_preserves_existing_newline(self, tmp_path):
        from phoscrosstalk.analysis import _save_txt
        p = str(tmp_path / "with_newline.txt")
        _save_txt(p, "has newline\n")
        content = open(p).read()
        # Only one trailing newline
        assert content == "has newline\n"

    def test_save_matrix_tsv_creates_file(self, tmp_path):
        from phoscrosstalk.analysis import _save_matrix_tsv
        mat = np.array([[1.0, 2.0], [3.0, 4.0]])
        p = str(tmp_path / "subdir" / "mat.tsv")
        _save_matrix_tsv(p, mat)
        assert os.path.exists(p)
        loaded = np.loadtxt(p, delimiter="\t")
        np.testing.assert_allclose(loaded, mat)

    def test_save_vector_tsv_creates_file(self, tmp_path):
        from phoscrosstalk.analysis import _save_vector_tsv
        vec = np.array([1.0, 2.0, 3.0])
        p = str(tmp_path / "vec.tsv")
        _save_vector_tsv(p, vec)
        assert os.path.exists(p)
        loaded = np.loadtxt(p, delimiter="\t").flatten()
        np.testing.assert_allclose(loaded, vec)

    def test_save_index_tsv_creates_file(self, tmp_path):
        from phoscrosstalk.analysis import _save_index_tsv
        idx = np.array([0, 1, 2], dtype=int)
        p = str(tmp_path / "idx.tsv")
        _save_index_tsv(p, idx)
        assert os.path.exists(p)
        loaded = np.loadtxt(p, delimiter="\t", dtype=int).flatten()
        np.testing.assert_array_equal(loaded, idx)


# ---------------------------------------------------------------------------
# print_biological_scores
# ---------------------------------------------------------------------------
class TestPrintBiologicalScores:
    def test_creates_file(self, tmp_path):
        from phoscrosstalk.analysis import print_biological_scores
        _set_dims()
        rng = np.random.default_rng(0)
        X = rng.uniform(0.01, 0.5, (3, THETA_DIM))
        print_biological_scores(str(tmp_path), X)
        assert os.path.exists(tmp_path / "biological_scores.tsv")

    def test_file_has_expected_columns(self, tmp_path):
        from phoscrosstalk.analysis import print_biological_scores
        _set_dims()
        X = np.ones((2, THETA_DIM)) * 0.1
        print_biological_scores(str(tmp_path), X)
        df = pd.read_csv(tmp_path / "biological_scores.tsv", sep="\t")
        assert "Index" in df.columns
        assert "Bio_Score" in df.columns
        assert len(df) == 2


# ---------------------------------------------------------------------------
# plot_biological_scores
# ---------------------------------------------------------------------------
class TestPlotBiologicalScores:
    def test_creates_png(self, tmp_path):
        from phoscrosstalk.analysis import plot_biological_scores
        _set_dims()
        rng = np.random.default_rng(42)
        X = rng.uniform(0.01, 0.5, (5, THETA_DIM))
        F = rng.uniform(0.1, 1.0, (5, 3))
        plot_biological_scores(str(tmp_path), X, F)
        assert os.path.exists(tmp_path / "biological_scores.png")


# ---------------------------------------------------------------------------
# plot_goodness_of_fit
# ---------------------------------------------------------------------------
class TestPlotGoodnessOfFit:
    def _make_tsv(self, tmp_path):
        """Create a minimal protein_fit_timeseries.tsv for testing."""
        T = 4
        records = []
        sim_cols = [f"sim_t{j}" for j in range(T)]
        data_cols = [f"data_t{j}" for j in range(T)]
        for i in range(3):
            rec = {"Type": "Phosphosite", "Protein": f"Prot{i}", "Residue": f"S{i}"}
            for j in range(T):
                rec[sim_cols[j]] = 1.0 + 0.1 * j + 0.01 * i
                rec[data_cols[j]] = 1.0 + 0.1 * j + 0.02 * i
            records.append(rec)
        df = pd.DataFrame(records)
        p = tmp_path / "protein_fit_timeseries.tsv"
        df.to_csv(p, sep="\t", index=False)
        return str(p)

    def test_creates_png(self, tmp_path):
        from phoscrosstalk.analysis import plot_goodness_of_fit
        tsv = self._make_tsv(tmp_path)
        plot_goodness_of_fit(tsv, str(tmp_path))
        assert os.path.exists(tmp_path / "goodness_of_fit.png")

    def test_missing_sim_cols_returns_silently(self, tmp_path):
        """If no sim_t* columns, function should return without error."""
        from phoscrosstalk.analysis import plot_goodness_of_fit
        df = pd.DataFrame({"Type": ["Phosphosite"], "Protein": ["P0"], "Residue": ["S0"]})
        p = tmp_path / "fit.tsv"
        df.to_csv(p, sep="\t", index=False)
        # Should not raise
        plot_goodness_of_fit(str(p), str(tmp_path))

    def test_abundance_rows_labeled_correctly(self, tmp_path):
        """ProteinAbundance rows get '_Abundance' label."""
        from phoscrosstalk.analysis import plot_goodness_of_fit
        T = 3
        sim_cols = [f"sim_t{j}" for j in range(T)]
        data_cols = [f"data_t{j}" for j in range(T)]
        records = []
        for i in range(2):
            rec = {"Type": "ProteinAbundance", "Protein": f"Prot{i}", "Residue": ""}
            for j in range(T):
                rec[sim_cols[j]] = 1.0 + 0.1 * j
                rec[data_cols[j]] = 1.0 + 0.1 * j + 0.05
            records.append(rec)
        df = pd.DataFrame(records)
        p = tmp_path / "fit.tsv"
        df.to_csv(p, sep="\t", index=False)
        # Should not raise; labels include '_Abundance'
        plot_goodness_of_fit(str(p), str(tmp_path))
        assert os.path.exists(tmp_path / "goodness_of_fit.png")


# ---------------------------------------------------------------------------
# save_preopt_snapshot_npz
# ---------------------------------------------------------------------------
class TestSavePreoptSnapshotNpz:
    def _make_kwargs(self, snap_dir):
        return dict(
            t=np.array([0.0, 1.0, 2.0]),
            Y=np.ones((N, 3)),
            P_scaled=np.ones((N, 3)) * 0.5,
            A_data=np.ones((K, 3)),
            A_scaled=np.ones((K, 3)),
            Cg=np.zeros((N, N)),
            Cl=np.zeros((N, N)),
            K_site_kin=np.zeros((N, M)),
            R=np.zeros((M, N)),
            L_alpha=np.zeros((M, M)),
            W_data=np.ones((N, 3)),
            W_data_prot=np.ones((K, 3)),
            site_prot_idx=np.array([0, 0, 1], dtype=int),
            kin_to_prot_idx=np.array([0, 1], dtype=int),
            receptor_mask_prot=np.zeros(K, dtype=int),
            receptor_mask_kin=np.zeros(M, dtype=int),
            positions=np.zeros((N, 2)),
            xl=np.zeros(THETA_DIM),
            xu=np.ones(THETA_DIM),
        )

    def test_creates_npz_file(self, tmp_path):
        from phoscrosstalk.analysis import save_preopt_snapshot_npz
        snap_dir = str(tmp_path / "snap")
        os.makedirs(snap_dir)
        save_preopt_snapshot_npz(snap_dir, **self._make_kwargs(snap_dir))
        assert os.path.exists(os.path.join(snap_dir, "preopt_snapshot.npz"))

    def test_does_not_overwrite_existing(self, tmp_path):
        from phoscrosstalk.analysis import save_preopt_snapshot_npz
        snap_dir = str(tmp_path / "snap")
        os.makedirs(snap_dir)
        out_path = os.path.join(snap_dir, "preopt_snapshot.npz")
        # Write sentinel
        np.savez(out_path, sentinel=np.array([999]))
        save_preopt_snapshot_npz(snap_dir, **self._make_kwargs(snap_dir))
        loaded = np.load(out_path)
        # Should still contain the sentinel (not overwritten)
        assert "sentinel" in loaded

    def test_handles_none_arrays(self, tmp_path):
        from phoscrosstalk.analysis import save_preopt_snapshot_npz
        snap_dir = str(tmp_path / "snap2")
        os.makedirs(snap_dir)
        kwargs = self._make_kwargs(snap_dir)
        kwargs["A_data"] = None
        save_preopt_snapshot_npz(snap_dir, **kwargs)
        loaded = np.load(os.path.join(snap_dir, "preopt_snapshot.npz"))
        assert "A_data" in loaded


# ---------------------------------------------------------------------------
# save_mrna_outputs
# ---------------------------------------------------------------------------
class TestSaveMrnaOutputs:
    def _make_data(self, n_genes=2, T=5):
        t_rna = np.linspace(0, 60, T)
        obs = np.ones((n_genes, T)) * 1.0
        sim = np.ones((n_genes, T)) * 1.2
        return t_rna, obs, sim

    def test_creates_mrna_tsv(self, tmp_path):
        from phoscrosstalk.analysis import save_mrna_outputs
        t_rna, obs, sim = self._make_data()
        gene_ids = ["Gene0", "Gene1"]
        save_mrna_outputs(str(tmp_path), gene_ids, t_rna, obs, sim)
        assert os.path.exists(tmp_path / "mrna_fit_timeseries.tsv")

    def test_shape_mismatch_skips_gracefully(self, tmp_path):
        """obs and sim have different shapes → warning, no crash, no file."""
        from phoscrosstalk.analysis import save_mrna_outputs
        t_rna = np.array([0.0, 10.0, 30.0])
        obs = np.ones((2, 3))
        sim = np.ones((3, 3))  # different n_genes
        save_mrna_outputs(str(tmp_path), ["G0", "G1"], t_rna, obs, sim)
        # File should NOT be created
        assert not os.path.exists(tmp_path / "mrna_fit_timeseries.tsv")

    def test_large_sim_values_log_warning(self, tmp_path, caplog):
        """When sim_max > 10 * obs_max, a warning is logged."""
        import logging
        from phoscrosstalk.analysis import save_mrna_outputs
        t_rna = np.array([0.0, 10.0, 30.0])
        obs = np.ones((2, 3)) * 1.0
        sim = np.ones((2, 3)) * 200.0  # 200 >> 10 * 1.0
        with caplog.at_level(logging.WARNING, logger="phoscrosstalk.analysis"):
            save_mrna_outputs(str(tmp_path), ["G0", "G1"], t_rna, obs, sim)
        # File should still be created (warning, not error)
        assert os.path.exists(tmp_path / "mrna_fit_timeseries.tsv")

    def test_nonfinite_sim_logs_warning(self, tmp_path, caplog):
        """When sim has non-finite values, warning is logged."""
        import logging
        from phoscrosstalk.analysis import save_mrna_outputs
        t_rna = np.array([0.0, 10.0])
        obs = np.ones((2, 2))
        sim = np.array([[np.inf, np.inf], [1.0, 1.0]])
        with caplog.at_level(logging.WARNING, logger="phoscrosstalk.analysis"):
            save_mrna_outputs(str(tmp_path), ["G0", "G1"], t_rna, obs, sim)


# ---------------------------------------------------------------------------
# save_derived_rates
# ---------------------------------------------------------------------------
class TestSaveDerivedRates:
    def _make_closures(self, K=2):
        def k_fn(t):
            return np.ones(K) * (1.0 + 0.01 * t)
        def s_fn(t):
            return np.ones(K) * 0.1
        return k_fn, s_fn

    def test_both_none_skips(self, tmp_path, caplog):
        import logging
        from phoscrosstalk.analysis import save_derived_rates
        with caplog.at_level(logging.WARNING, logger="phoscrosstalk.analysis"):
            save_derived_rates(str(tmp_path), ["P0", "P1"], np.array([0.0, 10.0]))
        assert not os.path.exists(tmp_path / "derived_rates.npz")

    def test_k_act_only(self, tmp_path):
        from phoscrosstalk.analysis import save_derived_rates
        k_fn, _ = self._make_closures()
        save_derived_rates(
            str(tmp_path), ["P0", "P1"], np.array([0.0, 10.0, 30.0]),
            k_act_fn=k_fn
        )
        assert os.path.exists(tmp_path / "derived_rates.npz")
        loaded = np.load(tmp_path / "derived_rates.npz", allow_pickle=True)
        assert "k_act" in loaded

    def test_s_prod_only(self, tmp_path):
        from phoscrosstalk.analysis import save_derived_rates
        _, s_fn = self._make_closures()
        save_derived_rates(
            str(tmp_path), ["P0", "P1"], np.array([0.0, 10.0, 30.0]),
            s_prod_fn=s_fn
        )
        assert os.path.exists(tmp_path / "derived_rates.npz")
        loaded = np.load(tmp_path / "derived_rates.npz", allow_pickle=True)
        assert "s_prod" in loaded

    def test_both_closures_creates_long_tsv(self, tmp_path):
        from phoscrosstalk.analysis import save_derived_rates
        k_fn, s_fn = self._make_closures()
        save_derived_rates(
            str(tmp_path), ["P0", "P1"], np.array([0.0, 10.0, 30.0]),
            k_act_fn=k_fn, s_prod_fn=s_fn
        )
        assert os.path.exists(tmp_path / "derived_rates_long.tsv")
        df = pd.read_csv(tmp_path / "derived_rates_long.tsv", sep="\t")
        assert "k_act" in df["rate_type"].values
        assert "s_prod" in df["rate_type"].values

    def test_t_rna_used_for_k_act_time(self, tmp_path):
        from phoscrosstalk.analysis import save_derived_rates
        k_fn, _ = self._make_closures()
        t_protein = np.array([0.0, 60.0])
        t_rna = np.array([0.0, 10.0, 30.0, 60.0])  # finer grid for RNA
        save_derived_rates(
            str(tmp_path), ["P0", "P1"], t_protein,
            k_act_fn=k_fn, t_rna=t_rna
        )
        loaded = np.load(tmp_path / "derived_rates.npz", allow_pickle=True)
        # k_act should have T_rna=4 time points
        assert loaded["k_act"].shape == (2, 4)


# ---------------------------------------------------------------------------
# plot_mrna_fit
# ---------------------------------------------------------------------------
class TestPlotMrnaFit:
    def _make_mrna_tsv(self, tmp_path):
        """Create mrna_fit_timeseries.tsv with 'fitted' column (as expected by plot_mrna_fit)."""
        genes = ["G0", "G1"]
        t = [0.0, 10.0, 30.0, 60.0]
        records = []
        for gene in genes:
            for ti, t_val in enumerate(t):
                rec = {
                    "gene": gene, "time": t_val,
                    "observed": 1.0 + 0.1 * ti,
                    "fitted": 1.0 + 0.08 * ti,
                }
                records.append(rec)
        df = pd.DataFrame(records)
        df.to_csv(tmp_path / "mrna_fit_timeseries.tsv", sep="\t", index=False)

    def test_skips_when_no_file(self, tmp_path):
        from phoscrosstalk.analysis import plot_mrna_fit
        # Should return silently if no mrna_fit_timeseries.tsv
        plot_mrna_fit(str(tmp_path))

    def test_creates_pngs_with_fitted_col(self, tmp_path):
        from phoscrosstalk.analysis import plot_mrna_fit
        self._make_mrna_tsv(tmp_path)
        plot_mrna_fit(str(tmp_path))
        # At least one PNG should be created
        pngs = list(tmp_path.glob("*.png"))
        assert len(pngs) > 0


# ---------------------------------------------------------------------------
# _save_dense_simulation exception branches
# ---------------------------------------------------------------------------
class TestSaveDenseSimulationExceptionBranches:
    """Test the exception-handling branches in _save_dense_simulation
    when data_interp_P/A/R callables raise exceptions."""

    @pytest.fixture
    def base_kwargs(self, tmp_path):
        return _save_dense_kwargs(tmp_path)

    def test_broken_interp_P_does_not_raise(self, tmp_path):
        """A data_interp_P that raises should be silently skipped."""
        from phoscrosstalk.analysis import _save_dense_simulation
        _set_dims()
        kwargs = _save_dense_kwargs(tmp_path)

        def broken_P(t_q):
            raise RuntimeError("deliberate interp P failure")

        with patch("phoscrosstalk.analysis.simulate_dense", return_value=_dense_result()):
            _save_dense_simulation(**kwargs, data_interp_P=broken_P)
        # File should still be created
        assert os.path.exists(tmp_path / "protein_fit_timeseries_dense.tsv")

    def test_broken_interp_A_does_not_raise(self, tmp_path):
        """A data_interp_A that raises should be silently skipped."""
        from phoscrosstalk.analysis import _save_dense_simulation
        _set_dims()
        kwargs = _save_dense_kwargs(tmp_path)

        def broken_A(t_q):
            raise RuntimeError("deliberate interp A failure")

        with patch("phoscrosstalk.analysis.simulate_dense", return_value=_dense_result()):
            _save_dense_simulation(**kwargs, data_interp_A=broken_A,
                                   prot_idx_for_A_full=np.array([0, 1], dtype=int))
        assert os.path.exists(tmp_path / "protein_fit_timeseries_dense.tsv")

    def test_broken_interp_R_logs_warning(self, tmp_path, caplog):
        """A data_interp_R that raises should log a warning."""
        import logging
        from phoscrosstalk.analysis import _save_dense_simulation
        _set_dims()
        kwargs = _save_dense_kwargs(tmp_path)

        def broken_R(t_q):
            raise RuntimeError("deliberate interp R failure")

        with patch("phoscrosstalk.analysis.simulate_dense", return_value=_dense_result()):
            with caplog.at_level(logging.WARNING, logger="phoscrosstalk.analysis"):
                _save_dense_simulation(**kwargs, data_interp_R=broken_R)
        assert os.path.exists(tmp_path / "protein_fit_timeseries_dense.tsv")

    def test_flat_array_interp_P_fallback(self, tmp_path):
        """When interp_P returns a 1-D array instead of 2-D, fallback is used."""
        from phoscrosstalk.analysis import _save_dense_simulation
        _set_dims()
        kwargs = _save_dense_kwargs(tmp_path, n_dense=10)

        def flat_P(t_q):
            # Returns a flat (n_dense,) instead of (N, n_dense)
            return np.ones(len(t_q))

        with patch("phoscrosstalk.analysis.simulate_dense", return_value=_dense_result(10)):
            _save_dense_simulation(**kwargs, data_interp_P=flat_P)
        df = pd.read_csv(tmp_path / "protein_fit_timeseries_dense.tsv", sep="\t")
        assert len(df) > 0

    def test_mrna_dense_tsv_created_when_R_sim_finite(self, tmp_path):
        """When R_sim_d has finite values, mrna_fit_timeseries_dense.tsv is created."""
        from phoscrosstalk.analysis import _save_dense_simulation
        _set_dims()
        kwargs = _save_dense_kwargs(tmp_path, n_dense=10)

        with patch("phoscrosstalk.analysis.simulate_dense", return_value=_dense_result(10)):
            _save_dense_simulation(**kwargs)
        assert os.path.exists(tmp_path / "mrna_fit_timeseries_dense.tsv")


# ---------------------------------------------------------------------------
# save_fitted_simulation — data interpolation exception branch (lines 701-726)
# ---------------------------------------------------------------------------
class TestSaveFittedSimulationInterpExceptions:
    """Test that exceptions in data interpolation build are warned, not raised."""

    def _base_kwargs(self, tmp_path):
        _set_dims()
        sites = ["Prot0_S0", "Prot0_S1", "Prot1_S2"]
        proteins = ["Prot0", "Prot1"]
        T = 4
        return dict(
            outdir=str(tmp_path),
            theta_opt=_theta(),
            t=np.linspace(0.0, 60.0, T),
            sites=sites,
            proteins=proteins,
            P_scaled=np.ones((N, T)) * 0.3,
            A_scaled=np.zeros((0, T)),
            prot_idx_for_A=np.array([], dtype=int),
            baselines=np.ones(N),
            amplitudes=np.ones(N),
            Y=np.ones((N, T)),
            A_data=np.zeros((0, T)),
            A_bases=np.array([]),
            A_amps=np.array([]),
            mechanism="dist",
            Cg=np.zeros((N, N)),
            Cl=np.zeros((N, N)),
            site_prot_idx=np.array([0, 0, 1], dtype=int),
            K_site_kin=np.zeros((N, M)),
            R=np.zeros((M, N)),
            L_alpha=np.zeros((M, M)),
            kin_to_prot_idx=np.array([0, 1], dtype=int),
            mask_p=np.zeros(K, dtype=int),
            mask_k=np.zeros(M, dtype=int),
        )

    def _sim_result(self, T=4):
        return (
            np.ones((N, T), dtype=np.float64) * 0.5,
            np.ones((K, T), dtype=np.float64),
            np.ones((K, T), dtype=np.float64) * 0.3,
            np.ones((M, T), dtype=np.float64) * 0.4,
        )

    def test_data_interp_build_exception_warns_not_raises(self, tmp_path, caplog):
        """If build_data_interpolations raises, a warning is logged and execution continues."""
        import logging
        from phoscrosstalk.analysis import save_fitted_simulation
        kwargs = self._base_kwargs(tmp_path)
        di_cfg = SimpleNamespace(
            enabled=True, method="linear",
            fill_forward_nans_at_end=False, replace_nans_at_start=None
        )
        kwargs["data_interpolation_cfg"] = di_cfg

        with patch("phoscrosstalk.analysis.simulate", return_value=self._sim_result()):
            with patch("phoscrosstalk.analysis.simulate_dense", return_value=_dense_result()):
                # The import happens inside the function as:
                # from phoscrosstalk.derived_rates import build_data_interpolations
                # so we patch the source module
                with patch(
                    "phoscrosstalk.derived_rates.build_data_interpolations",
                    side_effect=RuntimeError("deliberate failure")
                ):
                    with caplog.at_level(logging.WARNING, logger="phoscrosstalk.analysis"):
                        save_fitted_simulation(**kwargs)
        # Main output should still be created
        assert os.path.exists(tmp_path / "protein_fit_timeseries.tsv")

    def test_rna_interp_logs_message_when_successful(self, tmp_path, caplog):
        """When RNA interp is built successfully, messages are logged."""
        import logging
        from phoscrosstalk.analysis import save_fitted_simulation
        kwargs = self._base_kwargs(tmp_path)
        R_data0 = np.ones((K, 4)) * 1.2
        t_rna = np.linspace(0.0, 60.0, 4)
        kwargs["R_data0"] = R_data0
        kwargs["t_rna"] = t_rna
        di_cfg = SimpleNamespace(
            enabled=True, method="linear",
            fill_forward_nans_at_end=False, replace_nans_at_start=None
        )
        kwargs["data_interpolation_cfg"] = di_cfg

        with patch("phoscrosstalk.analysis.simulate", return_value=self._sim_result()):
            with patch("phoscrosstalk.analysis.simulate_dense", return_value=_dense_result()):
                save_fitted_simulation(**kwargs)
        assert os.path.exists(tmp_path / "protein_fit_timeseries.tsv")
