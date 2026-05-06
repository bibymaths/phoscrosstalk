"""
test_analysis.py

Tests for analysis.py covering the previously uncovered lines.
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

# Minimal viable dimensions: K=2, M=2, N=3
# Parameter vector dim = 2*K + 2 + 3*M + N + 4 = 4+2+6+3+4 = 19
K, M, N = 2, 2, 3
THETA_DIM = 2 * K + 2 + 3 * M + N + 4  # 19


def _set_dims():
    return ModelDims.set_dims(K, M, N)


def _theta():
    return np.zeros(THETA_DIM, dtype=np.float64)


def _sim_result(T=5):
    """Tuple returned by simulate(full_output=True)."""
    _set_dims()
    return (
        np.ones((N, T), dtype=np.float64),   # P_sim
        np.ones((K, T), dtype=np.float64),   # A_sim
        np.ones((K, T), dtype=np.float64),   # S_sim
        np.ones((M, T), dtype=np.float64),   # Kdyn_sim
    )


def _dense_result(n=200):
    """Dict returned by simulate_dense()."""
    _set_dims()
    return {
        "P_sim": np.ones((N, n), dtype=np.float64),
        "A_sim": np.ones((K, n), dtype=np.float64),
        "R_sim": np.ones((K, n), dtype=np.float64),
        "S_sim": np.ones((K, n), dtype=np.float64),
        "Kdyn_sim": np.ones((M, n), dtype=np.float64),
        "success": True,
    }


# ---------------------------------------------------------------------------
# save_run_results
# ---------------------------------------------------------------------------

class TestSaveRunResults:
    def test_creates_all_four_files(self, tmp_path):
        from phoscrosstalk.analysis import save_run_results
        _set_dims()
        n = 5
        F = np.random.default_rng(0).random((n, 3))
        X = np.zeros((n, THETA_DIM))
        f1, f2, f3 = F[:, 0], F[:, 1], F[:, 2]
        J = f1 + f2 + f3
        F_best = F[0]
        save_run_results(str(tmp_path), F, X, f1, f2, f3, J, F_best)
        assert (tmp_path / "pareto_stats.tsv").exists()
        assert (tmp_path / "pareto_front_with_J.tsv").exists()
        assert (tmp_path / "pareto_points.tsv").exists()
        assert (tmp_path / "pareto_front.npz").exists()

    def test_n_obj_3_objective_names(self, tmp_path):
        from phoscrosstalk.analysis import save_run_results
        _set_dims()
        n = 4
        F = np.random.default_rng(1).random((n, 3))
        X = np.zeros((n, THETA_DIM))
        f1, f2, f3 = F[:, 0], F[:, 1], F[:, 2]
        J = f1 + f2 + f3
        save_run_results(str(tmp_path), F, X, f1, f2, f3, J, F[0])
        df = pd.read_csv(tmp_path / "pareto_stats.tsv", sep="\t")
        assert set(df["objective"]) == {"f1_P_sites", "f2_protein", "f3_complexity"}

    def test_n_obj_4_includes_f4_mrna(self, tmp_path):
        from phoscrosstalk.analysis import save_run_results
        _set_dims()
        n = 5
        F = np.random.default_rng(2).random((n, 4))
        X = np.zeros((n, THETA_DIM))
        f1, f2, f3, f4 = F[:, 0], F[:, 1], F[:, 2], F[:, 3]
        J = f1 + f2 + f3 + f4
        save_run_results(str(tmp_path), F, X, f1, f2, f3, J, F[0], f4=f4)
        df = pd.read_csv(tmp_path / "pareto_stats.tsv", sep="\t")
        assert "f4_mrna" in df["objective"].values

    def test_npz_contains_correct_keys(self, tmp_path):
        from phoscrosstalk.analysis import save_run_results
        _set_dims()
        n = 3
        F = np.eye(3)
        X = np.zeros((n, THETA_DIM))
        f1, f2, f3 = F[:, 0], F[:, 1], F[:, 2]
        J = f1 + f2 + f3
        save_run_results(str(tmp_path), F, X, f1, f2, f3, J, F[0])
        data = np.load(tmp_path / "pareto_front.npz")
        assert "F" in data and "X" in data and "J" in data

    def test_bio_score_column_in_pareto_points(self, tmp_path):
        from phoscrosstalk.analysis import save_run_results
        _set_dims()
        n = 4
        F = np.random.default_rng(3).random((n, 3))
        X = np.zeros((n, THETA_DIM))
        f1, f2, f3 = F[:, 0], F[:, 1], F[:, 2]
        J = f1 + f2 + f3
        save_run_results(str(tmp_path), F, X, f1, f2, f3, J, F[0])
        df = pd.read_csv(tmp_path / "pareto_points.tsv", sep="\t")
        assert "bio_score" in df.columns


# ---------------------------------------------------------------------------
# plot_run_diagnostics
# ---------------------------------------------------------------------------

class TestPlotRunDiagnostics:
    def test_creates_f1_f2_and_param_corr_png(self, tmp_path):
        from phoscrosstalk.analysis import plot_run_diagnostics
        _set_dims()
        n = 5
        rng = np.random.default_rng(4)
        F = rng.random((n, 3))
        X = rng.random((n, THETA_DIM))
        f1, f2, f3 = F[:, 0], F[:, 1], F[:, 2]
        plot_run_diagnostics(str(tmp_path), F, F[0], f1, f2, f3, X)
        assert (tmp_path / "pareto_f1_f2.png").exists()
        assert (tmp_path / "pareto_param_corr.png").exists()

    def test_no_f4_png_when_f4_none(self, tmp_path):
        from phoscrosstalk.analysis import plot_run_diagnostics
        _set_dims()
        n = 4
        F = np.random.default_rng(5).random((n, 3))
        X = np.random.default_rng(5).random((n, THETA_DIM))
        f1, f2, f3 = F[:, 0], F[:, 1], F[:, 2]
        plot_run_diagnostics(str(tmp_path), F, F[0], f1, f2, f3, X, f4=None)
        assert not (tmp_path / "pareto_f1_f4.png").exists()

    def test_no_f4_png_when_f4_all_zero(self, tmp_path):
        from phoscrosstalk.analysis import plot_run_diagnostics
        _set_dims()
        n = 4
        F = np.random.default_rng(6).random((n, 3))
        X = np.random.default_rng(6).random((n, THETA_DIM))
        f1, f2, f3 = F[:, 0], F[:, 1], F[:, 2]
        plot_run_diagnostics(str(tmp_path), F, F[0], f1, f2, f3, X, f4=np.zeros(n))
        assert not (tmp_path / "pareto_f1_f4.png").exists()

    def test_creates_f4_png_when_f4_nonzero(self, tmp_path):
        from phoscrosstalk.analysis import plot_run_diagnostics
        _set_dims()
        n = 5
        rng = np.random.default_rng(7)
        F = rng.random((n, 4))
        X = rng.random((n, THETA_DIM))
        f1, f2, f3, f4 = F[:, 0], F[:, 1], F[:, 2], F[:, 3] + 0.1
        F_best = np.array([0.4, 0.3, 0.2, 0.1])
        plot_run_diagnostics(str(tmp_path), F, F_best, f1, f2, f3, X, f4=f4)
        assert (tmp_path / "pareto_f1_f4.png").exists()

    def test_f4_panel_with_3_element_F_best(self, tmp_path):
        """When F_best has only 3 elements, f4 panel skips the red marker."""
        from phoscrosstalk.analysis import plot_run_diagnostics
        _set_dims()
        n = 5
        rng = np.random.default_rng(8)
        F = rng.random((n, 4))
        X = rng.random((n, THETA_DIM))
        f1, f2, f3, f4 = F[:, 0], F[:, 1], F[:, 2], F[:, 3] + 0.1
        F_best = np.array([0.4, 0.3, 0.2])  # Only 3 elements
        plot_run_diagnostics(str(tmp_path), F, F_best, f1, f2, f3, X, f4=f4)
        assert (tmp_path / "pareto_f1_f4.png").exists()


# ---------------------------------------------------------------------------
# print_parameter_summary
# ---------------------------------------------------------------------------

class TestPrintParameterSummary:
    def test_creates_four_output_files(self, tmp_path):
        from phoscrosstalk.analysis import print_parameter_summary
        _set_dims()
        proteins = ["ProtA", "ProtB"]
        kinases = ["KinX", "KinY"]
        sites = ["ProtA_T1", "ProtA_S2", "ProtB_Y3"]
        print_parameter_summary(str(tmp_path), _set_dims(), _theta(), proteins, kinases, sites)
        assert (tmp_path / "parameter_summary_proteins.tsv").exists()
        assert (tmp_path / "parameter_summary_kinases.tsv").exists()
        assert (tmp_path / "parameter_summary_sites.tsv").exists()
        assert (tmp_path / "parameter_summary_global.txt").exists()

    def test_protein_tsv_has_expected_columns(self, tmp_path):
        from phoscrosstalk.analysis import print_parameter_summary
        _set_dims()
        proteins = ["ProtA", "ProtB"]
        kinases = ["KinX", "KinY"]
        sites = ["ProtA_T1", "ProtA_S2", "ProtB_Y3"]
        print_parameter_summary(str(tmp_path), _set_dims(), _theta(), proteins, kinases, sites)
        df = pd.read_csv(tmp_path / "parameter_summary_proteins.tsv", sep="\t")
        assert "Protein" in df.columns
        assert len(df) == K

    def test_global_txt_contains_beta(self, tmp_path):
        from phoscrosstalk.analysis import print_parameter_summary
        _set_dims()
        proteins = ["ProtA", "ProtB"]
        kinases = ["KinX", "KinY"]
        sites = ["ProtA_T1", "ProtA_S2", "ProtB_Y3"]
        print_parameter_summary(str(tmp_path), _set_dims(), _theta(), proteins, kinases, sites)
        txt = (tmp_path / "parameter_summary_global.txt").read_text()
        assert "beta_g" in txt
        assert "beta_l" in txt


# ---------------------------------------------------------------------------
# _save_dense_simulation
# ---------------------------------------------------------------------------

def _dense_common_args(tmp_path):
    _set_dims()
    T = 5
    t_obs = np.linspace(0, 60, T)
    sites = ["ProtA_T1", "ProtA_S2", "ProtB_Y3"]
    proteins = ["ProtA", "ProtB"]
    P_scaled = np.ones((N, T), dtype=np.float64)
    A_scaled = np.ones((K, T), dtype=np.float64)
    prot_idx_for_A = np.array([0, 1])
    Cg = np.eye(K)
    Cl = np.eye(K)
    site_prot_idx = np.array([0, 0, 1])
    K_site_kin = np.ones((M, N))
    R = np.eye(K)
    L_alpha = np.zeros((K, K))
    kin_to_prot_idx = np.array([0, 1])
    mask_p = np.ones(K)
    mask_k = np.ones(M)
    return dict(
        outdir=str(tmp_path),
        theta_opt=_theta(),
        t_obs=t_obs,
        sites=sites,
        proteins=proteins,
        P_scaled=P_scaled,
        A_scaled=A_scaled,
        prot_idx_for_A=prot_idx_for_A,
        Cg=Cg,
        Cl=Cl,
        site_prot_idx=site_prot_idx,
        K_site_kin=K_site_kin,
        R=R,
        L_alpha=L_alpha,
        kin_to_prot_idx=kin_to_prot_idx,
        mask_p=mask_p,
        mask_k=mask_k,
        mechanism="dist",
    )


class TestSaveDenseSimulation:
    def test_basic_creates_dense_tsv(self, tmp_path):
        from phoscrosstalk.analysis import _save_dense_simulation
        args = _dense_common_args(tmp_path)
        with patch("phoscrosstalk.analysis.simulate_dense", return_value=_dense_result()):
            _save_dense_simulation(**args)
        assert (tmp_path / "fit_timeseries_dense.tsv").exists()

    def test_creates_mrna_dense_tsv(self, tmp_path):
        from phoscrosstalk.analysis import _save_dense_simulation
        args = _dense_common_args(tmp_path)
        with patch("phoscrosstalk.analysis.simulate_dense", return_value=_dense_result()):
            _save_dense_simulation(**args)
        assert (tmp_path / "mrna_fit_timeseries_dense.tsv").exists()

    def test_with_data_interp_P_adds_rows(self, tmp_path):
        from phoscrosstalk.analysis import _save_dense_simulation
        args = _dense_common_args(tmp_path)

        def _interp_P(t_q):
            return np.ones((N, len(t_q)))

        with patch("phoscrosstalk.analysis.simulate_dense", return_value=_dense_result()):
            _save_dense_simulation(**args, data_interp_P=_interp_P)
        df = pd.read_csv(tmp_path / "fit_timeseries_dense.tsv", sep="\t")
        assert "observed_interpolated_dense" in df["series_type"].values

    def test_with_data_interp_A_adds_rows(self, tmp_path):
        from phoscrosstalk.analysis import _save_dense_simulation
        args = _dense_common_args(tmp_path)

        def _interp_A(t_q):
            return np.ones((K, len(t_q)))

        with patch("phoscrosstalk.analysis.simulate_dense", return_value=_dense_result()):
            _save_dense_simulation(
                **args,
                data_interp_A=_interp_A,
                prot_idx_for_A_full=np.array([0, 1]),
            )
        df = pd.read_csv(tmp_path / "fit_timeseries_dense.tsv", sep="\t")
        assert "observed_interpolated_dense" in df["series_type"].values

    def test_with_data_interp_R_adds_mrna_rows(self, tmp_path):
        from phoscrosstalk.analysis import _save_dense_simulation
        args = _dense_common_args(tmp_path)

        def _interp_R(t_q):
            return np.ones((K, len(t_q)))

        with patch("phoscrosstalk.analysis.simulate_dense", return_value=_dense_result()):
            _save_dense_simulation(**args, data_interp_R=_interp_R)
        df = pd.read_csv(tmp_path / "fit_timeseries_dense.tsv", sep="\t")
        mrna_rows = df[df["entity_type"] == "mRNA"]
        assert len(mrna_rows) > 0

    def test_interp_P_flat_fallback(self, tmp_path):
        """Test fallback when data_interp_P returns a 1-D array."""
        from phoscrosstalk.analysis import _save_dense_simulation
        args = _dense_common_args(tmp_path)

        def _interp_P_flat(t_q):
            # Returns flat array (single-site edge case)
            return np.ones(len(t_q))

        with patch("phoscrosstalk.analysis.simulate_dense", return_value=_dense_result()):
            _save_dense_simulation(**args, data_interp_P=_interp_P_flat)
        assert (tmp_path / "fit_timeseries_dense.tsv").exists()


# ---------------------------------------------------------------------------
# save_fitted_simulation
# ---------------------------------------------------------------------------

def _fit_common_args(tmp_path, T=5):
    dims = _set_dims()
    t = np.linspace(0, 60, T)
    sites = ["ProtA_T1", "ProtA_S2", "ProtB_Y3"]
    proteins = ["ProtA", "ProtB"]
    P_scaled = np.ones((N, T), dtype=np.float64)
    A_scaled = np.ones((K, T), dtype=np.float64)
    Y = np.ones((N, T), dtype=np.float64)
    A_data = np.ones((K, T), dtype=np.float64)
    baselines = np.zeros(N, dtype=np.float64)
    amplitudes = np.ones(N, dtype=np.float64)
    A_bases = np.zeros(K, dtype=np.float64)
    A_amps = np.ones(K, dtype=np.float64)
    prot_idx_for_A = np.array([0, 1])
    Cg = np.eye(K)
    Cl = np.eye(K)
    site_prot_idx = np.array([0, 0, 1])
    K_site_kin = np.ones((M, N))
    R = np.eye(K)
    L_alpha = np.zeros((K, K))
    kin_to_prot_idx = np.array([0, 1])
    mask_p = np.ones(K)
    mask_k = np.ones(M)
    return dict(
        outdir=str(tmp_path),
        dims=dims,
        theta_opt=_theta(),
        t=t,
        sites=sites,
        proteins=proteins,
        P_scaled=P_scaled,
        A_scaled=A_scaled,
        prot_idx_for_A=prot_idx_for_A,
        baselines=baselines,
        amplitudes=amplitudes,
        Y=Y,
        A_data=A_data,
        A_bases=A_bases,
        A_amps=A_amps,
        mechanism="dist",
        Cg=Cg,
        Cl=Cl,
        site_prot_idx=site_prot_idx,
        K_site_kin=K_site_kin,
        R=R,
        L_alpha=L_alpha,
        kin_to_prot_idx=kin_to_prot_idx,
        mask_p=mask_p,
        mask_k=mask_k,
    )


class TestSaveFittedSimulation:
    def test_creates_fit_timeseries_and_params(self, tmp_path):
        from phoscrosstalk.analysis import save_fitted_simulation
        args = _fit_common_args(tmp_path)
        with patch("phoscrosstalk.analysis.simulate", return_value=_sim_result()), \
             patch("phoscrosstalk.analysis.simulate_dense", return_value=_dense_result()):
            save_fitted_simulation(**args)
        assert (tmp_path / "fit_timeseries.tsv").exists()
        assert (tmp_path / "fitted_params.npz").exists()

    def test_creates_internal_states_tsv(self, tmp_path):
        from phoscrosstalk.analysis import save_fitted_simulation
        args = _fit_common_args(tmp_path)
        with patch("phoscrosstalk.analysis.simulate", return_value=_sim_result()), \
             patch("phoscrosstalk.analysis.simulate_dense", return_value=_dense_result()):
            save_fitted_simulation(**args)
        assert (tmp_path / "internal_states.tsv").exists()

    def test_dense_output_created_by_default(self, tmp_path):
        from phoscrosstalk.analysis import save_fitted_simulation
        args = _fit_common_args(tmp_path)
        with patch("phoscrosstalk.analysis.simulate", return_value=_sim_result()), \
             patch("phoscrosstalk.analysis.simulate_dense", return_value=_dense_result()):
            save_fitted_simulation(**args)
        assert (tmp_path / "fit_timeseries_dense.tsv").exists()

    def test_simulation_cfg_save_dense_false_skips_dense(self, tmp_path):
        from phoscrosstalk.analysis import save_fitted_simulation
        args = _fit_common_args(tmp_path)
        sim_cfg = SimpleNamespace(save_dense=False, dense_n_points=50,
                                  dense_interpolation="diffrax_dense")
        with patch("phoscrosstalk.analysis.simulate", return_value=_sim_result()):
            save_fitted_simulation(**args, simulation_cfg=sim_cfg)
        assert not (tmp_path / "fit_timeseries_dense.tsv").exists()

    def test_data_interpolation_cfg_enabled(self, tmp_path):
        """data_interpolation_cfg.enabled=True triggers build_data_interpolations."""
        from phoscrosstalk.analysis import save_fitted_simulation
        args = _fit_common_args(tmp_path)
        di_cfg = SimpleNamespace(
            enabled=True,
            method="linear",
            fill_forward_nans_at_end=False,
            replace_nans_at_start=None,
        )
        with patch("phoscrosstalk.analysis.simulate", return_value=_sim_result()), \
             patch("phoscrosstalk.analysis.simulate_dense", return_value=_dense_result()):
            save_fitted_simulation(**args, data_interpolation_cfg=di_cfg)
        assert (tmp_path / "fit_timeseries_dense.tsv").exists()

    def test_data_interpolation_with_rna_data(self, tmp_path):
        """data_interp_R branch: R_data0 and t_rna provided."""
        from phoscrosstalk.analysis import save_fitted_simulation
        _set_dims()
        T = 5
        T_rna = 4
        args = _fit_common_args(tmp_path, T=T)
        di_cfg = SimpleNamespace(
            enabled=True,
            method="linear",
            fill_forward_nans_at_end=False,
            replace_nans_at_start=None,
        )
        R_data0 = np.ones((K, T_rna), dtype=np.float64)
        t_rna = np.array([0.0, 10.0, 30.0, 60.0])
        with patch("phoscrosstalk.analysis.simulate", return_value=_sim_result(T=T)), \
             patch("phoscrosstalk.analysis.simulate_dense", return_value=_dense_result()):
            save_fitted_simulation(
                **args,
                R_data0=R_data0,
                t_rna=t_rna,
                data_interpolation_cfg=di_cfg,
            )
        assert (tmp_path / "fit_timeseries.tsv").exists()

    def test_kinases_parameter_used_for_labels(self, tmp_path):
        from phoscrosstalk.analysis import save_fitted_simulation
        args = _fit_common_args(tmp_path)
        with patch("phoscrosstalk.analysis.simulate", return_value=_sim_result()), \
             patch("phoscrosstalk.analysis.simulate_dense", return_value=_dense_result()):
            save_fitted_simulation(**args, kinases=["KinX", "KinY"])
        df = pd.read_csv(tmp_path / "internal_states.tsv", sep="\t")
        kdyn_ids = df[df["Type"] == "Kdyn_sim"]["ID"].tolist()
        assert "KinX" in kdyn_ids or "KinY" in kdyn_ids


# ---------------------------------------------------------------------------
# plot_fitted_simulation
# ---------------------------------------------------------------------------

def _write_timeseries_tsv(path, proteins, sites, T=5):
    records = []
    sim_cols = {f"sim_t{j}": 1.0 for j in range(T)}
    data_cols = {f"data_t{j}": 1.0 for j in range(T)}
    for site in sites:
        prot, res = site.split("_", 1)
        rec = {"Type": "Phosphosite", "Protein": prot, "Residue": res}
        rec.update(sim_cols)
        rec.update(data_cols)
        records.append(rec)
    for prot in proteins:
        rec = {"Type": "ProteinAbundance", "Protein": prot, "Residue": ""}
        rec.update(sim_cols)
        rec.update(data_cols)
        records.append(rec)
    pd.DataFrame(records).to_csv(path, sep="\t", index=False)


class TestPlotFittedSimulation:
    def test_no_tsv_returns_silently(self, tmp_path):
        from phoscrosstalk.analysis import plot_fitted_simulation
        # Should not raise
        plot_fitted_simulation(str(tmp_path))

    def test_creates_per_protein_pngs(self, tmp_path):
        from phoscrosstalk.analysis import plot_fitted_simulation
        proteins = ["ProtA", "ProtB"]
        sites = ["ProtA_T1", "ProtA_S2", "ProtB_Y3"]
        _write_timeseries_tsv(tmp_path / "fit_timeseries.tsv", proteins, sites)
        plot_fitted_simulation(str(tmp_path))
        pngs = list(tmp_path.glob("*.png"))
        assert len(pngs) >= 1

    def test_with_mrna_file_present(self, tmp_path):
        from phoscrosstalk.analysis import plot_fitted_simulation
        proteins = ["ProtA"]
        sites = ["ProtA_T1"]
        _write_timeseries_tsv(tmp_path / "fit_timeseries.tsv", proteins, sites)
        mrna_records = [
            {"gene": "ProtA", "time": 0.0, "fitted": 1.0, "observed": 1.0},
            {"gene": "ProtA", "time": 10.0, "fitted": 1.2, "observed": 1.1},
            {"gene": "ProtA", "time": 30.0, "fitted": 1.3, "observed": 1.2},
        ]
        pd.DataFrame(mrna_records).to_csv(
            tmp_path / "mrna_fit_timeseries.tsv", sep="\t", index=False
        )
        plot_fitted_simulation(str(tmp_path))
        pngs = list(tmp_path.glob("*.png"))
        assert len(pngs) >= 1

    def test_with_mrna_file_using_simulated_column(self, tmp_path):
        from phoscrosstalk.analysis import plot_fitted_simulation
        proteins = ["ProtA"]
        sites = ["ProtA_T1"]
        _write_timeseries_tsv(tmp_path / "fit_timeseries.tsv", proteins, sites)
        mrna_records = [
            {"gene": "ProtA", "time": 0.0, "simulated": 1.0, "observed": 1.0},
            {"gene": "ProtA", "time": 10.0, "simulated": 1.2, "observed": 1.1},
        ]
        pd.DataFrame(mrna_records).to_csv(
            tmp_path / "mrna_fit_timeseries.tsv", sep="\t", index=False
        )
        plot_fitted_simulation(str(tmp_path))
        pngs = list(tmp_path.glob("*.png"))
        assert len(pngs) >= 1

    def test_empty_mrna_file_treated_as_absent(self, tmp_path):
        from phoscrosstalk.analysis import plot_fitted_simulation
        proteins = ["ProtA"]
        sites = ["ProtA_T1"]
        _write_timeseries_tsv(tmp_path / "fit_timeseries.tsv", proteins, sites)
        # Write empty mrna file (just the header)
        pd.DataFrame(columns=["gene", "time", "fitted", "observed"]).to_csv(
            tmp_path / "mrna_fit_timeseries.tsv", sep="\t", index=False
        )
        plot_fitted_simulation(str(tmp_path))
        pngs = list(tmp_path.glob("*.png"))
        assert len(pngs) >= 1


# ---------------------------------------------------------------------------
# plot_internal_states
# ---------------------------------------------------------------------------

class TestPlotInternalStates:
    def test_creates_png(self, tmp_path):
        from phoscrosstalk.analysis import plot_internal_states
        _set_dims()
        t = np.linspace(0, 60, 5)
        S_sim = np.random.default_rng(0).random((K, 5))
        Kdyn_sim = np.random.default_rng(0).random((M, 5))
        proteins = ["ProtA", "ProtB"]
        plot_internal_states(str(tmp_path), t, S_sim, Kdyn_sim, proteins)
        assert (tmp_path / "fitted_internal_states.png").exists()

    def test_with_kinase_names(self, tmp_path):
        from phoscrosstalk.analysis import plot_internal_states
        _set_dims()
        t = np.linspace(0, 60, 5)
        S_sim = np.random.default_rng(1).random((K, 5))
        Kdyn_sim = np.random.default_rng(1).random((M, 5))
        proteins = ["ProtA", "ProtB"]
        kinases = ["KinX", "KinY"]
        plot_internal_states(str(tmp_path), t, S_sim, Kdyn_sim, proteins, kinases=kinases)
        assert (tmp_path / "fitted_internal_states.png").exists()

    def test_without_kinase_names_uses_generic_labels(self, tmp_path):
        from phoscrosstalk.analysis import plot_internal_states
        _set_dims()
        t = np.linspace(0, 60, 5)
        S_sim = np.random.default_rng(2).random((K, 5))
        Kdyn_sim = np.random.default_rng(2).random((M, 5))
        proteins = ["ProtA", "ProtB"]
        # Should not raise even without kinase names
        plot_internal_states(str(tmp_path), t, S_sim, Kdyn_sim, proteins, kinases=None)
        assert (tmp_path / "fitted_internal_states.png").exists()
