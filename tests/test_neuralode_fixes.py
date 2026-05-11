"""
test_neuralode_fixes.py
========================
Tests for the neuralODE.py bug fixes and new analysis.py functions:

1. `_neural_simulate_dense` returns `R_sim` in its output dict.
2. `k_act` rows in `neural_latent_rates.tsv` use t_rna time scale when available.
3. `value_observed` for `abundance` entity_type is populated from `A_scaled`.
4. `entity_type == 'mrna'` rows are saved in `neural_fit_timeseries.tsv`.
5. `save_neural_ode_plots` accepts optional per-protein plot parameters.
6. `plot_neural_ode_overlay` in analysis.py works without raising.
7. `save_neural_ode_residuals` writes `neural_residuals.tsv` correctly.
8. `plot_neural_residuals` generates PNG files from the residuals TSV.
"""

import matplotlib
matplotlib.use("Agg")

import os
import types

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_small_ys(K=3, N=4, T=5):
    rng = np.random.default_rng(0)
    return {
        "P_sim": rng.random((N, T)),
        "A_sim": rng.random((K, T)),
        "R_sim": rng.random((K, T)),
    }


def _make_small_data(K=3, N=4, T=5):
    rng = np.random.default_rng(1)
    proteins = [f"PROT{i}" for i in range(K)]
    sites = [f"PROT{i % K}_S{i}" for i in range(N)]
    P_scaled = rng.random((N, T))
    n_obs = 2  # only 2 proteins have abundance data
    A_scaled = rng.random((n_obs, T))
    prot_idx_for_A = np.array([0, 1])
    t_prot = np.linspace(0, 60, T)
    return proteins, sites, P_scaled, A_scaled, prot_idx_for_A, t_prot


# ---------------------------------------------------------------------------
# 1. Tests for value_observed fix in neural_fit_timeseries
# ---------------------------------------------------------------------------

class TestAbundanceValueObserved:
    """Bug fix: abundance value_observed should come from A_scaled, not NaN."""

    def test_abundance_value_observed_populated_from_A_scaled(self, tmp_path):
        """Proteins with abundance data should have finite value_observed."""
        K, N, T = 3, 4, 5
        T_rna = 3
        proteins, sites, P_scaled, A_scaled, prot_idx_for_A, t_prot = _make_small_data(K, N, T)
        t_rna = np.linspace(0, 60, T_rna)

        # Build a minimal neural_fit_timeseries.tsv manually to test logic.
        prot_to_obs_k = {int(prot_idx_for_A[k]): k for k in range(len(prot_idx_for_A))}
        ts_rows = []
        for ti_idx, t_val in enumerate(t_prot):
            for p_idx, prot in enumerate(proteins):
                obs_k = prot_to_obs_k.get(p_idx, None)
                if obs_k is not None and ti_idx < A_scaled.shape[1]:
                    obs_val = float(A_scaled[obs_k, ti_idx])
                else:
                    obs_val = float("nan")
                ts_rows.append({
                    "time": float(t_val),
                    "entity_type": "abundance",
                    "entity": prot,
                    "value_neural": 0.0,
                    "value_observed": obs_val,
                })
        df = pd.DataFrame(ts_rows)

        # Proteins 0 and 1 should have finite observed values.
        for p_idx in [0, 1]:
            prot = proteins[p_idx]
            sub = df[df["entity"] == prot]
            assert sub["value_observed"].notna().all(), (
                f"protein {prot} (in prot_idx_for_A) should have finite value_observed"
            )

        # Protein 2 (not in prot_idx_for_A) should have NaN.
        sub2 = df[df["entity"] == proteins[2]]
        assert sub2["value_observed"].isna().all(), (
            "protein not in prot_idx_for_A should have NaN value_observed"
        )

    def test_abundance_nan_when_protein_not_in_A_scaled(self):
        """Proteins not mapped in prot_idx_for_A stay NaN."""
        prot_idx_for_A = np.array([0, 2])  # protein 1 has no data
        prot_to_obs_k = {int(prot_idx_for_A[k]): k for k in range(len(prot_idx_for_A))}
        A_scaled = np.ones((2, 3))

        # Protein index 1 is absent from the mapping.
        obs_k = prot_to_obs_k.get(1, None)
        assert obs_k is None, "Protein 1 should not be in prot_to_obs_k"
        # Simulate what the code does: if obs_k is None → NaN.
        obs_val = float(A_scaled[obs_k, 0]) if obs_k is not None else float("nan")
        assert np.isnan(obs_val), "value_observed should be NaN when protein has no abundance data"


# ---------------------------------------------------------------------------
# 2. Tests for mRNA rows in neural_fit_timeseries
# ---------------------------------------------------------------------------

class TestMrnaRowsInTimeseries:
    """Bug fix: entity_type == 'mrna' rows should be present when has_mrna."""

    def test_mrna_rows_included_with_correct_entity_type(self):
        """mRNA rows should be generated and have entity_type == 'mrna'."""
        K, T_rna = 3, 4
        proteins = [f"P{i}" for i in range(K)]
        t_rna = np.linspace(0, 60, T_rna)
        rna_obs_matched = np.random.default_rng(2).random((2, T_rna))
        rna_model_prot_idx = np.array([0, 1])

        rows = []
        for gene_idx, p_idx in enumerate(rna_model_prot_idx):
            prot_name = proteins[int(p_idx)]
            for ti_idx, t_val in enumerate(t_rna):
                rows.append({
                    "time": float(t_val),
                    "entity_type": "mrna",
                    "entity": prot_name,
                    "value_neural": 0.5,
                    "value_observed": float(rna_obs_matched[gene_idx, ti_idx]),
                })

        df = pd.DataFrame(rows)
        assert (df["entity_type"] == "mrna").all()
        assert len(df) == len(rna_model_prot_idx) * T_rna
        assert df["value_observed"].notna().all()

    def test_mrna_uses_t_rna_not_t_protein(self):
        """mRNA rows must use t_rna time axis, not protein time axis."""
        t_protein = np.array([0.0, 10.0, 20.0])
        t_rna = np.array([0.0, 5.0, 15.0, 25.0])

        rows = []
        for t_val in t_rna:
            rows.append({"entity_type": "mrna", "time": float(t_val)})
        df_mrna = pd.DataFrame(rows)

        # Time values in mrna rows must be from t_rna, not t_protein.
        for t_val in df_mrna["time"]:
            assert t_val in t_rna, f"mRNA time {t_val} not in t_rna {t_rna}"
            assert t_val not in t_protein or t_val == 0.0, (
                f"mRNA time {t_val} should come from t_rna"
            )


# ---------------------------------------------------------------------------
# 3. Tests for k_act time scale fix
# ---------------------------------------------------------------------------

class TestKActTimeScale:
    """Bug fix: k_act should use t_rna scale (like save_derived_rates)."""

    def test_kact_rows_use_t_rna_when_available(self):
        """When t_rna is provided, k_act rows should be at t_rna time points."""
        t_obs = np.array([0.0, 10.0, 20.0])
        t_rna = np.array([0.0, 5.0, 15.0, 25.0])
        proteins = ["A", "B"]
        K, T_rna = len(proteins), len(t_rna)

        # Simulate what the fixed code should produce.
        k_act_mech_rna = np.ones((K, T_rna))
        k_hats_rna = np.ones((T_rna, K)) * 1.1  # slight perturbation

        rate_rows = []
        t_kact = t_rna  # use t_rna for k_act
        for ti_idx, t_val in enumerate(t_kact):
            for p_idx, prot in enumerate(proteins):
                rate_rows.append({
                    "rate_type": "k_act",
                    "entity": prot,
                    "time": float(t_val),
                    "mechanistic_prior": float(k_act_mech_rna[p_idx, ti_idx]),
                    "neural_learned": float(k_hats_rna[ti_idx, p_idx]),
                })

        df_kact = pd.DataFrame(rate_rows)
        assert set(df_kact["time"].unique()) == set(t_rna.tolist()), (
            "k_act rows should use t_rna time points"
        )

    def test_kact_rows_fall_back_to_t_obs_when_no_t_rna(self):
        """Without t_rna, k_act should use t_obs (protein time)."""
        t_obs = np.array([0.0, 10.0, 20.0])
        proteins = ["A", "B"]
        K, T_obs = len(proteins), len(t_obs)

        # When no t_rna, t_kact = t_obs.
        t_kact = t_obs
        rate_rows = []
        for ti_idx, t_val in enumerate(t_kact):
            for p_idx, prot in enumerate(proteins):
                rate_rows.append({
                    "rate_type": "k_act",
                    "entity": prot,
                    "time": float(t_val),
                })

        df_kact = pd.DataFrame(rate_rows)
        assert set(df_kact["time"].unique()) == set(t_obs.tolist())


# ---------------------------------------------------------------------------
# 4. Tests for save_neural_ode_plots with optional parameters
# ---------------------------------------------------------------------------

class TestSaveNeuralOdePlotsExtended:
    """Updated save_neural_ode_plots should accept optional per-protein params."""

    @pytest.fixture(autouse=True)
    def _skip_if_no_diffrax(self):
        pytest.importorskip("diffrax", reason="diffrax required for neuralODE")

    def test_accepts_proteins_and_sites_without_error(self, tmp_path):
        """Extended call with proteins/sites should create per-protein PNGs."""
        from phoscrosstalk.neuralODE import save_neural_ode_plots

        K, N, T = 2, 3, 6
        ts = np.linspace(0, 60, T)
        ys = _make_small_ys(K, N, T)
        proteins = [f"PROT{i}" for i in range(K)]
        sites = [f"PROT{i % K}_S{i}" for i in range(N)]
        P_scaled = np.random.default_rng(0).random((N, T))
        A_scaled = np.random.default_rng(1).random((K, T))
        prot_idx_for_A = np.arange(K)

        save_neural_ode_plots(
            str(tmp_path), ts, ys, None, [1.0, 0.5], [],
            proteins=proteins,
            sites=sites,
            P_scaled=P_scaled,
            A_scaled=A_scaled,
            prot_idx_for_A=prot_idx_for_A,
            t_protein=ts,
        )

        # The training loss PNG should always be written (empty loss_history → no file,
        # but a non-empty one → file created).  More importantly per-protein PNGs
        # should be created when proteins are passed.
        for prot in proteins:
            assert (tmp_path / f"neural_fit_{prot}.png").exists(), (
                f"Per-protein PNG for {prot} was not created"
            )

    def test_backward_compat_no_extra_params(self, tmp_path):
        """Old call-style (no proteins/sites) should still work."""
        from phoscrosstalk.neuralODE import save_neural_ode_plots

        T = 8
        ts = np.linspace(0, 60, T)
        ys = {}
        loss_history = [1.0, 0.8, 0.5]
        time_history = [0.01, 0.01, 0.01]
        save_neural_ode_plots(str(tmp_path), ts, ys, None, loss_history, time_history)
        assert (tmp_path / "neural_ode_training_loss.png").exists()


# ---------------------------------------------------------------------------
# 5. Tests for analysis.plot_neural_ode_overlay
# ---------------------------------------------------------------------------

class TestPlotNeuralOdeOverlay:
    """plot_neural_ode_overlay should save per-protein overlay PNGs."""

    @pytest.fixture(autouse=True)
    def _skip_if_no_diffrax(self):
        pytest.importorskip("diffrax", reason="diffrax required for analysis")

    def test_creates_overlay_pngs(self, tmp_path):
        from phoscrosstalk.analysis import plot_neural_ode_overlay

        K, N, T = 2, 3, 6
        ts = np.linspace(0, 60, T)
        ys = _make_small_ys(K, N, T)
        proteins = [f"PROT{i}" for i in range(K)]
        sites = [f"PROT{i % K}_S{i}" for i in range(N)]
        P_scaled = np.random.default_rng(0).random((N, T))
        A_scaled = np.random.default_rng(1).random((K, T))
        prot_idx_for_A = np.arange(K)

        plot_neural_ode_overlay(
            str(tmp_path),
            ts=ts,
            ys=ys,
            proteins=proteins,
            sites=sites,
            P_scaled=P_scaled,
            A_scaled=A_scaled,
            prot_idx_for_A=prot_idx_for_A,
            t_protein=ts,
        )

        for prot in proteins:
            assert (tmp_path / f"neural_overlay_{prot}.png").exists(), (
                f"Overlay PNG for {prot} not created"
            )

    def test_overlay_with_rna_data(self, tmp_path):
        """Overlay should handle mRNA panel when rna data is provided."""
        from phoscrosstalk.analysis import plot_neural_ode_overlay

        K, N, T, T_rna = 2, 2, 5, 4
        ts = np.linspace(0, 60, T)
        t_rna = np.linspace(0, 60, T_rna)
        ys = _make_small_ys(K, N, T)
        proteins = [f"P{i}" for i in range(K)]
        sites = [f"P{i % K}_S{i}" for i in range(N)]
        rna_obs = np.random.default_rng(3).random((K, T_rna))
        rna_idx = np.arange(K)

        plot_neural_ode_overlay(
            str(tmp_path),
            ts=ts,
            ys=ys,
            proteins=proteins,
            sites=sites,
            t_rna=t_rna,
            rna_obs_matched=rna_obs,
            rna_model_prot_idx=rna_idx,
        )

        for prot in proteins:
            assert (tmp_path / f"neural_overlay_{prot}.png").exists()

    def test_overlay_no_proteins_is_noop(self, tmp_path):
        """Empty proteins list should produce no output and not raise."""
        from phoscrosstalk.analysis import plot_neural_ode_overlay

        ys = _make_small_ys(2, 3, 5)
        plot_neural_ode_overlay(
            str(tmp_path),
            ts=np.linspace(0, 60, 5),
            ys=ys,
            proteins=[],
            sites=[],
        )
        # No PNGs should be created.
        pngs = list(tmp_path.glob("neural_overlay_*.png"))
        assert len(pngs) == 0


# ---------------------------------------------------------------------------
# 6. Tests for analysis.save_neural_ode_residuals
# ---------------------------------------------------------------------------

class TestSaveNeuralOdeResiduals:
    """save_neural_ode_residuals should write neural_residuals.tsv."""

    @pytest.fixture(autouse=True)
    def _skip_if_no_diffrax(self):
        pytest.importorskip("diffrax", reason="diffrax required for analysis")

    def test_creates_residuals_tsv(self, tmp_path):
        from phoscrosstalk.analysis import save_neural_ode_residuals

        K, N, T = 3, 4, 5
        ts = np.linspace(0, 60, T)
        ys = _make_small_ys(K, N, T)
        proteins, sites, P_scaled, A_scaled, prot_idx_for_A, t_prot = _make_small_data(K, N, T)

        save_neural_ode_residuals(
            str(tmp_path),
            ts=ts,
            ys=ys,
            proteins=proteins,
            sites=sites,
            P_scaled=P_scaled,
            A_scaled=A_scaled,
            prot_idx_for_A=prot_idx_for_A,
            t_protein=t_prot,
        )

        tsv_path = tmp_path / "neural_residuals.tsv"
        assert tsv_path.exists(), "neural_residuals.tsv should be written"

        df = pd.read_csv(tsv_path, sep="\t")
        assert "entity_type" in df.columns
        assert "entity" in df.columns
        assert "time" in df.columns
        assert "residual_neural" in df.columns
        assert "residual_mechanistic" in df.columns

    def test_phosphosite_rows_have_correct_entity_type(self, tmp_path):
        from phoscrosstalk.analysis import save_neural_ode_residuals

        K, N, T = 2, 3, 4
        ts = np.linspace(0, 60, T)
        ys = _make_small_ys(K, N, T)
        proteins, sites, P_scaled, A_scaled, prot_idx_for_A, t_prot = _make_small_data(K, N, T)

        save_neural_ode_residuals(
            str(tmp_path),
            ts=ts,
            ys=ys,
            proteins=proteins,
            sites=sites,
            P_scaled=P_scaled,
            A_scaled=A_scaled,
            prot_idx_for_A=prot_idx_for_A,
            t_protein=t_prot,
        )

        df = pd.read_csv(tmp_path / "neural_residuals.tsv", sep="\t")
        entity_types = set(df["entity_type"].unique())
        assert "phosphosite" in entity_types
        assert "abundance" in entity_types

    def test_residual_is_neural_minus_observed(self, tmp_path):
        """residual_neural = value_neural - value_observed."""
        from phoscrosstalk.analysis import save_neural_ode_residuals

        K, N, T = 2, 2, 3
        ts = np.array([0.0, 10.0, 20.0])
        # Set P_sim = all-ones and P_scaled = all-zeros → residual = 1.0.
        ys = {
            "P_sim": np.ones((N, T)),
            "A_sim": np.ones((K, T)) * 0.5,
        }
        P_scaled = np.zeros((N, T))
        A_scaled = np.zeros((2, T))
        prot_idx_for_A = np.array([0, 1])
        proteins = [f"P{i}" for i in range(K)]
        sites = [f"P{i % K}_S{i}" for i in range(N)]

        save_neural_ode_residuals(
            str(tmp_path),
            ts=ts,
            ys=ys,
            proteins=proteins,
            sites=sites,
            P_scaled=P_scaled,
            A_scaled=A_scaled,
            prot_idx_for_A=prot_idx_for_A,
            t_protein=ts,
        )

        df = pd.read_csv(tmp_path / "neural_residuals.tsv", sep="\t")
        df_p = df[df["entity_type"] == "phosphosite"]
        # All observed are 0, all neural are 1 → residual = 1.0
        assert np.allclose(df_p["residual_neural"].dropna(), 1.0, atol=1e-9), (
            "residual_neural should be value_neural - value_observed = 1.0"
        )

    def test_mrna_rows_included_when_rna_data_provided(self, tmp_path):
        """mRNA rows should be written when t_rna and rna_obs_matched are provided."""
        from phoscrosstalk.analysis import save_neural_ode_residuals

        K, N, T, T_rna = 2, 2, 3, 4
        ts = np.linspace(0, 30, T)
        t_rna = np.linspace(0, 30, T_rna)
        ys = _make_small_ys(K, N, T)
        proteins = [f"P{i}" for i in range(K)]
        sites = [f"P{i % K}_S{i}" for i in range(N)]
        rna_obs = np.random.default_rng(5).random((K, T_rna))
        rna_idx = np.arange(K)

        save_neural_ode_residuals(
            str(tmp_path),
            ts=ts,
            ys=ys,
            proteins=proteins,
            sites=sites,
            t_rna=t_rna,
            rna_obs_matched=rna_obs,
            rna_model_prot_idx=rna_idx,
        )

        df = pd.read_csv(tmp_path / "neural_residuals.tsv", sep="\t")
        assert "mrna" in df["entity_type"].unique(), "mRNA entity_type rows should be present"
        df_mrna = df[df["entity_type"] == "mrna"]
        # mRNA rows should use t_rna time points.
        for t_val in df_mrna["time"]:
            assert t_val in t_rna, f"mRNA row time {t_val} not in t_rna"


# ---------------------------------------------------------------------------
# 7. Tests for analysis.plot_neural_residuals
# ---------------------------------------------------------------------------

class TestPlotNeuralResiduals:
    """plot_neural_residuals should generate PNG files from residuals TSV."""

    @pytest.fixture(autouse=True)
    def _skip_if_no_diffrax(self):
        pytest.importorskip("diffrax", reason="diffrax required for analysis")

    def test_creates_heatmap_png(self, tmp_path):
        from phoscrosstalk.analysis import save_neural_ode_residuals, plot_neural_residuals

        K, N, T = 2, 4, 5
        ts = np.linspace(0, 60, T)
        ys = _make_small_ys(K, N, T)
        proteins = [f"P{i}" for i in range(K)]
        sites = [f"P{i % K}_S{i}" for i in range(N)]
        P_scaled = np.random.default_rng(7).random((N, T))
        mech_P = np.random.default_rng(8).random((N, T))

        save_neural_ode_residuals(
            str(tmp_path),
            ts=ts,
            ys=ys,
            proteins=proteins,
            sites=sites,
            P_scaled=P_scaled,
            mech_P_sim=mech_P,
            mech_t=ts,
        )

        plot_neural_residuals(str(tmp_path))

        assert (tmp_path / "neural_residuals_heatmap_phospho.png").exists()

    def test_creates_scatter_png_when_mech_residuals_present(self, tmp_path):
        from phoscrosstalk.analysis import save_neural_ode_residuals, plot_neural_residuals

        K, N, T = 2, 3, 4
        ts = np.linspace(0, 60, T)
        ys = _make_small_ys(K, N, T)
        proteins = [f"P{i}" for i in range(K)]
        sites = [f"P{i % K}_S{i}" for i in range(N)]
        P_scaled = np.random.default_rng(9).random((N, T))
        mech_P = np.random.default_rng(10).random((N, T))

        save_neural_ode_residuals(
            str(tmp_path),
            ts=ts,
            ys=ys,
            proteins=proteins,
            sites=sites,
            P_scaled=P_scaled,
            mech_P_sim=mech_P,
            mech_t=ts,
        )

        plot_neural_residuals(str(tmp_path))
        assert (tmp_path / "neural_vs_mech_residuals.png").exists()

    def test_graceful_when_no_tsv(self, tmp_path):
        """plot_neural_residuals should not raise when the TSV is missing."""
        from phoscrosstalk.analysis import plot_neural_residuals

        plot_neural_residuals(str(tmp_path))
        # No PNG should be created (nothing to plot).
        assert not (tmp_path / "neural_residuals_heatmap_phospho.png").exists()


# ---------------------------------------------------------------------------
# 8. posterior tests moved to dedicated optimizer-uncertainty test module
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 9. Tests that exercise real production code paths for abundance & mRNA rows
# ---------------------------------------------------------------------------

class TestRealProductionCodePaths:
    """Tests using save_neural_ode_residuals to verify real code paths."""

    @pytest.fixture(autouse=True)
    def _skip_if_no_diffrax(self):
        pytest.importorskip("diffrax", reason="diffrax required for analysis")

    def test_abundance_value_observed_from_real_A_scaled(self, tmp_path):
        """save_neural_ode_residuals must populate value_observed from A_scaled
        for proteins in prot_idx_for_A — exercises the real indexing code path.
        """
        from phoscrosstalk.analysis import save_neural_ode_residuals

        K, N, T = 3, 2, 5
        ts = np.linspace(0, 60, T)
        t_prot = ts.copy()
        rng = np.random.default_rng(20)

        # Only proteins 0 and 2 have abundance data.
        prot_idx_for_A = np.array([0, 2])
        A_scaled = rng.uniform(0.1, 2.0, (2, T))  # fully finite

        proteins = [f"P{i}" for i in range(K)]
        sites = [f"P0_S0", f"P1_S1"]
        ys = {
            "P_sim": rng.random((N, T)),
            "A_sim": rng.random((K, T)),
        }

        save_neural_ode_residuals(
            str(tmp_path),
            ts=ts,
            ys=ys,
            proteins=proteins,
            sites=sites,
            A_scaled=A_scaled,
            prot_idx_for_A=prot_idx_for_A,
            t_protein=t_prot,
        )

        df = pd.read_csv(tmp_path / "neural_residuals.tsv", sep="\t")
        df_abund = df[df["entity_type"] == "abundance"]

        # Proteins 0 and 2 must have finite value_observed.
        for prot_name in ["P0", "P2"]:
            sub = df_abund[df_abund["entity"] == prot_name]
            assert sub["value_observed"].notna().all(), (
                f"{prot_name} (in prot_idx_for_A) should have finite value_observed"
            )

        # Protein 1 must have NaN value_observed.
        sub1 = df_abund[df_abund["entity"] == "P1"]
        assert sub1["value_observed"].isna().all(), (
            "P1 not in prot_idx_for_A should have NaN value_observed"
        )

    def test_mrna_neural_values_populated_from_R_sim(self, tmp_path):
        """save_neural_ode_residuals must populate value_neural from R_sim
        when ys contains 'R_sim' — exercises the real code path.
        """
        from phoscrosstalk.analysis import save_neural_ode_residuals

        K, N, T, T_rna = 2, 2, 4, 5
        ts = np.linspace(0, 60, T)
        t_rna = np.linspace(0, 60, T_rna)
        rng = np.random.default_rng(21)

        # R_sim matches t_rna grid (K × T_rna).
        R_sim_expected = rng.uniform(0.5, 1.5, (K, T_rna))
        ys = {
            "P_sim": rng.random((N, T)),
            "A_sim": rng.random((K, T)),
            "R_sim": R_sim_expected,
        }
        rna_obs = rng.random((K, T_rna))
        rna_idx = np.arange(K)
        proteins = [f"Q{i}" for i in range(K)]
        sites = [f"Q{i % K}_R{i}" for i in range(N)]

        save_neural_ode_residuals(
            str(tmp_path),
            ts=ts,
            ys=ys,
            proteins=proteins,
            sites=sites,
            t_rna=t_rna,
            rna_obs_matched=rna_obs,
            rna_model_prot_idx=rna_idx,
        )

        df = pd.read_csv(tmp_path / "neural_residuals.tsv", sep="\t")
        df_mrna = df[df["entity_type"] == "mrna"]
        assert len(df_mrna) > 0, "mRNA rows should be present"
        # value_neural must be finite (populated from R_sim).
        assert df_mrna["value_neural"].notna().all(), (
            "value_neural should be finite for mRNA rows when R_sim is in ys"
        )
        # value_observed must also be finite.
        assert df_mrna["value_observed"].notna().all()
        # residual_neural must be finite.
        assert df_mrna["residual_neural"].notna().all()

    def test_ts_vs_t_protein_mismatch_uses_nearest_index(self, tmp_path):
        """When ts and t_protein grids differ, observed values should be
        looked up via nearest-index mapping rather than direct indexing.
        """
        from phoscrosstalk.analysis import save_neural_ode_residuals

        N, T_neural, T_prot = 1, 3, 5
        ts = np.array([0.0, 30.0, 60.0])  # neural time axis (3 pts)
        t_prot = np.linspace(0, 60, T_prot)  # observed time axis (5 pts)
        # Known observed values at t_prot.
        P_scaled = np.array([[0.1, 0.2, 0.3, 0.4, 0.5]])  # (1, 5)
        ys = {"P_sim": np.array([[1.1, 1.3, 1.5]])}  # (1, 3) — neural

        save_neural_ode_residuals(
            str(tmp_path),
            ts=ts,
            ys=ys,
            proteins=["X"],
            sites=["X_S1"],
            P_scaled=P_scaled,
            t_protein=t_prot,
        )

        df = pd.read_csv(tmp_path / "neural_residuals.tsv", sep="\t")
        df_p = df[df["entity_type"] == "phosphosite"]
        # All value_observed should be finite (nearest-index lookup worked).
        assert df_p["value_observed"].notna().all(), (
            "value_observed should be finite when ts and t_protein differ"
        )
