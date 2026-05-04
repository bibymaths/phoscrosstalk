"""
test_rna.py

Tests for RNA-as-ODE-state features:
  1. test_load_rna_data_x1_x9_timepoints
  2. test_tf_network_header_case_and_direction
  3. test_match_rna_to_model_proteins
  4. test_simulate_ode_returns_R_state
  5. test_rhs_state_dimension_with_R
  6. test_rna_loss_nonzero_when_sim_differs
  7. test_save_mrna_outputs_requires_simulated
  8. test_network_problem_loss_includes_rna
  9. test_main_cli_with_rna_args
 10. test_no_old_theta_indices
 11. test_plot_three_panel_fit
 12. test_backward_simulate_alias
"""

import sys
import tempfile
import os

import numpy as np
import pandas as pd
import pytest

from phoscrosstalk.config import ModelDims


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_model_dims():
    """Restore ModelDims after each test."""
    saved = (ModelDims.K, ModelDims.M, ModelDims.N)
    yield
    ModelDims.K, ModelDims.M, ModelDims.N = saved


def _make_tiny_model(K=2, M=3, N=4, T=6, seed=0):
    """Build small synthetic model arrays for testing."""
    ModelDims.set_dims(K, M, N)
    rng = np.random.default_rng(seed)

    dim = 2 * K + 2 + 3 * M + N + 4
    t = np.linspace(0.0, 60.0, T)
    P_data = rng.uniform(0.1, 0.9, (N, T))
    A_data = rng.uniform(0.5, 1.5, (K, T))
    theta = rng.uniform(-3.0, -0.1, dim)

    Cg = np.eye(N) * 0.1
    Cl = np.eye(N) * 0.05
    site_prot_idx = np.array([0, 0, 1, 1], dtype=int)
    K_site_kin = rng.uniform(0, 1, (N, M))
    K_site_kin /= K_site_kin.sum(axis=1, keepdims=True) + 1e-8
    R_mat = K_site_kin.T.copy()
    R_mat /= R_mat.sum(axis=1, keepdims=True) + 1e-8
    L_alpha = np.zeros((M, M))
    kin2prot = np.array([0, 1, -1], dtype=int)
    rm_prot = np.zeros(K, dtype=int)
    rm_kin = np.zeros(M, dtype=int)

    return dict(
        K=K, M=M, N=N, T=T, t=t,
        P_data=P_data, A_data=A_data, theta=theta,
        Cg=Cg, Cl=Cl, site_prot_idx=site_prot_idx,
        K_site_kin=K_site_kin, R=R_mat, L_alpha=L_alpha,
        kin_to_prot_idx=kin2prot,
        receptor_mask_prot=rm_prot,
        receptor_mask_kin=rm_kin,
    )


# ---------------------------------------------------------------------------
# 1. RNA time-point mapping (x1..x9 -> [4,8,15,30,60,120,240,480,960])
# ---------------------------------------------------------------------------


def test_load_rna_data_x1_x9_timepoints(tmp_path):
    """Columns x1..x9 must map to the canonical RNA time points."""
    from phoscrosstalk.data_loader import load_rna_data

    RNA_TIMES = [4, 8, 15, 30, 60, 120, 240, 480, 960]
    rng = np.random.default_rng(1)
    vals = rng.uniform(0.5, 2.0, 9)
    df = pd.DataFrame(
        {
            "gene": ["EGFR"],
            **{f"x{i+1}": [float(vals[i])] for i in range(9)},
        }
    )
    p = tmp_path / "rna_x1_x9.csv"
    df.to_csv(p, index=False)

    gene_ids, t_rna, matrix = load_rna_data(str(p))
    assert list(t_rna) == RNA_TIMES, f"Expected {RNA_TIMES}, got {list(t_rna)}"
    assert matrix.shape == (1, 9)


# ---------------------------------------------------------------------------
# 2. TF network header case / direction
# ---------------------------------------------------------------------------


def test_tf_network_header_case_and_direction(tmp_path):
    """Source/Target/Weight (capitalized) should load correctly."""
    from phoscrosstalk.data_loader import load_tf_network

    df = pd.DataFrame(
        {"Source": ["EGFR", "AKT1"], "Target": ["MAPK1", "MAPK1"], "Weight": [0.5, 1.0]}
    )
    p = tmp_path / "tf_caps.csv"
    df.to_csv(p, index=False)

    result = load_tf_network(str(p))
    assert list(result.columns) == ["source", "target", "weight"]
    assert len(result) == 2
    weights = sorted(result["weight"].tolist())
    assert weights == pytest.approx([0.5, 1.0])


# ---------------------------------------------------------------------------
# 3. match_rna_to_model_proteins
# ---------------------------------------------------------------------------


def test_match_rna_to_model_proteins_overlap():
    """Only genes overlapping with model proteins are fitted."""
    from phoscrosstalk.data_loader import match_rna_to_model_proteins

    gene_ids = ["EGFR", "TF1_source", "MAPK1", "AKT1"]
    rna_matrix = np.ones((4, 5))
    proteins = ["MAPK1", "AKT1", "PTEN"]

    matched, obs_mat, prot_idx, gene_idx = match_rna_to_model_proteins(
        gene_ids, rna_matrix, proteins
    )

    assert set(matched) == {"MAPK1", "AKT1"}
    assert obs_mat.shape == (2, 5)
    assert len(prot_idx) == 2
    assert set(prot_idx.tolist()) == {0, 1}  # MAPK1=0, AKT1=1 in proteins


def test_match_rna_to_model_proteins_no_overlap():
    """Returns empty results when there is no overlap."""
    from phoscrosstalk.data_loader import match_rna_to_model_proteins

    matched, obs_mat, prot_idx, gene_idx = match_rna_to_model_proteins(
        ["TF_source"], np.ones((1, 3)), ["MAPK1"]
    )
    assert matched == []
    assert obs_mat.shape == (0, 3)
    assert len(prot_idx) == 0


# ---------------------------------------------------------------------------
# 4. simulate_ode returns R state
# ---------------------------------------------------------------------------


def test_simulate_ode_returns_R_state():
    """return_full=True must include R_sim and R_sim_rna with correct shapes."""
    from phoscrosstalk.simulation import simulate_ode

    m = _make_tiny_model(K=2, M=3, N=4, T=6)
    K, M, N = m["K"], m["M"], m["N"]
    t_rna = np.array([4.0, 8.0, 30.0, 60.0])

    result = simulate_ode(
        m["t"],
        m["P_data"],
        m["A_data"],
        m["theta"],
        m["Cg"],
        m["Cl"],
        m["site_prot_idx"],
        m["K_site_kin"],
        m["R"],
        m["L_alpha"],
        m["kin_to_prot_idx"],
        m["receptor_mask_prot"],
        m["receptor_mask_kin"],
        mechanism="dist",
        return_full=True,
        t_rna=t_rna,
    )

    assert isinstance(result, dict), "return_full=True must return a dict"
    assert "R_sim" in result, "R_sim missing from full result"
    assert "R_sim_rna" in result, "R_sim_rna missing from full result"

    R_sim = result["R_sim"]
    R_sim_rna = result["R_sim_rna"]

    assert R_sim.shape == (K, m["T"]), f"R_sim shape wrong: {R_sim.shape}"
    assert R_sim_rna.shape == (K, len(t_rna)), f"R_sim_rna shape wrong: {R_sim_rna.shape}"
    assert np.all(np.isfinite(R_sim)), "R_sim contains NaN/Inf"
    assert np.all(np.isfinite(R_sim_rna)), "R_sim_rna contains NaN/Inf"
    # RNA state should be non-negative (fold-change)
    assert np.all(R_sim >= 0.0), "R_sim has negative values"


# ---------------------------------------------------------------------------
# 5. RHS state dimension with R
# ---------------------------------------------------------------------------


def test_rhs_state_dimension_with_R():
    """RHS must return shape 3*K + M + N when R_rna is included."""
    import jax.numpy as jnp
    from phoscrosstalk.jax_mechanisms import make_rhs, compute_prev_site_idx

    K, M, N = 2, 3, 4
    rhs = make_rhs(K, M, N, "dist")

    dim = 2 * K + 2 + 3 * M + N + 4  # theta dim
    theta_j = jnp.zeros(dim, dtype=jnp.float32)
    spi = np.array([0, 0, 1, 1], dtype=np.int32)
    prev = compute_prev_site_idx(spi, N)
    Cg = jnp.eye(N, dtype=jnp.float32)
    Cl = jnp.eye(N, dtype=jnp.float32)
    K_sk = jnp.ones((N, M), dtype=jnp.float32) / M
    R_mat = jnp.ones((M, N), dtype=jnp.float32) / N
    La = jnp.zeros((M, M), dtype=jnp.float32)
    k2p = jnp.array([0, 1, -1], dtype=jnp.int32)
    rmp = jnp.zeros(K, dtype=jnp.float32)
    rmk = jnp.zeros(M, dtype=jnp.float32)

    args = (
        theta_j, Cg, Cl,
        jnp.asarray(spi, dtype=jnp.int32),
        K_sk, R_mat, La, k2p, rmp, rmk,
        jnp.asarray(prev, dtype=jnp.int32),
    )

    expected_state_dim = 3 * K + M + N
    y = jnp.zeros(expected_state_dim, dtype=jnp.float32)
    dy = rhs(0.0, y, args)
    assert dy.shape == (expected_state_dim,), (
        f"RHS output shape {dy.shape}, expected ({expected_state_dim},)"
    )


# ---------------------------------------------------------------------------
# 6. RNA loss is nonzero when simulated differs from observed
# ---------------------------------------------------------------------------


def test_rna_loss_nonzero_when_sim_differs(tmp_path):
    """save_mrna_outputs must not produce zero residual unless sim == obs."""
    from phoscrosstalk.analysis import save_mrna_outputs

    rng = np.random.default_rng(42)
    t_rna = np.array([4.0, 8.0, 30.0])
    obs = rng.uniform(0.5, 2.0, (2, 3))
    sim = obs + rng.uniform(0.05, 0.3, obs.shape)  # deliberately different

    save_mrna_outputs(
        outdir=str(tmp_path),
        gene_ids=["MAPK1", "AKT1"],
        t_rna=t_rna,
        rna_data_obs=obs,
        rna_simulated=sim,
    )

    diag = pd.read_csv(tmp_path / "mrna_diagnostics.tsv", sep="\t")
    assert "rmse" in diag.columns
    # Residuals must be nonzero since sim != obs
    assert (diag["rmse"] > 1e-8).all(), "RMSE should be nonzero when sim != obs"

    ts = pd.read_csv(tmp_path / "mrna_fit_timeseries.tsv", sep="\t")
    residuals = (ts["observed"] - ts["fitted"]).abs()
    assert residuals.max() > 1e-8, "Residuals should be nonzero"


# ---------------------------------------------------------------------------
# 7. save_mrna_outputs raises ValueError when rna_simulated is None
# ---------------------------------------------------------------------------


def test_save_mrna_outputs_requires_simulated(tmp_path):
    """save_mrna_outputs must raise ValueError if rna_simulated is None."""
    from phoscrosstalk.analysis import save_mrna_outputs

    with pytest.raises(ValueError, match="rna_simulated must not be None"):
        save_mrna_outputs(
            outdir=str(tmp_path),
            gene_ids=["EGFR"],
            t_rna=np.array([4.0, 8.0]),
            rna_data_obs=np.ones((1, 2)),
            rna_simulated=None,
        )


# ---------------------------------------------------------------------------
# 8. NetworkProblem total loss changes when RNA observations are perturbed
# ---------------------------------------------------------------------------


def test_network_problem_loss_includes_rna():
    """Total loss must differ when RNA observations are perturbed."""
    import jax.numpy as jnp
    from phoscrosstalk.optimization import make_loss_fn, create_bounds

    m = _make_tiny_model(K=2, M=3, N=4, T=6)
    K, M, N = m["K"], m["M"], m["N"]
    xl, xu, _ = create_bounds(K, M, N)
    theta0 = jnp.asarray(0.5 * (xl + xu), dtype=jnp.float32)

    t_rna = np.array([4.0, 8.0, 30.0, 60.0])
    rna_obs = np.ones((1, len(t_rna)), dtype=np.float32)  # 1 matched gene
    rna_prot_idx = np.array([0], dtype=int)  # maps to protein 0

    loss_no_rna = make_loss_fn(
        t=m["t"], P_data=m["P_data"],
        A_scaled=np.zeros((0, m["T"])), prot_idx_for_A=np.array([], dtype=int),
        W_data=np.ones((N, m["T"])), W_data_prot=np.zeros((0, m["T"])),
        Cg=m["Cg"], Cl=m["Cl"], site_prot_idx=m["site_prot_idx"],
        K_site_kin=m["K_site_kin"], R=m["R"], L_alpha=m["L_alpha"],
        kin_to_prot_idx=m["kin_to_prot_idx"],
        receptor_mask_prot=m["receptor_mask_prot"],
        receptor_mask_kin=m["receptor_mask_kin"],
        mechanism="dist", lambda_net=1e-4, reg_lambda=1e-4,
    )

    loss_with_rna = make_loss_fn(
        t=m["t"], P_data=m["P_data"],
        A_scaled=np.zeros((0, m["T"])), prot_idx_for_A=np.array([], dtype=int),
        W_data=np.ones((N, m["T"])), W_data_prot=np.zeros((0, m["T"])),
        Cg=m["Cg"], Cl=m["Cl"], site_prot_idx=m["site_prot_idx"],
        K_site_kin=m["K_site_kin"], R=m["R"], L_alpha=m["L_alpha"],
        kin_to_prot_idx=m["kin_to_prot_idx"],
        receptor_mask_prot=m["receptor_mask_prot"],
        receptor_mask_kin=m["receptor_mask_kin"],
        mechanism="dist", lambda_net=1e-4, reg_lambda=1e-4,
        t_mrna=t_rna, rna_data_scaled=rna_obs * 100.0,  # large mismatch
        rna_model_prot_idx=rna_prot_idx, w_mrna=1.0,
    )

    total_no, _ = loss_no_rna(theta0, None)
    total_rna, (_, _, _, f4) = loss_with_rna(theta0, None)

    assert float(f4) > 0.0, "f4 (RNA loss) should be nonzero with mismatched RNA"
    assert float(total_rna) != float(total_no), (
        "Total loss should change when RNA observations are included"
    )


# ---------------------------------------------------------------------------
# 9. CLI parses --rna-data and --tf-net
# ---------------------------------------------------------------------------


def test_main_cli_with_rna_args(monkeypatch):
    """CLI must accept --rna-data and --tf-net without argparse error."""
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "phoscrosstalk",
            "--data", "fake.csv",
            "--ptm-intra", "fake.db",
            "--ptm-inter", "fake.db",
            "--rna-data", "fake_rna.csv",
            "--tf-net", "fake_tf.csv",
            "--n-starts", "1",
            "--max-steps", "5",
        ],
    )
    from phoscrosstalk.main import main

    with pytest.raises((SystemExit, Exception)) as exc_info:
        main()
    exc = exc_info.value
    if isinstance(exc, SystemExit):
        assert exc.code != 2, "--rna-data / --tf-net rejected by argparse (exit 2)"


# ---------------------------------------------------------------------------
# 10. No old hard-coded theta indices (alpha must be at 2*K+2)
# ---------------------------------------------------------------------------


def test_no_old_theta_indices():
    """idx_alpha in knockouts.py must equal 2*K+2, not 4*K+2."""
    import importlib.util
    import re

    spec = importlib.util.find_spec("phoscrosstalk.knockouts")
    with open(spec.origin) as f:
        src = f.read()

    # Must NOT contain '4 * K + 2' or '4*K+2'
    assert not re.search(r"4\s*\*\s*K\s*\+\s*2", src), (
        "knockouts.py still has hard-coded 4*K+2 alpha index"
    )

    # Must contain the correct 2*K+2 index
    assert re.search(r"2\s*\*\s*K\s*\+\s*2", src), (
        "knockouts.py must use 2*K+2 for alpha index"
    )


# ---------------------------------------------------------------------------
# 11. Three-panel plot is created when RNA data is present
# ---------------------------------------------------------------------------


def test_plot_three_panel_fit(tmp_path):
    """plot_fitted_simulation must create a plot file when mRNA data present."""
    from phoscrosstalk.analysis import plot_fitted_simulation

    outdir = str(tmp_path)

    # Write minimal fit_timeseries.tsv (2 phosphosites)
    t_cols_sim = [f"sim_t{j}" for j in range(3)]
    t_cols_dat = [f"data_t{j}" for j in range(3)]
    rows = []
    for i in range(2):
        row = {"Type": "Phosphosite", "Protein": "MAPK1", "Residue": f"S{i+1}"}
        for j in range(3):
            row[t_cols_sim[j]] = 0.5
            row[t_cols_dat[j]] = 0.6
        rows.append(row)
    # Protein abundance row
    row = {"Type": "ProteinAbundance", "Protein": "MAPK1", "Residue": ""}
    for j in range(3):
        row[t_cols_sim[j]] = 1.1
        row[t_cols_dat[j]] = 1.0
    rows.append(row)
    pd.DataFrame(rows).to_csv(os.path.join(outdir, "fit_timeseries.tsv"), sep="\t", index=False)

    # Write minimal mrna_fit_timeseries.tsv
    rna_rows = [
        {"gene": "MAPK1", "time": 4.0, "observed": 1.0, "fitted": 1.1},
        {"gene": "MAPK1", "time": 30.0, "observed": 1.5, "fitted": 1.4},
    ]
    pd.DataFrame(rna_rows).to_csv(
        os.path.join(outdir, "mrna_fit_timeseries.tsv"), sep="\t", index=False
    )

    plot_fitted_simulation(outdir)

    # At least one plot file should be created
    png_files = [f for f in os.listdir(outdir) if f.endswith(".png")]
    assert len(png_files) > 0, "No PNG file created by plot_fitted_simulation"


# ---------------------------------------------------------------------------
# 12. Backward-compatible simulate alias
# ---------------------------------------------------------------------------


def test_backward_simulate_alias():
    """simulate_p_scipy must be the same function as simulate_ode."""
    from phoscrosstalk.simulation import simulate_p_scipy, simulate_ode

    assert simulate_p_scipy is simulate_ode, (
        "simulate_p_scipy must be an alias for simulate_ode"
    )

    # Also verify it returns old-compatible 2-tuple by default
    m = _make_tiny_model(K=2, M=3, N=4, T=4)
    result = simulate_p_scipy(
        m["t"],
        m["P_data"],
        m["A_data"],
        m["theta"],
        m["Cg"],
        m["Cl"],
        m["site_prot_idx"],
        m["K_site_kin"],
        m["R"],
        m["L_alpha"],
        m["kin_to_prot_idx"],
        m["receptor_mask_prot"],
        m["receptor_mask_kin"],
        "dist",
    )
    assert len(result) == 2, "Default simulate_p_scipy should return (P_sim, A_sim)"
    P_sim, A_sim = result
    assert P_sim.shape == (m["N"], m["T"])
    assert A_sim.shape == (m["K"], m["T"])
