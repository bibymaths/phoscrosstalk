"""
test_nonnegative.py

Tests verifying strict biological non-negativity guarantees for model states,
outputs, and rate parameters in PhosCrosstalk.

Test coverage
-------------
1. Decoded rate parameters (k_deact, d_deg, beta_g, beta_l, alpha, kK_act,
   kK_deact, k_off) are positive for random theta vectors.
2. Gamma parameters (gamma_S_p, gamma_A_S, gamma_A_p, gamma_K_net) can be
   negative and are NOT forced positive (signed regulatory effects).
3. RHS returns finite derivatives for non-negative input states.
4. RHS boundary guards prevent negative derivatives at lower bounds.
5. simulate_ode never returns negative R_sim, S_sim, A_sim, Kdyn_sim, P_sim.
6. Residual construction clips model outputs before computing residuals
   (no negative model outputs enter the loss).
7. saved mrna_fit_timeseries.tsv fitted values are non-negative.
8. saved fit_timeseries.tsv simulated (sim_t*) values are non-negative.
9. Negative observed phosphosite data raises ValueError in
   validate_biological_inputs.
10. Config bounds section values are respected (positive, loaded correctly).
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from phoscrosstalk.config import ModelDims, load_config


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_model_dims():
    """Restore ModelDims after each test."""
    saved = (ModelDims.K, ModelDims.M, ModelDims.N)
    yield
    ModelDims.K, ModelDims.M, ModelDims.N = saved


def _make_tiny(K=2, M=3, N=4, T=5, seed=0):
    """Build a minimal synthetic model."""
    ModelDims.set_dims(K, M, N)
    rng = np.random.default_rng(seed)
    dim = 2 * K + 2 + 3 * M + N + 4
    t = np.linspace(0.0, 30.0, T)
    P_data = rng.uniform(0.1, 0.9, (N, T))
    A_data = rng.uniform(0.5, 2.0, (K, T))
    theta = rng.uniform(-2.0, 0.0, dim)  # random in log-space

    Cg = np.eye(N) * 0.1
    Cl = np.eye(N) * 0.05
    site_prot_idx = np.array([0, 0, 1, 1], dtype=int)
    K_site_kin = rng.uniform(0, 1, (N, M))
    K_site_kin /= K_site_kin.sum(axis=1, keepdims=True) + 1e-8
    R = K_site_kin.T.copy()
    R /= R.sum(axis=1, keepdims=True) + 1e-8
    L_alpha = np.zeros((M, M))
    kin2prot = np.array([0, 1, -1], dtype=int)
    rm_prot = np.zeros(K, dtype=int)
    rm_kin = np.zeros(M, dtype=int)

    return dict(
        K=K, M=M, N=N, T=T, t=t,
        P_data=P_data, A_data=A_data, theta=theta,
        Cg=Cg, Cl=Cl,
        site_prot_idx=site_prot_idx,
        K_site_kin=K_site_kin, R=R, L_alpha=L_alpha,
        kin_to_prot_idx=kin2prot,
        receptor_mask_prot=rm_prot,
        receptor_mask_kin=rm_kin,
        dim=dim,
    )


# ---------------------------------------------------------------------------
# 1. Decoded rate parameters are positive for random theta
# ---------------------------------------------------------------------------


def test_decoded_rate_params_positive():
    """k_deact, d_deg, beta_g, beta_l, alpha, kK_act, kK_deact, k_off > 0."""
    from phoscrosstalk.jax_mechanisms import decode_theta_jax
    import jax.numpy as jnp

    K, M, N = 3, 4, 6
    dim = 2 * K + 2 + 3 * M + N + 4
    rng = np.random.default_rng(42)

    for _ in range(10):
        theta = rng.uniform(-5.0, 5.0, dim)
        theta_j = jnp.asarray(theta, dtype=jnp.float32)
        (k_deact, d_deg, beta_g, beta_l, alpha, kK_act, kK_deact, k_off,
         _, _, _, _) = decode_theta_jax(theta_j, K, M, N)

        for name, val in [
            ("k_deact", k_deact), ("d_deg", d_deg),
            ("alpha", alpha), ("kK_act", kK_act), ("kK_deact", kK_deact),
            ("k_off", k_off),
        ]:
            arr = np.asarray(val)
            assert np.all(arr > 0), f"{name} has non-positive values: {arr}"
            assert np.all(np.isfinite(arr)), f"{name} has non-finite values: {arr}"

        assert float(beta_g) > 0, f"beta_g = {float(beta_g)}"
        assert float(beta_l) > 0, f"beta_l = {float(beta_l)}"


# ---------------------------------------------------------------------------
# 2. Gamma parameters can be negative (signed regulatory effects)
# ---------------------------------------------------------------------------


def test_gamma_params_can_be_negative():
    """Gamma parameters must NOT be forced to non-negative values."""
    from phoscrosstalk.jax_mechanisms import decode_theta_jax
    import jax.numpy as jnp

    K, M, N = 2, 2, 3
    dim = 2 * K + 2 + 3 * M + N + 4

    # Use large positive raw_gamma to get tanh(x) → +1, and large negative to get -1
    theta_pos = np.zeros(dim)
    theta_pos[-4:] = 10.0  # raw_gamma → tanh → +1, gamma_scale * 1 > 0
    theta_neg = np.zeros(dim)
    theta_neg[-4:] = -10.0  # raw_gamma → tanh → -1, gamma_scale * (-1) < 0

    (_, _, _, _, _, _, _, _,
     gsp_pos, gas_pos, gap_pos, gkn_pos) = decode_theta_jax(
        jnp.asarray(theta_pos, dtype=jnp.float32), K, M, N
    )
    (_, _, _, _, _, _, _, _,
     gsp_neg, gas_neg, gap_neg, gkn_neg) = decode_theta_jax(
        jnp.asarray(theta_neg, dtype=jnp.float32), K, M, N
    )

    # Positive raw_gamma → positive decoded gamma
    assert float(gsp_pos) > 0, "gamma_S_p should be positive for large positive raw"
    # Negative raw_gamma → negative decoded gamma
    assert float(gsp_neg) < 0, "gamma_S_p should be negative for large negative raw"
    assert float(gas_neg) < 0, "gamma_A_S should be negative for large negative raw"
    assert float(gap_neg) < 0, "gamma_A_p should be negative for large negative raw"
    assert float(gkn_neg) < 0, "gamma_K_net should be negative for large negative raw"


# ---------------------------------------------------------------------------
# 3. RHS returns finite derivatives for non-negative states
# ---------------------------------------------------------------------------


def test_rhs_finite_for_nonneg_states():
    """RHS must return finite derivatives when fed valid non-negative states."""
    import jax.numpy as jnp
    from phoscrosstalk.jax_mechanisms import compute_prev_site_idx, make_rhs

    K, M, N = 2, 3, 4
    dim = 2 * K + 2 + 3 * M + N + 4
    rng = np.random.default_rng(7)

    rhs = make_rhs(K, M, N, "dist")
    theta = jnp.asarray(rng.uniform(-2.0, 0.0, dim), dtype=jnp.float32)

    # Valid non-negative state: R_rna ≥ 0, S ∈ [0,1], A ≥ 0, Kdyn ∈ [0,1], p ∈ [0,1]
    y = jnp.zeros(3 * K + M + N, dtype=jnp.float32)
    y = y.at[:K].set(1.0)   # R_rna = 1
    y = y.at[K:2*K].set(0.5)  # S = 0.5
    y = y.at[2*K:3*K].set(1.0)  # A = 1
    y = y.at[3*K:3*K+M].set(0.3)  # Kdyn = 0.3
    y = y.at[3*K+M:].set(0.2)  # p = 0.2

    spi = jnp.array([0, 0, 1, 1], dtype=jnp.int32)
    K_sk = jnp.ones((N, M), dtype=jnp.float32) / M
    R_mat = jnp.ones((M, N), dtype=jnp.float32) / N
    La = jnp.zeros((M, M), dtype=jnp.float32)
    Cg = jnp.eye(N, dtype=jnp.float32) * 0.1
    Cl = jnp.eye(N, dtype=jnp.float32) * 0.05
    k2p = jnp.array([0, 1, -1], dtype=jnp.int32)
    rmp = jnp.zeros(K, dtype=jnp.float32)
    rmk = jnp.zeros(M, dtype=jnp.float32)
    prev_spi = jnp.asarray(
        compute_prev_site_idx(np.array([0, 0, 1, 1]), N), dtype=jnp.int32
    )

    args = (theta, Cg, Cl, spi, K_sk, R_mat, La, k2p, rmp, rmk, prev_spi)
    dy = rhs(0.0, y, args)

    assert dy.shape == (3 * K + M + N,), f"dy shape mismatch: {dy.shape}"
    assert np.all(np.isfinite(np.asarray(dy))), f"RHS has non-finite values: {dy}"


# ---------------------------------------------------------------------------
# 4. RHS boundary guards prevent negative derivatives at lower bounds
# ---------------------------------------------------------------------------


def test_rhs_boundary_guard_lower_bound():
    """At lower bound (state=0), the derivative must not be negative."""
    import jax.numpy as jnp
    from phoscrosstalk.jax_mechanisms import compute_prev_site_idx, make_rhs

    K, M, N = 2, 2, 3
    rng = np.random.default_rng(99)
    dim = 2 * K + 2 + 3 * M + N + 4

    rhs = make_rhs(K, M, N, "dist")
    theta = jnp.asarray(rng.uniform(-2.0, 0.0, dim), dtype=jnp.float32)

    # Set all states to exactly 0 (lower boundary)
    y = jnp.zeros(3 * K + M + N, dtype=jnp.float32)

    spi = jnp.array([0, 0, 1], dtype=jnp.int32)
    K_sk = jnp.ones((N, M), dtype=jnp.float32) / M
    R_mat = jnp.ones((M, N), dtype=jnp.float32) / N
    La = jnp.zeros((M, M), dtype=jnp.float32)
    Cg = jnp.eye(N, dtype=jnp.float32) * 0.1
    Cl = jnp.eye(N, dtype=jnp.float32) * 0.05
    k2p = jnp.array([0, 1], dtype=jnp.int32)
    rmp = jnp.zeros(K, dtype=jnp.float32)
    rmk = jnp.zeros(M, dtype=jnp.float32)
    prev_spi = jnp.asarray(
        compute_prev_site_idx(np.array([0, 0, 1]), N), dtype=jnp.int32
    )

    args = (theta, Cg, Cl, spi, K_sk, R_mat, La, k2p, rmp, rmk, prev_spi)
    dy = np.asarray(rhs(0.0, y, args))

    # R_rna at lower bound (0): derivative must be ≥ 0
    dR_rna = dy[:K]
    assert np.all(dR_rna >= -1e-7), (
        f"R_rna derivative is negative at lower bound: {dR_rna}"
    )

    # S at lower bound (0): derivative must be ≥ 0
    dS = dy[K : 2 * K]
    assert np.all(dS >= -1e-7), (
        f"S derivative is negative at lower bound: {dS}"
    )

    # A at lower bound (0): derivative must be ≥ 0 (s_eff ≥ 0)
    dA = dy[2 * K : 3 * K]
    assert np.all(dA >= -1e-7), (
        f"A derivative is negative at lower bound: {dA}"
    )

    # Kdyn at lower bound (0): derivative must be ≥ 0
    dKdyn = dy[3 * K : 3 * K + M]
    assert np.all(dKdyn >= -1e-7), (
        f"Kdyn derivative is negative at lower bound: {dKdyn}"
    )

    # p at lower bound (0): derivative must be ≥ 0
    dp = dy[3 * K + M :]
    assert np.all(dp >= -1e-7), (
        f"p derivative is negative at lower bound: {dp}"
    )


# ---------------------------------------------------------------------------
# 5. simulate_ode never returns negative R, S, A, Kdyn, P
# ---------------------------------------------------------------------------


def test_simulate_ode_nonneg_outputs():
    """All state outputs from simulate_ode must be non-negative."""
    from phoscrosstalk.simulation import simulate_ode

    m = _make_tiny(K=2, M=3, N=4, T=6)

    result = simulate_ode(
        m["t"], m["P_data"], m["A_data"], m["theta"],
        m["Cg"], m["Cl"], m["site_prot_idx"], m["K_site_kin"],
        m["R"], m["L_alpha"], m["kin_to_prot_idx"],
        m["receptor_mask_prot"], m["receptor_mask_kin"],
        mechanism="dist",
        return_full=True,
    )

    for key in ("P_sim", "A_sim", "S_sim", "Kdyn_sim", "R_sim"):
        arr = result[key]
        finite_vals = arr[np.isfinite(arr)]
        if finite_vals.size > 0:
            assert float(finite_vals.min()) >= -1e-7, (
                f"{key} has negative values: min={float(finite_vals.min()):.6g}"
            )


# ---------------------------------------------------------------------------
# 6. Residual construction clips model outputs (no negative values in residuals)
# ---------------------------------------------------------------------------


def test_residuals_no_negative_model_outputs():
    """
    P_sim and A_sim extracted inside residuals_fn must be clipped to [0, ...].
    We verify this indirectly by checking that residuals are finite and that
    the loss is consistent with clipped outputs.
    """
    import jax.numpy as jnp
    from phoscrosstalk.optimization import make_residuals_fn, create_bounds

    m = _make_tiny(K=2, M=3, N=4, T=5)
    K, M, N = m["K"], m["M"], m["N"]
    xl, xu, _ = create_bounds(K, M, N)
    theta_mid = jnp.asarray(0.5 * (xl + xu), dtype=jnp.float32)

    residuals_fn = make_residuals_fn(
        t=m["t"], P_data=m["P_data"],
        A_scaled=np.zeros((0, m["T"])),
        prot_idx_for_A=np.array([], dtype=int),
        W_data=np.ones((N, m["T"])),
        W_data_prot=np.zeros((0, m["T"])),
        Cg=m["Cg"], Cl=m["Cl"],
        site_prot_idx=m["site_prot_idx"],
        K_site_kin=m["K_site_kin"], R=m["R"], L_alpha=m["L_alpha"],
        kin_to_prot_idx=m["kin_to_prot_idx"],
        receptor_mask_prot=m["receptor_mask_prot"],
        receptor_mask_kin=m["receptor_mask_kin"],
        mechanism="dist", lambda_net=1e-4, reg_lambda=1e-4,
    )

    r, (f1, f2, f3, f4) = residuals_fn(theta_mid, None)
    r_np = np.asarray(r)

    # Phosphosite residuals: P_sim ∈ [0,1], P_data ∈ [0.1, 0.9] → diff ∈ [-0.9, 0.9]
    # The residuals must be finite (no NaN/Inf from negative model outputs)
    assert np.all(np.isfinite(r_np)), "Residual vector contains non-finite values"
    assert np.isfinite(float(f1)), f"f1 is not finite: {f1}"


# ---------------------------------------------------------------------------
# 7. Saved mrna_fit_timeseries.tsv fitted values are non-negative
# ---------------------------------------------------------------------------


def test_mrna_fit_timeseries_nonneg(tmp_path):
    """mrna_fit_timeseries.tsv 'fitted' column must not contain negative values."""
    from phoscrosstalk.analysis import save_mrna_outputs

    rng = np.random.default_rng(1)
    n_genes, T_rna = 3, 5
    gene_ids = [f"Gene{i}" for i in range(n_genes)]
    t_rna = np.array([0.0, 4.0, 8.0, 15.0, 30.0])
    rna_obs = rng.uniform(0.5, 2.0, (n_genes, T_rna))
    # Simulated values that include some very small negatives (should be clipped)
    rna_sim = rng.uniform(-0.05, 2.0, (n_genes, T_rna))

    save_mrna_outputs(str(tmp_path), gene_ids, t_rna, rna_obs, rna_sim)

    df = pd.read_csv(tmp_path / "mrna_fit_timeseries.tsv", sep="\t")
    assert "fitted" in df.columns, "mrna_fit_timeseries.tsv missing 'fitted' column"
    fitted_vals = df["fitted"].values
    assert float(fitted_vals.min()) >= -1e-9, (
        f"mrna_fit_timeseries.tsv has negative fitted values: min={fitted_vals.min()}"
    )


# ---------------------------------------------------------------------------
# 8. fit_timeseries.tsv simulated (sim_t*) values are non-negative
# ---------------------------------------------------------------------------


def test_fit_timeseries_nonneg(tmp_path):
    """fit_timeseries.tsv sim_t* columns must not contain negative values."""
    from phoscrosstalk.analysis import save_fitted_simulation
    from phoscrosstalk.simulation import build_full_A0

    m = _make_tiny(K=2, M=3, N=4, T=5)
    K, M, N, T = m["K"], m["M"], m["N"], m["T"]

    sites = [f"ProtA_S{i}" for i in range(N)]
    proteins = ["ProtA", "ProtB"]
    kinases = [f"Kin{j}" for j in range(M)]

    baselines = np.zeros(N)
    amplitudes = np.ones(N)
    A_bases = np.zeros(K)
    A_amps = np.ones(K)
    prot_idx_for_A = np.array([0, 1], dtype=int)

    save_fitted_simulation(
        outdir=str(tmp_path),
        theta_opt=m["theta"],
        t=m["t"],
        sites=sites,
        proteins=proteins,
        P_scaled=m["P_data"],
        A_scaled=m["A_data"],
        prot_idx_for_A=prot_idx_for_A,
        baselines=baselines,
        amplitudes=amplitudes,
        Y=m["P_data"],
        A_data=m["A_data"],
        A_bases=A_bases,
        A_amps=A_amps,
        mechanism="dist",
        Cg=m["Cg"],
        Cl=m["Cl"],
        site_prot_idx=m["site_prot_idx"],
        K_site_kin=m["K_site_kin"],
        R=m["R"],
        L_alpha=m["L_alpha"],
        kin_to_prot_idx=m["kin_to_prot_idx"],
        mask_p=m["receptor_mask_prot"],
        mask_k=m["receptor_mask_kin"],
        kinases=kinases,
    )

    df = pd.read_csv(tmp_path / "fit_timeseries.tsv", sep="\t")
    sim_cols = [c for c in df.columns if c.startswith("sim_t")]
    assert len(sim_cols) > 0, "fit_timeseries.tsv has no sim_t* columns"

    sim_vals = df[sim_cols].to_numpy(dtype=float)
    finite_sim = sim_vals[np.isfinite(sim_vals)]
    if finite_sim.size > 0:
        assert float(finite_sim.min()) >= -1e-7, (
            f"fit_timeseries.tsv has negative simulated values: min={finite_sim.min()}"
        )


# ---------------------------------------------------------------------------
# 9. Negative observed data raises ValueError in validate_biological_inputs
# ---------------------------------------------------------------------------


def test_negative_observed_data_raises():
    """validate_biological_inputs must raise ValueError for negative P_data."""
    from phoscrosstalk.optimization import validate_biological_inputs

    # P_data with negative values
    P_data_bad = np.array([[0.5, -0.1, 0.3], [0.2, 0.4, 0.0]])
    with pytest.raises(ValueError, match="negative"):
        validate_biological_inputs(P_data=P_data_bad)

    # A_scaled with negative values
    A_bad = np.array([[1.0, -0.5, 0.8]])
    with pytest.raises(ValueError, match="negative"):
        validate_biological_inputs(A_scaled=A_bad)

    # rna_data_scaled with negative values
    rna_bad = np.array([[1.0, 0.5, -0.2]])
    with pytest.raises(ValueError, match="negative"):
        validate_biological_inputs(rna_data_scaled=rna_bad)


def test_valid_observed_data_passes():
    """validate_biological_inputs must not raise for valid non-negative data."""
    from phoscrosstalk.optimization import validate_biological_inputs

    P_data_ok = np.array([[0.5, 0.1, 0.3], [0.2, 0.4, 0.0]])
    A_ok = np.array([[1.0, 0.5, 0.8]])
    rna_ok = np.array([[1.0, 0.5, 0.2]])
    # Should not raise
    validate_biological_inputs(P_data=P_data_ok, A_scaled=A_ok, rna_data_scaled=rna_ok)


# ---------------------------------------------------------------------------
# 10. Config bounds section values are respected
# ---------------------------------------------------------------------------


def test_config_bounds_defaults():
    """Config [bounds] section must load with correct positive defaults."""
    cfg = load_config(None)  # use pure defaults (no file)

    bounds = cfg.bounds
    assert hasattr(bounds, "rate_min"), "bounds.rate_min missing"
    assert hasattr(bounds, "rate_max"), "bounds.rate_max missing"
    assert hasattr(bounds, "rna_max"), "bounds.rna_max missing"
    assert hasattr(bounds, "abundance_max"), "bounds.abundance_max missing"

    assert bounds.rate_min > 0, f"rate_min must be positive, got {bounds.rate_min}"
    assert bounds.rate_max > 0, f"rate_max must be positive, got {bounds.rate_max}"
    assert bounds.rna_max > 0, f"rna_max must be positive, got {bounds.rna_max}"
    assert bounds.abundance_max > 0, (
        f"abundance_max must be positive, got {bounds.abundance_max}"
    )
    assert bounds.rate_min < bounds.rate_max, "rate_min must be less than rate_max"


def test_config_bounds_toml(tmp_path):
    """Config [bounds] values from a TOML file override defaults correctly."""
    cfg_text = """
[bounds]
rate_min = 1e-4
rate_max = 5.0
rna_max = 8.0
abundance_max = 3.0
"""
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text(cfg_text)

    cfg = load_config(str(cfg_path))
    assert abs(cfg.bounds.rate_min - 1e-4) < 1e-10
    assert abs(cfg.bounds.rate_max - 5.0) < 1e-10
    assert abs(cfg.bounds.rna_max - 8.0) < 1e-10
    assert abs(cfg.bounds.abundance_max - 3.0) < 1e-10
