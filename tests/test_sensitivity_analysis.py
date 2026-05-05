# SPDX-License-Identifier: MIT
"""
test_sensitivity_analysis.py

Tests for build_parameter_labels and compute_second_order_sensitivities.

Coverage:
  A. build_parameter_labels
     1. Length matches 2*K + 2 + 3*M + N + 4.
     2. First K labels are log_k_deact[0..K-1].
     3. Coupling labels (log_beta_g, log_beta_l) are at correct positions.
     4. Final 4 labels are gamma_raw[0..3].
     5. Full ordering matches documented theta layout.

  B. compute_second_order_sensitivities with a fake quadratic loss.
     1. Returned array shape is (n, n).
     2. Values are close to 2 * I (Hessian of sum(theta^2)).
     3. .npy file exists and contents match returned Hessian.
     4. .tsv file exists, contains param labels in header and row labels.
     5. _heatmap.png file exists.

  C. Regression smoke test: scan_kind="bounded" change does not break
     make_loss_fn interface (tiny model, check finite output).
"""

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

K, M, N = 2, 3, 4


@pytest.fixture(autouse=True)
def reset_model_dims():
    """Restore ModelDims after each test."""
    from phoscrosstalk.config import ModelDims

    saved = (ModelDims.K, ModelDims.M, ModelDims.N)
    yield
    ModelDims.K, ModelDims.M, ModelDims.N = saved


# ---------------------------------------------------------------------------
# A. build_parameter_labels
# ---------------------------------------------------------------------------


class TestBuildParameterLabels:
    def test_length(self):
        from phoscrosstalk.optimization import build_parameter_labels

        labels = build_parameter_labels(K, M, N)
        expected_len = 2 * K + 2 + 3 * M + N + 4
        assert len(labels) == expected_len, (
            f"Expected {expected_len} labels, got {len(labels)}"
        )

    def test_first_labels_are_k_deact(self):
        from phoscrosstalk.optimization import build_parameter_labels

        labels = build_parameter_labels(K, M, N)
        for k in range(K):
            assert labels[k] == f"log_k_deact[{k}]", (
                f"labels[{k}]={labels[k]!r} expected 'log_k_deact[{k}]'"
            )

    def test_d_deg_labels(self):
        from phoscrosstalk.optimization import build_parameter_labels

        labels = build_parameter_labels(K, M, N)
        for k in range(K):
            assert labels[K + k] == f"log_d_deg[{k}]", (
                f"labels[{K + k}]={labels[K + k]!r} expected 'log_d_deg[{k}]'"
            )

    def test_coupling_labels(self):
        from phoscrosstalk.optimization import build_parameter_labels

        labels = build_parameter_labels(K, M, N)
        assert labels[2 * K] == "log_beta_g"
        assert labels[2 * K + 1] == "log_beta_l"

    def test_kinase_labels_positions(self):
        from phoscrosstalk.optimization import build_parameter_labels

        labels = build_parameter_labels(K, M, N)
        base = 2 * K + 2
        for m in range(M):
            assert labels[base + m] == f"log_alpha[{m}]"
        for m in range(M):
            assert labels[base + M + m] == f"log_kK_act[{m}]"
        for m in range(M):
            assert labels[base + 2 * M + m] == f"log_kK_deact[{m}]"

    def test_k_off_labels(self):
        from phoscrosstalk.optimization import build_parameter_labels

        labels = build_parameter_labels(K, M, N)
        base = 2 * K + 2 + 3 * M
        for n in range(N):
            assert labels[base + n] == f"log_k_off[{n}]"

    def test_final_labels_are_gamma_raw(self):
        from phoscrosstalk.optimization import build_parameter_labels

        labels = build_parameter_labels(K, M, N)
        for i in range(4):
            assert labels[-(4 - i)] == f"gamma_raw[{i}]"

    def test_exact_ordering(self):
        """Full ordering matches the documented theta layout."""
        from phoscrosstalk.optimization import build_parameter_labels

        labels = build_parameter_labels(K, M, N)
        expected = (
            [f"log_k_deact[{k}]" for k in range(K)]
            + [f"log_d_deg[{k}]" for k in range(K)]
            + ["log_beta_g", "log_beta_l"]
            + [f"log_alpha[{m}]" for m in range(M)]
            + [f"log_kK_act[{m}]" for m in range(M)]
            + [f"log_kK_deact[{m}]" for m in range(M)]
            + [f"log_k_off[{n}]" for n in range(N)]
            + [f"gamma_raw[{i}]" for i in range(4)]
        )
        assert labels == expected


# ---------------------------------------------------------------------------
# B. compute_second_order_sensitivities with fake quadratic loss
# ---------------------------------------------------------------------------


class TestComputeSecondOrderSensitivities:
    def _fake_loss_fn(self):
        """loss_fn(theta, _) = (sum(theta^2), None). Hessian = 2*I."""
        import jax.numpy as jnp

        def loss_fn(theta, _args):
            return jnp.sum(theta**2), None

        return loss_fn

    def test_returned_shape(self, tmp_path):
        from phoscrosstalk.optimization import (
            build_parameter_labels,
            compute_second_order_sensitivities,
        )

        labels = build_parameter_labels(K, M, N)
        n = len(labels)
        rng = np.random.default_rng(42)
        theta = rng.uniform(-1, 1, n).astype(np.float64)

        H = compute_second_order_sensitivities(
            theta=theta,
            loss_fn=self._fake_loss_fn(),
            param_labels=labels,
            out_dir=tmp_path,
            prefix="test_hessian",
        )
        assert H.shape == (n, n), f"Expected ({n},{n}), got {H.shape}"

    def test_values_close_to_2I(self, tmp_path):
        from phoscrosstalk.optimization import (
            build_parameter_labels,
            compute_second_order_sensitivities,
        )

        labels = build_parameter_labels(K, M, N)
        n = len(labels)
        theta = np.zeros(n, dtype=np.float64)

        H = compute_second_order_sensitivities(
            theta=theta,
            loss_fn=self._fake_loss_fn(),
            param_labels=labels,
            out_dir=tmp_path,
            prefix="test_hessian",
        )
        np.testing.assert_allclose(H, 2.0 * np.eye(n), atol=1e-4)

    def test_npy_file_exists_and_matches(self, tmp_path):
        from phoscrosstalk.optimization import (
            build_parameter_labels,
            compute_second_order_sensitivities,
        )

        labels = build_parameter_labels(K, M, N)
        theta = np.zeros(len(labels), dtype=np.float64)

        H = compute_second_order_sensitivities(
            theta=theta,
            loss_fn=self._fake_loss_fn(),
            param_labels=labels,
            out_dir=tmp_path,
            prefix="test_hessian",
        )
        npy_path = tmp_path / "test_hessian.npy"
        assert npy_path.exists(), ".npy file not created"
        H_loaded = np.load(npy_path)
        np.testing.assert_array_equal(H, H_loaded)

    def test_tsv_file_exists_and_has_labels(self, tmp_path):
        from phoscrosstalk.optimization import (
            build_parameter_labels,
            compute_second_order_sensitivities,
        )

        labels = build_parameter_labels(K, M, N)
        theta = np.zeros(len(labels), dtype=np.float64)

        compute_second_order_sensitivities(
            theta=theta,
            loss_fn=self._fake_loss_fn(),
            param_labels=labels,
            out_dir=tmp_path,
            prefix="test_hessian",
        )
        tsv_path = tmp_path / "test_hessian.tsv"
        assert tsv_path.exists(), ".tsv file not created"

        lines = tsv_path.read_text().splitlines()
        # First line: tab + column labels
        header = lines[0]
        assert header.startswith("\t"), "Header must start with a tab"
        col_labels = header.lstrip("\t").split("\t")
        assert col_labels == list(labels), "Column labels in TSV header mismatch"

        # Data rows: first token is row label
        for i, label in enumerate(labels):
            row_parts = lines[i + 1].split("\t")
            assert row_parts[0] == label, (
                f"Row label mismatch at row {i}: got {row_parts[0]!r}, "
                f"expected {label!r}"
            )

    def test_heatmap_png_exists(self, tmp_path):
        from phoscrosstalk.optimization import (
            build_parameter_labels,
            compute_second_order_sensitivities,
        )

        labels = build_parameter_labels(K, M, N)
        theta = np.zeros(len(labels), dtype=np.float64)

        compute_second_order_sensitivities(
            theta=theta,
            loss_fn=self._fake_loss_fn(),
            param_labels=labels,
            out_dir=tmp_path,
            prefix="test_hessian",
        )
        assert (tmp_path / "test_hessian_heatmap.png").exists(), (
            "_heatmap.png not created"
        )

    def test_dtype_is_float64(self, tmp_path):
        from phoscrosstalk.optimization import (
            build_parameter_labels,
            compute_second_order_sensitivities,
        )

        labels = build_parameter_labels(K, M, N)
        theta = np.zeros(len(labels), dtype=np.float64)

        H = compute_second_order_sensitivities(
            theta=theta,
            loss_fn=self._fake_loss_fn(),
            param_labels=labels,
            out_dir=tmp_path,
            prefix="test_hessian",
        )
        assert H.dtype == np.float64, f"Expected float64, got {H.dtype}"


# ---------------------------------------------------------------------------
# C. Regression: make_loss_fn with scan_kind="bounded" still produces finite output
# ---------------------------------------------------------------------------


def _make_tiny_model_for_loss(K=2, M=3, N=4, T=6, seed=7):
    """Build a tiny synthetic model for loss_fn smoke test."""
    from phoscrosstalk.config import ModelDims

    ModelDims.set_dims(K, M, N)
    rng = np.random.default_rng(seed)

    t = np.linspace(0.0, 60.0, T)
    P_data = rng.uniform(0.1, 0.9, (N, T))
    A_data = np.zeros((0, T))
    theta = rng.uniform(-2.0, -0.1, 2 * K + 2 + 3 * M + N + 4)

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
        K=K,
        M=M,
        N=N,
        T=T,
        t=t,
        P_data=P_data,
        A_data=A_data,
        theta=theta,
        Cg=Cg,
        Cl=Cl,
        site_prot_idx=site_prot_idx,
        K_site_kin=K_site_kin,
        R=R_mat,
        L_alpha=L_alpha,
        kin_to_prot_idx=kin2prot,
        receptor_mask_prot=rm_prot,
        receptor_mask_kin=rm_kin,
    )


def test_make_loss_fn_scan_kind_bounded_produces_finite_output():
    """
    Regression test: make_loss_fn with Tsit5(scan_kind="bounded") must
    return a finite scalar loss for a valid tiny model.
    """
    import jax.numpy as jnp

    from phoscrosstalk.optimization import make_loss_fn

    m = _make_tiny_model_for_loss()
    _, _, N2, T2 = m["K"], m["M"], m["N"], m["T"]

    loss_fn = make_loss_fn(
        t=m["t"],
        P_data=m["P_data"],
        A_scaled=m["A_data"],
        prot_idx_for_A=np.array([], dtype=int),
        W_data=np.ones((N2, T2)),
        W_data_prot=np.zeros((0, T2)),
        Cg=m["Cg"],
        Cl=m["Cl"],
        site_prot_idx=m["site_prot_idx"],
        K_site_kin=m["K_site_kin"],
        R=m["R"],
        L_alpha=m["L_alpha"],
        kin_to_prot_idx=m["kin_to_prot_idx"],
        receptor_mask_prot=m["receptor_mask_prot"],
        receptor_mask_kin=m["receptor_mask_kin"],
        mechanism="dist",
        lambda_net=0.0,
        reg_lambda=1e-4,
    )

    theta0 = jnp.asarray(m["theta"], dtype=jnp.float64)
    total, (f1, f2, f3, f4) = loss_fn(theta0, None)

    assert np.isfinite(float(total)), f"total loss is not finite: {total}"
    assert np.isfinite(float(f1)), f"f1 is not finite: {f1}"
    assert np.isfinite(float(f3)), f"f3 is not finite: {f3}"
