"""
Regression tests for data_loader.py

Covers:
- FileNotFoundError raised when input files are missing
- ValueError raised when value-column count does not match the supplied timepoints
- ValueError raised when required columns are absent
- load_site_data returns correct shapes on valid minimal input
"""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from phoscrosstalk.data_loader import (
    apply_scaling,
    build_alpha_laplacian_from_unified_graph,
    build_C_matrices_from_db,
    build_kinase_site_from_kea,
    load_kinase_site_matrix,
    load_site_data,
    row_normalize,
)

# Number of time points used throughout the test suite.
# Matches the example time-point array that tests supply explicitly.
_N_TP = 14

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_valid_csv(path, n_sites=3, n_timepoints=None):
    """Write a minimal valid phosphosite CSV with the expected column structure."""
    if n_timepoints is None:
        n_timepoints = _N_TP
    value_cols = {f"v{i}": np.random.rand(n_sites) for i in range(n_timepoints)}
    df = pd.DataFrame(
        {
            "Protein": [f"PROT{i}" for i in range(n_sites)],
            "Residue": [f"S{i * 10 + 1}" for i in range(n_sites)],
            **value_cols,
        }
    )
    df.to_csv(path, index=False)


def _make_timepoints(n=_N_TP):
    """Return a valid strictly-increasing time-point list of length *n*."""
    return list(range(n))


# ---------------------------------------------------------------------------
# load_site_data
# ---------------------------------------------------------------------------


def test_load_site_data_missing_file():
    with pytest.raises(FileNotFoundError, match="not found"):
        load_site_data("/nonexistent/path/data.csv", timepoints=[0, 1, 2])


def test_load_site_data_wrong_column_count():
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
        fname = f.name
    try:
        # Write CSV with 2 value columns but pass 14 timepoints
        df = pd.DataFrame(
            {
                "Protein": ["PROT1"],
                "Residue": ["S123"],
                "v0": [1.0],
                "v1": [2.0],
            }
        )
        df.to_csv(fname, index=False)
        with pytest.raises(ValueError, match="value columns"):
            load_site_data(fname, timepoints=_make_timepoints(_N_TP))
    finally:
        os.unlink(fname)


def test_load_site_data_missing_protein_column():
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
        fname = f.name
    try:
        n = _N_TP
        value_cols = {f"v{i}": [float(i)] for i in range(n)}
        df = pd.DataFrame({"Residue": ["S1"], **value_cols})
        df.to_csv(fname, index=False)
        with pytest.raises(ValueError, match="'Protein' or 'GeneID'"):
            load_site_data(fname, timepoints=_make_timepoints(n))
    finally:
        os.unlink(fname)


def test_load_site_data_missing_residue_column():
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
        fname = f.name
    try:
        n = _N_TP
        value_cols = {f"v{i}": [float(i)] for i in range(n)}
        df = pd.DataFrame({"Protein": ["PROT1"], **value_cols})
        df.to_csv(fname, index=False)
        with pytest.raises(ValueError, match="'Residue' or 'Psite'"):
            load_site_data(fname, timepoints=_make_timepoints(n))
    finally:
        os.unlink(fname)


def test_load_site_data_valid(tmp_path):
    csv_path = str(tmp_path / "data.csv")
    n_sites = 4
    _make_valid_csv(csv_path, n_sites=n_sites)
    tp = _make_timepoints(_N_TP)
    sites, proteins, site_prot_idx, positions, t, Y, A_data, A_proteins = (
        load_site_data(csv_path, timepoints=tp)
    )
    assert len(sites) == n_sites
    assert Y.shape == (n_sites, _N_TP)
    assert len(t) == _N_TP
    assert site_prot_idx.shape == (n_sites,)
    assert A_data is None
    assert A_proteins is None


# ---------------------------------------------------------------------------
# load_kinase_site_matrix
# ---------------------------------------------------------------------------


def test_load_kinase_site_matrix_missing_file():
    with pytest.raises(FileNotFoundError, match="not found"):
        load_kinase_site_matrix("/nonexistent/kinase.tsv", ["PROT1_S1"])


# ---------------------------------------------------------------------------
# build_kinase_site_from_kea
# ---------------------------------------------------------------------------


def test_build_kinase_site_from_kea_missing_file():
    with pytest.raises(FileNotFoundError, match="not found"):
        build_kinase_site_from_kea("/nonexistent/kea.tsv", ["PROT1_S1"])


# ---------------------------------------------------------------------------
# build_C_matrices_from_db
# ---------------------------------------------------------------------------


def test_build_c_matrices_missing_intra():
    with pytest.raises(FileNotFoundError, match="Intra-protein"):
        build_C_matrices_from_db(
            "/nonexistent/intra.db",
            "/nonexistent/inter.db",
            [],
            np.array([], dtype=int),
            np.array([], dtype=float),
            [],
        )


def test_build_c_matrices_missing_inter(tmp_path):
    # Create an empty intra DB so only inter is missing
    intra_path = str(tmp_path / "intra.db")
    import sqlite3

    conn = sqlite3.connect(intra_path)
    conn.execute(
        "CREATE TABLE intra_pairs "
        "(id INTEGER PRIMARY KEY, protein TEXT, residue1 TEXT, score1 REAL, residue2 TEXT, score2 REAL)"  # noqa: E501
    )
    conn.commit()
    conn.close()

    with pytest.raises(FileNotFoundError, match="Inter-protein"):
        build_C_matrices_from_db(
            intra_path,
            "/nonexistent/inter.db",
            [],
            np.array([], dtype=int),
            np.array([], dtype=float),
            [],
        )


# ---------------------------------------------------------------------------
# build_alpha_laplacian_from_unified_graph
# ---------------------------------------------------------------------------


def test_build_alpha_laplacian_missing_pkl():
    with pytest.raises(FileNotFoundError, match="not found"):
        build_alpha_laplacian_from_unified_graph("/nonexistent/graph.pkl", ["K1"])


# ---------------------------------------------------------------------------
# apply_scaling
# ---------------------------------------------------------------------------


def test_apply_scaling_unknown_mode():
    Y = np.ones((3, 5))
    with pytest.raises(ValueError, match="Unknown scale mode"):
        apply_scaling(Y, mode="unknown_mode")


def test_apply_scaling_none_mode():
    Y = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    P, baselines, amplitudes = apply_scaling(Y, mode="none")
    np.testing.assert_array_equal(P, Y)


def test_apply_scaling_minmax_range():
    Y = np.array([[0.0, 5.0, 10.0]])
    P, baselines, amplitudes = apply_scaling(Y, mode="minmax")
    assert float(P[0, 0]) == pytest.approx(0.0)
    assert float(P[0, -1]) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# row_normalize
# ---------------------------------------------------------------------------


def test_row_normalize_basic():
    C = np.array([[2.0, 2.0], [0.0, 0.0]])
    Cn = row_normalize(C)
    np.testing.assert_allclose(Cn[0], [0.5, 0.5])
    # Zero row should not produce NaN (divided by 1.0)
    assert np.all(np.isfinite(Cn))


# ---------------------------------------------------------------------------
# load_rna_data tests
# ---------------------------------------------------------------------------


def test_load_rna_data_valid(tmp_path):
    from phoscrosstalk.data_loader import load_rna_data

    df = pd.DataFrame(
        {
            "gene": ["EGFR", "MAPK1", "AKT1"],
            "0": [1.0, 0.8, 1.2],
            "10": [1.5, 0.9, 1.1],
            "30": [2.0, 1.0, 1.0],
        }
    )
    p = tmp_path / "rna.csv"
    df.to_csv(p, index=False)

    gene_ids, t_rna, matrix = load_rna_data(str(p))
    assert gene_ids == ["EGFR", "MAPK1", "AKT1"]
    assert len(t_rna) == 3
    assert matrix.shape == (3, 3)
    assert not np.any(np.isnan(matrix))


def test_load_rna_data_missing_file():
    from phoscrosstalk.data_loader import load_rna_data

    with pytest.raises(FileNotFoundError):
        load_rna_data("/nonexistent/path/rna.csv")


def test_load_rna_data_too_few_time_cols(tmp_path):
    from phoscrosstalk.data_loader import load_rna_data

    df = pd.DataFrame({"gene": ["EGFR"], "0": [1.0]})
    p = tmp_path / "rna_bad.csv"
    df.to_csv(p, index=False)

    with pytest.raises(ValueError, match="at least 2"):
        load_rna_data(str(p))


def test_load_rna_data_nan_values(tmp_path):
    from phoscrosstalk.data_loader import load_rna_data

    df = pd.DataFrame({"gene": ["EGFR"], "0": [float("nan")], "10": [1.0]})
    p = tmp_path / "rna_nan.csv"
    df.to_csv(p, index=False)

    with pytest.raises(ValueError, match="non-finite"):
        load_rna_data(str(p))


# ---------------------------------------------------------------------------
# load_tf_network tests
# ---------------------------------------------------------------------------


def test_load_tf_network_valid(tmp_path):
    from phoscrosstalk.data_loader import load_tf_network

    df = pd.DataFrame(
        {"source": ["EGFR", "AKT1"], "target": ["MAPK1", "MAPK1"], "weight": [0.5, 1.0]}
    )
    p = tmp_path / "tf.csv"
    df.to_csv(p, index=False)

    result = load_tf_network(str(p))
    assert list(result.columns) == ["source", "target", "weight"]
    assert len(result) == 2


def test_load_tf_network_default_weight(tmp_path):
    from phoscrosstalk.data_loader import load_tf_network

    df = pd.DataFrame({"source": ["EGFR"], "target": ["MAPK1"]})
    p = tmp_path / "tf_no_weight.csv"
    df.to_csv(p, index=False)

    result = load_tf_network(str(p))
    assert result["weight"].iloc[0] == 1.0


def test_load_tf_network_missing_column(tmp_path):
    from phoscrosstalk.data_loader import load_tf_network

    df = pd.DataFrame({"src": ["EGFR"], "tgt": ["MAPK1"]})
    p = tmp_path / "tf_bad.csv"
    df.to_csv(p, index=False)

    with pytest.raises(ValueError, match="'source'"):
        load_tf_network(str(p))


def test_load_tf_network_warns_on_unknown_tf(tmp_path, recwarn):
    from phoscrosstalk.data_loader import load_tf_network

    df = pd.DataFrame({"source": ["UNKNOWN_GENE"], "target": ["MAPK1"]})
    p = tmp_path / "tf_warn.csv"
    df.to_csv(p, index=False)

    # Should not raise – just warn
    result = load_tf_network(str(p), gene_ids=["EGFR", "MAPK1"])
    assert len(result) == 1


# ---------------------------------------------------------------------------
# build_tf_prot_weights tests
# ---------------------------------------------------------------------------


def test_build_tf_prot_weights_shape():
    from phoscrosstalk.data_loader import build_tf_prot_weights

    tf_net_df = pd.DataFrame(
        {"source": ["EGFR", "EGFR"], "target": ["MAPK1", "AKT1"], "weight": [1.0, 0.5]}
    )
    gene_ids = ["EGFR", "MAPK1", "AKT1"]
    proteins = ["MAPK1", "AKT1", "PTEN"]

    W = build_tf_prot_weights(tf_net_df, gene_ids, proteins)
    assert W.shape == (3, 3)
    # EGFR (gene 0) → MAPK1 (prot 0) with weight 1.0
    assert W[0, 0] == pytest.approx(1.0)
    # EGFR (gene 0) → AKT1 (prot 1) with weight 0.5
    assert W[1, 0] == pytest.approx(0.5)
