"""
Regression tests for data_loader.py

Covers:
- FileNotFoundError raised when input files are missing
- ValueError raised when value-column count mismatches DEFAULT_TIMEPOINTS
- ValueError raised when required columns are absent
- load_site_data returns correct shapes on valid minimal input
"""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from phoscrosstalk.config import DEFAULT_TIMEPOINTS
from phoscrosstalk.data_loader import (
    load_site_data,
    load_kinase_site_matrix,
    build_kinase_site_from_kea,
    build_C_matrices_from_db,
    build_alpha_laplacian_from_unified_graph,
    apply_scaling,
    row_normalize,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_valid_csv(path, n_sites=3, n_timepoints=None):
    """Write a minimal valid phosphosite CSV with the expected column structure."""
    if n_timepoints is None:
        n_timepoints = len(DEFAULT_TIMEPOINTS)
    value_cols = {f"v{i}": np.random.rand(n_sites) for i in range(n_timepoints)}
    df = pd.DataFrame(
        {
            "Protein": [f"PROT{i}" for i in range(n_sites)],
            "Residue": [f"S{i * 10 + 1}" for i in range(n_sites)],
            **value_cols,
        }
    )
    df.to_csv(path, index=False)


# ---------------------------------------------------------------------------
# load_site_data
# ---------------------------------------------------------------------------

def test_load_site_data_missing_file():
    with pytest.raises(FileNotFoundError, match="not found"):
        load_site_data("/nonexistent/path/data.csv")


def test_load_site_data_wrong_column_count():
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
        fname = f.name
    try:
        # Write CSV with wrong number of value columns (only 2 instead of 14)
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
            load_site_data(fname)
    finally:
        os.unlink(fname)


def test_load_site_data_missing_protein_column():
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
        fname = f.name
    try:
        n = len(DEFAULT_TIMEPOINTS)
        value_cols = {f"v{i}": [float(i)] for i in range(n)}
        df = pd.DataFrame({"Residue": ["S1"], **value_cols})
        df.to_csv(fname, index=False)
        with pytest.raises(ValueError, match="'Protein' or 'GeneID'"):
            load_site_data(fname)
    finally:
        os.unlink(fname)


def test_load_site_data_missing_residue_column():
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
        fname = f.name
    try:
        n = len(DEFAULT_TIMEPOINTS)
        value_cols = {f"v{i}": [float(i)] for i in range(n)}
        df = pd.DataFrame({"Protein": ["PROT1"], **value_cols})
        df.to_csv(fname, index=False)
        with pytest.raises(ValueError, match="'Residue' or 'Psite'"):
            load_site_data(fname)
    finally:
        os.unlink(fname)


def test_load_site_data_valid(tmp_path):
    csv_path = str(tmp_path / "data.csv")
    n_sites = 4
    _make_valid_csv(csv_path, n_sites=n_sites)
    sites, proteins, site_prot_idx, positions, t, Y, A_data, A_proteins = load_site_data(csv_path)
    assert len(sites) == n_sites
    assert Y.shape == (n_sites, len(DEFAULT_TIMEPOINTS))
    assert len(t) == len(DEFAULT_TIMEPOINTS)
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
        "(id INTEGER PRIMARY KEY, protein TEXT, residue1 TEXT, score1 REAL, residue2 TEXT, score2 REAL)"
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
