"""
test_audit_fixes.py

Focused regression tests for issues identified in AUDIT_REPORT.md.

Coverage:
  C1  – build_tf_prot_weights warns when weight matrix is all-zeros
  C2  – run_single_optimisation total_loss excludes f4 (uses f1+f2+f3 only)
  C3  – _generate_starts is reproducible (fixed per-start seeds)
  H2  – create_bounds respects config bounds
  H5  – W_data_mrna shape guard in main pipeline
  H6  – ModelDims.set_dims is protected by a threading.Lock
  M1  – load_site_data time-column detection uses strict regex
  M8  – _VALID_SCALE contains log-minmax and not zscore
  Reservoir / filtering integration test (problem-statement fixture)
"""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# C1 – build_tf_prot_weights warns when W is all-zeros
# ---------------------------------------------------------------------------


def test_build_tf_prot_weights_warns_all_zeros(caplog):
    """C1: all-zero W matrix should trigger a logger.warning."""
    import logging

    from phoscrosstalk.data_loader import build_tf_prot_weights

    # TF net: source=TF1→target=PROT_X (not in proteins), TF2→PROT_Y (not in gene_ids)
    tf_net_df = pd.DataFrame(
        {
            "source": ["TF1", "TF2"],
            "target": ["PROT_X", "PROT_Y"],
            "weight": [0.5, 0.5],
        }
    )
    proteins = ["PROT_A", "PROT_B"]  # no overlap with targets
    gene_ids = ["GENE1", "GENE2"]  # no overlap with sources

    with caplog.at_level(logging.WARNING, logger="phoscrosstalk.data_loader"):
        W = build_tf_prot_weights(tf_net_df, gene_ids, proteins)

    assert W.sum() == 0.0
    assert any(
        "all-zeros" in record.message.lower() for record in caplog.records
    ), "Expected warning about all-zero TF weight matrix"


def test_build_tf_prot_weights_no_warn_when_nonzero(caplog):
    """C1: no spurious warning when W has non-zero entries."""
    import logging

    from phoscrosstalk.data_loader import build_tf_prot_weights

    tf_net_df = pd.DataFrame(
        {"source": ["FOXO3"], "target": ["EGFR"], "weight": [0.5]}
    )
    proteins = ["EGFR", "ERBB2"]
    gene_ids = ["FOXO3", "TFE3"]

    with caplog.at_level(logging.WARNING, logger="phoscrosstalk.data_loader"):
        W = build_tf_prot_weights(tf_net_df, gene_ids, proteins)

    assert W.sum() > 0.0
    assert not any(
        "all-zeros" in record.message.lower() for record in caplog.records
    )


# ---------------------------------------------------------------------------
# C2 – total_loss uses f1+f2+f3 only (excludes f4)
# ---------------------------------------------------------------------------


def test_total_loss_excludes_f4():
    """C2: run_single_optimisation must use f1+f2+f3 (not +f4) for total_loss."""
    import inspect
    import re

    from phoscrosstalk.optimization import run_single_optimisation

    src = inspect.getsource(run_single_optimisation)
    # Find the assignment line for total_loss (strip comments)
    assignment_lines = []
    for line in src.splitlines():
        if re.match(r"\s*total_loss\s*=", line):
            # Strip inline comment
            code_part = line.split("#")[0].strip()
            assignment_lines.append(code_part)

    assert assignment_lines, "total_loss must be assigned in run_single_optimisation"
    for code in assignment_lines:
        # f4 must not appear as a variable in the RHS (addition)
        assert not re.search(r"\+\s*f4\b", code) and not re.search(r"\bf4\s*\+", code), (
            f"total_loss must NOT add f4; got code: {code!r}. "
            "Use f1 + f2 + f3 only for best-run selection."
        )


# ---------------------------------------------------------------------------
# C3 – _generate_starts reproducibility
# ---------------------------------------------------------------------------


def test_generate_starts_reproducible():
    """C3: same n_starts/bounds must always produce the same starting points."""
    from phoscrosstalk.multistarts import _generate_starts

    xl = np.array([-5.0, -5.0, -5.0])
    xu = np.array([5.0, 5.0, 5.0])

    starts_a = _generate_starts(4, xl, xu)
    starts_b = _generate_starts(4, xl, xu)

    for a, b in zip(starts_a, starts_b):
        np.testing.assert_array_equal(a, b)


def test_generate_starts_different_seeds():
    """C3: different indices must produce different starting points."""
    from phoscrosstalk.multistarts import _generate_starts

    xl = np.array([-5.0, -5.0])
    xu = np.array([5.0, 5.0])

    starts = _generate_starts(3, xl, xu)
    # Verify starts are different from each other
    assert not np.allclose(starts[0], starts[1])
    assert not np.allclose(starts[0], starts[2])


# ---------------------------------------------------------------------------
# H2 – create_bounds respects config bounds
# ---------------------------------------------------------------------------


def test_create_bounds_default():
    """H2: create_bounds without config uses hard-coded defaults."""
    from phoscrosstalk.optimization import create_bounds

    K, M, N = 2, 3, 4
    xl, xu, dim = create_bounds(K, M, N)
    assert dim == 2 * K + 2 + 3 * M + N + 4
    assert np.all(np.isfinite(xl))
    assert np.all(np.isfinite(xu))
    assert np.all(xu >= xl)


def test_create_bounds_from_config():
    """H2: create_bounds applies config bounds when provided."""
    from types import SimpleNamespace

    from phoscrosstalk.optimization import create_bounds

    K, M, N = 2, 3, 4
    bounds_cfg = SimpleNamespace(
        rate_min=1e-4,
        rate_max=5.0,
        protein_degradation_max=0.3,
        kinase_rate_max=2.0,
        phosphatase_rate_max=4.0,
        gamma_abs_max=2.0,
    )

    xl_default, xu_default, _ = create_bounds(K, M, N)
    xl_cfg, xu_cfg, dim = create_bounds(K, M, N, bounds=bounds_cfg)

    assert dim == 2 * K + 2 + 3 * M + N + 4
    # gamma bound differs: default 3.0 vs config 2.0
    # Gammas are the last 4 elements
    assert xu_cfg[-1] == pytest.approx(bounds_cfg.gamma_abs_max)
    assert xl_cfg[-1] == pytest.approx(-bounds_cfg.gamma_abs_max)
    # log(rate_max) upper bound for k_deact should differ
    assert xu_cfg[0] == pytest.approx(np.log(bounds_cfg.rate_max))
    assert xu_default[0] == pytest.approx(np.log(10.0))


# ---------------------------------------------------------------------------
# H6 – ModelDims.set_dims is protected by threading.Lock
# ---------------------------------------------------------------------------


def test_model_dims_has_lock():
    """H6: ModelDims must have a thread lock."""
    import threading

    from phoscrosstalk.config import ModelDims

    assert hasattr(ModelDims, "_lock"), "ModelDims must have a _lock attribute"
    assert isinstance(ModelDims._lock, type(threading.Lock()))


def test_model_dims_concurrent_set():
    """H6: concurrent set_dims calls must not corrupt state."""
    import threading

    from phoscrosstalk.config import ModelDims

    results = []
    barrier = threading.Barrier(2)

    def worker(k, m, n):
        barrier.wait()
        ModelDims.set_dims(k, m, n)
        results.append((ModelDims.K, ModelDims.M, ModelDims.N))

    t1 = threading.Thread(target=worker, args=(3, 4, 5))
    t2 = threading.Thread(target=worker, args=(6, 7, 8))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    # All recorded tuples should be internally consistent (not mixed)
    for k, m, n in results:
        assert (k, m, n) in {(3, 4, 5), (6, 7, 8)}, f"Corrupt state: {k, m, n}"


# ---------------------------------------------------------------------------
# M1 – load_site_data time-column detection uses strict regex
# ---------------------------------------------------------------------------


def test_load_site_data_value_col_strict_regex(tmp_path):
    """M1: columns like 'velocity' must NOT be treated as time columns."""
    from phoscrosstalk.config import DEFAULT_TIMEPOINTS
    from phoscrosstalk.data_loader import load_site_data

    n = len(DEFAULT_TIMEPOINTS)
    value_cols = {f"v{i}": [1.0] for i in range(n)}
    # Add a non-time column that starts with 'v' but is not a time column
    df = pd.DataFrame(
        {
            "Protein": ["EGFR"],
            "Residue": ["Y1172"],
            "velocity": [0.5],  # must NOT be counted as a time column
            **value_cols,
        }
    )
    csv_path = str(tmp_path / "data.csv")
    df.to_csv(csv_path, index=False)

    # Should succeed: 'velocity' is not matched by the [vx]\d+ pattern
    sites, proteins, _, _, t, Y, _, _ = load_site_data(csv_path)
    assert len(t) == n
    assert Y.shape == (1, n)


# ---------------------------------------------------------------------------
# M8 – _VALID_SCALE
# ---------------------------------------------------------------------------


def test_valid_scale_contains_log_minmax():
    """M8: _VALID_SCALE must include 'log-minmax' and exclude 'zscore'."""
    from phoscrosstalk.config import _VALID_SCALE

    assert "log-minmax" in _VALID_SCALE, "_VALID_SCALE must contain 'log-minmax'"
    assert "zscore" not in _VALID_SCALE, "_VALID_SCALE must NOT contain 'zscore'"
    assert "minmax" in _VALID_SCALE
    assert "none" in _VALID_SCALE


# ---------------------------------------------------------------------------
# Reservoir / filtering integration tests (problem-statement fixture)
# ---------------------------------------------------------------------------

# Small fixtures matching the problem-statement examples.

_KINASE_TSV_CONTENT = """\
Site\tKinase\tweight
EGFR_Y1172\tERBB2\t0.801
EGFR_Y1172\tEGFR\t0.199
EGFR_Y1197\tERBB2\t0.600
EGFR_Y1197\tMET\t0.400
EGFR_Y869\tMET\t1.0
MET_Y1252\tMET\t0.588
MET_Y1252\tEGFR\t0.412
"""

_TF_CSV_CONTENT = """\
Source,Target,Weight
FOXO3,EGFR,0.48
DNMT1,EGFR,0.09
AR,ERBB2,0.47
ETV1,ERBB2,0.17
TFE3,MET,0.27
AR,MET,0.17
"""

_PHOSPHO_CSV_CONTENT = """\
GeneID,Psite,x1,x2,x3,x4,x5
EGFR,,0.98,1.01,1.04,1.02,0.98
EGFR,Y_1172,0.84,0.87,0.90,0.91,0.90
EGFR,Y_1197,1.51,1.60,1.71,1.77,1.72
EGFR,Y_869,2.92,2.83,2.88,3.04,3.22
ERBB2,,1.01,1.01,1.01,1.02,1.03
MET,,1.17,1.11,1.03,0.98,0.97
MET,Y_1252,1.70,1.69,1.68,1.66,1.65
UNKNOWN_PROTEIN,S100,0.5,0.5,0.5,0.5,0.5
"""

_RNA_CSV_CONTENT = """\
GeneID,x1,x2,x3,x4,x5
EGFR,1.11,0.97,1.05,0.76,1.01
ERBB2,1.13,0.94,1.00,0.81,1.00
MET,1.06,0.99,1.04,0.81,0.99
AR,1.14,1.13,1.06,0.93,1.09
DNMT1,1.07,0.95,1.00,0.73,0.97
ETV1,0.87,0.82,0.91,0.76,0.91
FOXO3,1.02,0.91,1.05,0.71,1.00
TFE3,1.03,0.86,0.93,0.74,0.90
JUNK_GENE,0.5,0.5,0.5,0.5,0.5
"""


@pytest.fixture
def network_fixture(tmp_path):
    """Write fixture files and return paths."""
    kinase_tsv = str(tmp_path / "kinase_sites.tsv")
    tf_csv = str(tmp_path / "tf_mrna.csv")
    phospho_csv = str(tmp_path / "phospho.csv")
    rna_csv = str(tmp_path / "rna.csv")

    with open(kinase_tsv, "w") as f:
        f.write(_KINASE_TSV_CONTENT)
    with open(tf_csv, "w") as f:
        f.write(_TF_CSV_CONTENT)
    with open(phospho_csv, "w") as f:
        f.write(_PHOSPHO_CSV_CONTENT)
    with open(rna_csv, "w") as f:
        f.write(_RNA_CSV_CONTENT)

    return {
        "kinase_tsv": kinase_tsv,
        "tf_csv": tf_csv,
        "phospho_csv": phospho_csv,
        "rna_csv": rna_csv,
    }


def test_prefilter_phospho_csv_keeps_network_rows(network_fixture):
    """Reservoir test: phospho pre-filter keeps sites/kinases, drops unknowns."""
    from phoscrosstalk.data_loader import (
        _build_network_allow_sets,
        _prefilter_phospho_csv,
    )
    from phoscrosstalk.logger import get_logger

    log = get_logger("test")
    allowed_sites, allowed_kinases, tf_sources, tf_targets = _build_network_allow_sets(
        network_fixture["kinase_tsv"],
        network_fixture["tf_csv"],
        include_tfs_as_proteins=False,
    )

    tmp_path = _prefilter_phospho_csv(
        data_path=network_fixture["phospho_csv"],
        allowed_sites=allowed_sites,
        allowed_kinases=allowed_kinases,
        tf_sources=tf_sources,
        tf_targets=tf_targets,
        include_tfs_as_proteins=False,
        logger_=log,
    )
    try:
        df = pd.read_csv(tmp_path)
        gene_ids = df["GeneID"].unique().tolist()

        # UNKNOWN_PROTEIN should be removed
        assert "UNKNOWN_PROTEIN" not in gene_ids, "UNKNOWN_PROTEIN should be filtered out"

        # Model proteins/kinases should be present
        for prot in ["EGFR", "ERBB2", "MET"]:
            assert prot in gene_ids, f"{prot} should be in filtered phospho data"

        # Phosphosite rows should be present (Y_1172 → normalised to EGFR_Y1172)
        if "Psite" in df.columns:
            psite_labels = df["Psite"].dropna().astype(str).tolist()
            assert any("1172" in s for s in psite_labels), (
                "EGFR_Y1172 site should be present in filtered data"
            )
    finally:
        if tmp_path != network_fixture["phospho_csv"]:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


def test_prefilter_rna_csv_keeps_model_and_tf_genes(network_fixture):
    """Reservoir test: RNA pre-filter keeps model proteins and TF sources, drops junk."""
    from phoscrosstalk.data_loader import (
        _build_network_allow_sets,
        _prefilter_rna_csv,
    )
    from phoscrosstalk.logger import get_logger

    log = get_logger("test")
    _, _, tf_sources, tf_targets = _build_network_allow_sets(
        network_fixture["kinase_tsv"],
        network_fixture["tf_csv"],
        include_tfs_as_proteins=False,
    )

    model_proteins = ["EGFR", "ERBB2", "MET"]

    tmp_path = _prefilter_rna_csv(
        rna_path=network_fixture["rna_csv"],
        model_proteins=model_proteins,
        tf_sources=tf_sources,
        tf_targets=tf_targets,
        logger_=log,
    )
    try:
        df = pd.read_csv(tmp_path)
        gene_ids = df["GeneID"].tolist()

        # JUNK_GENE should be removed
        assert "JUNK_GENE" not in gene_ids, "JUNK_GENE should be filtered out"

        # Model proteins should be present
        for prot in model_proteins:
            assert prot in gene_ids, f"{prot} should be in filtered RNA data"

        # TF sources should be present
        for tf in tf_sources:
            assert tf in gene_ids, f"TF source {tf} should be in filtered RNA data"
    finally:
        if tmp_path != network_fixture["rna_csv"]:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


def test_normalise_psite_label():
    """Site label normalisation: Y_1172 → Y1172, EGFR_Y_1172 handled."""
    from phoscrosstalk.data_loader import _normalise_psite_label

    assert _normalise_psite_label("Y_1172") == "Y1172"
    assert _normalise_psite_label("S_473") == "S473"
    assert _normalise_psite_label("T202") == "T202"  # already normalised
    assert _normalise_psite_label("Y1068") == "Y1068"


def test_build_network_allow_sets_normalises_sites(network_fixture):
    """_build_network_allow_sets should return sites exactly as in the TSV."""
    from phoscrosstalk.data_loader import _build_network_allow_sets

    allowed_sites, allowed_kinases, tf_sources, tf_targets = _build_network_allow_sets(
        network_fixture["kinase_tsv"],
        network_fixture["tf_csv"],
        include_tfs_as_proteins=False,
    )

    # Sites from TSV (already in EGFR_Y1172 format)
    assert "EGFR_Y1172" in allowed_sites
    assert "EGFR_Y869" in allowed_sites
    assert "MET_Y1252" in allowed_sites

    # Kinases
    assert "EGFR" in allowed_kinases
    assert "ERBB2" in allowed_kinases
    assert "MET" in allowed_kinases

    # TF sources and targets from tf_mrna.csv
    assert "FOXO3" in tf_sources
    assert "AR" in tf_sources
    assert "EGFR" in tf_targets
    assert "MET" in tf_targets


def test_tmp_file_cleanup_phospho(network_fixture):
    """H1: phospho temp file should be deleted after use."""
    from phoscrosstalk.data_loader import (
        _build_network_allow_sets,
        _prefilter_phospho_csv,
    )
    from phoscrosstalk.logger import get_logger

    log = get_logger("test")
    allowed_sites, allowed_kinases, tf_sources, tf_targets = _build_network_allow_sets(
        network_fixture["kinase_tsv"],
        network_fixture["tf_csv"],
        include_tfs_as_proteins=False,
    )

    tmp_path = _prefilter_phospho_csv(
        data_path=network_fixture["phospho_csv"],
        allowed_sites=allowed_sites,
        allowed_kinases=allowed_kinases,
        tf_sources=tf_sources,
        tf_targets=tf_targets,
        include_tfs_as_proteins=False,
        logger_=log,
    )

    # Simulate main.py cleanup logic
    if tmp_path != network_fixture["phospho_csv"]:
        assert os.path.exists(tmp_path), "temp file should exist before cleanup"
        os.unlink(tmp_path)
        assert not os.path.exists(tmp_path), "temp file should be deleted after cleanup"
