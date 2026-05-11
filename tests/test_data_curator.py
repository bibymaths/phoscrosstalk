import gzip
import io
import pickle
import sqlite3
import zipfile
from pathlib import Path

import pandas as pd
import requests

from phoscrosstalk.data_curator import DataCurator, main


def _write_gzip(path: Path, text: str):
    with gzip.open(path, "wt") as fh:
        fh.write(text)


def test_data_curator_init_and_download_helpers(tmp_path):
    curator = DataCurator(str(tmp_path))
    assert curator.raw_dir.exists()
    assert curator.processed_dir.exists()

    class DummyResponse:
        def __init__(self, content: bytes):
            self._content = content
            self.raw = io.BytesIO(content)

        def iter_content(self, chunk_size=1024):
            for i in range(0, len(self._content), chunk_size):
                yield self._content[i : i + chunk_size]

    plain = DummyResponse(b"abc123")
    out = tmp_path / "plain.bin"
    curator._download_file(plain, out)
    assert out.read_bytes() == b"abc123"

    gz_bytes = gzip.compress(b"hello")
    gz = DummyResponse(gz_bytes)
    gz_path = tmp_path / "file.txt.gz"
    curator._download_and_decompress_file(gz, gz_path)
    assert gz_path.with_suffix("").read_text() == "hello"


def test_download_resources_handles_download_skip_and_request_failure(tmp_path, monkeypatch):
    curator = DataCurator(str(tmp_path))
    existing = curator.raw_dir / "kea" / "gene_attribute_matrix.txt.gz"
    existing.parent.mkdir(parents=True, exist_ok=True)
    existing.write_text("cached", encoding="utf-8")

    class DummyResponse:
        def __init__(self, status_code=200, content=b"payload"):
            self.status_code = status_code
            self._content = content
            self.raw = io.BytesIO(gzip.compress(content))

        def iter_content(self, chunk_size=1024):
            for i in range(0, len(self._content), chunk_size):
                yield self._content[i : i + chunk_size]

    def fake_get(url, stream=True):
        if url.endswith("gene_attribute_edges.txt.gz"):
            raise requests.RequestException("boom")
        if url.endswith("gene_set_library_crisp.gmt.gz"):
            return DummyResponse(status_code=404)
        return DummyResponse()

    monkeypatch.setattr("phoscrosstalk.data_curator.requests.get", fake_get)
    curator.download_resources(decompress=True)
    assert existing.read_text() == "cached"
    downloaded = curator.raw_dir / "kea" / "kinase_substrate_phospho-site_level_pmid_resource_database.txt"
    assert downloaded.exists()


def test_build_ptm_databases_and_convert_custom_csv(tmp_path):
    curator = DataCurator(str(tmp_path))
    within = tmp_path / "within.gz"
    between = tmp_path / "between.gz"
    _write_gzip(
        within,
        "\n".join(
            [
                "# comment",
                "P1\tHomo sapiens\tphosphorylation\tY10\t0.1\tx\tphosphorylation\tS20\t0.2\tx\tx\tx\tx",
                "P2\tMus musculus\tphosphorylation\tY1\t0.1\tx\tphosphorylation\tS2\t0.2\tx\tx\tx\tx",
            ]
        ),
    )
    _write_gzip(
        between,
        "\n".join(
            [
                "# comment",
                "P1\tP2\tHomo sapiens\tphosphorylation\tY10\t0.1\tx\tphosphorylation\tS20\t0.2\tx\tx\tx",
                "P3\tP4\tMus musculus\tphosphorylation\tY1\t0.1\tx\tphosphorylation\tS2\t0.2\tx\tx\tx",
            ]
        ),
    )

    curator.build_ptm_databases(str(within), str(between))
    intra = sqlite3.connect(curator.processed_dir / "ptm_intra.db")
    inter = sqlite3.connect(curator.processed_dir / "ptm_inter.db")
    try:
        assert intra.execute("select count(*) from intra_pairs").fetchone()[0] == 1
        assert inter.execute("select count(*) from inter_pairs").fetchone()[0] == 1
    finally:
        intra.close()
        inter.close()

    custom = tmp_path / "custom.csv"
    pd.DataFrame(
        [{"GeneID": "EGFR", "Psite": "Y_1172", "Kinase": "{SRC, ABL1}"}]
    ).to_csv(custom, index=False)
    curator.convert_custom_kinase_csv(str(custom))
    out = pd.read_csv(curator.processed_dir / "kinase_sites.tsv", sep="\t")
    assert set(out["Kinase"]) == {"SRC", "ABL1"}
    assert set(out["Site"]) == {"EGFR_Y1172"}


def test_build_kinase_networks_and_ks_map(tmp_path):
    curator = DataCurator(str(tmp_path))
    kea_dir = curator.raw_dir / "kea"
    kea_dir.mkdir(parents=True, exist_ok=True)

    kin_zip = kea_dir / "kinase_networks.zip"
    with zipfile.ZipFile(kin_zip, "w") as zf:
        zf.writestr("kk_layer1.tsv", "source\ttarget\tweight\nA\tB\t1.0\n")
        zf.writestr("kk_layer2.tsv", "source\ttarget\tweight\nA\tB\t2.0\nB\tC\t1.0\n")

    curator.build_kinase_networks()
    graph_path = curator.processed_dir / "unified_kinase_graph.gpickle"
    assert graph_path.exists()
    with open(graph_path, "rb") as fh:
        graph = pickle.load(fh)
    assert graph["A"]["B"]["weight_sum"] == 3.0
    assert graph["A"]["B"]["support"] == 2
    assert (curator.processed_dir / "unified_kinase_graph.graphml").exists()

    gsl_zip = kea_dir / "gsl_database.zip"
    with zipfile.ZipFile(gsl_zip, "w") as zf:
        zf.writestr(
            "kinase_substrate_phospho-site_level_pmid_resource_database.txt",
            "SRC\tEGFR_Y1172\t123\tPSP\nABL1\tEGFRY999\t999\tOTHER\n",
        )

    curator.build_ks_map()
    ks_table = pd.read_csv(curator.processed_dir / "ks_psite_table.tsv", sep="\t")
    assert list(ks_table["substrate_site_upper"]) == ["EGFR_Y1172"]
    with open(curator.processed_dir / "ks_psite_index.pkl", "rb") as fh:
        index = pickle.load(fh)
    assert index["EGFR_Y1172"]["kinases"] == {"SRC"}


def test_data_curator_error_paths_and_main_cli(tmp_path, monkeypatch):
    curator = DataCurator(str(tmp_path))
    curator.build_kinase_networks(str(tmp_path / "missing.zip"))
    curator.build_ks_map(str(tmp_path / "missing.zip"))

    bad_zip = curator.raw_dir / "kea"
    bad_zip.mkdir(parents=True, exist_ok=True)
    broken = bad_zip / "gsl_database.zip"
    broken.write_text("not-a-zip", encoding="utf-8")
    curator.build_ks_map(str(broken))

    csv = tmp_path / "custom.csv"
    pd.DataFrame([{"Protein": "P", "Residue": "S1", "Kinase": "{K1}"}]).to_csv(csv, index=False)

    calls = []

    def record(name):
        def _inner(*args, **kwargs):
            calls.append(name)
        return _inner

    monkeypatch.setattr("sys.argv", ["prog", "--dir", str(tmp_path), "--download", "--kea", "--convert-csv", str(csv)])
    monkeypatch.setattr(DataCurator, "download_resources", record("download"))
    monkeypatch.setattr(DataCurator, "build_kinase_networks", record("net"))
    monkeypatch.setattr(DataCurator, "build_ks_map", record("ks"))
    monkeypatch.setattr(DataCurator, "convert_custom_kinase_csv", record("convert"))
    main()
    assert calls == ["download", "net", "ks", "convert"]
