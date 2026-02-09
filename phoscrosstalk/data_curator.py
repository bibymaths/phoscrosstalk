"""
data_curator.py
A unified pipeline module to download, process, and curate all necessary data artifacts
for PhosCrosstalk (SQLite DBs, Kinase Graphs, KS Mappings).

Consolidates logic from:
- harmonizomedownloader.py
- build_db_from_ptmcode2.py
- build_kinase_networks.py
- build_site_level_ks_map.py
"""

import os
import argparse
import gzip
import json
import pickle
import sqlite3
import zipfile
import zlib
from pathlib import Path
from typing import Dict, List, Set, Optional

import pandas as pd
import numpy as np
import networkx as nx
import requests

from phoscrosstalk.logger import get_logger

logger = get_logger()

# --- Configuration Constants ---
HARMONIZOME_BASE_URL = 'https://maayanlab.cloud/static/hdfs/harmonizome/data'

# Datasets to fetch automatically
DEFAULT_DATASETS = [
    ('KEA Substrates of Kinases', 'kea'),
    ('PhosphoSitePlus Phosphosite-Disease Associations', 'phosphositeplusdisease'),
    ('PhosphoSitePlus Substrates of Kinases', 'phosphositeplus'),
]

DEFAULT_DOWNLOADABLES = [
    'gene_attribute_matrix.txt.gz',
    'gene_attribute_edges.txt.gz',
    'gene_set_library_crisp.gmt.gz',
    'kinase_substrate_phospho-site_level_pmid_resource_database.txt',  # Specific to KEA/PSP
    'kinase_networks.zip'  # Specific to KEA
]


class DataCurator:
    """
    Orchestrates the downloading and processing of raw biological data into
    PhosCrosstalk-compatible artifacts (SQLite, Pickle, GraphML).
    """

    def __init__(self, data_dir: str = "data_curated"):
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"

        # Ensure directories exist
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # 1. DOWNLOADING (Harmonizome)
    # =========================================================================

    def _download_file(self, response, filepath: Path):
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=1024):
                f.write(chunk)

    def _download_and_decompress_file(self, response, filepath: Path):
        decompressor = zlib.decompressobj(16 + zlib.MAX_WBITS)
        # Strip .gz extension for target
        target_path = filepath.with_suffix('')
        with open(target_path, 'w+') as f:
            while True:
                chunk = response.raw.read(1024)
                if not chunk:
                    break
                string = decompressor.decompress(chunk)
                f.write(string.decode('utf-8'))  # Decode bytes to string

    def download_resources(self, decompress: bool = False):
        """
        Downloads required datasets from Harmonizome.
        """
        logger.header("[*] Downloading Data Resources")

        for dataset_name, path_suffix in DEFAULT_DATASETS:
            dataset_dir = self.raw_dir / path_suffix
            dataset_dir.mkdir(exist_ok=True)

            logger.info(f"Checking dataset: {dataset_name} ({path_suffix})")

            # Note: Not all datasets have all files. We try typical ones + specific KEA ones
            # For simplicity, we try to fetch specific files if they exist
            targets = DEFAULT_DOWNLOADABLES
            if path_suffix == 'kea':
                targets += ['kinase_networks.zip', 'gsl_database.zip']

            for downloadable in targets:
                url = f"{HARMONIZOME_BASE_URL}/{path_suffix}/{downloadable}"
                local_path = dataset_dir / downloadable

                if local_path.exists():
                    logger.info(f"  - Skipping {downloadable} (already exists)")
                    continue

                try:
                    response = requests.get(url, stream=True)
                    if response.status_code == 200:
                        logger.info(f"  - Downloading {downloadable}...")
                        if decompress and downloadable.endswith('.gz') and 'zip' not in downloadable:
                            self._download_and_decompress_file(response, local_path)
                        else:
                            self._download_file(response, local_path)
                    else:
                        # File might not exist for this specific dataset, which is fine
                        pass
                except Exception as e:
                    logger.warning(f"  ! Failed to download {url}: {e}")

    # =========================================================================
    # 2. PTM DATABASE BUILDER (PTMcode2)
    # =========================================================================

    def build_ptm_databases(self, within_path: str, between_path: str):
        """
        Builds SQLite databases for intra- and inter-protein PTM pairs from PTMcode2 data.

        Args:
            within_path: Path to PTMcode2 'within' file (gzipped).
            between_path: Path to PTMcode2 'between' file (gzipped).
        """
        logger.header("[*] Building PTM SQLite Databases")

        out_intra = self.processed_dir / "ptm_intra.db"
        out_inter = self.processed_dir / "ptm_inter.db"

        # --- Helper: Load Within ---
        def load_within(path):
            rows = []
            logger.info(f"Parsing intra-protein pairs from {path}...")
            with gzip.open(path, "rt") as f:
                for line in f:
                    if line.startswith("#") or line.strip() == "": continue
                    parts = line.strip().split("\t")
                    if len(parts) < 13: continue

                    # Strict filtering for Human Phosphorylation
                    if parts[1] != "Homo sapiens": continue
                    if parts[2] != "phosphorylation" or parts[6] != "phosphorylation": continue

                    protein = parts[0]
                    res1 = parts[3]
                    r1 = float(parts[4])
                    res2 = parts[7]
                    r2 = float(parts[8])
                    rows.append((protein, res1, r1, protein, res2, r2))
            return rows

        # --- Helper: Load Between ---
        def load_between(path):
            rows = []
            logger.info(f"Parsing inter-protein pairs from {path}...")
            with gzip.open(path, "rt") as f:
                for line in f:
                    if line.startswith("#") or line.strip() == "": continue
                    parts = line.strip().split("\t")
                    if len(parts) < 13: continue

                    if parts[2] != "Homo sapiens": continue
                    if parts[3] != "phosphorylation" or parts[7] != "phosphorylation": continue

                    p1, p2 = parts[0], parts[1]
                    res1 = parts[4]
                    r1 = float(parts[5])
                    res2 = parts[8]
                    r2 = float(parts[9])
                    rows.append((p1, res1, r1, p2, res2, r2))
            return rows

        # --- Execution ---
        W = load_within(within_path)
        B = load_between(between_path)

        # Write Intra DB
        if out_intra.exists(): out_intra.unlink()
        conn_i = sqlite3.connect(out_intra)
        conn_i.execute("""
                       CREATE TABLE intra_pairs
                       (
                           id       INTEGER PRIMARY KEY AUTOINCREMENT,
                           protein  TEXT NOT NULL,
                           residue1 TEXT NOT NULL,
                           score1   REAL NOT NULL,
                           residue2 TEXT NOT NULL,
                           score2   REAL NOT NULL
                       )
                       """)
        conn_i.execute("CREATE INDEX idx_intra_protein ON intra_pairs(protein)")
        conn_i.executemany(
            "INSERT INTO intra_pairs (protein, residue1, score1, residue2, score2) VALUES (?, ?, ?, ?, ?)",
            [(p, r1, s1, r2, s2) for (p, r1, s1, _, r2, s2) in W]
        )
        conn_i.commit()
        conn_i.close()
        logger.success(f"Saved {out_intra}")

        # Write Inter DB
        if out_inter.exists(): out_inter.unlink()
        conn_e = sqlite3.connect(out_inter)
        conn_e.execute("""
                       CREATE TABLE inter_pairs
                       (
                           id       INTEGER PRIMARY KEY AUTOINCREMENT,
                           protein1 TEXT NOT NULL,
                           residue1 TEXT NOT NULL,
                           score1   REAL NOT NULL,
                           protein2 TEXT NOT NULL,
                           residue2 TEXT NOT NULL,
                           score2   REAL NOT NULL
                       )
                       """)
        conn_e.execute("CREATE INDEX idx_inter_proteins ON inter_pairs(protein1, protein2)")
        conn_e.executemany(
            "INSERT INTO inter_pairs (protein1, residue1, score1, protein2, residue2, score2) VALUES (?, ?, ?, ?, ?, ?)",
            B
        )
        conn_e.commit()
        conn_e.close()
        logger.success(f"Saved {out_inter}")

    # =========================================================================
    # 3. KINASE NETWORK BUILDER
    # =========================================================================

    def build_kinase_networks(self, zip_path: Optional[str] = None):
        """
        Builds multiplex kinase networks and a unified kinase graph.
        Defaults to looking for 'kinase_networks.zip' in raw/kea/.
        """
        logger.header("[*] Building Kinase Networks")

        if zip_path is None:
            zip_path = self.raw_dir / "kea" / "kinase_networks.zip"
        else:
            zip_path = Path(zip_path)

        if not zip_path.exists():
            logger.error(f"Kinase networks zip not found at {zip_path}")
            return

        layer_graphs: Dict[str, nx.Graph] = {}

        # Load TSVs from Zip
        with zipfile.ZipFile(zip_path) as zf:
            for name in zf.namelist():
                if not name.endswith(".tsv"): continue

                with zf.open(name) as f:
                    df = pd.read_csv(f, sep="\t")

                layer_name = name.replace(".tsv", "").replace("kk_", "").strip()

                G = nx.Graph()
                for _, row in df.iterrows():
                    s, t = row["source"], row["target"]
                    w = float(row.get("weight", 1.0))
                    if G.has_edge(s, t):
                        G[s][t]["weight"] += w
                    else:
                        G.add_edge(s, t, weight=w, layer=layer_name)

                layer_graphs[layer_name] = G
                logger.info(f"  - Loaded layer '{layer_name}': {G.number_of_nodes()} nodes")

        # Unified Graph
        U = nx.Graph()
        for layer, G in layer_graphs.items():
            for u, v, data in G.edges(data=True):
                w = float(data.get("weight", 1.0))
                if not U.has_node(u): U.add_node(u)
                if not U.has_node(v): U.add_node(v)

                if U.has_edge(u, v):
                    e = U[u][v]
                    e["layers"].add(layer)
                    e["weights"][layer] = w
                else:
                    U.add_edge(u, v, layers={layer}, weights={layer: w})

        # Aggregation
        for u, v, data in U.edges(data=True):
            ws = list(data["weights"].values())
            data["weight_sum"] = float(sum(ws))
            data["weight_mean"] = float(sum(ws) / len(ws))
            data["support"] = len(ws)

        # Save Native
        out_pkl = self.processed_dir / "unified_kinase_graph.gpickle"
        with open(out_pkl, "wb") as f:
            pickle.dump(U, f)
        logger.success(f"Saved {out_pkl}")

        # Save GraphML (Sanitized)
        U_gml = nx.Graph()
        U_gml.add_nodes_from(U.nodes())
        for u, v, data in U.edges(data=True):
            d = dict(data)
            if "layers" in d: d["layers"] = ",".join(sorted(d["layers"]))
            if "weights" in d: d["weights"] = json.dumps(d["weights"])
            U_gml.add_edge(u, v, **d)

        nx.write_graphml(U_gml, self.processed_dir / "unified_kinase_graph.graphml")

    # =========================================================================
    # 4. KINASE-SUBSTRATE MAPPING
    # =========================================================================

    def build_ks_map(self, gsl_zip: Optional[str] = None):
        """
        Builds the Site-Level Kinase-Substrate Map from KEA data.
        """
        logger.header("[*] Building Kinase-Substrate Map")

        if gsl_zip is None:
            gsl_zip = self.raw_dir / "kea" / "gsl_database.zip"
        else:
            gsl_zip = Path(gsl_zip)

        if not gsl_zip.exists():
            logger.error(f"GSL database zip not found at {gsl_zip}")
            return

        fname = "kinase_substrate_phospho-site_level_pmid_resource_database.txt"

        # Load
        try:
            with zipfile.ZipFile(gsl_zip) as zf:
                if fname not in zf.namelist():
                    logger.error(f"Required file {fname} not found in zip.")
                    return
                with zf.open(fname) as f:
                    df = pd.read_csv(f, sep="\t", header=None, names=["kinase", "substrate_site", "pmid", "source"])
        except Exception as e:
            logger.error(f"Failed to read KS zip: {e}")
            return

        # Process
        mask = df["substrate_site"].astype(str).str.contains("_")
        df = df[mask].copy()
        df[["substrate_gene", "site"]] = df["substrate_site"].str.rsplit("_", n=1, expand=True)

        df["substrate_site_upper"] = df["substrate_site"].str.upper()
        df["substrate_gene"] = df["substrate_gene"].str.upper()
        df["kinase"] = df["kinase"].str.upper()

        # Build Index
        index = {}
        for _, row in df.iterrows():
            key = row["substrate_site_upper"]
            entry = index.setdefault(key, {"kinases": set(), "pmids": set(), "sources": set()})
            entry["kinases"].add(str(row["kinase"]))
            entry["pmids"].add(str(row["pmid"]))
            entry["sources"].add(str(row["source"]))

        # Save Table
        out_table = self.processed_dir / "ks_psite_table.tsv"
        df.to_csv(out_table, sep="\t", index=False)
        logger.success(f"Saved {out_table}")

        # Save Index
        out_index = self.processed_dir / "ks_psite_index.pkl"
        with open(out_index, "wb") as f:
            pickle.dump(index, f)
        logger.success(f"Saved {out_index}")

    # =========================================================================
    # 5. UTILITIES (Custom conversion)
    # =========================================================================

    def convert_custom_kinase_csv(self, csv_path: str, out_name: str = "kinase_sites.tsv"):
        """
        Converts a user-supplied CSV with columns [GeneID, Psite, Kinase={K1,K2}]
        into a model-ready TSV [Site, Kinase, weight].
        """
        logger.info(f"Converting custom kinase file: {csv_path}")
        df = pd.read_csv(csv_path)
        rows = []

        for _, row in df.iterrows():
            gene = str(row.get("GeneID", row.get("Protein")))
            psite = str(row.get("Psite", row.get("Residue")))

            # Normalize Site ID
            psite_clean = psite.replace("_", "")
            site_id = f"{gene}_{psite_clean}"

            # Parse Kinase List "{CDK1, CDK2}"
            kin_str = str(row.get("Kinase", "")).strip("{} ")
            kin_list = [k.strip() for k in kin_str.split(",") if k.strip()]

            for k in kin_list:
                rows.append({"Site": site_id, "Kinase": k, "weight": 1.0})

        out_df = pd.DataFrame(rows)
        out_path = self.processed_dir / out_name
        out_df.to_csv(out_path, sep="\t", index=False)
        logger.success(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser(description="PhosCrosstalk Data Curation Pipeline")
    parser.add_argument("--dir", default="data_curated", help="Root directory for data storage")
    parser.add_argument("--download", action="store_true", help="Run downloader")
    parser.add_argument("--ptmcode", nargs=2, metavar=('WITHIN', 'BETWEEN'),
                        help="Build PTM databases from provided PTMcode2 .gz files")
    parser.add_argument("--kea", action="store_true",
                        help="Build Kinase Networks and KS Map (requires downloaded KEA data)")
    parser.add_argument("--convert-csv", type=str, help="Convert a custom kinase CSV to TSV")
    parser.add_argument("--all", action="store_true",
                        help="Run full standard pipeline (assuming files are in place/downloaded)")

    args = parser.parse_args()

    curator = DataCurator(args.dir)

    if args.download or args.all:
        curator.download_resources()

    if args.kea or args.all:
        curator.build_kinase_networks()
        curator.build_ks_map()

    if args.ptmcode:
        curator.build_ptm_databases(args.ptmcode[0], args.ptmcode[1])
    elif args.all:
        # Check if user put them manually in raw?
        # PTMcode data usually isn't in Harmonizome default downloads, so we warn if missing
        logger.warning("Skipping PTMcode DB build (requires explicit paths via --ptmcode).")

    if args.convert_csv:
        curator.convert_custom_kinase_csv(args.convert_csv)

    if not any([args.download, args.ptmcode, args.kea, args.convert_csv, args.all]):
        parser.print_help()


if __name__ == "__main__":
    main()