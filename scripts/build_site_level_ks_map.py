#!/usr/bin/env python3
"""
Build site-level kinase–substrate map from KEA gsl_database.zip.

Outputs:
  - ks_psite_table.tsv
      kinase, substrate_site, substrate_gene, site, pmid, source

  - ks_psite_index.pkl
      dict:
        substrate_site (uppercased) -> {
            'kinases': set[str],
            'pmids': set[str],
            'sources': set[str],
        }

  - ks_hits_for_psites.tsv   (optional, if --psites-file is provided)
"""

from __future__ import annotations

import argparse
import pickle
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd


def load_site_level_ks(gsl_zip: str | Path) -> pd.DataFrame:
    """
    Load site-level kinase–substrate database with PMIDs from gsl_database.zip.

    Uses:
      kinase_substrate_phospho-site_level_pmid_resource_database.txt

    Returns
    -------
    df : DataFrame
      Columns: kinase, substrate_site, pmid, source,
               substrate_gene, site, substrate_site_upper
    """
    gsl_zip = Path(gsl_zip)

    fname = "kinase_substrate_phospho-site_level_pmid_resource_database.txt"
    with zipfile.ZipFile(gsl_zip) as zf:
        with zf.open(fname) as f:
            df = pd.read_csv(
                f,
                sep="\t",
                header=None,
                names=["kinase", "substrate_site", "pmid", "source"],
            )

    # keep only entries that look like GENE_SITE
    mask = df["substrate_site"].astype(str).str.contains("_")
    df = df[mask].copy()

    # split substrate_site -> gene + site
    df[["substrate_gene", "site"]] = df["substrate_site"].str.rsplit(
        "_", n=1, expand=True
    )

    # upper-case for robust matching
    df["substrate_site_upper"] = df["substrate_site"].str.upper()
    df["substrate_gene"] = df["substrate_gene"].str.upper()
    df["kinase"] = df["kinase"].str.upper()

    return df


def build_psite_index(df: pd.DataFrame) -> Dict[str, Dict[str, set]]:
    """
    Build an index: substrate_site_upper -> {kinases, pmids, sources}
    """
    index: Dict[str, Dict[str, set]] = {}

    for _, row in df.iterrows():
        key = row["substrate_site_upper"]
        entry = index.setdefault(
            key,
            {"kinases": set(), "pmids": set(), "sources": set()},
        )
        entry["kinases"].add(str(row["kinase"]))
        entry["pmids"].add(str(row["pmid"]))
        entry["sources"].add(str(row["source"]))

    return index


def parse_psites_file(path: str | Path) -> List[str]:
    """
    Parse a file of Psites.

    Accepted formats per non-comment line:
      - "EGFR_Y1068"
      - "EGFR   Y1068"
    Blank lines and lines starting with '#' are ignored.
    """
    psites: List[str] = []
    path = Path(path)

    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) == 1:
                psites.append(parts[0])
            else:
                # assume gene + site
                psites.append(f"{parts[0]}_{parts[1]}")
    return psites


def filter_psites(df: pd.DataFrame, psites: Iterable[str]) -> pd.DataFrame:
    """
    Filter the KS table for a given list of Psites (GENE_SITE).

    Matching is done in upper case on 'substrate_site_upper'.
    """
    norm = {p.upper() for p in psites}
    return df[df["substrate_site_upper"].isin(norm)].copy()


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build site-level kinase–substrate map from KEA gsl_database.zip"
    )
    ap.add_argument(
        "--gsl-zip",
        type=str,
        default="../data/kea2/gsl_database.zip",
        help="Path to gsl_database.zip (from KEA all gene set libraries)",
    )
    ap.add_argument(
        "--out-table",
        type=str,
        default="ks_psite_table.tsv",
        help="Output TSV with full site-level KS table",
    )
    ap.add_argument(
        "--out-index",
        type=str,
        default="ks_psite_index.pkl",
        help="Output pickle with substrate_site -> {kinases, pmids, sources}",
    )
    ap.add_argument(
        "--psites-file",
        type=str,
        default="../data/my_psites.txt",
        help=(
            "Optional file with Psites (one per line, e.g. 'EGFR_Y1068' or 'EGFR Y1068'). "
            "If given, a ks_hits_for_psites.tsv will be written."
        ),
    )
    args = ap.parse_args()

    gsl_zip = Path(args.gsl_zip)

    print(f"[INFO] Loading site-level KS data from {gsl_zip} ...")
    df = load_site_level_ks(gsl_zip)
    print(f"[INFO] Loaded {len(df):,} KS site-level rows.")

    # Save full table
    out_table = Path(args.out_table)
    print(f"[INFO] Writing full KS psite table to {out_table} ...")
    df[["kinase", "substrate_site", "substrate_gene", "site", "pmid", "source"]].to_csv(
        out_table, sep="\t", index=False
    )

    # Build and save index
    print("[INFO] Building psite index ...")
    index = build_psite_index(df)
    out_index = Path(args.out_index)
    print(f"[INFO] Writing psite index (pickle) to {out_index} ...")
    with out_index.open("wb") as f:
        pickle.dump(index, f)

    # Optional: filter for user-provided Psites
    if args.psites_file:
        print(f"[INFO] Parsing Psites from {args.psites_file} ...")
        psites = parse_psites_file(args.psites_file)
        print(f"[INFO] {len(psites)} Psites read.")
        hits = filter_psites(df, psites)
        out_hits = Path("ks_hits_for_psites.tsv")
        print(f"[INFO] Writing hits for provided Psites to {out_hits} ...")
        hits.to_csv(out_hits, sep="\t", index=False)

    print("[INFO] Done.")


if __name__ == "__main__":
    main()
