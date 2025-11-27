#!/usr/bin/env python3
import argparse
import gzip
from io import StringIO
import subprocess
from pathlib import Path
import pandas as pd


def load_dbptm(dbptm_path: str) -> pd.DataFrame:
    """Load dbPTM phospho.gz and prepare a join key."""
    phospho = pd.read_csv(
        dbptm_path,
        sep="\t",
        compression="gzip"
    )

    # Keep phosphorylation only (case-insensitive)
    phospho = phospho[phospho["PTMType"].str.lower() == "phosphorylation"].copy()

    phospho["Position"] = phospho["Position"].astype(int)
    phospho["UniProtID"] = phospho["UniProtID"].astype(str)
    phospho["key"] = phospho["UniProtID"] + ":" + phospho["Position"].astype(str)

    return phospho


def load_idmap(mapping_path: str) -> pd.DataFrame:
    """
    Load mapped_ids table in a very forgiving way.

    Expected examples:
        ##  NA
        1   P48742
        16-5-5 Q9NS63
        CBFA2T2 O43439  # extra trailing columns are ignored

    Rules:
      - skip empty lines
      - skip lines starting with '#'
      - split on any whitespace
      - take the first two tokens as (Protein, UniProtID)
      - ignore rows where UniProtID == 'NA'
      - drop duplicate Protein entries
    """
    rows: list[tuple[str, str]] = []

    with open(mapping_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                continue

            parts = line.split()
            if len(parts) < 2:
                continue

            protein = parts[0]
            uniprot = parts[1]
            rows.append((protein, uniprot))

    if not rows:
        raise ValueError(f"No valid mapping rows found in {mapping_path}")

    idmap = pd.DataFrame(rows, columns=["Protein", "UniProtID"], dtype=str)

    # Drop NA mappings
    idmap = idmap[idmap["UniProtID"] != "NA"].copy()

    # Ensure uniqueness per PTMcode protein
    idmap = idmap.drop_duplicates(subset=["Protein"])

    return idmap

def parse_residue(res_str: str):
    """
    Convert 'Y535' -> ('Y', 535).
    Assumes first char is aa, rest is integer position.
    """
    if not isinstance(res_str, str) or len(res_str) < 2:
        return None, None
    aa = res_str[0]
    try:
        pos = int(res_str[1:])
    except ValueError:
        return aa, None
    return aa, pos

def read_ptmcode_table(path: str) -> pd.DataFrame:
    """
    Read a PTMcode2 .gz file (within/between) that starts with
    '##' metadata lines and then a header line.

    We:
      - skip leading comment lines
      - detect the header line by presence of 'Protein' and 'Residue1'
      - pass header + data to pandas
      - strip leading '#' from column names
    """
    lines = []
    with gzip.open(path, "rt") as f:
        for line in f:
            # skip pure metadata / blank lines
            if line.strip().startswith("## Sept") or line.strip().startswith("## propagated"):
                continue
            if line.strip() == "":
                continue
            # first non-metadata line containing column names
            lines.append(line)
            break
        # read the rest (data rows)
        for line in f:
            lines.append(line)

    if not lines:
        raise ValueError(f"No header/data found in PTMcode file: {path}")

    buf = StringIO("".join(lines))
    df = pd.read_csv(buf, sep="\t")

    # Normalize column names: remove leading '#', trim spaces
    df.rename(columns=lambda c: c.lstrip("#").strip(), inplace=True)
    return df


def load_ptmcode_within(within_path: str) -> pd.DataFrame:
    """
    Load PTMcode2 within.gz and return per-site long-form table.
    Each row in the input yields two site rows (PTM1, PTM2).
    """
    df = read_ptmcode_table(within_path)
    # Expect columns: Protein, Species, PTM1, Residue1, rRCS1, Propagated1, PTM2, Residue2, ...

    rows = []
    for _, row in df.iterrows():
        protein = str(row["Protein"])
        species = str(row["Species"])

        # PTM1
        aa1, pos1 = parse_residue(str(row["Residue1"]))
        rows.append({
            "Protein": protein,
            "Species": species,
            "ptm_type": str(row["PTM1"]),
            "aa": aa1,
            "pos": pos1,
            "rRCS": row["rRCS1"],
            "propagated": row["Propagated1"],
            "context": "within",
        })

        # PTM2
        aa2, pos2 = parse_residue(str(row["Residue2"]))
        rows.append({
            "Protein": protein,
            "Species": species,
            "ptm_type": str(row["PTM2"]),
            "aa": aa2,
            "pos": pos2,
            "rRCS": row["rRCS2"],
            "propagated": row["Propagated2"],
            "context": "within",
        })

    return pd.DataFrame(rows)

def load_ptmcode_between(between_path: str) -> pd.DataFrame:
    """
    Load PTMcode2 between.gz and return per-site long-form table.
    Each edge yields two site rows (one on Protein1, one on Protein2).
    """
    df = read_ptmcode_table(between_path)
    # Expect columns: Protein1, Protein2, Species, PTM1, Residue1, rRCS1, Propagated1, PTM2, Residue2, ...

    rows = []
    for _, row in df.iterrows():
        species = str(row["Species"])

        # Site on Protein1
        aa1, pos1 = parse_residue(str(row["Residue1"]))
        rows.append({
            "Protein": str(row["Protein1"]),
            "partner": str(row["Protein2"]),
            "Species": species,
            "ptm_type": str(row["PTM1"]),
            "aa": aa1,
            "pos": pos1,
            "rRCS": row["rRCS1"],
            "propagated": row["Propagated1"],
            "context": "between",
        })

        # Site on Protein2
        aa2, pos2 = parse_residue(str(row["Residue2"]))
        rows.append({
            "Protein": str(row["Protein2"]),
            "partner": str(row["Protein1"]),
            "Species": species,
            "ptm_type": str(row["PTM2"]),
            "aa": aa2,
            "pos": pos2,
            "rRCS": row["rRCS2"],
            "propagated": row["Propagated2"],
            "context": "between",
        })

    return pd.DataFrame(rows)


def build_ptm_sites(within_path: str, between_path: str) -> pd.DataFrame:
    """Combine within + between PTMcode2 into one per-site table."""
    within_sites = load_ptmcode_within(within_path)
    between_sites = load_ptmcode_between(between_path)

    ptm_sites = pd.concat([within_sites, between_sites], ignore_index=True)

    # Keep phosphorylation only
    ptm_sites["ptm_type_lower"] = ptm_sites["ptm_type"].str.lower()
    ptm_sites = ptm_sites[ptm_sites["ptm_type_lower"] == "phosphorylation"].copy()
    ptm_sites.drop(columns=["ptm_type_lower"], inplace=True)

    # Drop sites with no position parsed
    ptm_sites = ptm_sites[ptm_sites["pos"].notna()].copy()
    ptm_sites["pos"] = ptm_sites["pos"].astype(int)

    return ptm_sites

def compress_output_tsv(tsv_path: str):
    """Compress merged TSV using maximum xz compression."""
    tsv_path = Path(tsv_path)
    out_path = tsv_path.with_suffix(".tar.xz")

    print(f"[*] Compressing {tsv_path} -> {out_path} with xz -9e ...")

    subprocess.run(
        [
            "tar",
            "-cvf",
            str(out_path),
            "--use-compress-program=xz -9e",
            str(tsv_path)
        ],
        check=True,
        shell=False,
    )

    print(f"[✓] Saved: {out_path}")

def merge_all(dbptm_path: str,
              within_path: str,
              between_path: str,
              mapping_path: str,
              out_path: str):
    print("[*] Loading dbPTM...")
    phospho = load_dbptm(dbptm_path)

    print("[*] Loading PTMcode2 within/between...")
    ptm_sites = build_ptm_sites(within_path, between_path)

    print(f"    PTMcode sites total: {ptm_sites.shape[0]}")

    print("[*] Loading ID mapping...")
    idmap = load_idmap(mapping_path)
    print(f"    Mapped proteins: {idmap.shape[0]}")

    print("[*] Attaching UniProt IDs to PTMcode sites...")
    ptm_sites = ptm_sites.merge(idmap, on="Protein", how="left")

    before = ptm_sites.shape[0]
    ptm_sites = ptm_sites[ptm_sites["UniProtID"].notna()].copy()
    after = ptm_sites.shape[0]
    print(f"    Sites with mapped UniProtID: {after}/{before}")

    # Build join key and merge with dbPTM
    ptm_sites["UniProtID"] = ptm_sites["UniProtID"].astype(str)
    ptm_sites["key"] = ptm_sites["UniProtID"] + ":" + ptm_sites["pos"].astype(str)

    print("[*] Merging PTMcode sites with dbPTM annotations...")
    merged = ptm_sites.merge(
        phospho[["key", "EntryName", "PMIDs", "Peptide"]],
        on="key",
        how="left",
        suffixes=("", "_dbptm"),
    )

    print(f"    Merged table rows: {merged.shape[0]}")

    print(f"[*] Writing merged table to {out_path}")
    merged.to_csv(out_path, sep="\t", index=False)

    # Quick stats
    n_with_dbptm = merged["EntryName"].notna().sum()
    print(f"    Sites with dbPTM match: {n_with_dbptm}/{merged.shape[0]}")

    compress_output_tsv(out_path)
    print(f"[*] Compressed output saved.")

def main():
    ap = argparse.ArgumentParser(
        description="Merge dbPTM phospho.gz, PTMcode2 within/between, and mapped_ids."
    )
    ap.add_argument("--dbptm", required=True, help="Path to dbPTM phospho.gz")
    ap.add_argument("--within", required=True, help="Path to PTMcode2 within.gz")
    ap.add_argument("--between", required=True, help="Path to PTMcode2 between.gz")
    ap.add_argument("--mapping", required=True, help="Path to mapped_ids file (Protein -> UniProtID, NA allowed)")
    ap.add_argument("--out", required=True, help="Output TSV path for merged table")

    args = ap.parse_args()

    merge_all(
        dbptm_path=args.dbptm,
        within_path=args.within,
        between_path=args.between,
        mapping_path=args.mapping,
        out_path=args.out,
    )

if __name__ == "__main__":
    main()
