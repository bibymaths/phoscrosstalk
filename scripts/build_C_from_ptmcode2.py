#!/usr/bin/env python3
import gzip
import argparse
import sqlite3


def load_within(path):
    """
    PTMcode2 'within' file.
    Returns list of (protein, residue1, score1, protein, residue2, score2)
    for Homo sapiens phosphorylation–phosphorylation pairs.
    """
    rows = []
    with gzip.open(path, "rt") as f:
        for line in f:
            if line.startswith("#") or line.strip() == "":
                continue
            parts = line.strip().split("\t")
            if len(parts) < 13:
                continue

            protein = parts[0]
            species = parts[1]
            if species != "Homo sapiens":
                continue
            if parts[2] != "phosphorylation" or parts[6] != "phosphorylation":
                continue

            res1 = parts[3]   # e.g. Y535
            r1 = float(parts[4])
            res2 = parts[7]
            r2 = float(parts[8])

            rows.append((protein, res1, r1, protein, res2, r2))
    return rows


def load_between(path):
    """
    PTMcode2 'between' file.
    Returns list of (protein1, residue1, score1, protein2, residue2, score2)
    for Homo sapiens phosphorylation–phosphorylation pairs.
    """
    rows = []
    with gzip.open(path, "rt") as f:
        for line in f:
            if line.startswith("#") or line.strip() == "":
                continue
            parts = line.strip().split("\t")
            if len(parts) < 13:
                continue

            p1, p2 = parts[0], parts[1]
            species = parts[2]
            if species != "Homo sapiens":
                continue
            if parts[3] != "phosphorylation" or parts[7] != "phosphorylation":
                continue

            res1 = parts[4]   # e.g. S588
            r1 = float(parts[5])
            res2 = parts[8]   # e.g. S380
            r2 = float(parts[9])

            rows.append((p1, res1, r1, p2, res2, r2))
    return rows


def init_intra_db(path):
    conn = sqlite3.connect(path)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS intra_pairs (
            id       INTEGER PRIMARY KEY AUTOINCREMENT,
            protein  TEXT NOT NULL,
            residue1 TEXT NOT NULL,
            score1   REAL NOT NULL,
            residue2 TEXT NOT NULL,
            score2   REAL NOT NULL
        )
        """
    )
    cur.execute("CREATE INDEX IF NOT EXISTS idx_intra_protein ON intra_pairs(protein)")
    conn.commit()
    return conn


def init_inter_db(path):
    conn = sqlite3.connect(path)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS inter_pairs (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            protein1  TEXT NOT NULL,
            residue1  TEXT NOT NULL,
            score1    REAL NOT NULL,
            protein2  TEXT NOT NULL,
            residue2  TEXT NOT NULL,
            score2    REAL NOT NULL
        )
        """
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_inter_proteins "
        "ON inter_pairs(protein1, protein2)"
    )
    conn.commit()
    return conn


def main():
    parser = argparse.ArgumentParser(
        description="Build SQLite DBs of PTMcode2 intra/inter phospho pairs."
    )
    parser.add_argument("--within", required=True,
                        help="Gzipped PTMcode2 'within' file.")
    parser.add_argument("--between", required=True,
                        help="Gzipped PTMcode2 'between' file.")
    parser.add_argument("--out-intra", default="ptm_intra.db",
                        help="Output SQLite DB for intra-protein pairs.")
    parser.add_argument("--out-inter", default="ptm_inter.db",
                        help="Output SQLite DB for inter-protein pairs.")
    args = parser.parse_args()

    print("[*] Loading PTMcode2 within (intra-protein) ...")
    W = load_within(args.within)
    print(f"[*] Loaded {len(W)} intra-protein phosphorylation pairs.")

    print("[*] Loading PTMcode2 between (inter-protein) ...")
    B = load_between(args.between)
    print(f"[*] Loaded {len(B)} inter-protein phosphorylation pairs.")

    # ------------------------------------------------------------------
    # Write intra-protein pairs DB
    # ------------------------------------------------------------------
    print(f"[*] Initialising intra DB: {args.out_intra}")
    conn_intra = init_intra_db(args.out_intra)
    cur_intra = conn_intra.cursor()

    print("[*] Inserting intra-protein pairs into DB ...")
    cur_intra.executemany(
        """
        INSERT INTO intra_pairs (protein, residue1, score1, residue2, score2)
        VALUES (?, ?, ?, ?, ?)
        """,
        [(p, res1, r1, res2, r2) for (p, res1, r1, _, res2, r2) in W],
    )
    conn_intra.commit()
    conn_intra.close()
    print(f"[*] Saved intra-protein pairs to {args.out_intra}")

    # ------------------------------------------------------------------
    # Write inter-protein pairs DB
    # ------------------------------------------------------------------
    print(f"[*] Initialising inter DB: {args.out_inter}")
    conn_inter = init_inter_db(args.out_inter)
    cur_inter = conn_inter.cursor()

    print("[*] Inserting inter-protein pairs into DB ...")
    cur_inter.executemany(
        """
        INSERT INTO inter_pairs
            (protein1, residue1, score1, protein2, residue2, score2)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        B,
    )
    conn_inter.commit()
    conn_inter.close()
    print(f"[*] Saved inter-protein pairs to {args.out_inter}")

    print("[*] Done.")


if __name__ == "__main__":
    main()