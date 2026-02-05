"""
data_loader.py
Handles data ingestion, scaling, database connectivity, and matrix construction.
"""
import re
import sqlite3
import pickle
import numpy as np
import pandas as pd
from phoscrosstalk.config import DEFAULT_TIMEPOINTS


def load_site_data(path, timepoints=DEFAULT_TIMEPOINTS):
    """Load time-series from a file."""
    df = pd.read_csv(path, sep=None, engine="python")

    value_cols = [c for c in df.columns if c.startswith("v") or c.startswith("x")]
    if len(value_cols) != len(timepoints):
        raise ValueError(f"Expected {len(timepoints)} value columns, found {len(value_cols)}")

    if "Protein" in df.columns:
        prot_col = "Protein"
    elif "GeneID" in df.columns:
        prot_col = "GeneID"
    else:
        raise ValueError("Need either 'Protein' or 'GeneID' column in data.")

    if "Psite" in df.columns:
        has_site = df["Psite"].notna()
    elif "Residue" in df.columns:
        has_site = df["Residue"].notna()
    else:
        raise ValueError("Need either 'Residue' or 'Psite' column in data.")

    df_sites = df[has_site].copy()
    df_prot = df[~has_site].copy()

    proteins_raw = df_sites[prot_col].astype(str).tolist()
    residues_raw = []
    positions = []

    if "Residue" in df_sites.columns:
        for r in df_sites["Residue"].astype(str):
            residues_raw.append(r)
            m = re.match(r"[A-Z]([0-9]+)", r)
            positions.append(int(m.group(1)) if m else np.nan)
    elif "Psite" in df_sites.columns:
        for psite in df_sites["Psite"]:
            psite = str(psite)
            if "_" in psite:
                aa, pos = psite.split("_", 1)
                residues_raw.append(f"{aa}{pos}")
                try:
                    positions.append(int(pos))
                except ValueError:
                    positions.append(np.nan)
            else:
                residues_raw.append(psite)
                m = re.match(r"[A-Z]([0-9]+)", psite)
                positions.append(int(m.group(1)) if m else np.nan)

    positions = np.array(positions, dtype=float)
    sites = [f"{p}_{r}" for p, r in zip(proteins_raw, residues_raw)]
    proteins = sorted(set(proteins_raw))
    prot_index = {p: k for k, p in enumerate(proteins)}
    site_prot_idx = np.array([prot_index[p] for p in proteins_raw], dtype=int)

    Y = df_sites[value_cols].values.astype(float)
    t = np.array(timepoints, dtype=float)

    A_data = None
    A_proteins = None

    if not df_prot.empty:
        A_rows = []
        A_prots = []
        for p, sub in df_prot.groupby(prot_col):
            p_str = str(p)
            if p_str not in prot_index:
                continue
            A_prots.append(p_str)
            A_rows.append(sub[value_cols].values.astype(float).mean(axis=0))

        if A_rows:
            A_data = np.vstack(A_rows)
            A_proteins = np.array(A_prots, dtype=object)

    return sites, proteins, site_prot_idx, positions, t, Y, A_data, A_proteins


def scale_fc_to_unit_interval(Y, use_log=False):
    """Min-Max scaling to [0, 1] based on the whole row."""
    N, T = Y.shape
    P = np.zeros_like(Y, dtype=float)
    baselines = np.zeros(N, dtype=float)
    amplitudes = np.zeros(N, dtype=float)

    for i in range(N):
        y = Y[i].astype(float)
        if use_log:
            y = np.log1p(np.maximum(y, 0))

        min_y = np.min(y)
        max_y = np.max(y)
        A = max_y - min_y

        if A < 1e-6:
            A = 1.0

        P[i] = (y - min_y) / A
        baselines[i] = min_y
        amplitudes[i] = A

    return P, baselines, amplitudes


def apply_scaling(Y, mode="minmax"):
    if mode == "minmax":
        return scale_fc_to_unit_interval(Y, use_log=False)
    elif mode == "log-minmax":
        return scale_fc_to_unit_interval(Y, use_log=True)
    elif mode == "none":
        N, T = Y.shape
        baselines = np.zeros(N, dtype=float)
        amplitudes = np.ones(N, dtype=float)
        return Y.astype(float).copy(), baselines, amplitudes
    else:
        raise ValueError(f"Unknown scale mode: {mode}")


def row_normalize(C):
    row_sums = C.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0.0] = 1.0
    return C / row_sums


def build_C_matrices_from_db(ptm_intra_path, ptm_inter_path, sites, site_prot_idx, positions, proteins,
                             length_scale=50.0):
    N = len(sites)
    idx = {s: i for i, s in enumerate(sites)}
    Cg = np.zeros((N, N), dtype=float)

    # Intra
    conn_i = sqlite3.connect(ptm_intra_path)
    cur_i = conn_i.cursor()
    for protein, res1, r1, res2, r2 in cur_i.execute(
            "SELECT protein, residue1, score1, residue2, score2 FROM intra_pairs"):
        s1 = f"{protein}_{res1}"
        s2 = f"{protein}_{res2}"
        if s1 in idx and s2 in idx:
            i, j = idx[s1], idx[s2]
            score = 0.8 * (r1 + r2) / 200.0
            if score > Cg[i, j]:
                Cg[i, j] = Cg[j, i] = score
    conn_i.close()

    # Inter
    conn_e = sqlite3.connect(ptm_inter_path)
    cur_e = conn_e.cursor()
    for p1, res1, r1, p2, res2, r2 in cur_e.execute(
            "SELECT protein1, residue1, score1, protein2, residue2, score2 FROM inter_pairs"):
        s1 = f"{p1}_{res1}"
        s2 = f"{p2}_{res2}"
        if s1 in idx and s2 in idx:
            i, j = idx[s1], idx[s2]
            score = 0.8 * (r1 + r2) / 200.0
            if score > Cg[i, j]:
                Cg[i, j] = Cg[j, i] = score
    conn_e.close()

    # Local
    Cl = np.zeros((N, N), dtype=float)
    L = float(length_scale)
    for i in range(N):
        for j in range(N):
            if i == j: continue
            if site_prot_idx[i] != site_prot_idx[j]: continue
            if np.isfinite(positions[i]) and np.isfinite(positions[j]):
                d = abs(positions[i] - positions[j])
                Cl[i, j] = np.exp(-d / L)

    return Cg, Cl


def load_kinase_site_matrix(path, sites):
    df = pd.read_csv(path, sep="\t")
    if "weight" not in df.columns: df["weight"] = 0.0
    site_to_idx = {s: i for i, s in enumerate(sites)}
    kinases = sorted(df["Kinase"].astype(str).unique())
    kin_index = {k: j for j, k in enumerate(kinases)}
    N, M = len(sites), len(kinases)
    K_site_kin = np.zeros((N, M), dtype=float)
    for _, row in df.iterrows():
        s, k, w = str(row["Site"]), str(row["Kinase"]), float(row["weight"])
        if s in site_to_idx and k in kin_index:
            i, j = site_to_idx[s], kin_index[k]
            if w > K_site_kin[i, j]:
                K_site_kin[i, j] = w
    return K_site_kin, kinases


def build_kinase_site_from_kea(ks_psite_table_path, sites):
    df = pd.read_csv(ks_psite_table_path, sep="\t")
    df["substrate_site"] = df["substrate_site"].astype(str).str.upper()
    df["kinase"] = df["kinase"].astype(str).str.upper()
    sites_upper = [s.upper() for s in sites]
    site_to_idx = {s: i for i, s in enumerate(sites_upper)}
    df = df[df["substrate_site"].isin(site_to_idx.keys())].copy()
    if df.empty:
        raise ValueError("No overlap between ks_psite_table and model sites.")
    grouped = df.groupby(["substrate_site", "kinase"]).agg(weight=("pmid", "nunique")).reset_index()
    kinases = sorted(grouped["kinase"].unique())
    kin_index = {k: j for j, k in enumerate(kinases)}
    N, M = len(sites), len(kinases)
    K_site_kin = np.zeros((N, M), dtype=float)
    for _, row in grouped.iterrows():
        s, k, w = row["substrate_site"], row["kinase"], float(row["weight"])
        i, j = site_to_idx[s], kin_index[k]
        if w > K_site_kin[i, j]:
            K_site_kin[i, j] = w
    row_sums = K_site_kin.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0.0] = 1.0
    return K_site_kin / row_sums, kinases


def build_alpha_laplacian_from_unified_graph(pkl_path, kinases, weight_attr="weight_mean"):
    with open(pkl_path, "rb") as f:
        G_full = pickle.load(f)
    kin_to_idx = {k: i for i, k in enumerate(kinases)}
    M = len(kinases)
    A = np.zeros((M, M), dtype=float)
    for u, v, data in G_full.edges(data=True):
        if u in kin_to_idx and v in kin_to_idx and u != v:
            i, j = kin_to_idx[u], kin_to_idx[v]
            w = float(data.get(weight_attr, 1.0))
            if w > A[i, j]:
                A[i, j] = A[j, i] = w
    L = np.diag(A.sum(axis=1)) - A
    return L
