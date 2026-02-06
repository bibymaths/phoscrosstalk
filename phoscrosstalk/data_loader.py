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
from phoscrosstalk.logger import get_logger

logger = get_logger()

def load_site_data(path, timepoints=DEFAULT_TIMEPOINTS):
    """
    Parses a time-series CSV file to extract phosphosite data and optional protein abundance.

    This function expects columns starting with 'v' or 'x' for intensity values matching the
    provided timepoints. It identifies proteins and residues/sites, parsing position information
    (e.g., 'S473') into numeric arrays where possible.

    Args:
        path (str): File path to the dataset (CSV).
        timepoints (list/array): Expected time points corresponding to value columns.

    Returns:
        tuple:
            - sites (list): formatted site labels (Protein_Residue).
            - proteins (list): unique sorted protein names.
            - site_prot_idx (np.ndarray): indices mapping sites to the 'proteins' list.
            - positions (np.ndarray): numeric residue positions (NaN if parsing fails).
            - t (np.ndarray): time points array.
            - Y (np.ndarray): Phosphosite intensity matrix (N_sites x T).
            - A_data (np.ndarray or None): Protein abundance matrix (N_proteins x T) if available.
            - A_proteins (np.ndarray or None): Names of proteins in A_data.
    """

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
    """
    Performs Min-Max scaling on time-series data to normalize it to the [0, 1] interval.

    Args:
        Y (np.ndarray): Raw data matrix (N x T).
        use_log (bool): If True, applies log1p transform before scaling.

    Returns:
        tuple:
            - P (np.ndarray): Scaled data matrix.
            - baselines (np.ndarray): Minimum values per row (for inverse scaling).
            - amplitudes (np.ndarray): Range (max-min) per row (for inverse scaling).
    """
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
    """
    Wrapper function to apply a specific scaling strategy to the data.

    Args:
        Y (np.ndarray): Raw data matrix.
        mode (str): Scaling mode ('minmax', 'log-minmax', or 'none').

    Returns:
        tuple: (Scaled Matrix, Baselines, Amplitudes)
    """
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
    """
    Normalizes a matrix such that each row sums to 1.0.
    Rows summing to zero are set to 1.0 to avoid division errors.

    Args:
        C (np.ndarray): Input matrix.

    Returns:
        np.ndarray: Row-normalized matrix.
    """
    row_sums = C.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0.0] = 1.0
    return C / row_sums


def build_C_matrices_from_db(ptm_intra_path, ptm_inter_path, sites, site_prot_idx, positions, proteins,
                             length_scale=50.0):
    """
    Constructs global (Cg) and local (Cl) crosstalk connectivity matrices using SQLite databases.

    Global crosstalk (Cg) is derived from PTM functional association databases (intra- and inter-protein).
    Local crosstalk (Cl) is calculated based on sequence proximity between sites on the same protein,
    decaying exponentially with distance.

    Args:
        ptm_intra_path (str): Path to SQLite DB for intra-protein associations.
        ptm_inter_path (str): Path to SQLite DB for inter-protein associations.
        sites (list): List of site labels.
        site_prot_idx (np.ndarray): Mapping of sites to proteins.
        positions (np.ndarray): Numeric positions of sites.
        proteins (list): List of protein names.
        length_scale (float): Decay length for local sequence-based coupling.

    Returns:
        tuple: (Cg, Cl) - The global and local adjacency matrices.
    """
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
    """
    Loads a pre-defined Kinase-Substrate interaction matrix from a file.

    Args:
        path (str): Path to the TSV file containing 'Kinase', 'Site', and 'weight'.
        sites (list): List of target sites in the model.

    Returns:
        tuple:
            - K_site_kin (np.ndarray): Matrix of weights (Sites x Kinases).
            - kinases (list): Sorted list of kinase names found in the file.
    """
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
    """
    Constructs a Kinase-Substrate matrix by aggregating citation counts from a KEA (Kinase Enrichment Analysis) table.

    Args:
        ks_psite_table_path (str): Path to KEA TSV table.
        sites (list): List of target sites in the model.

    Returns:
        tuple:
            - K_site_kin (np.ndarray): Row-normalized interaction matrix (Sites x Kinases).
            - kinases (list): Sorted list of kinase names.
    """
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
    """
    Constructs a Laplacian matrix representing the kinase-kinase interaction network from a NetworkX graph.

    The Laplacian (L = D - A) is used to model diffusion or regulatory coupling between kinases
    in the network topology.

    Args:
        pkl_path (str): Path to a pickled NetworkX graph file.
        kinases (list): List of kinase nodes to include in the matrix.
        weight_attr (str): Edge attribute name to use as weight.

    Returns:
        np.ndarray: The Laplacian matrix (M_kinases x M_kinases).
    """
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
