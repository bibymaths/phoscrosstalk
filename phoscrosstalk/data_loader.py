"""
data_loader.py
Handles data ingestion, scaling, database connectivity, and matrix construction.
"""

import re
import os
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
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")

    df = pd.read_csv(path, sep=None, engine="python")

    value_cols = [c for c in df.columns if c.startswith("v") or c.startswith("x")]
    if len(value_cols) != len(timepoints):
        raise ValueError(
            f"Expected {len(timepoints)} value columns, found {len(value_cols)}"
        )

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


def build_C_matrices_from_db(
    ptm_intra_path,
    ptm_inter_path,
    sites,
    site_prot_idx,
    positions,
    proteins,
    length_scale=50.0,
):
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
    if not os.path.exists(ptm_intra_path):
        raise FileNotFoundError(
            f"Intra-protein PTM database not found: {ptm_intra_path}"
        )
    if not os.path.exists(ptm_inter_path):
        raise FileNotFoundError(
            f"Inter-protein PTM database not found: {ptm_inter_path}"
        )
    N = len(sites)
    idx = {s: i for i, s in enumerate(sites)}
    Cg = np.zeros((N, N), dtype=float)

    # Intra
    conn_i = sqlite3.connect(ptm_intra_path)
    cur_i = conn_i.cursor()
    for protein, res1, r1, res2, r2 in cur_i.execute(
        "SELECT protein, residue1, score1, residue2, score2 FROM intra_pairs"
    ):
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
        "SELECT protein1, residue1, score1, protein2, residue2, score2 FROM inter_pairs"
    ):
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
            if i == j:
                continue
            if site_prot_idx[i] != site_prot_idx[j]:
                continue
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
    if not os.path.exists(path):
        raise FileNotFoundError(f"Kinase-site TSV not found: {path}")
    df = pd.read_csv(path, sep="\t")
    if "weight" not in df.columns:
        df["weight"] = 0.0
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
    if not os.path.exists(ks_psite_table_path):
        raise FileNotFoundError(
            f"KEA kinase-substrate table not found: {ks_psite_table_path}"
        )
    df = pd.read_csv(ks_psite_table_path, sep="\t")
    df["substrate_site"] = df["substrate_site"].astype(str).str.upper()
    df["kinase"] = df["kinase"].astype(str).str.upper()
    sites_upper = [s.upper() for s in sites]
    site_to_idx = {s: i for i, s in enumerate(sites_upper)}
    df = df[df["substrate_site"].isin(site_to_idx.keys())].copy()
    if df.empty:
        raise ValueError("No overlap between ks_psite_table and model sites.")
    grouped = (
        df.groupby(["substrate_site", "kinase"])
        .agg(weight=("pmid", "nunique"))
        .reset_index()
    )
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

RNA_TIMEPOINTS_9 = np.array(
    [4, 8, 15, 30, 60, 120, 240, 480, 960],
    dtype=float,
)

_MAX_WARNING_ITEMS = 20


def load_rna_data(path, timepoints=None):
    """
    Load mRNA time-series data from a CSV file.

    Expected format for current data:

        GeneID,x1,x2,x3,x4,x5,x6,x7,x8,x9
        SMAD7,...

    The columns x1..x9 are mapped to mRNA time points:

        [4, 8, 15, 30, 60, 120, 240, 480, 960]

    Parameters
    ----------
    path : str
        Path to the mRNA CSV file.

    timepoints : array-like | None
        Optional explicit time points. If None and columns are x1..x9,
        RNA_TIMEPOINTS_9 is used.

    Returns
    -------
    tuple
        gene_ids : list[str]
        t_rna : np.ndarray
        rna_matrix : np.ndarray
    """

    if not os.path.exists(path):
        raise FileNotFoundError(f"mRNA data file not found: {path}")

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError(f"mRNA data file is empty: {path}")

    # Prefer explicit GeneID for your current file.
    if "GeneID" in df.columns:
        id_col = "GeneID"
    else:
        # Fallback: first non-numeric / non-time-like column.
        id_col = None
        for col in df.columns:
            col_str = str(col).strip()
            if not _is_time_column(col_str):
                id_col = col
                break

        if id_col is None:
            raise ValueError(
                f"mRNA data file '{path}' must contain a gene identifier column, "
                "for example 'GeneID'."
            )

    if df[id_col].isna().any():
        raise ValueError(
            f"mRNA data file '{path}': gene-ID column '{id_col}' contains NaN."
        )

    gene_ids = df[id_col].astype(str).str.strip().tolist()

    if len(set(gene_ids)) != len(gene_ids):
        duplicated = pd.Series(gene_ids)[pd.Series(gene_ids).duplicated()].unique()
        raise ValueError(
            f"mRNA data file '{path}' contains duplicated gene IDs: "
            + ", ".join(map(str, duplicated[:_MAX_WARNING_ITEMS]))
        )

    data_cols = [c for c in df.columns if c != id_col]

    if len(data_cols) < 2:
        raise ValueError(
            f"mRNA data file '{path}' must contain at least 2 time-series columns; "
            f"found {len(data_cols)}."
        )

    # Current expected format: x1..x9
    x_cols = [c for c in data_cols if re.fullmatch(r"x\d+", str(c).strip(), flags=re.I)]

    if len(x_cols) == len(data_cols):
        x_cols = sorted(x_cols, key=lambda c: int(re.findall(r"\d+", str(c))[0]))

        if timepoints is None:
            if len(x_cols) != len(RNA_TIMEPOINTS_9):
                raise ValueError(
                    f"mRNA data file '{path}' has {len(x_cols)} x-columns, but the "
                    f"default RNA time vector has {len(RNA_TIMEPOINTS_9)} values. "
                    "Pass explicit timepoints or fix the input file."
                )
            t_rna = RNA_TIMEPOINTS_9.copy()
        else:
            t_rna = np.asarray(timepoints, dtype=float)
            if len(t_rna) != len(x_cols):
                raise ValueError(
                    f"Provided {len(t_rna)} RNA time points but found "
                    f"{len(x_cols)} expression columns."
                )

        time_cols = x_cols

    else:
        # Fallback: parse numeric column names directly.
        numeric_cols = []
        numeric_times = []

        for col in data_cols:
            try:
                numeric_times.append(float(str(col).strip()))
                numeric_cols.append(col)
            except ValueError:
                continue

        if len(numeric_cols) < 2:
            raise ValueError(
                f"mRNA data file '{path}' must either use x1..x9 columns or numeric "
                f"time-point column names. Found columns: {list(df.columns)}"
            )

        order = np.argsort(numeric_times)
        time_cols = [numeric_cols[i] for i in order]
        t_rna = np.asarray([numeric_times[i] for i in order], dtype=float)

    try:
        rna_matrix = df[time_cols].to_numpy(dtype=float)
    except ValueError as exc:
        raise ValueError(
            f"mRNA data file '{path}' contains non-numeric expression values "
            f"in columns {time_cols}."
        ) from exc

    if not np.isfinite(rna_matrix).all():
        bad = int((~np.isfinite(rna_matrix)).sum())
        raise ValueError(
            f"mRNA data file '{path}' contains {bad} non-finite value(s) "
            "in the expression matrix."
        )

    if len(t_rna) != rna_matrix.shape[1]:
        raise ValueError(
            f"Internal RNA shape mismatch: {len(t_rna)} time points but "
            f"RNA matrix has {rna_matrix.shape[1]} columns."
        )

    return gene_ids, t_rna, rna_matrix


def _is_time_column(col):
    """
    Return True for numeric columns or x1/x2/... style time columns.
    """
    col = str(col).strip()

    if re.fullmatch(r"x\d+", col, flags=re.I):
        return True

    try:
        float(col)
        return True
    except ValueError:
        return False


def load_tf_network(path, gene_ids=None):
    """
    Load a TF-mRNA interaction network from a CSV file.

    Expected current format:

        Source,Target,Weight
        AR,CLK3,0.827...

    Column names are accepted case-insensitively and normalized to:

        source, target, weight

    Parameters
    ----------
    path : str
        Path to the TF-mRNA CSV file.

    gene_ids : list[str] | None
        Optional mRNA gene IDs from the RNA file. Used for warnings only.

    Returns
    -------
    pd.DataFrame
        Columns: source, target, weight
    """

    if not os.path.exists(path):
        raise FileNotFoundError(f"TF-mRNA network file not found: {path}")

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError(f"TF-mRNA network file is empty: {path}")

    # Normalize column names case-insensitively.
    col_map = {str(c).strip().lower(): c for c in df.columns}

    required = {
        "source": None,
        "target": None,
    }

    for key in required:
        if key not in col_map:
            raise ValueError(
                f"TF network file '{path}' is missing required column '{key}'. "
                f"Found columns: {list(df.columns)}"
            )
        required[key] = col_map[key]

    source_col = required["source"]
    target_col = required["target"]
    weight_col = col_map.get("weight", None)

    out = pd.DataFrame()
    out["source"] = df[source_col].astype(str).str.strip()
    out["target"] = df[target_col].astype(str).str.strip()

    if out["source"].eq("").any():
        raise ValueError(f"TF network file '{path}': empty source values found.")

    if out["target"].eq("").any():
        raise ValueError(f"TF network file '{path}': empty target values found.")

    if df[source_col].isna().any():
        raise ValueError(f"TF network file '{path}': source column contains NaN.")

    if df[target_col].isna().any():
        raise ValueError(f"TF network file '{path}': target column contains NaN.")

    if weight_col is None:
        out["weight"] = 1.0
    else:
        out["weight"] = pd.to_numeric(df[weight_col], errors="coerce").fillna(1.0)

    if not np.isfinite(out["weight"].to_numpy(dtype=float)).all():
        raise ValueError(
            f"TF network file '{path}': weight column contains non-finite values."
        )

    # Collapse duplicate edges by summing weights.
    out = (
        out.groupby(["source", "target"], as_index=False, sort=False)["weight"]
        .sum()
    )

    if gene_ids is not None:
        gene_set = {str(g).strip() for g in gene_ids}

        unknown_targets = sorted(set(out["target"]) - gene_set)
        if unknown_targets:
            logger.warning(
                f"[!] {len(unknown_targets)} TF target(s) in '{path}' not found "
                "in mRNA data: "
                + ", ".join(unknown_targets[:_MAX_WARNING_ITEMS])
                + ("..." if len(unknown_targets) > _MAX_WARNING_ITEMS else "")
            )

        unknown_sources = sorted(set(out["source"]) - gene_set)
        if unknown_sources:
            logger.warning(
                f"[!] {len(unknown_sources)} TF source(s) in '{path}' not found "
                "in mRNA data. This is allowed if TF activity is inferred from "
                "protein/kinase data: "
                + ", ".join(unknown_sources[:_MAX_WARNING_ITEMS])
                + ("..." if len(unknown_sources) > _MAX_WARNING_ITEMS else "")
            )

    return out[["source", "target", "weight"]].copy()


def build_tf_prot_weights(tf_net_df, gene_ids, proteins):
    """
    Build the TF → protein weight matrix from a TF–mRNA edge table.

    Maps TF gene names (``source`` column) to model proteins via the target
    mRNA name, assuming TF name == protein name in the model namespace.

    Args:
        tf_net_df (pd.DataFrame): Edge table with columns
            ``['source', 'target', 'weight']``.
        gene_ids (list[str]): Gene IDs in the mRNA dataset (column order).
        proteins (list[str]): Protein names in the model.

    Returns:
        np.ndarray: Weight matrix of shape ``(K_proteins, n_genes)`` where
            entry ``[p, g]`` is the sum of edge weights from gene *g* (as TF)
            to protein *p* (matched by target name).
    """
    K = len(proteins)
    G = len(gene_ids)
    gene_idx = {g: i for i, g in enumerate(gene_ids)}
    prot_idx = {p: k for k, p in enumerate(proteins)}

    W = np.zeros((K, G), dtype=float)
    for _, row in tf_net_df.iterrows():
        src = str(row["source"])
        tgt = str(row["target"])
        w = float(row["weight"])
        # target maps to a gene; source TF maps to a protein of same name
        g_idx = gene_idx.get(src)
        p_idx = prot_idx.get(tgt)
        if g_idx is not None and p_idx is not None:
            W[p_idx, g_idx] += w

    return W


def build_alpha_laplacian_from_unified_graph(
    pkl_path, kinases, weight_attr="weight_mean"
):
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
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"Unified kinase graph pickle not found: {pkl_path}")
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
