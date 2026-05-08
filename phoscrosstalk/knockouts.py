"""
knockouts.py
Systematic in-silico knockout screens (Kinase, Protein, and Phosphosite KO).
"""

import os

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from tqdm import tqdm

from phoscrosstalk.config import ModelDims
from phoscrosstalk.logger import get_logger
from phoscrosstalk.simulation import build_full_A0, simulate

logger = get_logger()


def run_live_knockout(
    t_eval,
    theta_opt,
    ko_type,
    ko_target,
    proteins,
    kinases,
    sites,
    snap,
    a_proteins=None,
    k_act_fn=None,
    s_prod_fn=None,
):
    """
    Run a pair of ODE simulations (WT vs. single knockout) and return results.

    This is a clean public helper called from the dashboard; all biology is
    delegated to ``simulate``.

    Perturbation types
    ------------------
    ``ko_type`` may be ``"kinase"``, ``"protein"``, or ``"site"``:
      - kinase KO : sets log(alpha) for that kinase to −20 (≈ zero activity).
      - protein KO: sets log(k_deact) for that protein to +10 (fast deactivation).
      - site KO   : zeros all kinase edges for that site in K_site_kin.

    Args:
        t_eval (np.ndarray): Time points for the simulation.
        theta_opt (np.ndarray): Fitted parameter vector (WT).
        ko_type (str): One of ``"kinase"``, ``"protein"``, ``"site"``.
        ko_target (str): Name of the entity to knock out.
        proteins (list[str]): Protein names.
        kinases (list[str]): Kinase names.
        sites (list[str]): Phosphosite names.
        snap (dict): Preopt snapshot dict (from ``load_preopt_snapshot``).
        a_proteins (list[str] | None): Names of proteins that have abundance data,
            in the same row order as ``snap["A_scaled"]``. Used to build the initial
            protein-abundance state vector.  If ``None``, A₀ is initialised to zeros.
        k_act_fn (callable | None): Optional JAX closure for derived k_act(t).
        s_prod_fn (callable | None): Optional JAX closure for derived s_prod(t).

    Returns:
        (dict): Keys ``"wt"`` and ``"ko"`` (each a dict with ``"P_sim"``,
            ``"A_sim"``, ``"S_sim"``, ``"Kdyn_sim"`` of shape entities × T),
            plus ``"t_eval"`` (np.ndarray), ``"ko_type"``, and ``"ko_target"``.
    """
    K = len(proteins)
    M = len(kinases)
    N = len(sites)
    dims = ModelDims.set_dims(K, M, N)

    A_scaled = snap.get("A_scaled", np.empty((0, 0)))
    prot_idx_for_A = _prot_idx_for_A(A_scaled, proteins, a_proteins)
    A0 = _build_A0(K, t_eval, A_scaled, prot_idx_for_A)

    common_kwargs = dict(
        dims=dims,
        t_arr=t_eval,
        P_data0=snap["P_scaled"],
        A_data0=A0,
        Cg=snap["Cg"],
        Cl=snap["Cl"],
        site_prot_idx=snap["site_prot_idx"],
        K_site_kin=snap["K_site_kin"],
        R=snap["R"],
        L_alpha=snap["L_alpha"],
        kin_to_prot_idx=snap["kin_to_prot_idx"],
        receptor_mask_prot=snap["receptor_mask_prot"],
        receptor_mask_kin=snap["receptor_mask_kin"],
        mechanism=snap.get("meta", {}).get("mechanism", "dist"),
        full_output=True,
        k_act_fn=k_act_fn,
        s_prod_fn=s_prod_fn,
    )

    # WT baseline
    P_wt, A_wt, S_wt, Kdyn_wt = simulate(theta=theta_opt, **common_kwargs)

    # Build KO theta / K_site_kin
    theta_ko = theta_opt.copy()
    K_site_kin_ko = snap["K_site_kin"].copy()

    ko_type_lower = ko_type.lower()
    if ko_type_lower == "kinase":
        idx_alpha = 2 * K + 2  # layout: k_deact[K], d_deg[K], beta_g, beta_l, alpha[M]
        try:
            m_idx = kinases.index(ko_target)
            theta_ko[idx_alpha + m_idx] = -20.0
        except (ValueError, IndexError) as exc:
            raise ValueError(f"Kinase '{ko_target}' not found in kinase list.") from exc

    elif ko_type_lower == "protein":
        # Fast-deactivate: push k_deact very high (index: first K elements)
        try:
            p_idx = proteins.index(ko_target)
            theta_ko[p_idx] = 10.0  # log(k_deact) = 10 → k_deact ≈ 22026
        except (ValueError, IndexError) as exc:
            raise ValueError(
                f"Protein '{ko_target}' not found in protein list."
            ) from exc

    elif ko_type_lower == "site":
        try:
            s_idx = sites.index(ko_target)
            K_site_kin_ko[s_idx, :] = 0.0
        except (ValueError, IndexError) as exc:
            raise ValueError(f"Site '{ko_target}' not found in site list.") from exc

    else:
        raise ValueError(
            f"Unknown ko_type '{ko_type}'. Use 'kinase', 'protein', or 'site'."
        )

    common_kwargs["K_site_kin"] = K_site_kin_ko
    P_ko, A_ko, S_ko, Kdyn_ko = simulate(theta=theta_ko, **common_kwargs)

    return {
        "wt": {"P_sim": P_wt, "A_sim": A_wt, "S_sim": S_wt, "Kdyn_sim": Kdyn_wt},
        "ko": {"P_sim": P_ko, "A_sim": A_ko, "S_sim": S_ko, "Kdyn_sim": Kdyn_ko},
        "t_eval": t_eval,
        "ko_type": ko_type,
        "ko_target": ko_target,
    }


def _prot_idx_for_A(A_scaled, proteins, a_proteins):
    """Derive protein indices for A_scaled rows, given the A_proteins name list.

    Args:
        A_scaled (np.ndarray): Protein abundance matrix (n_A_proteins × T).
        proteins (list[str]): Full protein list (length K).
        a_proteins (list[str] | None): Names of proteins with abundance data,
            in the same row order as A_scaled.  When None or empty the function
            returns an empty index array.

    Returns:
        np.ndarray of dtype int, length n_A_proteins.
    """
    if a_proteins is None or len(a_proteins) == 0:
        return np.array([], dtype=int)
    prot_map = {p: i for i, p in enumerate(proteins)}
    indices = []
    for name in a_proteins:
        if name in prot_map:
            indices.append(prot_map[name])
    return np.array(indices, dtype=int)


def _build_A0(K, t_eval, A_scaled, prot_idx_for_A):
    """Construct initial A matrix compatible with simulate."""
    A0 = np.zeros((K, len(t_eval)), dtype=float)
    if A_scaled is not None and A_scaled.size > 0 and len(prot_idx_for_A) > 0:
        for k, p_idx in enumerate(prot_idx_for_A):
            if A_scaled.ndim == 2 and A_scaled.shape[1] > 0:
                A0[p_idx, 0] = A_scaled[k, 0]
            elif A_scaled.ndim == 1:
                A0[p_idx, 0] = A_scaled[k]
    return A0


def run_knockout_screen(outdir, dims: ModelDims, problem, theta_opt, sites, proteins, kinases):
    """
    Perform a systematic in-silico knockout screen for kinases, proteins, and phosphosites.

    This function iterates through every component in the network, virtually "deleting" it
    by modifying parameters (setting rates to zero) or topology matrices, and simulating
    the new steady-state. It calculates the Log2 Fold Change compared to the Wild Type (WT)
    baseline and generates a clustered heatmap of the results.

    Perturbation types:
    - **Kinase KO**: Sets global strength (alpha) to ~0.
    - **Protein KO**: Sets synthesis rate (s_prod) to ~0.
    - **Site KO**: Removes all upstream kinase inputs (simulating Alanine mutation).

    Args:
        outdir (str): Output directory for TSV and PNG files.
        problem (NetworkOptimizationProblem): The initialized optimization problem object containing matrices and config.
        theta_opt (np.ndarray): The optimized parameter vector (Wild Type).
        sites (list): List of phosphosite IDs.
        proteins (list): List of protein IDs.
        kinases (list): List of kinase IDs.

    Returns:
        (None): Saves 'knockout_l2fc.tsv' and 'knockout_clustermap.png' to disk.
    """  # noqa: E501
    logger.info("\n[*] Running Systematic Knockout Screen (Fold Change)...")
    ko_dir = os.path.join(outdir, "knockouts")
    os.makedirs(ko_dir, exist_ok=True)

    K = dims.K

    # 1. Establish Wild Type (WT) Baseline
    t_eval = np.array([0, 240.0])

    def get_steady_state(theta_in, K_mat_in=None):
        if K_mat_in is None:
            K_mat_in = problem.K_site_kin

        A0 = build_full_A0(
            K,
            len(t_eval),
            problem.A_scaled[:, 0:1] if problem.A_scaled.size > 0 else np.array([]),
            problem.prot_idx_for_A,
        )

        P, _, S, Kdyn = simulate(
            t_eval,
            problem.P_data,
            A0,
            theta_in,
            problem.Cg,
            problem.Cl,
            problem.site_prot_idx,
            K_mat_in,
            problem.R,
            problem.L_alpha,
            problem.kin_to_prot_idx,
            problem.receptor_mask_prot,
            problem.receptor_mask_kin,
            problem.mechanism,
            full_output=True,
            dims=dims,
        )
        return P[:, -1], S[:, -1], Kdyn[:, -1]

    wt_P, wt_S, wt_Kdyn = get_steady_state(theta_opt)

    # Storage for results (Fold Change)
    res_P, res_S, res_Kdyn = {}, {}, {}

    # 2. Kinase Knockouts (Set Alpha -> 0)
    # Alpha index in parameter vector: 2*K + 2  (k_deact: K, d_deg: K, beta_g: 1, beta_l: 1)  # noqa: E501
    idx_alpha = 2 * K + 2
    for m, kin_name in enumerate(tqdm(kinases, desc="Kinase KOs")):
        theta_ko = theta_opt.copy()
        theta_ko[idx_alpha + m] = -20.0  # Effectively zero in log space

        ko_P, ko_S, ko_Kdyn = get_steady_state(theta_ko)

        # Calculate Fold Change: (KO / WT)
        # Add epsilon to denominator to avoid division by zero
        res_P[f"KO_Kin_{kin_name}"] = (ko_P + 1e-9) / (wt_P + 1e-9)
        res_S[f"KO_Kin_{kin_name}"] = (ko_S + 1e-9) / (wt_S + 1e-9)
        res_Kdyn[f"KO_Kin_{kin_name}"] = (ko_Kdyn + 1e-9) / (wt_Kdyn + 1e-9)

    # --- Helper to Save & Plot FC ---
    def process_and_save(res_dict, cols, name):
        df = pd.DataFrame(res_dict).T

        if df.empty:
            logger.warning(f"[!] No knockout results available for {name}.")
            return

        df.columns = cols

        # Keep rows where at least one value deviates meaningfully from 1.0.
        mask = ((df < 0.95) | (df > 1.05)).any(axis=1)
        df = df.loc[mask]

        if df.empty:
            logger.info(f"[*] No significant knockout effects for {name}; skipping plot.")
            return

        out_tsv = os.path.join(ko_dir, f"knockout_fc_{name}.tsv")
        df.to_csv(out_tsv, sep="\t")

        # Keep only finite numeric values for plotting.
        df_plot = df.replace([np.inf, -np.inf], np.nan)
        df_plot = df_plot.dropna(axis=0, how="all")
        df_plot = df_plot.dropna(axis=1, how="all")

        if df_plot.empty:
            logger.warning(f"[!] No finite knockout values for {name}; skipping plot.")
            return

        # Drop rows/columns with no variation. These break or trivialize clustering.
        if df_plot.shape[0] > 1:
            row_var = df_plot.var(axis=1, skipna=True)
            df_plot = df_plot.loc[row_var > 0]

        if df_plot.shape[1] > 1:
            col_var = df_plot.var(axis=0, skipna=True)
            df_plot = df_plot.loc[:, col_var > 0]

        if df_plot.empty:
            logger.info(
                f"[*] Knockout matrix for {name} has no variable rows/columns; "
                "saved TSV but skipped plot."
            )
            return

        vals = df_plot.to_numpy(dtype=float).ravel()
        vals = vals[np.isfinite(vals)]

        if vals.size == 0:
            logger.warning(f"[!] No finite values for {name}; skipping plot.")
            return

        vmax = float(np.percentile(vals, 98))
        if not np.isfinite(vmax) or vmax < 1.5:
            vmax = 1.5

        try:
            # clustermap requires at least 2 rows/cols for clustering.
            row_cluster = df_plot.shape[0] >= 2
            col_cluster = df_plot.shape[1] >= 2

            g = sns.clustermap(
                df_plot,
                cmap="vlag",
                center=1.0,
                vmin=0.0,
                vmax=vmax,
                figsize=(10, 10),
                row_cluster=row_cluster,
                col_cluster=col_cluster,
                cbar_kws={"label": "Fold Change (KO/WT)"},
            )
            g.fig.suptitle(f"Fold Change {name} upon Knockout")
            plt.savefig(os.path.join(ko_dir, f"clustermap_{name}_fc.png"), dpi=300)
            plt.close(g.fig)

        except (ValueError, np.linalg.LinAlgError, RuntimeError) as e:
            logger.warning(f"Clustermap {name} failed: {e}")

    process_and_save(res_S, proteins, "S_sim")
    process_and_save(res_Kdyn, kinases, "Kdyn_sim")

    # 4. Phosphosite Knockouts (Remove Input Edges)
    logger.info("   -> Simulating Phosphosite Knockouts (Alanine Scanning)...")
    for i, site_name in enumerate(tqdm(sites)):
        K_site_kin_ko = problem.K_site_kin.copy()
        K_site_kin_ko[i, :] = 0.0  # Remove all kinase inputs

        # Fix: unpacking tuple, using correct baseline (wt_P)
        ko_P_vals, _, _ = get_steady_state(theta_opt, K_mat_in=K_site_kin_ko)

        # Calculate FC
        fc_val = (ko_P_vals + 1e-9) / (wt_P + 1e-9)
        res_P[f"KO_Site_{site_name}"] = fc_val

    # 5. Compile Main P Results
    df_res = pd.DataFrame(res_P).T
    df_res.columns = sites

    # Filter: Drop perturbations close to 1.0 (no change)
    mask_rows = ((df_res < 0.9) | (df_res > 1.1)).any(axis=1)
    df_filtered = df_res.loc[mask_rows]

    if df_filtered.empty:
        logger.warning("[!] No perturbations caused significant changes > 10% FC.")
        return

    df_filtered.to_csv(os.path.join(ko_dir, "knockout_fc.tsv"), sep="\t")

    # 6. Plotting Main Clustermap
    logger.info("   -> Generating Clustermap...")

    df_plot = df_filtered.replace([np.inf, -np.inf], np.nan)
    df_plot = df_plot.dropna(axis=0, how="all")
    df_plot = df_plot.dropna(axis=1, how="all")

    if df_plot.empty:
        logger.warning("[!] No finite phosphosite knockout values to plot.")
        return

    if df_plot.shape[0] > 1:
        row_var = df_plot.var(axis=1, skipna=True)
        df_plot = df_plot.loc[row_var > 0]

    if df_plot.shape[1] > 1:
        col_var = df_plot.var(axis=0, skipna=True)
        df_plot = df_plot.loc[:, col_var > 0]

    if df_plot.empty:
        logger.warning(
            "[!] Phosphosite knockout matrix has no variable rows/columns; "
            "saved TSV but skipped clustermap."
        )
        return

    try:
        vals = df_plot.to_numpy(dtype=float).ravel()
        vals = vals[np.isfinite(vals)]

        if vals.size == 0:
            logger.warning("[!] No finite phosphosite knockout values to plot.")
            return

        vmax = float(np.percentile(vals, 98))
        if not np.isfinite(vmax) or vmax < 2.0:
            vmax = 2.0

        row_cluster = df_plot.shape[0] >= 2
        col_cluster = df_plot.shape[1] >= 2

        g = sns.clustermap(
            df_plot,
            cmap="vlag",
            center=1.0,
            vmin=0.0,
            vmax=vmax,
            figsize=(14, 14),
            xticklabels=False,
            yticklabels=True,
            row_cluster=row_cluster,
            col_cluster=col_cluster,
            dendrogram_ratio=(0.1, 0.2),
            cbar_pos=(0.02, 0.8, 0.03, 0.18),
            cbar_kws={"label": "Fold Change (KO/WT)"},
        )
        g.ax_heatmap.set_xlabel("Downstream Phosphosites")
        g.ax_heatmap.set_ylabel("Perturbation (KO)")
        plt.savefig(os.path.join(ko_dir, "knockout_clustermap_fc.png"), dpi=300)
        plt.close(g.fig)

    except (ValueError, np.linalg.LinAlgError, RuntimeError) as e:
        logger.critical(f"[!] Clustermap generation failed: {e}")
