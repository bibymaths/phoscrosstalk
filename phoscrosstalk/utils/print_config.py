import os
import numpy as np
import pandas as pd
from phoscrosstalk.logger import get_logger

logger = get_logger()

def _save_model_entities_table(
    outdir, proteins, sites, site_prot_idx, entity_masks, gene_ids, A_proteins
):
    """
    Save a model_entities.tsv file recording each protein's prior support and
    inclusion reason.

    Args:
        outdir (str): Output directory.
        proteins (list[str]): Model protein names (length K).
        sites (list[str]): Phosphosite labels (length N).
        site_prot_idx (np.ndarray): Maps each site to its protein index.
        entity_masks (dict): Output of ``build_protein_entity_masks()``.
        gene_ids (list[str] or None): Gene IDs in the mRNA dataset.
        A_proteins (np.ndarray or None): Protein names with abundance data.
    """
    K = len(proteins)
    # Count phosphosites per protein
    n_psites = np.zeros(K, dtype=int)
    for p_idx in site_prot_idx:
        n_psites[p_idx] += 1

    # Which proteins have abundance observations
    prot_has_abundance = np.zeros(K, dtype=bool)
    if A_proteins is not None:
        prot_idx_map = {p: i for i, p in enumerate(proteins)}
        for pname in A_proteins:
            if pname in prot_idx_map:
                prot_has_abundance[prot_idx_map[pname]] = True

    rows = []
    for p_idx, p_name in enumerate(proteins):
        rows.append(
            {
                "protein": p_name,
                "has_rna_observation": bool(entity_masks["protein_has_rna"][p_idx]),
                "has_protein_observation": bool(prot_has_abundance[p_idx]),
                "n_phosphosites": int(n_psites[p_idx]),
                "is_tf_source": bool(entity_masks["protein_is_tf_source"][p_idx]),
                "is_tf_target": bool(entity_masks["protein_is_tf_target"][p_idx]),
                "has_tf_input": bool(entity_masks["protein_has_tf_input"][p_idx]),
                "has_kinase_prior": bool(
                    entity_masks["protein_has_kinase_prior"][p_idx]
                ),
                "included_by_extended_mode": bool(
                    entity_masks["included_by_extended_mode"][p_idx]
                ),
            }
        )

    df = pd.DataFrame(rows)
    path = os.path.join(outdir, "model_entities.tsv")
    df.to_csv(path, sep="\t", index=False)

    # Also save masks to preopt_snapshot/
    snap_dir = os.path.join(outdir, "preopt_snapshot")
    os.makedirs(snap_dir, exist_ok=True)
    df.to_csv(os.path.join(snap_dir, "model_entities.tsv"), sep="\t", index=False)

    # Save site-level kinase prior mask
    site_mask_rows = [
        {
            "site": s,
            "has_kinase_prior": bool(entity_masks["site_has_kinase_prior"][i]),
        }
        for i, s in enumerate(sites)
    ]
    pd.DataFrame(site_mask_rows).to_csv(
        os.path.join(snap_dir, "site_kinase_prior_mask.tsv"), sep="\t", index=False
    )


def _print_config_summary(cfg, config_path: str) -> None:
    """Print a compact, human-readable summary of validated configuration."""
    p = cfg.paths
    m = cfg.model
    o = cfg.optimisation
    s = cfg.solver
    lw = cfg.loss_weights
    an = getattr(cfg, "analysis", None)

    def _opt(v):
        v = (v or "").strip()
        return v if v else "(not set)"

    lines = [
        "",
        f"  config: {config_path}",
        "",
        "  [paths]",
        f"    data            = {p.data}",
        f"    ptm_intra       = {p.ptm_intra}",
        f"    ptm_inter       = {p.ptm_inter}",
        f"    output_dir      = {p.output_dir}",
        f"    rna_data        = {_opt(p.rna_data)}",
        f"    tf_net          = {_opt(p.tf_net)}",
        f"    kinase_tsv      = {_opt(p.kinase_tsv)}",
        f"    kea_ks_table    = {_opt(p.kea_ks_table)}",
        f"    unified_graph   = {_opt(p.unified_graph_pkl)}",
        "",
        "  [model]",
        f"    mechanism               = {m.mechanism}",
        f"    scale_mode              = {m.scale_mode}",
        f"    length_scale            = {m.length_scale}",
        f"    weight_scheme           = {m.weight_scheme}",
        f"    include_tfs_as_proteins = {m.include_tfs_as_proteins}",
        f"    receptors               = {list(m.receptors)}",
        f"    receptor_kinases        = {list(m.receptor_kinases)}",
        "",
        "  [optimisation]",
        f"    n_starts  = {o.n_starts}",
        f"    max_steps = {o.max_steps}",
        f"    ls_solver = {getattr(o, 'ls_solver', 'lm')}",
        f"    optx_adjoint = {getattr(o, 'optx_adjoint', 'implicit')}",
        f"    jac_mode = {getattr(o, 'jac_mode', 'fwd')}",
        f"    lambda_net = {o.lambda_net}",
        f"    reg_lambda = {o.reg_lambda}",
        "",
        "  [loss_weights]",
        f"    phospho={lw.phospho}  abundance={lw.abundance}  "
        f"    mrna={lw.mrna}  reg={lw.reg}",
        "",
        "  [solver]",
        f"    ode_solver={getattr(s, 'ode_solver', 'tsit5')}  "
        f"    ode_adjoint={getattr(s, 'ode_adjoint', 'forward')}",
        f"    rtol={s.rtol}  atol={s.atol}  max_steps={s.max_steps}  "
        f"    dt0={getattr(s, 'dt0', 0.01)}  "
        f"    root_find_max_steps={getattr(s, 'root_find_max_steps', 10)}",
        "",
    ]
    if an is not None:
        lines += [
            "  [analysis]",
            f"    tune={an.tune}  steadystate={an.run_steadystate}  "
            f"knockouts={an.run_knockouts}  sensitivity={an.run_sensitivity}",
            "",
        ]
    logger.header("Configuration Summary")
    logger.info("\n".join(lines))

