#!/usr/bin/env python3
"""
Standalone symbolic equation generator for the PhosCrosstalk RHS.

This script reconstructs the algebraic relationships in mechanisms.py without
running JAX, Diffrax, or the ODE solver.

Markdown output uses only $$ ... $$ display math blocks.

Examples
--------
One protein, one kinase, one phosphosite:

    python scripts/rhs_algebra.py \\
      --proteins EGFR \\
      --kinases EGFR \\
      --sites EGFR_Y1068 \\
      --mechanism rand \\
      --out rhs_equations_EGFR_Y1068.md \\
      --graph rhs_dependency_edges_EGFR_Y1068.tsv

Equivalent explicit-dimension form:

    python scripts/rhs_algebra.py \\
      --K 1 --M 1 --N 1 \\
      --proteins EGFR \\
      --kinases EGFR \\
      --sites EGFR_Y1068 \\
      --site-prot-idx 0 \\
      --kin-to-prot-idx 0 \\
      --mechanism rand \\
      --out rhs_equations_EGFR_Y1068.md

Multiple proteins/sites:

    python scripts/rhs_algebra.py \\
      --proteins EGFR,MET,ERBB2 \\
      --kinases EGFR,MET,ERBB2 \\
      --sites EGFR_Y1068,EGFR_Y1173,MET_Y1234,ERBB2_Y1248 \\
      --kin-to-prot-idx 0,1,2 \\
      --mechanism rand \\
      --out rhs_equations_subnetwork.md
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable

import sympy as sp


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate symbolic algebraic relationships for the PhosCrosstalk RHS."
    )

    parser.add_argument("--K", type=int, default=None, help="Number of proteins.")
    parser.add_argument("--M", type=int, default=None, help="Number of kinases.")
    parser.add_argument("--N", type=int, default=None, help="Number of phosphosites.")

    parser.add_argument(
        "--proteins",
        type=str,
        default=None,
        help="Comma-separated protein/gene symbols. Example: EGFR,MET,ERBB2",
    )

    parser.add_argument(
        "--kinases",
        type=str,
        default=None,
        help="Comma-separated kinase symbols. Example: EGFR,MET,ERBB2",
    )

    parser.add_argument(
        "--sites",
        type=str,
        default=None,
        help="Comma-separated phosphosite labels. Example: EGFR_Y1068,MET_Y1234",
    )

    parser.add_argument(
        "--mechanism",
        choices=("dist", "seq", "rand"),
        required=True,
        help="Phosphorylation mechanism.",
    )

    parser.add_argument(
        "--site-prot-idx",
        type=str,
        default=None,
        help=(
            "Comma-separated site-to-protein mapping of length N. "
            "Example: 0,0,0,1,1,2. If omitted and --sites/--proteins are "
            "provided, mapping is inferred from site prefix before '_'."
        ),
    )

    parser.add_argument(
        "--kin-to-prot-idx",
        type=str,
        default=None,
        help=(
            "Comma-separated kinase-to-protein mapping of length M. "
            "Use -1 for unmapped kinases. Example: 0,1,-1. If omitted and "
            "--kinases/--proteins are provided, exact symbol matches are mapped."
        ),
    )

    parser.add_argument(
        "--out",
        type=Path,
        default=Path("rhs_equations.md"),
        help="Output Markdown report.",
    )

    parser.add_argument(
        "--graph",
        type=Path,
        default=None,
        help="Optional TSV file containing dependency edges.",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _parse_str_list(raw: str | None) -> list[str] | None:
    if raw is None:
        return None
    vals = [x.strip() for x in raw.split(",") if x.strip() != ""]
    if not vals:
        return None
    return vals


def _parse_int_list(
    raw: str | None,
    n: int,
    name: str,
    default: list[int],
) -> list[int]:
    if raw is None:
        if len(default) != n:
            raise ValueError(f"Default for {name} must have length {n}.")
        return default

    vals = [int(x.strip()) for x in raw.split(",") if x.strip() != ""]
    if len(vals) != n:
        raise ValueError(f"{name} must contain exactly {n} values. Got {len(vals)}.")

    return vals


def _infer_dim(explicit_dim: int | None, labels: list[str] | None, name: str) -> int:
    if explicit_dim is not None:
        if explicit_dim <= 0 and name != "M":
            raise ValueError(f"--{name} must be positive.")
        if explicit_dim < 0 and name == "M":
            raise ValueError("--M must be non-negative.")

        if labels is not None and len(labels) != explicit_dim:
            raise ValueError(
                f"--{name}={explicit_dim}, but {name.lower()} labels contain "
                f"{len(labels)} entries: {labels}"
            )
        return explicit_dim

    if labels is not None:
        return len(labels)

    raise ValueError(
        f"Could not infer {name}. Provide --{name} or provide the matching label list."
    )


def _default_labels(prefix: str, n: int) -> list[str]:
    return [f"{prefix}_{i}" for i in range(n)]


def _site_prefix(site: str) -> str:
    """
    Extract protein prefix from a phosphosite label.

    Examples:
        EGFR_Y1068       -> EGFR
        MAPK1_T185_Y187  -> MAPK1
        EGFR:Y1068       -> EGFR
        EGFR-Y1068       -> EGFR
    """
    for sep in ("_", ":", "-"):
        if sep in site:
            return site.split(sep, 1)[0]
    return site


def _infer_site_prot_idx(sites: list[str], proteins: list[str]) -> list[int]:
    prot_to_idx = {p: i for i, p in enumerate(proteins)}
    out: list[int] = []

    missing: list[str] = []
    for site in sites:
        prefix = _site_prefix(site)
        if prefix in prot_to_idx:
            out.append(prot_to_idx[prefix])
        else:
            missing.append(site)

    if missing:
        raise ValueError(
            "Could not infer site_prot_idx for these sites because their prefixes "
            f"do not match any protein label: {missing}. "
            "Either fix --sites/--proteins or pass --site-prot-idx explicitly."
        )

    return out


def _infer_kin_to_prot_idx(kinases: list[str], proteins: list[str]) -> list[int]:
    prot_to_idx = {p: i for i, p in enumerate(proteins)}
    return [prot_to_idx.get(k, -1) for k in kinases]


def _safe_tex_label(label: str) -> str:
    """
    Convert arbitrary biological label to a LaTeX-safe text token.

    We keep labels readable using \\mathrm{...}; underscores and symbols are escaped.
    """
    escaped = (
        label.replace("\\", r"\textbackslash{}")
        .replace("_", r"\_")
        .replace("%", r"\%")
        .replace("&", r"\&")
        .replace("#", r"\#")
        .replace("{", r"\{")
        .replace("}", r"\}")
    )
    return rf"\mathrm{{{escaped}}}"


def _safe_symbol_suffix(label: str) -> str:
    """
    Convert label to a safe internal SymPy symbol suffix.
    """
    suffix = re.sub(r"[^A-Za-z0-9]+", "_", label.strip())
    suffix = suffix.strip("_")
    if not suffix:
        suffix = "x"
    if suffix[0].isdigit():
        suffix = f"x_{suffix}"
    return suffix


# ---------------------------------------------------------------------------
# Symbol helpers
# ---------------------------------------------------------------------------

def sym_vec(prefix: str, labels: list[str]) -> list[sp.Symbol]:
    return [sp.Symbol(f"{prefix}_{_safe_symbol_suffix(label)}") for label in labels]


def sym_mat(prefix: str, row_labels: list[str], col_labels: list[str]) -> list[list[sp.Symbol]]:
    return [
        [
            sp.Symbol(
                f"{prefix}_{_safe_symbol_suffix(r)}_{_safe_symbol_suffix(c)}"
            )
            for c in col_labels
        ]
        for r in row_labels
    ]


def add_sum(items: Iterable[sp.Expr]) -> sp.Expr:
    items = list(items)
    if not items:
        return sp.Integer(0)
    return sp.Add(*items)


def smooth_pos(x: sp.Expr, eps: sp.Symbol) -> sp.Expr:
    return sp.Rational(1, 2) * (x + sp.sqrt(x**2 + eps**2)) - sp.Rational(1, 2) * eps


def sigmoid(x: sp.Expr) -> sp.Expr:
    return 1 / (1 + sp.exp(-x))


def row_l1_norm(row: list[sp.Expr], eps: sp.Symbol) -> sp.Expr:
    return sp.sqrt(add_sum(sp.Abs(v) for v in row) ** 2 + eps**2)


def prev_site_idx_from_mapping(site_prot_idx: list[int]) -> list[int]:
    prev = [-1] * len(site_prot_idx)
    last_seen: dict[int, int] = {}

    for i, p_idx in enumerate(site_prot_idx):
        if p_idx in last_seen:
            prev[i] = last_seen[p_idx]
        last_seen[p_idx] = i

    return prev


def md_math(lhs: str, rhs: sp.Expr) -> str:
    return f"$$\n{lhs} = {sp.latex(rhs)}\n$$\n"


def md_raw_math(expr: str) -> str:
    return f"$$\n{expr}\n$$\n"


# ---------------------------------------------------------------------------
# Algebra builder
# ---------------------------------------------------------------------------

def build_rhs_algebra(
    *,
    proteins: list[str],
    kinases: list[str],
    sites: list[str],
    mechanism: str,
    site_prot_idx: list[int],
    kin_to_prot_idx: list[int],
) -> tuple[str, list[tuple[str, str, str]]]:
    """
    Build compact algebraic RHS report.

    The equations are unchanged relative to mechanisms.py. The output is made
    printable by introducing intermediate symbols and expanding them in later
    sections.
    """
    K = len(proteins)
    M = len(kinases)
    N = len(sites)

    prot_tex = [_safe_tex_label(x) for x in proteins]
    kin_tex = [_safe_tex_label(x) for x in kinases]
    site_tex = [_safe_tex_label(x) for x in sites]

    sites_by_prot: list[list[int]] = [[] for _ in range(K)]
    for site_i, prot_i in enumerate(site_prot_idx):
        if 0 <= prot_i < K:
            sites_by_prot[prot_i].append(site_i)

    prev_site_idx = prev_site_idx_from_mapping(site_prot_idx)

    edges: list[tuple[str, str, str]] = []

    def edge(src: str, dst: str, rel: str = "depends_on") -> None:
        edges.append((src, dst, rel))

    def block(expr: str) -> str:
        return f"$$\n{expr}\n$$\n"

    def section(title: str) -> None:
        lines.append(f"\n## {title}\n")

    def subsection(title: str) -> None:
        lines.append(f"\n### {title}\n")

    def subsubsection(title: str) -> None:
        lines.append(f"\n#### {title}\n")

    def tex_sum(terms: list[str]) -> str:
        if not terms:
            return "0"
        return " + ".join(terms)

    def site_set_tex(k: int) -> str:
        idxs = sites_by_prot[k]
        if not idxs:
            return r"\varnothing"
        return r"\left\{" + ", ".join(site_tex[i] for i in idxs) + r"\right\}"

    def site_count(k: int) -> int:
        return len(sites_by_prot[k])

    def q_sym(i: int) -> str:
        return rf"q_{{{site_tex[i]}}}"

    def p_sym(i: int) -> str:
        return rf"p_{{{site_tex[i]}}}"

    def ppos_sym(i: int) -> str:
        return rf"p_{{{site_tex[i]},+}}"

    def coup_sym(i: int) -> str:
        return rf"C_{{{site_tex[i]}}}"

    def phi_sym(i: int) -> str:
        return rf"\Phi_{{{site_tex[i]}}}"

    def qbar_sym(k: int) -> str:
        return rf"\bar q_{{{prot_tex[k]}}}"

    def cbar_sym(k: int) -> str:
        return rf"\bar C_{{{prot_tex[k]}}}"

    def rna_field_sym(k: int) -> str:
        return rf"F^R_{{{prot_tex[k]}}}"

    def s_field_sym(k: int) -> str:
        return rf"F^S_{{{prot_tex[k]}}}"

    def s_drive_sym(k: int) -> str:
        return rf"Z^S_{{{prot_tex[k]}}}"

    def a_mod_sym(k: int) -> str:
        return rf"H^A_{{{prot_tex[k]}}}"

    def s_eff_sym(k: int) -> str:
        return rf"S^{{\mathrm{{eff}}}}_{{{prot_tex[k]}}}"

    def rna_relax_sym(k: int) -> str:
        return rf"\rho^R_{{{prot_tex[k]}}}"

    def rna_reg_sym(k: int) -> str:
        return rf"\eta^R_{{{prot_tex[k]}}}"

    def k_basal_sym(m: int) -> str:
        return rf"B^K_{{{kin_tex[m]}}}"

    def k_u_sym(m: int) -> str:
        return rf"U^K_{{{kin_tex[m]}}}"

    def k_drive_sym(m: int) -> str:
        return rf"D^K_{{{kin_tex[m]}}}"

    def u_sub_sym(m: int) -> str:
        return rf"U^{{\mathrm{{sub}}}}_{{{kin_tex[m]}}}"

    def u_net_sym(m: int) -> str:
        return rf"U^{{\mathrm{{net}}}}_{{{kin_tex[m]}}}"

    def prot_contrib_sym(m: int) -> str:
        return rf"U^{{\mathrm{{prot}}}}_{{{kin_tex[m]}}}"

    def kon_sym(i: int) -> str:
        return rf"E^p_{{{site_tex[i]}}}"

    def gate_sym(i: int) -> str:
        return rf"G^p_{{{site_tex[i]}}}"

    def asite_sym(i: int) -> str:
        return rf"A^p_{{{site_tex[i]}}}"

    def vraw_sym(i: int) -> str:
        return rf"V^p_{{{site_tex[i]}}}"

    lines: list[str] = []

    lines.append("# PhosCrosstalk RHS Algebraic Relationships\n")
    lines.append(
        "This report uses compact substitution symbols so the RHS equations fit on an A4 page. "
        "The substitutions are expanded after the compact RHS blocks.\n"
    )

    section("Model dimensions and labels")
    lines.append(f"- `K = {K}` proteins")
    lines.append(f"- `M = {M}` kinases")
    lines.append(f"- `N = {N}` phosphosites")
    lines.append(f"- `mechanism = {mechanism}`")
    lines.append(f"- `proteins = {proteins}`")
    lines.append(f"- `kinases = {kinases}`")
    lines.append(f"- `sites = {sites}`")
    lines.append(f"- `site_prot_idx = {site_prot_idx}`")
    lines.append(f"- `kin_to_prot_idx = {kin_to_prot_idx}`\n")

    section("1. Decoded fitted parameters")
    lines.append(block(r"k_{\mathrm{deact},g} = \exp(\mathrm{clip}(\theta_{k_{\mathrm{deact},g}}))"))
    lines.append(block(r"d_{\mathrm{deg},g} = \exp(\mathrm{clip}(\theta_{d_{\mathrm{deg},g}}))"))
    lines.append(block(r"\beta_g = \exp(\mathrm{clip}(\theta_{\beta_g}))"))
    lines.append(block(r"\beta_l = \exp(\mathrm{clip}(\theta_{\beta_l}))"))
    lines.append(block(r"\alpha_m = \exp(\mathrm{clip}(\theta_{\alpha_m}))"))
    lines.append(block(r"kK_{\mathrm{act},m} = \exp(\mathrm{clip}(\theta_{kK_{\mathrm{act},m}}))"))
    lines.append(block(r"kK_{\mathrm{deact},m} = \exp(\mathrm{clip}(\theta_{kK_{\mathrm{deact},m}}))"))
    lines.append(block(r"k_{\mathrm{off},i} = \exp(\mathrm{clip}(\theta_{k_{\mathrm{off},i}}))"))
    lines.append(block(r"\gamma = 2\tanh(\theta_{\gamma})"))

    section("2. Universal helper definitions")
    lines.append(block(r"[x]_+ = \frac{1}{2}\left(x+\sqrt{x^2+\epsilon^2}\right)-\frac{1}{2}\epsilon"))
    lines.append(block(r"\sigma(x) = \frac{1}{1+\exp(-x)}"))
    lines.append(block(r"u(t) = \sigma(t/0.1)"))
    lines.append(block(r"p_{i,+} = [p_i]_+"))
    lines.append(block(r"q_i = \frac{p_{i,+}}{1+p_{i,+}}"))

    section("3. Compact RHS equations")

    subsection("3.1 Latent mRNA / transcriptional state")
    for k, prot in enumerate(proteins):
        pt = prot_tex[k]
        subsubsection(prot)
        lines.append(block(
            rf"\frac{{dR_{{{pt}}}}}{{dt}}"
            rf" = {rna_relax_sym(k)}\left({rna_reg_sym(k)} - R_{{{pt}}}\right)"
        ))

    subsection("3.2 Protein signalling state")
    for k, prot in enumerate(proteins):
        pt = prot_tex[k]
        subsubsection(prot)
        lines.append(block(
            rf"\frac{{dS_{{{pt}}}}}{{dt}}"
            rf" = k_{{\mathrm{{act}},{pt}}}(t)\,{s_drive_sym(k)}"
            rf"\left(1-S_{{{pt}}}\right)"
            rf" - k_{{\mathrm{{deact}},{pt}}}S_{{{pt}}}"
        ))

    subsection("3.3 Protein abundance state")
    for k, prot in enumerate(proteins):
        pt = prot_tex[k]
        subsubsection(prot)
        lines.append(block(
            rf"\frac{{dA_{{{pt}}}}}{{dt}}"
            rf" = {s_eff_sym(k)} - d_{{\mathrm{{deg}},{pt}}}A_{{{pt}}}"
        ))

    subsection("3.4 Kinase activity state")
    for m, kin in enumerate(kinases):
        kt = kin_tex[m]
        subsubsection(kin)
        lines.append(block(
            rf"\frac{{dK_{{{kt}}}}}{{dt}}"
            rf" = kK_{{\mathrm{{act}},{kt}}}\,{k_drive_sym(m)}"
            rf"\left(1-K_{{{kt}}}\right)"
            rf" - kK_{{\mathrm{{deact}},{kt}}}K_{{{kt}}}"
        ))

    subsection("3.5 Phosphosite state")
    for i, site in enumerate(sites):
        st = site_tex[i]
        subsubsection(site)
        lines.append(block(
            rf"\frac{{dp_{{{st}}}}}{{dt}}"
            rf" = \frac{{{vraw_sym(i)}}}{{1+{vraw_sym(i)}}}"
            rf" - k_{{\mathrm{{off}},{st}}}\,{ppos_sym(i)}"
        ))

    section("4. Phosphosite positivity and occupancy substitutions")
    for i, site in enumerate(sites):
        st = site_tex[i]
        subsubsection(site)
        lines.append(block(rf"{ppos_sym(i)} = [p_{{{st}}}]_+"))
        lines.append(block(rf"{q_sym(i)} = \frac{{{ppos_sym(i)}}}{{1+{ppos_sym(i)}}}"))

    section("5. Crosstalk substitutions")
    lines.append("For each phosphosite, define row-normalised global and local crosstalk fields.\n")

    for i, site in enumerate(sites):
        st = site_tex[i]
        subsubsection(site)

        dcg = rf"D^g_{{{st}}}"
        dcl = rf"D^l_{{{st}}}"
        xg = rf"X^g_{{{st}}}"
        xl = rf"X^l_{{{st}}}"

        dcg_terms = [rf"\left|C^g_{{{st},{site_tex[j]}}}\right|" for j in range(N)]
        dcl_terms = [rf"\left|C^l_{{{st},{site_tex[j]}}}\right|" for j in range(N)]

        xg_terms = [rf"C^g_{{{st},{site_tex[j]}}}{q_sym(j)}" for j in range(N)]
        xl_terms = [rf"C^l_{{{st},{site_tex[j]}}}{q_sym(j)}" for j in range(N)]

        lines.append(block(rf"{dcg} = \sqrt{{\left({tex_sum(dcg_terms)}\right)^2+\epsilon^2}}"))
        lines.append(block(rf"{dcl} = \sqrt{{\left({tex_sum(dcl_terms)}\right)^2+\epsilon^2}}"))
        lines.append(block(rf"{xg} = \frac{{{tex_sum(xg_terms)}}}{{{dcg}}}"))
        lines.append(block(rf"{xl} = \frac{{{tex_sum(xl_terms)}}}{{{dcl}}}"))
        lines.append(block(rf"{coup_sym(i)} = \tanh\left(\beta_g{xg}+\beta_l{xl}\right)"))
        lines.append(block(rf"{phi_sym(i)} = \exp\left({coup_sym(i)}\right)"))

    section("6. Per-protein phosphosite summary substitutions")
    for k, prot in enumerate(proteins):
        pt = prot_tex[k]
        idxs = sites_by_prot[k]
        subsubsection(prot)

        if not idxs:
            lines.append(block(rf"\mathcal{{I}}_{{{pt}}} = \varnothing"))
            lines.append(block(rf"{qbar_sym(k)} = 0"))
            lines.append(block(rf"{cbar_sym(k)} = 0"))
            continue

        q_terms = [q_sym(i) for i in idxs]
        c_terms = [coup_sym(i) for i in idxs]
        n_k = site_count(k)

        lines.append(block(rf"\mathcal{{I}}_{{{pt}}} = {site_set_tex(k)}"))
        lines.append(block(
            rf"{qbar_sym(k)}"
            rf" = \frac{{{tex_sum(q_terms)}}}{{\sqrt{{{n_k}^2+\epsilon^2}}}}"
        ))
        lines.append(block(
            rf"{cbar_sym(k)}"
            rf" = \frac{{{tex_sum(c_terms)}}}{{\sqrt{{{n_k}^2+\epsilon^2}}}}"
        ))

    section("7. RNA regulatory substitutions")
    lines.append(block(
        r"\lambda_R"
        r" = \max\left("
        r"\frac{1}{2}\log\frac{k^{+}_{\max}+\epsilon}{k^{+}_{\min}+\epsilon},"
        r"\frac{\sqrt{(\gamma_{S,p}^2+\gamma_{A,S}^2+\gamma_{K,\mathrm{net}}^2)/3+\epsilon}}"
        r"{1+\sqrt{(\gamma_{S,p}^2+\gamma_{A,S}^2+\gamma_{K,\mathrm{net}}^2)/3+\epsilon}}"
        r"\right)"
    ))

    for k, prot in enumerate(proteins):
        pt = prot_tex[k]
        subsubsection(prot)
        lines.append(block(
            rf"{rna_relax_sym(k)}"
            rf" = \sqrt{{[k_{{\mathrm{{deact}},{pt}}}]_+"
            rf"[d_{{\mathrm{{deg}},{pt}}}]_+ + \epsilon}}"
        ))
        lines.append(block(
            rf"{rna_field_sym(k)}"
            rf" = \gamma_{{S,p}}{qbar_sym(k)}"
            rf" + {cbar_sym(k)}"
            rf" + r^P_{{{pt}}}u(t)"
        ))
        lines.append(block(
            rf"{rna_reg_sym(k)}"
            rf" = \exp\left(\lambda_R\tanh\left({rna_field_sym(k)}\right)\right)"
        ))

    section("8. Protein signalling substitutions")
    for k, prot in enumerate(proteins):
        pt = prot_tex[k]
        subsubsection(prot)
        lines.append(block(
            rf"{s_field_sym(k)}"
            rf" = \gamma_{{S,p}}{qbar_sym(k)}"
            rf" + {cbar_sym(k)}"
            rf" + r^P_{{{pt}}}u(t)"
        ))
        lines.append(block(
            rf"{s_drive_sym(k)}"
            rf" = \sigma\left({s_field_sym(k)}\right)"
        ))

    section("9. Protein abundance substitutions")
    for k, prot in enumerate(proteins):
        pt = prot_tex[k]
        subsubsection(prot)
        lines.append(block(
            rf"{a_mod_sym(k)}"
            rf" = 1 + \frac{{1}}{{2}}\tanh\left("
            rf"\gamma_{{A,S}}\left(S_{{{pt}}}-\frac{{1}}{{2}}\right)"
            rf"\right)"
        ))
        lines.append(block(
            rf"{s_eff_sym(k)}"
            rf" = \left[s_{{\mathrm{{prod}},{pt}}}(t)\,{a_mod_sym(k)}\right]_+"
        ))

    section("10. Kinase basal and kinase-drive substitutions")

    lines.append(block(
        r"e_m = \frac{kK_{\mathrm{act},m}}"
        r"{kK_{\mathrm{act},m}+kK_{\mathrm{deact},m}+\epsilon}"
    ))
    lines.append(block(
        r"B_{\min} = \max\left(10^{-6}, \frac{1}{4}Q_{0.10}(e_m)\right)"
    ))
    lines.append(block(
        r"B_{\max} = \min\left(0.95,\max(B_{\min}+\epsilon,\frac{3}{4}Q_{0.90}(e_m))\right)"
    ))
    lines.append(block(
        r"B_{\mathrm{bonus}} = \min\left((B_{\max}-B_{\min})"
        r"\frac{\overline{s}_{\mathrm{receptor}}}{\overline{s}+\epsilon}, B_{\max}\right)"
    ))

    for m, kin in enumerate(kinases):
        kt = kin_tex[m]
        subsubsection(kin)

        site_support_terms = [
            rf"\left|K^p_{{{site_tex[i]},{kt}}}\right|"
            for i in range(N)
        ]
        feedback_terms = [
            rf"\left|R^K_{{{kt},{site_tex[i]}}}\right|"
            for i in range(N)
        ]

        support_m = rf"s_{{{kt}}}"
        support_norm_m = rf"\tilde s_{{{kt}}}"

        lines.append(block(
            rf"{support_m}"
            rf" = {tex_sum(site_support_terms)}"
            rf" + {tex_sum(feedback_terms)}"
            rf" + B_{{\mathrm{{bonus}}}}r^K_{{{kt}}}"
        ))
        lines.append(block(
            rf"{support_norm_m}"
            rf" = \frac{{{support_m}}}{{s_{{\max}}+\epsilon}}"
        ))
        lines.append(block(
            rf"{k_basal_sym(m)}"
            rf" = B_{{\min}} + \left(B_{{\max}}-B_{{\min}}\right){support_norm_m}"
        ))

    section("11. Kinase dynamic substitutions")
    for m, kin in enumerate(kinases):
        kt = kin_tex[m]
        subsubsection(kin)

        r_den = rf"D^R_{{{kt}}}"
        l_den = rf"D^L_{{{kt}}}"

        r_den_terms = [
            rf"\left|R^K_{{{kt},{site_tex[i]}}}\right|"
            for i in range(N)
        ]
        l_den_terms = [
            rf"\left|L^\alpha_{{{kt},{kin_tex[j]}}}\right|"
            for j in range(M)
        ]

        usub_terms = [
            rf"R^K_{{{kt},{site_tex[i]}}}{q_sym(i)}"
            for i in range(N)
        ]
        unet_terms = [
            rf"L^\alpha_{{{kt},{kin_tex[j]}}}K_{{{kin_tex[j]}}}"
            for j in range(M)
        ]

        p_idx = kin_to_prot_idx[m]
        if 0 <= p_idx < K:
            prot_contrib = rf"\gamma_{{A,S}}S_{{{prot_tex[p_idx]}}}"
        else:
            prot_contrib = "0"

        lines.append(block(rf"{r_den} = \sqrt{{\left({tex_sum(r_den_terms)}\right)^2+\epsilon^2}}"))
        lines.append(block(rf"{l_den} = \sqrt{{\left({tex_sum(l_den_terms)}\right)^2+\epsilon^2}}"))
        lines.append(block(rf"{u_sub_sym(m)} = \frac{{{tex_sum(usub_terms)}}}{{{r_den}}}"))
        lines.append(block(rf"{u_net_sym(m)} = -\frac{{{tex_sum(unet_terms)}}}{{{l_den}}}"))
        lines.append(block(rf"{prot_contrib_sym(m)} = {prot_contrib}"))
        lines.append(block(
            rf"{k_u_sym(m)}"
            rf" = {u_sub_sym(m)}"
            rf" + \gamma_{{K,\mathrm{{net}}}}{u_net_sym(m)}"
            rf" + {prot_contrib_sym(m)}"
            rf" + r^K_{{{kt}}}u(t)"
        ))
        lines.append(block(
            rf"{k_drive_sym(m)}"
            rf" = {k_basal_sym(m)}"
            rf" + \left(1-{k_basal_sym(m)}\right)"
            rf"\sigma\left({k_u_sym(m)}\right)"
        ))

    section("12. Phosphosite flux substitutions")
    for i, site in enumerate(sites):
        st = site_tex[i]
        prot_i = site_prot_idx[i]
        pt = prot_tex[prot_i]

        subsubsection(site)

        ksk_den = rf"D^p_{{{st}}}"
        ksk_den_terms = [
            rf"\left|K^p_{{{st},{kin_tex[m]}}}\right|"
            for m in range(M)
        ]
        kon_terms = [
            rf"K^p_{{{st},{kin_tex[m]}}}\alpha_{{{kin_tex[m]}}}K_{{{kin_tex[m]}}}"
            for m in range(M)
        ]

        lines.append(block(rf"{ksk_den} = \sqrt{{\left({tex_sum(ksk_den_terms)}\right)^2+\epsilon^2}}"))
        lines.append(block(rf"{kon_sym(i)} = \left[\frac{{{tex_sum(kon_terms)}}}{{{ksk_den}}}\right]_+"))

        if mechanism == "dist":
            gate_expr = "1"

        elif mechanism == "seq":
            site_available = rf"\left(1-{q_sym(i)}\right)"
            prev_i = prev_site_idx[i]
            if prev_i >= 0:
                gate_expr = (
                    rf"\frac{{{q_sym(prev_i)}}}"
                    rf"{{h_{{\mathrm{{seq}}}}+{q_sym(prev_i)}+\epsilon}}"
                    rf"{site_available}"
                )
            else:
                gate_expr = site_available

        else:
            gate_expr = rf"\frac{{1}}{{1+{qbar_sym(prot_i)}}}"

        lines.append(block(rf"{gate_sym(i)} = {gate_expr}"))
        lines.append(block(rf"{asite_sym(i)} = \left[\frac{{A_{{{pt}}}}}{{A_{{\max}}+\epsilon}}\right]_+"))
        lines.append(block(
            rf"{vraw_sym(i)}"
            rf" = \left["
            rf"{kon_sym(i)}"
            rf"{phi_sym(i)}"
            rf"{gate_sym(i)}"
            rf"\left(1+\frac{{1}}{{2}}{asite_sym(i)}\right)"
            rf"\right]_+"
        ))

    section("13. Main biological dependency structure")
    lines.append("- `p -> q -> crosstalk -> R_rna, S, p`")
    lines.append("- `p -> qbar -> R_rna, S, p`")
    lines.append("- `S -> A`")
    lines.append("- `S -> Kdyn` through kinase-to-protein mapping")
    lines.append("- `Kdyn -> p` through kinase-site activation")
    lines.append("- `A -> p` through abundance-scaled phosphorylation flux")
    lines.append("- `k_act(t) -> S`")
    lines.append("- `s_prod(t) -> A`")
    lines.append("- `kK_act, kK_deact, K_site_kin, R -> kinase_basal -> Kdyn`")

    # ------------------------------------------------------------------
    # Dependency edges.
    # ------------------------------------------------------------------
    for i, site in enumerate(sites):
        edge(site, f"q:{site}")
        edge(f"q:{site}", f"crosstalk:{site}")
        edge(f"q:{site}", f"dp:{site}")
        edge(f"crosstalk:{site}", f"dp:{site}")

    for k, prot in enumerate(proteins):
        for i in sites_by_prot[k]:
            edge(f"q:{sites[i]}", f"qbar:{prot}")
            edge(f"crosstalk:{sites[i]}", f"cbar:{prot}")

        edge(f"qbar:{prot}", f"dR:{prot}")
        edge(f"cbar:{prot}", f"dR:{prot}")
        edge(f"qbar:{prot}", f"dS:{prot}")
        edge(f"cbar:{prot}", f"dS:{prot}")
        edge(f"S:{prot}", f"dA:{prot}")
        edge(f"A:{prot}", f"dp_sites_for:{prot}")

    for m, kin in enumerate(kinases):
        edge(f"Kdyn:{kin}", f"dKdyn:{kin}")
        edge(f"Kdyn:{kin}", "phosphosite_flux")

        p_idx = kin_to_prot_idx[m]
        if 0 <= p_idx < K:
            edge(f"S:{proteins[p_idx]}", f"dKdyn:{kin}")

    return "\n".join(lines) + "\n", edges

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    proteins_raw = _parse_str_list(args.proteins)
    kinases_raw = _parse_str_list(args.kinases)
    sites_raw = _parse_str_list(args.sites)

    K = _infer_dim(args.K, proteins_raw, "K")
    M = _infer_dim(args.M, kinases_raw, "M")
    N = _infer_dim(args.N, sites_raw, "N")

    if K <= 0:
        raise ValueError("--K must be positive.")
    if M < 0:
        raise ValueError("--M must be non-negative.")
    if N <= 0:
        raise ValueError("--N must be positive.")

    proteins = proteins_raw if proteins_raw is not None else _default_labels("Protein", K)
    kinases = kinases_raw if kinases_raw is not None else _default_labels("Kinase", M)
    sites = sites_raw if sites_raw is not None else _default_labels("Site", N)

    if len(set(proteins)) != len(proteins):
        raise ValueError(f"Protein labels must be unique. Got: {proteins}")
    if len(set(kinases)) != len(kinases):
        raise ValueError(f"Kinase labels must be unique. Got: {kinases}")
    if len(set(sites)) != len(sites):
        raise ValueError(f"Site labels must be unique. Got: {sites}")

    if args.site_prot_idx is not None:
        site_prot_idx = _parse_int_list(
            raw=args.site_prot_idx,
            n=N,
            name="site_prot_idx",
            default=[],
        )
    elif sites_raw is not None and proteins_raw is not None:
        site_prot_idx = _infer_site_prot_idx(sites, proteins)
    else:
        site_prot_idx = [i % K for i in range(N)]

    if args.kin_to_prot_idx is not None:
        kin_to_prot_idx = _parse_int_list(
            raw=args.kin_to_prot_idx,
            n=M,
            name="kin_to_prot_idx",
            default=[],
        )
    elif kinases_raw is not None and proteins_raw is not None:
        kin_to_prot_idx = _infer_kin_to_prot_idx(kinases, proteins)
    else:
        kin_to_prot_idx = [i % K for i in range(M)]

    for val in site_prot_idx:
        if val < 0 or val >= K:
            raise ValueError(
                f"site_prot_idx values must be in [0, K-1]. Got {val}."
            )

    for val in kin_to_prot_idx:
        if val != -1 and (val < 0 or val >= K):
            raise ValueError(
                f"kin_to_prot_idx values must be -1 or in [0, K-1]. Got {val}."
            )

    report, edges = build_rhs_algebra(
        proteins=proteins,
        kinases=kinases,
        sites=sites,
        mechanism=args.mechanism,
        site_prot_idx=site_prot_idx,
        kin_to_prot_idx=kin_to_prot_idx,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report, encoding="utf-8")
    print(f"[rhs_algebra] wrote equation report: {args.out}")

    if args.graph is not None:
        args.graph.parent.mkdir(parents=True, exist_ok=True)
        with args.graph.open("w", encoding="utf-8") as fh:
            fh.write("source\ttarget\trelationship\n")
            for src, dst, rel in edges:
                fh.write(f"{src}\t{dst}\t{rel}\n")

        print(f"[rhs_algebra] wrote dependency edges: {args.graph}")


if __name__ == "__main__":
    main()