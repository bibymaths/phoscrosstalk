#!/usr/bin/env python3
"""
rhs_algebra.py

Standalone symbolic equation generator for the PhosCrosstalk RHS.

This script reconstructs the algebraic relationships in mechanisms.py without
running JAX, Diffrax, or the ODE solver.

Output:
    - Markdown equation report using $$ ... $$ display math only.
    - Optional TSV dependency edge file.

Example:
    python scripts/rhs_algebra.py \
      --K 1 \
      --M 1 \
      --N 1 \
      --mechanism rand \
      --site-prot-idx 0 \
      --kin-to-prot-idx 0 \
      --out rhs_equations_K1_M1_N1.md \
      --graph rhs_dependency_edges_K1_M1_N1.tsv
"""

from __future__ import annotations

import argparse
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

    parser.add_argument("--K", type=int, required=True, help="Number of proteins.")
    parser.add_argument("--M", type=int, required=True, help="Number of kinases.")
    parser.add_argument("--N", type=int, required=True, help="Number of phosphosites.")

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
            "Example: 0,0,0,1,1,2"
        ),
    )

    parser.add_argument(
        "--kin-to-prot-idx",
        type=str,
        default=None,
        help=(
            "Comma-separated kinase-to-protein mapping of length M. "
            "Use -1 for unmapped kinases. Example: 0,1,-1"
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


# ---------------------------------------------------------------------------
# Symbol helpers
# ---------------------------------------------------------------------------

def sym_vec(prefix: str, n: int) -> list[sp.Symbol]:
    return [sp.Symbol(f"{prefix}_{i}") for i in range(n)]


def sym_mat(prefix: str, rows: int, cols: int) -> list[list[sp.Symbol]]:
    return [[sp.Symbol(f"{prefix}_{i}_{j}") for j in range(cols)] for i in range(rows)]


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
    """Return Markdown-safe display equation using $$ ... $$ only."""
    return f"$$\n{lhs} = {sp.latex(rhs)}\n$$\n"


def md_raw_math(expr: str) -> str:
    """Return raw LaTeX equation text as Markdown-safe display math."""
    return f"$$\n{expr}\n$$\n"


# ---------------------------------------------------------------------------
# Algebra builder
# ---------------------------------------------------------------------------

def build_rhs_algebra(
    *,
    K: int,
    M: int,
    N: int,
    mechanism: str,
    site_prot_idx: list[int],
    kin_to_prot_idx: list[int],
) -> tuple[str, list[tuple[str, str, str]]]:
    eps = sp.Symbol(r"\epsilon", positive=True)
    t = sp.Symbol("t")

    # State symbols.
    R_rna = sym_vec("R", K)
    S = sym_vec("S", K)
    A = sym_vec("A", K)
    Kdyn = sym_vec("Kdyn", M)
    p = sym_vec("p", N)

    # External derived-rate symbols.
    k_act = sym_vec("k_act", K)
    s_prod = sym_vec("s_prod", K)

    # Decoded fitted parameters.
    k_deact = sym_vec("k_deact", K)
    d_deg = sym_vec("d_deg", K)
    alpha = sym_vec("alpha", M)
    kK_act = sym_vec("kK_act", M)
    kK_deact = sym_vec("kK_deact", M)
    k_off = sym_vec("k_off", N)

    beta_g, beta_l = sp.symbols("beta_g beta_l")
    gamma_S_p, gamma_A_S, gamma_K_net = sp.symbols("gamma_S_p gamma_A_S gamma_K_net")
    A_max = sp.Symbol("A_max", positive=True)

    # Network matrices.
    Cg = sym_mat("Cg", N, N)
    Cl = sym_mat("Cl", N, N)
    K_site_kin = sym_mat("Ksitekin", N, M)
    Rmat = sym_mat("Rmat", M, N)
    L_alpha = sym_mat("Lalpha", M, M)

    receptor_mask_prot = sym_vec("receptor_prot", K)
    receptor_mask_kin = sym_vec("receptor_kin", M)

    # ------------------------------------------------------------------
    # Core transforms.
    # ------------------------------------------------------------------
    p_pos = [smooth_pos(p[i], eps) for i in range(N)]
    q = [p_pos[i] / (1 + p_pos[i]) for i in range(N)]

    u = sigmoid(t / sp.Rational(1, 10))

    Cg_row_scale = [row_l1_norm(Cg[i], eps) for i in range(N)]
    Cl_row_scale = [row_l1_norm(Cl[i], eps) for i in range(N)]
    Ksk_row_scale = [row_l1_norm(K_site_kin[i], eps) for i in range(N)]
    R_row_scale = [row_l1_norm(Rmat[m], eps) for m in range(M)]
    L_row_scale = [row_l1_norm(L_alpha[m], eps) for m in range(M)]

    Cg_q = [
        add_sum(Cg[i][j] * q[j] for j in range(N)) / Cg_row_scale[i]
        for i in range(N)
    ]

    Cl_q = [
        add_sum(Cl[i][j] * q[j] for j in range(N)) / Cl_row_scale[i]
        for i in range(N)
    ]

    coup_field = [beta_g * Cg_q[i] + beta_l * Cl_q[i] for i in range(N)]
    coup = [sp.tanh(coup_field[i]) for i in range(N)]
    coup_factor = [sp.exp(coup[i]) for i in range(N)]

    # ------------------------------------------------------------------
    # Per-protein phosphosite summaries.
    # ------------------------------------------------------------------
    sites_by_prot: list[list[int]] = [[] for _ in range(K)]
    for site_i, prot_i in enumerate(site_prot_idx):
        if 0 <= prot_i < K:
            sites_by_prot[prot_i].append(site_i)

    den = []
    mq = []
    mc = []

    for k in range(K):
        n_sites_k = len(sites_by_prot[k])
        den_k = sp.sqrt(sp.Integer(n_sites_k) ** 2 + eps**2)
        den.append(den_k)

        mq_k = add_sum(q[i] for i in sites_by_prot[k]) / den_k
        mc_k = add_sum(coup[i] for i in sites_by_prot[k]) / den_k

        mq.append(mq_k)
        mc.append(mc_k)

    # ------------------------------------------------------------------
    # Auto-derived RNA terms.
    # ------------------------------------------------------------------
    rna_relax_auto = [
        sp.sqrt(smooth_pos(k_deact[k], eps) * smooth_pos(d_deg[k], eps) + eps)
        for k in range(K)
    ]

    k_min = sp.Symbol("min_k_act_pos")
    k_max = sp.Symbol("max_k_act_pos")

    k_span_scale = sp.Rational(1, 2) * sp.log((k_max + eps) / (k_min + eps))

    gamma_rms = sp.sqrt(
        (gamma_S_p**2 + gamma_A_S**2 + gamma_K_net**2) / 3 + eps
    )
    gamma_scale = gamma_rms / (1 + gamma_rms)

    rna_exp_scale = sp.Symbol("rna_exp_scale")

    # ------------------------------------------------------------------
    # Kinase basal terms.
    # ------------------------------------------------------------------
    eq = [
        kK_act[m] / (kK_act[m] + kK_deact[m] + eps)
        for m in range(M)
    ]

    basal_min = sp.Symbol("basal_min")
    basal_max = sp.Symbol("basal_max")
    receptor_bonus = sp.Symbol("receptor_bonus")
    max_support = sp.Symbol("max_support")

    site_support = [
        add_sum(sp.Abs(K_site_kin[i][m]) for i in range(N))
        for m in range(M)
    ]

    feedback_support = [
        add_sum(sp.Abs(Rmat[m][i]) for i in range(N))
        for m in range(M)
    ]

    raw_support = [
        site_support[m] + feedback_support[m] + receptor_bonus * receptor_mask_kin[m]
        for m in range(M)
    ]

    support_norm = [raw_support[m] / (max_support + eps) for m in range(M)]

    kinase_basal = [
        basal_min + (basal_max - basal_min) * support_norm[m]
        for m in range(M)
    ]

    # ------------------------------------------------------------------
    # RHS blocks.
    # ------------------------------------------------------------------
    dR = []
    dS = []
    dA = []
    dKdyn = []
    dp = []

    for k in range(K):
        rna_field = gamma_S_p * mq[k] + mc[k] + receptor_mask_prot[k] * u
        rna_reg = sp.exp(rna_exp_scale * sp.tanh(rna_field))
        dR_k = rna_relax_auto[k] * (rna_reg - R_rna[k])
        dR.append(dR_k)

        S_field = gamma_S_p * mq[k] + mc[k] + receptor_mask_prot[k] * u
        S_drive = sigmoid(S_field)
        dS_k = k_act[k] * S_drive * (1 - S[k]) - k_deact[k] * S[k]
        dS.append(dS_k)

        A_signal_mod = sp.Rational(1, 2) * sp.tanh(
            gamma_A_S * (S[k] - sp.Rational(1, 2))
        )
        s_eff = smooth_pos(s_prod[k] * (1 + A_signal_mod), eps)
        dA_k = s_eff - d_deg[k] * A[k]
        dA.append(dA_k)

    for m in range(M):
        u_sub = add_sum(Rmat[m][i] * q[i] for i in range(N)) / R_row_scale[m]
        u_net = -add_sum(L_alpha[m][j] * Kdyn[j] for j in range(M)) / L_row_scale[m]

        p_idx = kin_to_prot_idx[m]
        if 0 <= p_idx < K:
            prot_contrib = gamma_A_S * S[p_idx]
        else:
            prot_contrib = sp.Integer(0)

        U = (
            u_sub
            + gamma_K_net * u_net
            + prot_contrib
            + receptor_mask_kin[m] * u
        )

        K_drive = kinase_basal[m] + (1 - kinase_basal[m]) * sigmoid(U)

        dKdyn_m = (
            kK_act[m] * K_drive * (1 - Kdyn[m])
            - kK_deact[m] * Kdyn[m]
        )
        dKdyn.append(dKdyn_m)

    prev_site_idx = prev_site_idx_from_mapping(site_prot_idx)

    for i in range(N):
        prot_i = site_prot_idx[i]

        kinase_signal_sum = add_sum(
            K_site_kin[i][m] * alpha[m] * Kdyn[m]
            for m in range(M)
        )

        k_on_eff = smooth_pos(kinase_signal_sum / Ksk_row_scale[i], eps)

        if mechanism == "dist":
            gate = sp.Integer(1)

        elif mechanism == "seq":
            prev_i = prev_site_idx[i]
            site_available = 1 - q[i]

            if prev_i >= 0:
                seq_half = sp.Symbol("h_seq", positive=True)
                pred_enable = q[prev_i] / (seq_half + q[prev_i] + eps)
                gate = pred_enable * site_available
            else:
                gate = site_available

        else:
            occupied_frac = mq[prot_i]
            gate = 1 / (1 + occupied_frac)

        A_site = smooth_pos(A[prot_i] / (A_max + eps), eps)

        v_on_raw = smooth_pos(
            k_on_eff
            * coup_factor[i]
            * gate
            * (1 + sp.Rational(1, 2) * A_site),
            eps,
        )

        v_on = v_on_raw / (1 + v_on_raw)
        v_off = k_off[i] * p_pos[i]

        dp_i = v_on - v_off
        dp.append(dp_i)

    # ------------------------------------------------------------------
    # Dependency edges.
    # ------------------------------------------------------------------
    edges: list[tuple[str, str, str]] = []

    def edge(src: str, dst: str, rel: str = "depends_on") -> None:
        edges.append((src, dst, rel))

    for i in range(N):
        edge(f"p_{i}", f"q_{i}")
        edge(f"q_{i}", f"coup_{i}")
        edge(f"q_{i}", f"dp_{i}")
        edge(f"coup_{i}", f"dp_{i}")

    for k in range(K):
        for i in sites_by_prot[k]:
            edge(f"q_{i}", f"mq_{k}")
            edge(f"coup_{i}", f"mc_{k}")

        edge(f"mq_{k}", f"dR_{k}")
        edge(f"mc_{k}", f"dR_{k}")
        edge(f"mq_{k}", f"dS_{k}")
        edge(f"mc_{k}", f"dS_{k}")
        edge(f"S_{k}", f"dA_{k}")
        edge(f"A_{k}", f"dp_sites_for_protein_{k}")

    for m in range(M):
        edge(f"Kdyn_{m}", f"dKdyn_{m}")
        edge(f"Kdyn_{m}", "kinase_signal")

        p_idx = kin_to_prot_idx[m]
        if 0 <= p_idx < K:
            edge(f"S_{p_idx}", f"dKdyn_{m}")

    # ------------------------------------------------------------------
    # Markdown report.
    # ------------------------------------------------------------------
    lines: list[str] = []

    lines.append("# PhosCrosstalk RHS Algebraic Relationships\n")

    lines.append("## Model dimensions\n")
    lines.append(f"- `K = {K}` proteins")
    lines.append(f"- `M = {M}` kinases")
    lines.append(f"- `N = {N}` phosphosites")
    lines.append(f"- `mechanism = {mechanism}`")
    lines.append(f"- `site_prot_idx = {site_prot_idx}`")
    lines.append(f"- `kin_to_prot_idx = {kin_to_prot_idx}`\n")

    lines.append("## 1. Decoded parameter relationships\n")
    lines.append(md_raw_math(r"k_{\mathrm{deact},k} = \exp(\mathrm{clip}(\theta_{k_{\mathrm{deact},k}}))"))
    lines.append(md_raw_math(r"d_{\mathrm{deg},k} = \exp(\mathrm{clip}(\theta_{d_{\mathrm{deg},k}}))"))
    lines.append(md_raw_math(r"\beta_g = \exp(\mathrm{clip}(\theta_{\beta_g}))"))
    lines.append(md_raw_math(r"\beta_l = \exp(\mathrm{clip}(\theta_{\beta_l}))"))
    lines.append(md_raw_math(r"\alpha_m = \exp(\mathrm{clip}(\theta_{\alpha_m}))"))
    lines.append(md_raw_math(r"kK_{\mathrm{act},m} = \exp(\mathrm{clip}(\theta_{kK_{\mathrm{act},m}}))"))
    lines.append(md_raw_math(r"kK_{\mathrm{deact},m} = \exp(\mathrm{clip}(\theta_{kK_{\mathrm{deact},m}}))"))
    lines.append(md_raw_math(r"k_{\mathrm{off},i} = \exp(\mathrm{clip}(\theta_{k_{\mathrm{off},i}}))"))
    lines.append(md_raw_math(r"\gamma = 2\tanh(\theta_{\gamma})"))

    lines.append("## 2. Smooth helper relationships\n")
    lines.append(md_raw_math(r"[x]_+ = \frac{1}{2}\left(x + \sqrt{x^2+\epsilon^2}\right) - \frac{1}{2}\epsilon"))
    lines.append(md_raw_math(r"\sigma(x) = \frac{1}{1+\exp(-x)}"))
    lines.append(md_raw_math(r"q_i = \frac{[p_i]_+}{1+[p_i]_+}"))
    lines.append(md_raw_math(r"u(t) = \sigma(t/0.1)"))

    lines.append("## 3. Crosstalk relationships\n")
    for i in range(N):
        lines.append(md_math(f"Cgq_{{{i}}}", Cg_q[i]))
        lines.append(md_math(f"Clq_{{{i}}}", Cl_q[i]))
        lines.append(md_math(f"coup_{{{i}}}", coup[i]))
        lines.append(md_math(f"\\Phi_{{{i}}}", coup_factor[i]))

    lines.append("## 4. Per-protein phosphosite summaries\n")
    for k in range(K):
        lines.append(md_math(f"\\bar q_{{{k}}}", mq[k]))
        lines.append(md_math(f"\\bar c_{{{k}}}", mc[k]))

    lines.append("## 5. Auto-derived RNA relationships\n")
    for k in range(K):
        lines.append(md_math(f"\\rho_{{R,{k}}}", rna_relax_auto[k]))

    lines.append(md_math(r"scale_{k\_act}", k_span_scale))
    lines.append(md_math(r"scale_{\gamma}", gamma_scale))
    lines.append(md_raw_math(r"rna\_exp\_scale = \max(scale_{k\_act}, scale_{\gamma})"))

    lines.append("## 6. Auto-derived kinase basal relationships\n")
    for m in range(M):
        lines.append(md_math(f"eq_{{K,{m}}}", eq[m]))
        lines.append(md_math(f"support_{{{m}}}", raw_support[m]))
        lines.append(md_math(f"basal_{{K,{m}}}", kinase_basal[m]))

    lines.append("## 7. RHS equations\n")

    lines.append("### 7.1 Latent mRNA / transcriptional state\n")
    for k in range(K):
        lines.append(md_math(f"\\frac{{dR_{{{k}}}}}{{dt}}", dR[k]))

    lines.append("### 7.2 Protein signalling state\n")
    for k in range(K):
        lines.append(md_math(f"\\frac{{dS_{{{k}}}}}{{dt}}", dS[k]))

    lines.append("### 7.3 Protein abundance state\n")
    for k in range(K):
        lines.append(md_math(f"\\frac{{dA_{{{k}}}}}{{dt}}", dA[k]))

    lines.append("### 7.4 Kinase activity state\n")
    for m in range(M):
        lines.append(md_math(f"\\frac{{dKdyn_{{{m}}}}}{{dt}}", dKdyn[m]))

    lines.append("### 7.5 Phosphosite state\n")
    for i in range(N):
        lines.append(md_math(f"\\frac{{dp_{{{i}}}}}{{dt}}", dp[i]))

    lines.append("## 8. Main biological dependency structure\n")
    lines.append("- `p -> q -> crosstalk -> R_rna, S, p`")
    lines.append("- `p -> mq -> R_rna, S, p`")
    lines.append("- `S -> A`")
    lines.append("- `S -> Kdyn` through kinase-to-protein mapping")
    lines.append("- `Kdyn -> p` through kinase-site activation")
    lines.append("- `A -> p` through abundance-scaled phosphorylation flux")
    lines.append("- `k_act(t) -> S`")
    lines.append("- `s_prod(t) -> A`")
    lines.append("- `kK_act, kK_deact, K_site_kin, R -> kinase_basal -> Kdyn`")

    return "\n".join(lines) + "\n", edges


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    if args.K <= 0:
        raise ValueError("--K must be positive.")
    if args.M < 0:
        raise ValueError("--M must be non-negative.")
    if args.N <= 0:
        raise ValueError("--N must be positive.")

    default_site_prot_idx = [i % args.K for i in range(args.N)]
    default_kin_to_prot_idx = [i % args.K for i in range(args.M)]

    site_prot_idx = _parse_int_list(
        raw=args.site_prot_idx,
        n=args.N,
        name="site_prot_idx",
        default=default_site_prot_idx,
    )

    kin_to_prot_idx = _parse_int_list(
        raw=args.kin_to_prot_idx,
        n=args.M,
        name="kin_to_prot_idx",
        default=default_kin_to_prot_idx,
    )

    for val in site_prot_idx:
        if val < 0 or val >= args.K:
            raise ValueError(
                f"site_prot_idx values must be in [0, K-1]. Got {val}."
            )

    for val in kin_to_prot_idx:
        if val != -1 and (val < 0 or val >= args.K):
            raise ValueError(
                f"kin_to_prot_idx values must be -1 or in [0, K-1]. Got {val}."
            )

    report, edges = build_rhs_algebra(
        K=args.K,
        M=args.M,
        N=args.N,
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