"""
Generates LaTeX documentation of the model's Ordinary Differential Equations.

Outputs:
  1. Numeric fitted report for the active mechanism.
  2. Symbolic reports for dist, seq, and rand mechanisms.

This version matches the reduced mechanisms.py model:

  - k_act(t) and s_prod(t) are bounded derived external rates.
  - k_act(t) is used directly in dS, without k_act/(1+k_act).
  - s_prod(t) is used as the main abundance synthesis drive.
  - gamma_A_p is retained in decoded parameters for backward compatibility,
    but is not used in the reduced abundance equation.
  - Kinase activation uses a bounded sigmoid drive.
  - Phosphosite dynamics use q = p_+/(1+p_+) and saturated v_on.

LaTeX safety rules:
  - Text labels and math identifiers are escaped separately.
  - Biological identifiers such as EPHA2_Y575 are rendered as
    \\mathrm{EPHA2\\_Y575} inside math mode.
  - \\mathbb requires amssymb/amsfonts.
  - Phosphosite equations are displayed directly with dmath*.
  - Large phosphosite production terms are split into V_site definitions.
"""

from __future__ import annotations

import os
import subprocess
from typing import Any

import numpy as np

from phoscrosstalk.mechanisms import decode_theta
from phoscrosstalk.logger import get_logger

logger = get_logger()


# ---------------------------------------------------------------------------
# LaTeX escaping helpers
# ---------------------------------------------------------------------------

def _tex_text(name: Any) -> str:
    """
    Escape a string for LaTeX text mode.

    Use for section titles, table labels, prose, and \\textbf{...}.
    """
    s = str(name)
    replacements = {
        "\\": r"\textbackslash{}",
        "_": r"\_",
        "%": r"\%",
        "&": r"\&",
        "#": r"\#",
        "$": r"\$",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in s)


def _tex_math_id(name: Any) -> str:
    """
    Escape a biological identifier for use inside math mode.

    Example:
        EPHA2_Y575 -> \\mathrm{EPHA2\\_Y575}
    """
    s = str(name)
    replacements = {
        "\\": r"\backslash{}",
        "_": r"\_",
        "%": r"\%",
        "&": r"\&",
        "#": r"\#",
        "$": r"\$",
        "{": r"\{",
        "}": r"\}",
    }
    escaped = "".join(replacements.get(ch, ch) for ch in s)
    return rf"\mathrm{{{escaped}}}"


def _fmt_float(x: float, *, zero_tol: float = 1e-10, sig: int = 4) -> str:
    """
    Format numeric coefficients for LaTeX equations.

    Very small values are printed as 0 to avoid noisy equations.
    """
    x = float(x)

    if not np.isfinite(x):
        return r"\mathrm{NaN}"

    if abs(x) < zero_tol:
        return "0"

    return f"{x:.{sig}g}"


def _safe_title(title: str) -> str:
    """Escape document title for LaTeX text mode."""
    return _tex_text(title)


# ---------------------------------------------------------------------------
# LaTeX document helpers
# ---------------------------------------------------------------------------

def _latex_preamble(
        *,
        title: str = "Phospho-Network Model Equations",
        paper: str = "a4paper",
        margin: str = "1in",
        landscape: bool = False,
        font_pt: int = 11,
) -> str:
    """
    Build LaTeX preamble with user-controlled page and font settings.

    Notes:
      - Standard article supports 10pt/11pt/12pt.
      - For <10pt, extarticle is used automatically.
      - amssymb/amsfonts are required for \\mathbb.
      - breqn is used for long display equations.
    """
    if font_pt < 10:
        docclass = rf"\documentclass[{font_pt}pt]{{extarticle}}"
    else:
        docclass = rf"\documentclass[{font_pt}pt]{{article}}"

    orient = "landscape" if landscape else "portrait"
    safe_title = _safe_title(title)

    return rf"""
{docclass}
\usepackage{{amsmath}}
\usepackage{{amssymb}}
\usepackage{{amsfonts}}
\usepackage{{geometry}}
\usepackage{{longtable}}
\usepackage{{breqn}}
\usepackage{{xcolor}}
\usepackage{{microtype}}
\sloppy
\setlength{{\emergencystretch}}{{3em}}
\allowdisplaybreaks
\geometry{{{paper}, {orient}, margin={margin}}}
\title{{{safe_title}}}
\date{{\today}}
\begin{{document}}
\maketitle
""".lstrip()


def _table_spec(eq_col: str) -> str:
    """
    Return longtable column spec.

    eq_col examples:
      - "12cm"
      - "0.78\\linewidth"
    """
    return rf"l p{{{eq_col}}}"


def _latex_footer() -> str:
    """Return document footer."""
    return r"\end{document}"


# ---------------------------------------------------------------------------
# Small equation-string helpers
# ---------------------------------------------------------------------------

def _nonzero_indices(row: np.ndarray, tol: float = 0.0) -> np.ndarray:
    """Return indices with absolute row values above tolerance."""
    row = np.asarray(row, dtype=float)
    return np.where(np.abs(row) > tol)[0]


def _mean_proxy_symbol(prot_id: str) -> str:
    """Symbol for protein-level mean bounded phosphosite proxy."""
    return rf"\bar{{q}}_{{{prot_id}}}"


def _crosstalk_symbol(site_id: str) -> str:
    """Symbol for site-level crosstalk field."""
    return rf"\mathcal{{C}}_{{{site_id}}}"


def _site_abundance_symbol(site_id: str) -> str:
    """Symbol for abundance value mapped to a phosphosite."""
    return rf"A_{{\pi({site_id})}}"


# ---------------------------------------------------------------------------
# Equation generation
# ---------------------------------------------------------------------------

def _generate_latex_source(
        mode,
        theta,
        proteins,
        kinases,
        sites,
        Cg,
        Cl,
        site_prot_idx,
        K_site_kin,
        R,
        L_alpha,
        kin_to_prot_idx,
        receptor_mask_prot,
        receptor_mask_kin,
        mechanism,
        *,
        latex_title: str,
        paper: str,
        margin: str,
        landscape: bool,
        font_pt: int,
        eq_col: str,
):
    """
    Build LaTeX source for the ODE system.

    Args:
        mode:
            "numeric" or "symbolic".
        theta:
            Fitted parameter vector when mode == "numeric".
        mechanism:
            "dist", "seq", or "rand".
    """
    K, M, N = len(proteins), len(kinases), len(sites)

    Cg = np.asarray(Cg, dtype=float)
    Cl = np.asarray(Cl, dtype=float)
    site_prot_idx = np.asarray(site_prot_idx, dtype=int)
    K_site_kin = np.asarray(K_site_kin, dtype=float)
    R = np.asarray(R, dtype=float)
    L_alpha = np.asarray(L_alpha, dtype=float)
    kin_to_prot_idx = np.asarray(kin_to_prot_idx, dtype=int)
    receptor_mask_prot = np.asarray(receptor_mask_prot, dtype=bool)
    receptor_mask_kin = np.asarray(receptor_mask_kin, dtype=bool)

    if mode not in {"numeric", "symbolic"}:
        raise ValueError(f"mode must be 'numeric' or 'symbolic', got {mode!r}")

    if mechanism not in {"dist", "seq", "rand"}:
        raise ValueError(
            f"mechanism must be one of 'dist', 'seq', 'rand', got {mechanism!r}"
        )

    # Decode fitted parameters only for numeric report.
    if mode == "numeric" and theta is not None:
        (
            k_deact,
            d_deg,
            beta_g,
            beta_l,
            alpha,
            kK_act,
            kK_deact,
            k_off,
            gamma_S_p,
            gamma_A_S,
            gamma_A_p,
            gamma_K_net,
        ) = decode_theta(theta, K, M, N)

    def get_p(arr_name: str, idx: int, symbol_base: str) -> str:
        """
        Parameter lookup helper.

        For symbolic mode, returns symbol_base.
        For numeric mode, returns fitted value when available.
        """
        if mode != "numeric" or theta is None:
            return symbol_base

        val = None

        if arr_name == "k_act":
            return symbol_base

        if arr_name == "s_prod":
            return symbol_base

        if arr_name == "k_deact":
            val = k_deact[idx]
        elif arr_name == "d_deg":
            val = d_deg[idx]
        elif arr_name == "alpha":
            val = alpha[idx]
        elif arr_name == "kK_act":
            val = kK_act[idx]
        elif arr_name == "kK_deact":
            val = kK_deact[idx]
        elif arr_name == "k_off":
            val = k_off[idx]
        elif arr_name == "gamma_S_p":
            val = gamma_S_p
        elif arr_name == "gamma_A_S":
            val = gamma_A_S
        elif arr_name == "gamma_A_p":
            val = gamma_A_p
        elif arr_name == "gamma_K_net":
            val = gamma_K_net
        elif arr_name == "beta_g":
            val = beta_g
        elif arr_name == "beta_l":
            val = beta_l
        else:
            return symbol_base

        return _fmt_float(float(val), zero_tol=1e-8, sig=4)

    lines: list[str] = []

    lines.append(
        _latex_preamble(
            title=latex_title,
            paper=paper,
            margin=margin,
            landscape=landscape,
            font_pt=font_pt,
        )
    )

    # ------------------------------------------------------------------
    # Header / configuration
    # ------------------------------------------------------------------
    lines.append(rf"\section*{{Model Configuration: {_tex_text(mode.capitalize())}}}")

    mech_desc = {
        "dist": "Distributive independent-site phosphorylation",
        "seq": r"Sequential ordered phosphorylation $p_1 \to p_2$",
        "rand": "Random/crowding-aware phosphorylation with bounded proxy feedback",
    }[mechanism]

    lines.append(rf"\textbf{{Mechanism:}} {mech_desc}\\")
    if mode == "numeric":
        lines.append(r"\textbf{Note:} Parameters shown are fitted numeric values.\\")
    else:
        lines.append(r"\textbf{Note:} Parameters shown are symbolic.\\")
    lines.append(
        r"\textbf{Reduced-rate convention:} "
        r"$k_{\mathrm{act}}(t)$ and $s_{\mathrm{prod}}(t)$ are bounded derived "
        r"external rates. They are generated outside the RHS from omics/network "
        r"signals and are not fitted entries in $\theta$.\\"
    )
    lines.append("")

    # ------------------------------------------------------------------
    # State layout
    # ------------------------------------------------------------------
    lines.append(r"\section{ODE State Layout}")
    lines.append(r"\begin{itemize}")
    lines.append(
        r"\item State vector: "
        r"$\mathbf{y} = [R_{\mathrm{rna}}, S, A, K_{\mathrm{dyn}}, p]$ "
        r"with dimension $3K + M + N$."
    )
    lines.append(
        r"\item $R_{\mathrm{rna}} \in \mathbb{R}^{K}$: latent mRNA / "
        r"transcriptional state."
    )
    lines.append(
        r"\item $S \in \mathbb{R}^{K}$: protein signalling state."
    )
    lines.append(r"\item $A \in \mathbb{R}^{K}$: protein abundance.")
    lines.append(
        r"\item $K_{\mathrm{dyn}} \in \mathbb{R}^{M}$: kinase activity state."
    )
    lines.append(
        r"\item $p \in \mathbb{R}^{N}$: relative phosphosite state."
    )
    lines.append(r"\end{itemize}")
    lines.append("")

    # ------------------------------------------------------------------
    # Global definitions
    # ------------------------------------------------------------------
    lines.append(r"\section{Global Definitions}")

    bg = get_p("beta_g", 0, r"\beta_g")
    bl = get_p("beta_l", 0, r"\beta_l")

    lines.append(r"\begin{itemize}")
    lines.append(r"\item Smooth receptor stimulus: $u(t) = \sigma(t/0.1)$.")
    lines.append(r"\item Sigmoid: $\sigma(x)=\frac{1}{1+e^{-x}}$.")
    lines.append(
        r"\item Smooth positive part used in implementation: "
        r"$x_{+} \approx \frac{1}{2}\left(x+\sqrt{x^2+\varepsilon^2}\right)"
        r"-\frac{1}{2}\varepsilon$."
    )
    lines.append(
        r"\item Bounded phosphosite proxy: "
        r"$q_i = \frac{p_{i,+}}{1+p_{i,+}} \in [0,1)$."
    )
    lines.append(
        r"\item Protein-level mean phosphosite proxy: "
        r"$\bar{q}_g = |\mathcal{P}(g)|^{-1}\sum_{i\in\mathcal{P}(g)}q_i$."
    )
    lines.append(
        r"\item Site crosstalk field: "
        rf"$\mathcal{{C}}_i = \tanh\left("
        rf"{bg}\frac{{(C_g\mathbf{{q}})_i}}{{\|C_{{g,i\cdot}}\|_1}}"
        rf" + {bl}\frac{{(C_l\mathbf{{q}})_i}}{{\|C_{{l,i\cdot}}\|_1}}"
        r"\right)$."
    )
    lines.append(
        r"\item Crosstalk multiplicative factor: "
        r"$\Phi_i = \exp(\mathcal{C}_i)$."
    )
    lines.append(
        r"\item $k_{\mathrm{act}}^g(t)$ is a bounded TF/RNA-derived activation "
        r"rate for protein $g$."
    )
    lines.append(
        r"\item $s_{\mathrm{prod}}^g(t)$ is a bounded kinase/phosphosite-derived "
        r"synthesis drive for protein $g$."
    )
    lines.append(r"\end{itemize}")
    lines.append("")

    # ------------------------------------------------------------------
    # Latent mRNA dynamics
    # ------------------------------------------------------------------
    lines.append(r"\section{Latent mRNA / Transcriptional Dynamics ($R_{\mathrm{rna}}$)}")
    lines.append(
        r"The latent transcriptional state relaxes toward a positive bounded "
        r"regulatory target. This state is separate from the external "
        r"$k_{\mathrm{act}}(t)$ derived-rate input."
    )
    lines.append(r"\begin{longtable}{" + _table_spec(eq_col) + r"}")
    lines.append(r"\textbf{Protein} & \textbf{Equation} \\ \hline")

    g_Sp = get_p("gamma_S_p", 0, r"\gamma_{S,p}")

    for k, prot in enumerate(proteins):
        prot_txt = _tex_text(prot)
        prot_id = _tex_math_id(prot)

        qbar = _mean_proxy_symbol(prot_id)

        field_terms = [
            rf"{g_Sp}\,{qbar}",
            rf"\bar{{\mathcal{{C}}}}_{{{prot_id}}}",
        ]
        if receptor_mask_prot[k]:
            field_terms.append(r"u(t)")

        field = " + ".join(field_terms)

        eq_R = (
            rf"\frac{{dR_{{\mathrm{{rna}},{prot_id}}}}}{{dt}} = "
            rf"\rho_R\left["
            rf"\exp\left(\frac{{1}}{{2}}\tanh\left({field}\right)\right)"
            rf" - R_{{\mathrm{{rna}},{prot_id}}}"
            rf"\right]"
        )

        lines.append(rf"\textbf{{{prot_txt}}} & ${eq_R}$ \\ \hline")

    lines.append(r"\end{longtable}")
    lines.append("")

    # ------------------------------------------------------------------
    # Protein signalling dynamics
    # ------------------------------------------------------------------
    lines.append(r"\section{Protein Signalling Dynamics ($S$)}")
    lines.append(
        r"The reduced RHS uses $k_{\mathrm{act}}(t)$ directly. "
        r"No additional $k_{\mathrm{act}}/(1+k_{\mathrm{act}})$ compression is applied."
    )
    lines.append(r"\begin{longtable}{" + _table_spec(eq_col) + r"}")
    lines.append(r"\textbf{Protein} & \textbf{Equation} \\ \hline")

    for k, prot in enumerate(proteins):
        prot_txt = _tex_text(prot)
        prot_id = _tex_math_id(prot)

        ka = get_p("k_act", k, rf"k_{{\mathrm{{act}}}}^{{{prot_id}}}(t)")
        kd = get_p("k_deact", k, rf"k_{{\mathrm{{deact}}}}^{{{prot_id}}}")

        qbar = _mean_proxy_symbol(prot_id)
        field_terms = [
            rf"{g_Sp}\,{qbar}",
            rf"\bar{{\mathcal{{C}}}}_{{{prot_id}}}",
        ]
        if receptor_mask_prot[k]:
            field_terms.append(r"u(t)")

        field = " + ".join(field_terms)

        eq_S = (
            rf"\frac{{dS_{{{prot_id}}}}}{{dt}} = "
            rf"{ka}\,\sigma\left({field}\right)"
            rf"\left(1-S_{{{prot_id}}}\right) - "
            rf"{kd}\,S_{{{prot_id}}}"
        )

        lines.append(rf"\textbf{{{prot_txt}}} & ${eq_S}$ \\ \hline")

    lines.append(r"\end{longtable}")
    lines.append("")

    # ------------------------------------------------------------------
    # Protein abundance dynamics
    # ------------------------------------------------------------------
    lines.append(r"\section{Protein Abundance Dynamics ($A$)}")
    lines.append(
        r"The synthesis drive $s_{\mathrm{prod}}(t)$ already contains the "
        r"bounded kinase/phosphosite-derived regulatory input. Therefore the "
        r"abundance equation only modulates this drive by the internal signalling "
        r"state $S$ and does not re-use $\gamma_{A,p}\bar{q}$."
    )
    lines.append(r"\begin{longtable}{" + _table_spec(eq_col) + r"}")
    lines.append(r"\textbf{Protein} & \textbf{Equation} \\ \hline")

    g_AS = get_p("gamma_A_S", 0, r"\gamma_{A,S}")

    for k, prot in enumerate(proteins):
        prot_txt = _tex_text(prot)
        prot_id = _tex_math_id(prot)

        sp = get_p("s_prod", k, rf"s_{{\mathrm{{prod}}}}^{{{prot_id}}}(t)")
        dd = get_p("d_deg", k, rf"d_{{\mathrm{{deg}}}}^{{{prot_id}}}")

        eq_A = (
            rf"\frac{{dA_{{{prot_id}}}}}{{dt}} = "
            rf"\left["
            rf"{sp}\left(1 + \frac{{1}}{{2}}\tanh\left("
            rf"{g_AS}\left(S_{{{prot_id}}}-\frac{{1}}{{2}}\right)"
            rf"\right)\right)"
            rf"\right]_+ - "
            rf"{dd}\,A_{{{prot_id}}}"
        )

        lines.append(rf"\textbf{{{prot_txt}}} & ${eq_A}$ \\ \hline")

    lines.append(r"\end{longtable}")
    lines.append("")

    # ------------------------------------------------------------------
    # Kinase dynamics
    # ------------------------------------------------------------------
    lines.append(r"\section{Kinase Dynamics ($K_{\mathrm{dyn}}$)}")

    g_Knet = get_p("gamma_K_net", 0, r"\gamma_{K,\mathrm{net}}")

    lines.append(
        r"The kinase drive is bounded by "
        r"$K_{\mathrm{drive}}=\delta+(1-\delta)\sigma(U_m)$ with $\delta=0.05$."
    )
    lines.append(r"\begin{longtable}{" + _table_spec(eq_col) + r"}")
    lines.append(r"\textbf{Kinase} & \textbf{Equation} \\ \hline")

    for m, kin in enumerate(kinases):
        kin_txt = _tex_text(kin)
        kin_id = _tex_math_id(kin)

        kka = get_p("kK_act", m, rf"k_{{K,\mathrm{{act}}}}^{{{kin_id}}}")
        kkd = get_p("kK_deact", m, rf"k_{{K,\mathrm{{deact}}}}^{{{kin_id}}}")

        u_terms = []

        feeding = _nonzero_indices(R[m, :]) if R.ndim == 2 and m < R.shape[0] else np.array([], dtype=int)
        if len(feeding) > 0:
            if mode == "symbolic":
                u_terms.append(
                    rf"\frac{{\sum_i R_{{{kin_id},i}}q_i}}{{\|R_{{{kin_id},\cdot}}\|_1}}"
                )
            else:
                top = feeding[:3]
                sub_terms = []
                for i_site in top:
                    site_id = _tex_math_id(sites[i_site])
                    weight = _fmt_float(float(R[m, i_site]), sig=3)
                    sub_terms.append(rf"{weight}\,q_{{{site_id}}}")

                sub_str = " + ".join(sub_terms)
                if len(feeding) > 3:
                    sub_str += r" + \cdots"

                u_terms.append(rf"\frac{{{sub_str}}}{{\|R_{{{kin_id},\cdot}}\|_1}}")

        if mode == "symbolic" or g_Knet != "0":
            u_terms.append(
                rf"{g_Knet}\left[-\frac{{(L_\alpha K)_{{{kin_id}}}}}"
                rf"{{\|L_{{\alpha,{kin_id},\cdot}}\|_1}}\right]"
            )

        p_idx = int(kin_to_prot_idx[m]) if m < len(kin_to_prot_idx) else -1
        if 0 <= p_idx < len(proteins):
            prot_id = _tex_math_id(proteins[p_idx])
            u_terms.append(rf"{g_AS}\,S_{{{prot_id}}}")

        if receptor_mask_kin[m]:
            u_terms.append(r"u(t)")

        u_str = " + ".join(u_terms) if u_terms else "0"

        eq_K = (
            rf"\frac{{dK_{{{kin_id}}}}}{{dt}} = "
            rf"{kka}\left["
            rf"\delta + (1-\delta)\sigma\left({u_str}\right)"
            rf"\right]\left(1-K_{{{kin_id}}}\right) - "
            rf"{kkd}\,K_{{{kin_id}}},"
            r"\quad \delta=0.05"
        )

        lines.append(rf"\textbf{{{kin_txt}}} & ${eq_K}$ \\ \hline")

    lines.append(r"\end{longtable}")
    lines.append("")

    # ------------------------------------------------------------------
    # Phosphosite dynamics
    # ------------------------------------------------------------------
    lines.append(r"\section{Phosphosite Dynamics ($p$)}")

    if mechanism == "seq":
        lines.append(
            r"\textcolor{blue}{\textbf{Sequential model:}} "
            r"The gate for non-first sites depends on the predecessor occupancy "
            r"$q_{\mathrm{prev}}/(h_{\mathrm{seq}}+q_{\mathrm{prev}})$ and the "
            r"site availability $(1-q_i)$.\\"
        )
    elif mechanism == "rand":
        lines.append(
            r"\textcolor{blue}{\textbf{Random/crowding-aware model:}} "
            r"The gate uses the mean bounded proxy $\bar{q}$ per protein.\\"
        )
    else:
        lines.append(
            r"\textcolor{blue}{\textbf{Distributive model:}} "
            r"The gate is $1$ for all sites.\\"
        )

    last_site_for_protein: dict[int, str] = {}

    for i, site in enumerate(sites):
        site_txt = _tex_text(site)
        site_id = _tex_math_id(site)

        koff = get_p("k_off", i, rf"k_{{\mathrm{{off}}}}^{{{site_id}}}")
        prot_idx = int(site_prot_idx[i])
        prot_id = _tex_math_id(proteins[prot_idx])

        kin_idxs = (
            _nonzero_indices(K_site_kin[i, :])
            if K_site_kin.ndim == 2
            else np.array([], dtype=int)
        )

        k_on_terms = []
        for kid in kin_idxs:
            if kid >= len(kinases):
                continue

            w = float(K_site_kin[i, kid])
            kin_id = _tex_math_id(kinases[kid])
            alpha_symbol = get_p("alpha", kid, rf"\alpha_{{{kin_id}}}")

            if mode == "symbolic":
                term = rf"w_{{{site_id},{kin_id}}}\,{alpha_symbol}\,K_{{{kin_id}}}"
            else:
                try:
                    eff_w = w * float(alpha_symbol)
                    term = rf"{_fmt_float(eff_w, sig=4)}\,K_{{{kin_id}}}"
                except ValueError:
                    term = rf"{_fmt_float(w, sig=3)}\,{alpha_symbol}\,K_{{{kin_id}}}"

            k_on_terms.append(term)

        k_on_str = " + ".join(k_on_terms) if k_on_terms else "0"

        c_term = rf"\Phi_{{{site_id}}}"

        if mechanism == "dist":
            gate = "1"

        elif mechanism == "seq":
            prev_site_id = last_site_for_protein.get(prot_idx)

            if prev_site_id:
                gate = (
                    rf"\frac{{q_{{{prev_site_id}}}}}"
                    rf"{{h_{{\mathrm{{seq}}}}+q_{{{prev_site_id}}}}}"
                    rf"\left(1-q_{{{site_id}}}\right)"
                )
            else:
                gate = rf"\left(1-q_{{{site_id}}}\right)"

        else:
            qbar = _mean_proxy_symbol(prot_id)
            gate = rf"\frac{{1}}{{1+{qbar}}}"

        A_site = _site_abundance_symbol(site_id)
        p_pos = rf"p_{{{site_id},+}}"

        v_raw = (
            rf"\left[{k_on_str}\right]"
            rf"\,{c_term}"
            rf"\,{gate}"
            rf"\left(1 + \frac{{1}}{{2}}\frac{{{A_site}}}{{A_{{\max}}}}\right)"
        )

        lines.append(rf"\subsection*{{{site_txt}}}")

        lines.append(r"\begin{dmath*}")
        lines.append(rf"V_{{{site_id}}} = \left[{v_raw}\right]_+")
        lines.append(r"\end{dmath*}")

        lines.append(r"\begin{dmath*}")
        lines.append(
            rf"\frac{{dp_{{{site_id}}}}}{{dt}} = "
            rf"\frac{{V_{{{site_id}}}}}{{1 + V_{{{site_id}}}}} - "
            rf"{koff}\,{p_pos}"
        )
        lines.append(r"\end{dmath*}")
        lines.append("")

        last_site_for_protein[prot_idx] = site_id

    lines.append(_latex_footer())

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# PDF compilation
# ---------------------------------------------------------------------------

def _compile_pdf(tex_path: str, outdir: str) -> None:
    """
    Compile LaTeX to PDF if pdflatex is available.

    Writes a .compile.log file next to the .tex file.
    """
    try:
        subprocess.run(
            ["pdflatex", "--version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )

        base = os.path.basename(tex_path)
        logger.info(f"    -> Compiling {base}...")

        result = subprocess.run(
            [
                "pdflatex",
                "-interaction=nonstopmode",
                "-halt-on-error",
                "-output-directory",
                outdir,
                tex_path,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )

        compile_log = os.path.join(outdir, base.replace(".tex", ".compile.log"))
        with open(compile_log, "w", encoding="utf-8") as fh:
            fh.write(result.stdout or "")

        pdf_name = base.replace(".tex", ".pdf")
        pdf_path = os.path.join(outdir, pdf_name)

        if result.returncode == 0 and os.path.exists(pdf_path):
            logger.success(f"    -> Generated {pdf_name}")
        else:
            logger.warning(
                f"    -> pdflatex failed for {base}. See compile log: {compile_log}"
            )

    except (OSError, subprocess.SubprocessError, FileNotFoundError):
        logger.warning("    -> 'pdflatex' not found or failed. PDF not generated.")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def generate_equations_report(
        outdir,
        theta_opt,
        proteins,
        kinases,
        sites,
        Cg,
        Cl,
        site_prot_idx,
        K_site_kin,
        R,
        L_alpha,
        kin_to_prot_idx,
        receptor_mask_prot,
        receptor_mask_kin,
        mechanism,
        *,
        latex_title: str = "Phospho-Network Model Equations",
        paper: str = "a4paper",
        margin: str = "1in",
        landscape: bool = False,
        font_pt: int = 11,
        eq_col: str = r"0.78\linewidth",
):
    """
    Generate equation reports.

    Outputs
    -------
    In {outdir}/equations:
      - model_equations_fitted_<mechanism>.tex
      - model_equations_fitted_<mechanism>.pdf, if pdflatex succeeds
      - model_equations_symbolic_dist.tex
      - model_equations_symbolic_seq.tex
      - model_equations_symbolic_rand.tex
      - corresponding PDFs when compilation succeeds
    """
    logger.info("[*] Generating Model Equation Reports...")

    eq_dir = os.path.join(outdir, "equations")
    os.makedirs(eq_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Fitted numeric report for active mechanism
    # ------------------------------------------------------------------
    tex_num = _generate_latex_source(
        "numeric",
        theta_opt,
        proteins,
        kinases,
        sites,
        Cg,
        Cl,
        site_prot_idx,
        K_site_kin,
        R,
        L_alpha,
        kin_to_prot_idx,
        receptor_mask_prot,
        receptor_mask_kin,
        mechanism,
        latex_title=latex_title,
        paper=paper,
        margin=margin,
        landscape=landscape,
        font_pt=font_pt,
        eq_col=eq_col,
    )

    path_num = os.path.join(eq_dir, f"model_equations_fitted_{mechanism}.tex")
    with open(path_num, "w", encoding="utf-8") as fh:
        fh.write(tex_num)

    _compile_pdf(path_num, eq_dir)

    # ------------------------------------------------------------------
    # 2. Symbolic reports for all mechanisms
    # ------------------------------------------------------------------
    for mech in ["dist", "seq", "rand"]:
        tex_sym = _generate_latex_source(
            "symbolic",
            None,
            proteins,
            kinases,
            sites,
            Cg,
            Cl,
            site_prot_idx,
            K_site_kin,
            R,
            L_alpha,
            kin_to_prot_idx,
            receptor_mask_prot,
            receptor_mask_kin,
            mech,
            latex_title=latex_title,
            paper=paper,
            margin=margin,
            landscape=landscape,
            font_pt=font_pt,
            eq_col=eq_col,
        )

        path_sym = os.path.join(eq_dir, f"model_equations_symbolic_{mech}.tex")
        with open(path_sym, "w", encoding="utf-8") as fh:
            fh.write(tex_sym)

        _compile_pdf(path_sym, eq_dir)