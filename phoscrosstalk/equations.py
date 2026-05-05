"""
equations.py

Generates LaTeX documentation of the model's Ordinary Differential Equations (ODEs).

Outputs:
  1. Numeric fitted report for the active mechanism.
  2. Symbolic reports for dist, seq, and rand mechanisms.

Key LaTeX safety rules:
  - Text labels and math identifiers are escaped separately.
  - Biological identifiers such as EPHA2_Y575 are rendered as \\mathrm{EPHA2\\_Y575}
    inside math mode.
  - \\mathbb requires amssymb/amsfonts.
  - Phosphosite equations are displayed directly with dmath*; they are not wrapped
    in longtable.
  - Large phosphosite production terms are split into V_site definitions to avoid
    fragile nested fractions.

Controls:
  - Font size
  - Portrait/landscape layout
  - Table equation column width
"""

from __future__ import annotations

import os
import subprocess
from typing import Any

import numpy as np

from phoscrosstalk.jax_mechanisms import decode_theta_jax
from phoscrosstalk.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# LaTeX escaping helpers
# ---------------------------------------------------------------------------


def _tex_text(name: Any) -> str:
    """
    Escape a string for LaTeX text mode.

    Use for:
      - section titles
      - table labels
      - prose text
      - \\textbf{...}
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

    Use inside subscripts/superscripts:
        p_{\\mathrm{EPHA2\\_Y575}}
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
    Build the LaTeX source for the ODE system.

    Parameters
    ----------
    mode : {"numeric", "symbolic"}
        numeric: uses fitted parameter values from theta.
        symbolic: uses symbolic parameter names.
    theta : np.ndarray | None
        Fitted parameter vector when mode == "numeric".
    mechanism : {"dist", "seq", "rand"}
        Phosphosite mechanism.
    """
    K, M, N = len(proteins), len(kinases), len(sites)

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
            # Derived from data; not optimized.
            return symbol_base
        if arr_name == "s_prod":
            # Derived from data; not optimized.
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
    lines.append("")

    # ------------------------------------------------------------------
    # State layout
    # ------------------------------------------------------------------
    lines.append(r"\section{ODE State Layout}")
    lines.append(r"\begin{itemize}")
    lines.append(
        r"\item State vector: $\mathbf{y} = [R, S, A, K_{\mathrm{dyn}}, p]$ "
        r"with dimension $3K + M + N$."
    )
    lines.append(r"\item $R \in \mathbb{R}^{K}$: mRNA levels, fitted ODE state.")
    lines.append(
        r"\item $S \in \mathbb{R}^{K}$: protein signalling/activation fraction."
    )
    lines.append(r"\item $A \in \mathbb{R}^{K}$: protein abundance.")
    lines.append(
        r"\item $K_{\mathrm{dyn}} \in \mathbb{R}^{M}$: kinase activity fraction."
    )
    lines.append(
        r"\item $p \in \mathbb{R}^{N}$: relative phosphosite signal, nonnegative."
    )
    lines.append(r"\end{itemize}")
    lines.append("")

    # ------------------------------------------------------------------
    # Global definitions
    # ------------------------------------------------------------------
    lines.append(r"\section{Global Definitions}")
    lines.append(r"\begin{itemize}")
    lines.append(r"\item Input stimulus: $u(t) = \frac{1}{1 + e^{-t/0.1}}$.")
    lines.append(
        r"\item Bounded regulatory proxy: "
        r"$q_i = \frac{p_i}{1+p_i} \in [0,1)$."
    )

    bg = get_p("beta_g", 0, r"\beta_g")
    bl = get_p("beta_l", 0, r"\beta_l")

    lines.append(
        r"\item Crosstalk coupling: "
        rf"$\mathcal{{C}}_i = \tanh\left("
        rf"{bg}(C_g\mathbf{{q}})_i + {bl}(C_l\mathbf{{q}})_i"
        r"\right)$."
    )
    lines.append(
        r"\item $k_{\mathrm{act}}(t)$ and $s_{\mathrm{prod}}(t)$ are derived "
        r"from TF/kinase signals and are not optimised parameters."
    )
    lines.append(r"\end{itemize}")
    lines.append("")

    # ------------------------------------------------------------------
    # mRNA dynamics
    # ------------------------------------------------------------------
    lines.append(r"\section{mRNA Dynamics ($R$)}")
    lines.append(
        r"The mRNA state $R_g$ is an explicit ODE variable. "
        r"$k_{\mathrm{act}}(t)$ drives transcription and "
        r"$d_{\mathrm{deg}}$ controls decay."
    )
    lines.append(r"\begin{longtable}{" + _table_spec(eq_col) + r"}")
    lines.append(r"\textbf{Protein} & \textbf{Equation} \\ \hline")

    for k, prot in enumerate(proteins):
        prot_txt = _tex_text(prot)
        prot_id = _tex_math_id(prot)

        dd = get_p("d_deg", k, rf"d_{{\mathrm{{deg}}}}^{{{prot_id}}}")

        eq_R = (
            rf"\frac{{dR_{{{prot_id}}}}}{{dt}} = "
            rf"k_{{\mathrm{{act}}}}^{{{prot_id}}}(t) - "
            rf"{dd}\,R_{{{prot_id}}}"
        )

        lines.append(rf"\textbf{{{prot_txt}}} & ${eq_R}$ \\ \hline")

    lines.append(r"\end{longtable}")
    lines.append("")

    # ------------------------------------------------------------------
    # Protein signalling dynamics
    # ------------------------------------------------------------------
    lines.append(r"\section{Protein Dynamics ($S$, $A$)}")

    g_Sp = get_p("gamma_S_p", 0, r"\gamma_{Sp}")
    g_AS = get_p("gamma_A_S", 0, r"\gamma_{AS}")

    lines.append(r"\subsection{Protein Signalling State ($S$)}")
    lines.append(r"\begin{longtable}{" + _table_spec(eq_col) + r"}")
    lines.append(r"\textbf{Protein} & \textbf{Equation} \\ \hline")

    for k, prot in enumerate(proteins):
        prot_txt = _tex_text(prot)
        prot_id = _tex_math_id(prot)

        ka = get_p("k_act", k, rf"k_{{\mathrm{{act}}}}^{{{prot_id}}}")
        kd = get_p("k_deact", k, rf"k_{{\mathrm{{deact}}}}^{{{prot_id}}}")

        if mode == "symbolic":
            fb_term = rf"{g_Sp}\,\langle p\rangle_{{{prot_id}}}"
        else:
            fb_term = rf"{g_Sp}\,\mathrm{{mean}}(p)"

        drive = ["1", fb_term]
        if receptor_mask_prot[k]:
            drive.append(r"u(t)")

        drive_str = " + ".join(drive)

        eq_S = (
            rf"\frac{{dS_{{{prot_id}}}}}{{dt}} = "
            rf"{ka}\left[{drive_str}\right]"
            rf"\left(1-S_{{{prot_id}}}\right) - "
            rf"{kd}\,S_{{{prot_id}}}"
        )

        lines.append(rf"\textbf{{{prot_txt}}} & ${eq_S}$ \\ \hline")

    lines.append(r"\end{longtable}")
    lines.append("")

    # ------------------------------------------------------------------
    # Protein abundance dynamics
    # ------------------------------------------------------------------
    lines.append(r"\subsection{Protein Abundance ($A$)}")
    lines.append(
        r"Protein synthesis is gated by the mRNA state $R_g$: "
        r"$s_{\mathrm{eff}} = s_{\mathrm{prod}}(t) R_g "
        r"(1 + \gamma_{AS} S_g)$."
    )
    lines.append(r"\begin{longtable}{" + _table_spec(eq_col) + r"}")
    lines.append(r"\textbf{Protein} & \textbf{Equation} \\ \hline")

    for k, prot in enumerate(proteins):
        prot_txt = _tex_text(prot)
        prot_id = _tex_math_id(prot)

        sp = get_p("s_prod", k, rf"s_{{\mathrm{{prod}}}}^{{{prot_id}}}(t)")
        dd = get_p("d_deg", k, rf"d_{{\mathrm{{deg}}}}^{{{prot_id}}}")

        eq_A = (
            rf"\frac{{dA_{{{prot_id}}}}}{{dt}} = "
            rf"\mathrm{{clip}}\left("
            rf"{sp}\,R_{{{prot_id}}}"
            rf"\left(1 + {g_AS}\,S_{{{prot_id}}}\right), "
            rf"0, \infty"
            rf"\right) - "
            rf"{dd}\,A_{{{prot_id}}}"
        )

        lines.append(rf"\textbf{{{prot_txt}}} & ${eq_A}$ \\ \hline")

    lines.append(r"\end{longtable}")
    lines.append("")

    # ------------------------------------------------------------------
    # Kinase dynamics
    # ------------------------------------------------------------------
    lines.append(r"\section{Kinase Dynamics}")

    g_Knet = get_p("gamma_K_net", 0, r"\gamma_{Knet}")

    lines.append(r"\begin{longtable}{" + _table_spec(eq_col) + r"}")
    lines.append(r"\textbf{Kinase} & \textbf{Equation} \\ \hline")

    for m, kin in enumerate(kinases):
        kin_txt = _tex_text(kin)
        kin_id = _tex_math_id(kin)

        kka = get_p("kK_act", m, rf"k_{{K,\mathrm{{act}}}}^{{{kin_id}}}")
        kkd = get_p("kK_deact", m, rf"k_{{K,\mathrm{{deact}}}}^{{{kin_id}}}")

        u_terms = []

        feeding = np.where(R[m, :] > 0)[0]
        if len(feeding) > 0:
            if mode == "symbolic":
                u_terms.append(r"\sum_i w_{mi}\,p_i")
            else:
                top = feeding[:3]
                sub_terms = []
                for i_site in top:
                    site_id = _tex_math_id(sites[i_site])
                    weight = _fmt_float(float(R[m, i_site]), sig=3)
                    sub_terms.append(rf"{weight}\,p_{{{site_id}}}")
                sub_str = " + ".join(sub_terms)
                if len(feeding) > 3:
                    sub_str += r" + \cdots"
                u_terms.append(sub_str)

        if mode == "symbolic" or g_Knet != "0":
            u_terms.append(rf"{g_Knet}\,\nabla^2 K")

        if receptor_mask_kin[m]:
            u_terms.append(r"u(t)")

        u_str = " + ".join(u_terms) if u_terms else "0"

        eq_K = (
            rf"\frac{{dK_{{{kin_id}}}}}{{dt}} = "
            rf"{kka}\left[\delta + (1-\delta)\sigma\left({u_str}\right)\right]"
            rf"\left(1-K_{{{kin_id}}}\right) - "
            rf"{kkd}\,K_{{{kin_id}}},"
            r"\quad \delta=0.05,\quad \sigma=\mathrm{sigmoid}"
        )

        lines.append(rf"\textbf{{{kin_txt}}} & ${eq_K}$ \\ \hline")

    lines.append(r"\end{longtable}")
    lines.append("")

    # ------------------------------------------------------------------
    # Phosphosite dynamics
    # ------------------------------------------------------------------
    lines.append(r"\section{Phosphosite Dynamics}")

    if mechanism == "seq":
        lines.append(
            r"\textcolor{blue}{\textbf{Sequential model:}} "
            r"The gate uses the bounded predecessor proxy "
            r"$q_{i-1} = p_{i-1}/(1+p_{i-1})$. "
            r"A small leak $\epsilon$ may be used in the implementation to avoid "
            r"structural blocking.\\"
        )
    elif mechanism == "rand":
        lines.append(
            r"\textcolor{blue}{\textbf{Random/crowding-aware model:}} "
            r"The gate uses the mean bounded proxy $\bar{q}$ per protein, "
            r"where $q_i = p_i/(1+p_i)$.\\"
        )
    else:
        lines.append(
            r"\textcolor{blue}{\textbf{Distributive model:}} "
            r"Sites are updated independently except for kinase activity, "
            r"abundance scaling, and crosstalk coupling.\\"
        )

    prev_prot_idx = -1
    prev_site_id = ""

    for i, site in enumerate(sites):
        site_txt = _tex_text(site)
        site_id = _tex_math_id(site)

        koff = get_p("k_off", i, rf"k_{{\mathrm{{off}}}}^{{{site_id}}}")
        prot_idx = int(site_prot_idx[i])

        kin_idxs = np.where(K_site_kin[i, :] > 0)[0]
        k_on_terms = []

        for kid in kin_idxs:
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

        c_term = rf"\left(1 + \mathcal{{C}}_{{{site_id}}}\right)"

        mech_term = ""
        if mechanism == "seq":
            if prot_idx == prev_prot_idx and prev_site_id:
                mech_term = (
                    rf"\,q_{{{prev_site_id}}}"
                    r"\quad\left(q=\frac{p}{1+p}\right)"
                )
        elif mechanism == "rand":
            prot_id = _tex_math_id(proteins[prot_idx])
            mech_term = (
                rf"\,\frac{{1}}{{1+\bar{{q}}_{{{prot_id}}}}}"
                r"\quad\left(\bar{q}=\mathrm{mean\ bounded\ proxy}\right)"
            )

        v_raw = (
            rf"{c_term}\,"
            rf"\left[{k_on_str}\right]"
            rf"{mech_term}\,"
            rf"\left(1 + A_{{\mathrm{{site}}}}\right)"
        )

        lines.append(rf"\subsection*{{{site_txt}}}")

        # Define V_site separately to avoid huge nested fractions.
        lines.append(r"\begin{dmath*}")
        lines.append(rf"V_{{{site_id}}} = {v_raw}")
        lines.append(r"\end{dmath*}")

        lines.append(r"\begin{dmath*}")
        lines.append(
            rf"\frac{{dp_{{{site_id}}}}}{{dt}} = "
            rf"\frac{{V_{{{site_id}}}}}{{1 + V_{{{site_id}}}}} - "
            rf"{koff}\,p_{{{site_id}}}"
        )
        lines.append(r"\end{dmath*}")
        lines.append("")

        prev_prot_idx = prot_idx
        prev_site_id = site_id

    # No \end{longtable} here. Phosphosite equations are not inside longtable.
    lines.append(_latex_footer())

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# PDF compilation
# ---------------------------------------------------------------------------


def _compile_pdf(tex_path: str, outdir: str) -> None:
    """
    Compile LaTeX to PDF if pdflatex is available.

    Writes a .compile.log file next to the .tex file. This is important because
    LaTeX errors otherwise get hidden when subprocess output is suppressed.
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

    Appearance controls
    -------------------
    latex_title:
        PDF title.
    paper:
        e.g. "a4paper", "letterpaper".
    margin:
        e.g. "1in", "2cm".
    landscape:
        Use landscape geometry if True.
    font_pt:
        Document font size. <10 uses extarticle.
    eq_col:
        Equation column width for longtable sections.
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
