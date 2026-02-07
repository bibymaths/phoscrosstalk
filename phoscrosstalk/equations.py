"""
equations.py
Generates LaTeX documentation of the model's Ordinary Differential Equations (ODEs).
Creates both symbolic (parameter names) and numeric (fitted values) reports.
Supports Distributive, Sequential, and Random/Cooperative mechanisms.
"""
import os
import subprocess
import numpy as np
from phoscrosstalk.core_mechanisms import decode_theta
from phoscrosstalk.logger import get_logger

logger = get_logger()

# LaTeX Header
LATEX_PREAMBLE = r"""
\documentclass{article}
\usepackage{amsmath}
\usepackage{geometry}
\usepackage{longtable}
\usepackage{breqn}
\usepackage{xcolor}
\geometry{a4paper, margin=1in}
\title{Phospho-Network Model Equations}
\date{\today}
\begin{document}
\maketitle
"""


def _clean_tex(name):
    """Escapes underscores for LaTeX text mode."""
    return name.replace("_", r"\_")


def _generate_latex_source(
        mode, theta, proteins, kinases, sites,
        Cg, Cl, site_prot_idx, K_site_kin, R, L_alpha, kin_to_prot_idx,
        receptor_mask_prot, receptor_mask_kin, mechanism
):
    """
    Core logic to build the LaTeX string for the entire system.
    """
    K, M, N = len(proteins), len(kinases), len(sites)

    # 1. Decode Parameters
    if mode == 'numeric' and theta is not None:
        (k_act, k_deact, s_prod, d_deg, beta_g, beta_l, alpha,
         kK_act, kK_deact, k_off,
         gamma_S_p, gamma_A_S, gamma_A_p, gamma_K_net) = decode_theta(theta, K, M, N)

    # Parameter lookup helper
    def get_p(arr_name, idx, symbol_base):
        if mode == 'numeric' and theta is not None:
            val = 0.0
            if arr_name == 'k_act':
                val = k_act[idx]
            elif arr_name == 'k_deact':
                val = k_deact[idx]
            elif arr_name == 's_prod':
                val = s_prod[idx]
            elif arr_name == 'd_deg':
                val = d_deg[idx]
            elif arr_name == 'alpha':
                val = alpha[idx]
            elif arr_name == 'kK_act':
                val = kK_act[idx]
            elif arr_name == 'kK_deact':
                val = kK_deact[idx]
            elif arr_name == 'k_off':
                val = k_off[idx]
            elif arr_name == 'gamma_S_p':
                val = gamma_S_p
            elif arr_name == 'gamma_A_S':
                val = gamma_A_S
            elif arr_name == 'gamma_A_p':
                val = gamma_A_p
            elif arr_name == 'gamma_K_net':
                val = gamma_K_net
            elif arr_name == 'beta_g':
                val = beta_g
            elif arr_name == 'beta_l':
                val = beta_l

            # Formatting: if extremely small, show 0, else 3 sig figs
            if abs(val) < 1e-6: return "0"
            return f"{val:.3g}"
        else:
            return symbol_base

    lines = []
    lines.append(LATEX_PREAMBLE)
    lines.append(f"\\section*{{Model Configuration: {mode.capitalize()}}}")

    # Mechanism Description
    mech_desc = "Unknown"
    if mechanism == "dist":
        mech_desc = "Distributive (Independent sites)"
    elif mechanism == "seq":
        mech_desc = "Sequential (Ordered phosphorylation $p_1 \\to p_2$)"
    elif mechanism == "rand":
        mech_desc = "Random/Cooperative (Mean occupancy feedback)"

    lines.append(f"\\textbf{{Mechanism:}} {mech_desc} \\\\")
    if mode == 'numeric':
        lines.append(r"\textbf{Note:} Parameters are fitted values.")

    # --- GLOBAL ---
    lines.append(r"\section{Global Definitions}")
    lines.append(r"\begin{itemize}")
    lines.append(r"\item Input Stimulus: $u(t) = \frac{1}{1 + e^{-t/0.1}}$")

    bg = get_p('beta_g', 0, r"\beta_g")
    bl = get_p('beta_l', 0, r"\beta_l")
    lines.append(
        r"\item Crosstalk coupling: $\mathcal{C}_i = \tanh(" + bg + r"(C_g \mathbf{p})_i + " + bl + r"(C_l \mathbf{p})_i)$")
    lines.append(r"\end{itemize}")

    # --- PROTEINS ---
    lines.append(r"\section{Protein Dynamics}")
    g_Sp = get_p('gamma_S_p', 0, r"\gamma_{Sp}")
    g_AS = get_p('gamma_A_S', 0, r"\gamma_{AS}")

    lines.append(r"\begin{longtable}{l p{12cm}}")
    for k, prot in enumerate(proteins):
        clean_prot = _clean_tex(prot)
        ka = get_p('k_act', k, rf"k_{{act}}^{{{clean_prot}}}")
        kd = get_p('k_deact', k, rf"k_{{deact}}^{{{clean_prot}}}")

        # Feedback term
        if mode == 'symbolic':
            fb_term = rf"{g_Sp} \langle p \rangle_{{{clean_prot}}}"
        else:
            fb_term = rf"{g_Sp} \cdot \text{{mean}}(p)"

        drive = [r"1", fb_term]
        if receptor_mask_prot[k]: drive.append(r"u(t)")
        drive_str = " + ".join(drive)

        eq_S = rf"\frac{{dS}}{{dt}} = {ka} [{drive_str}] (1 - S) - {kd} S"
        lines.append(rf"\textbf{{{clean_prot}}} & ${eq_S}$ \\ \hline")
    lines.append(r"\end{longtable}")

    # --- KINASES ---
    lines.append(r"\section{Kinase Dynamics}")
    g_Knet = get_p('gamma_K_net', 0, r"\gamma_{Knet}")

    lines.append(r"\begin{longtable}{l p{12cm}}")
    for m, kin in enumerate(kinases):
        clean_kin = _clean_tex(kin)
        kka = get_p('kK_act', m, rf"k_{{act}}^{{{clean_kin}}}")
        kkd = get_p('kK_deact', m, rf"k_{{deact}}^{{{clean_kin}}}")

        # Build Input U
        u_terms = []

        # 1. Substrates (R matrix)
        feeding = np.where(R[m, :] > 0)[0]
        if len(feeding) > 0:
            if mode == 'symbolic':
                u_terms.append(r"\sum w_{sub} p_{sub}")
            else:
                top = feeding[:2]
                sub_str = " + ".join([rf"{R[m, i]:.2f}p_{{{_clean_tex(sites[i])}}}" for i in top])
                if len(feeding) > 2: sub_str += "..."
                u_terms.append(sub_str)

        # 2. Network
        if mode == 'symbolic' or float(g_Knet) != 0:
            u_terms.append(rf"{g_Knet} \nabla^2 K")

        if receptor_mask_kin[m]:
            u_terms.append(r"u(t)")

        u_str = " + ".join(u_terms) if u_terms else "0"

        eq_K = rf"\frac{{dK}}{{dt}} = {kka} \tanh({u_str}) (1 - K) - {kkd} K"
        lines.append(rf"\textbf{{{clean_kin}}} & ${eq_K}$ \\ \hline")
    lines.append(r"\end{longtable}")

    # --- PHOSPHOSITES (MECHANISM SPECIFIC) ---
    lines.append(r"\section{Phosphosite Dynamics}")

    # Add Mechanism Header
    if mechanism == "seq":
        lines.append(r"\textcolor{blue}{\textbf{Sequential Model:}} Rate depends on predecessor $p_{i-1}$. \\")
    elif mechanism == "rand":
        lines.append(
            r"\textcolor{blue}{\textbf{Cooperative Model:}} Rate depends on mean protein occupancy $\bar{p}$. \\")

    lines.append(r"\begin{longtable}{l p{12cm}}")

    # Track previous site for Sequential logic
    # Assumes 'sites' list is ordered by protein!
    prev_prot_idx = -1
    prev_site_tex = ""

    for i, site in enumerate(sites):
        clean_site = _clean_tex(site)
        koff = get_p('k_off', i, rf"k_{{off}}^{{{clean_site}}}")
        prot_idx = site_prot_idx[i]

        # 1. Calculate Base Kinase Drive (k_on_eff)
        kin_idxs = np.where(K_site_kin[i, :] > 0)[0]
        k_on_terms = []
        for kid in kin_idxs:
            w = K_site_kin[i, kid]
            k_name = _clean_tex(kinases[kid])
            a_val = get_p('alpha', kid, rf"\alpha_{{{k_name}}}")

            if mode == 'symbolic':
                term = rf"w \cdot {a_val} K_{{{k_name}}}"
            else:
                eff_w = w * float(a_val)
                term = rf"{eff_w:.3f} K_{{{k_name}}}"
            k_on_terms.append(term)

        k_on_str = " + ".join(k_on_terms) if k_on_terms else "0"

        # 2. Crosstalk
        c_term = rf"(1 + \mathcal{{C}}_{{{clean_site}}})"

        # 3. Apply Mechanism Logic
        mech_term = ""

        if mechanism == "dist":
            # Distributive: No extra term
            mech_term = ""

        elif mechanism == "seq":
            # Sequential: Check if same protein as previous
            if prot_idx == prev_prot_idx:
                # Gated by previous site
                mech_term = rf" \cdot \underbrace{{p_{{{prev_site_tex}}}}}_{{\text{{gate}}}}"
            else:
                # First site on protein: No gate (or gate=1)
                mech_term = ""

        elif mechanism == "rand":
            # Random/Coop: Mean occupancy term
            prot_name = _clean_tex(proteins[prot_idx])
            mech_term = rf" \cdot (1 + \langle p \rangle_{{{prot_name}}})"

        # 4. Assemble V_raw
        v_raw = rf"{c_term} [{k_on_str}] {mech_term} (1 - p_{{{clean_site}}})"

        eq_p = (
            rf"\frac{{dp_{{{clean_site}}}}}{{dt}} &= "
            rf"\frac{{{v_raw}}}{{1 + |{v_raw}|}} - "
            rf"\frac{{{koff} p_{{{clean_site}}}}}{{1 + {koff} p_{{{clean_site}}}}}"
        )

        lines.append(rf"\textbf{{{clean_site}}} & $\displaystyle {eq_p}$ \\ \hline")

        # Update trackers for sequential
        prev_prot_idx = prot_idx
        prev_site_tex = clean_site

    lines.append(r"\end{longtable}")
    lines.append(r"\end{document}")

    return "\n".join(lines)


def _compile_pdf(tex_path, outdir):
    """Compiles LaTeX to PDF."""
    try:
        subprocess.run(["pdflatex", "--version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        logger.info(f"    -> Compiling {os.path.basename(tex_path)}...")
        subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-output-directory", outdir, tex_path],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        pdf_name = os.path.basename(tex_path).replace(".tex", ".pdf")
        if os.path.exists(os.path.join(outdir, pdf_name)):
            logger.success(f"    -> Generated {pdf_name}")
    except Exception:
        logger.warning("    -> 'pdflatex' not found or failed. PDF not generated.")


def generate_equations_report(outdir, theta_opt,
                              proteins, kinases, sites,
                              Cg, Cl, site_prot_idx, K_site_kin, R, L_alpha, kin_to_prot_idx,
                              receptor_mask_prot, receptor_mask_kin, mechanism):
    """
    Main entry point. Generates reports for the FITTED mechanism (numeric)
    and SYMBOLIC reports for ALL mechanisms (for documentation).
    """
    logger.info("[*] Generating Model Equation Reports...")

    eq_dir = os.path.join(outdir, "equations")
    os.makedirs(eq_dir, exist_ok=True)

    # 1. Fitted Numeric Report (Specific to run mechanism)
    tex_num = _generate_latex_source(
        'numeric', theta_opt, proteins, kinases, sites,
        Cg, Cl, site_prot_idx, K_site_kin, R, L_alpha, kin_to_prot_idx,
        receptor_mask_prot, receptor_mask_kin, mechanism
    )
    path_num = os.path.join(eq_dir, f"model_equations_fitted_{mechanism}.tex")
    with open(path_num, "w") as f: f.write(tex_num)
    _compile_pdf(path_num, eq_dir)

    # 2. Symbolic Reports (For all mechanisms, as reference)
    for mech in ['dist', 'seq', 'rand']:
        tex_sym = _generate_latex_source(
            'symbolic', None, proteins, kinases, sites,
            Cg, Cl, site_prot_idx, K_site_kin, R, L_alpha, kin_to_prot_idx,
            receptor_mask_prot, receptor_mask_kin, mech
        )
        path_sym = os.path.join(eq_dir, f"model_equations_symbolic_{mech}.tex")
        with open(path_sym, "w") as f: f.write(tex_sym)
        _compile_pdf(path_sym, eq_dir)