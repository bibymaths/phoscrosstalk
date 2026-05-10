# SPDX-License-Identifier: MIT
"""
Diagnostic-only module for jaxpr inspection, parsing, and report saving.

Captures JAX computation graphs (jaxprs) at key flow steps and renders
them as self-contained HTML and optional PDF reports.

This module is **purely diagnostic** — it never modifies any computation,
optimizer logic, loss semantics, or fitted outputs.  All entry points are
no-ops when `save_jaxpr_reports = false` (the default) in config.toml.

Usage::

    from phoscrosstalk.jaxpr_reporter import capture_and_save_jaxpr
    from pathlib import Path

    capture_and_save_jaxpr(
        fn=residuals_fn,
        example_args=(theta_example, None),
        step_label="multistart_residuals_fn",
        module_label="phoscrosstalk.optimization",
        out_dir=Path(output_dir) / "jaxpr_reports",
    )

PDF output requires WeasyPrint (``pip install weasyprint``).  When
WeasyPrint is unavailable the module degrades gracefully — HTML is always
saved and a ``WARNING`` is emitted for the skipped PDF.
"""

from __future__ import annotations

import datetime
import html
import json
import logging
from pathlib import Path

import jax

logger = logging.getLogger("phoscrosstalk.jaxpr_reporter")


# ---------------------------------------------------------------------------
# Jaxpr parsing
# ---------------------------------------------------------------------------


def parse_jaxpr(closed_jaxpr, fn_name: str) -> dict:
    """Parse a ``ClosedJaxpr`` into a plain Python dictionary.

    Parameters
    ----------
    closed_jaxpr : jax.core.ClosedJaxpr
        Result of ``jax.make_jaxpr(fn)(*example_args)``.
    fn_name : str
        Human-readable function name to embed in the report.

    Returns
    -------
    dict
        Structured metadata including input/output shapes, all equations,
        and flags for higher-order primitives (cond, while, scan, pjit).
    """
    try:
        from jax.extend import core  # JAX ≥ 0.4
    except ImportError:  # pragma: no cover
        from jax import core  # type: ignore[no-redef]

    jaxpr = closed_jaxpr.jaxpr

    def _fmt_var(v) -> dict:
        if isinstance(v, core.Literal):
            return {"kind": "literal", "value": str(v.val), "type": str(v.aval), "name": ""}
        return {"kind": "var", "name": str(v), "type": str(v.aval), "value": ""}

    invars = [_fmt_var(v) for v in jaxpr.invars]
    outvars = [_fmt_var(v) for v in jaxpr.outvars]
    constvars = [_fmt_var(v) for v in jaxpr.constvars]

    equations = []
    for eqn in jaxpr.eqns:
        equations.append({
            "primitive": str(eqn.primitive),
            "invars": [_fmt_var(v) for v in eqn.invars],
            "outvars": [_fmt_var(v) for v in eqn.outvars],
            "params": {k: str(v) for k, v in eqn.params.items()},
        })

    primitives_used = sorted(set(e["primitive"] for e in equations))
    return {
        "fn_name": fn_name,
        "n_invars": len(invars),
        "n_outvars": len(outvars),
        "n_constvars": len(constvars),
        "n_eqns": len(equations),
        "invars": invars,
        "outvars": outvars,
        "constvars": constvars,
        "equations": equations,
        "raw_jaxpr": str(closed_jaxpr.jaxpr),
        "primitives_used": primitives_used,
        "has_cond": any(e["primitive"] == "cond" for e in equations),
        "has_while": any(e["primitive"] == "while" for e in equations),
        "has_scan": any(e["primitive"] == "scan" for e in equations),
        "has_pjit": any(e["primitive"] in ("pjit", "jit") for e in equations),
    }


# ---------------------------------------------------------------------------
# HTML rendering
# ---------------------------------------------------------------------------

_CSS = """
<style>
  body{font-family:monospace;font-size:13px;background:#1c1b19;color:#cdccca;margin:0;padding:0}
  .page{max-width:1100px;margin:auto;padding:24px 32px}
  h1{font-size:18px;color:#4f98a3;border-bottom:1px solid #393836;padding-bottom:8px;margin-bottom:16px}
  h2{font-size:14px;color:#6daa45;margin:20px 0 8px}
  .meta{display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:8px;margin-bottom:20px}
  .kv{background:#201f1d;border:1px solid #393836;border-radius:4px;padding:8px 12px}
  .kv .k{color:#797876;font-size:11px;text-transform:uppercase;letter-spacing:.05em}
  .kv .v{color:#cdccca;font-size:14px;font-weight:bold;margin-top:2px}
  .badge{display:inline-block;padding:2px 8px;border-radius:10px;font-size:11px;margin:2px}
  .badge.yes{background:#3a4435;color:#6daa45}
  .badge.no{background:#22211f;color:#5a5957}
  table{width:100%;border-collapse:collapse;font-size:12px;margin-bottom:20px}
  th{background:#201f1d;color:#797876;text-align:left;padding:6px 10px;font-weight:normal;
     text-transform:uppercase;letter-spacing:.05em;border-bottom:1px solid #393836}
  td{padding:6px 10px;border-bottom:1px solid #262523;vertical-align:top;word-break:break-all}
  tr:hover td{background:#201f1d}
  .prim{color:#4f98a3}
  .var{color:#a86fdf}
  .lit{color:#fdab43}
  .raw{background:#171614;border:1px solid #393836;border-radius:4px;padding:14px;
       white-space:pre-wrap;font-size:11px;color:#797876;overflow-x:auto;max-height:400px;
       overflow-y:auto;margin-bottom:20px}
  .pill{display:inline-block;background:#313b3b;color:#4f98a3;border-radius:3px;
        padding:1px 6px;font-size:11px;margin:1px}
</style>
"""


def _var_html(v: dict) -> str:
    if v["kind"] == "literal":
        label = html.escape(v["value"])
        cls = "lit"
    else:
        label = html.escape(v["name"])
        cls = "var"
    return f'<span class="{cls}">{label}</span> <span style="color:#5a5957">:{html.escape(v["type"])}</span>'


def render_html_report(parsed: dict, step_label: str, module_label: str) -> str:
    """Render a parsed jaxpr dict into a standalone HTML string.

    Parameters
    ----------
    parsed : dict
        Output of :func:`parse_jaxpr`.
    step_label : str
        Human-readable step name, e.g. ``"multistart_residuals_fn"``.
    module_label : str
        Module name, e.g. ``"phoscrosstalk.optimization"``.

    Returns
    -------
    str
        Self-contained HTML document.
    """
    p = parsed
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    primitives_html = " ".join(
        f'<span class="pill">{html.escape(pr)}</span>' for pr in p["primitives_used"]
    )

    def badge(flag: bool, label: str) -> str:
        cls = "yes" if flag else "no"
        mark = "✓" if flag else "✗"
        return f'<span class="badge {cls}">{mark} {html.escape(label)}</span>'

    def var_table(vars_list: list, title: str) -> str:
        rows = "".join(
            f"<tr><td>{i}</td><td>{_var_html(v)}</td></tr>"
            for i, v in enumerate(vars_list)
        )
        return (
            f"<h2>{html.escape(title)} ({len(vars_list)})</h2>"
            f"<table><thead><tr><th>#</th><th>Variable</th></tr></thead>"
            f"<tbody>{rows}</tbody></table>"
        )

    eqn_rows = ""
    for i, e in enumerate(p["equations"]):
        out_str = ", ".join(_var_html(v) for v in e["outvars"])
        in_str = ", ".join(_var_html(v) for v in e["invars"])
        params_str = (
            "; ".join(
                f"{html.escape(k)}={html.escape(str(v))[:80]}"
                for k, v in e["params"].items()
            )
            if e["params"]
            else "&mdash;"
        )
        eqn_rows += (
            f"<tr><td>{i}</td>"
            f"<td><span class='prim'>{html.escape(e['primitive'])}</span></td>"
            f"<td>{out_str}</td><td>{in_str}</td>"
            f"<td style='color:#5a5957;font-size:11px'>{params_str}</td></tr>"
        )

    eqns_table = (
        f"<h2>Equations ({p['n_eqns']})</h2>"
        f"<table><thead><tr><th>#</th><th>Primitive</th><th>Out vars</th>"
        f"<th>In vars</th><th>Params</th></tr></thead>"
        f"<tbody>{eqn_rows}</tbody></table>"
    )

    return (
        f'<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">'
        f'<title>jaxpr: {html.escape(step_label)}</title>{_CSS}</head>'
        f'<body><div class="page">'
        f'<h1>Jaxpr Report &mdash; {html.escape(step_label)}</h1>'
        f'<div class="meta">'
        f'<div class="kv"><div class="k">Module</div><div class="v">{html.escape(module_label)}</div></div>'
        f'<div class="kv"><div class="k">Function</div><div class="v">{html.escape(p["fn_name"])}</div></div>'
        f'<div class="kv"><div class="k">Equations</div><div class="v">{p["n_eqns"]}</div></div>'
        f'<div class="kv"><div class="k">In vars</div><div class="v">{p["n_invars"]}</div></div>'
        f'<div class="kv"><div class="k">Out vars</div><div class="v">{p["n_outvars"]}</div></div>'
        f'<div class="kv"><div class="k">Const vars</div><div class="v">{p["n_constvars"]}</div></div>'
        f'<div class="kv"><div class="k">Generated</div><div class="v">{html.escape(ts)}</div></div>'
        f'</div>'
        f'<h2>Higher-Order Primitives</h2>'
        f'<p>{badge(p["has_cond"], "cond")} {badge(p["has_while"], "while")} '
        f'{badge(p["has_scan"], "scan")} {badge(p["has_pjit"], "pjit/jit")}</p>'
        f'<h2>All Primitives Used</h2><p>{primitives_html}</p>'
        f'{var_table(p["invars"], "Input Variables")}'
        f'{var_table(p["outvars"], "Output Variables")}'
        f'{var_table(p["constvars"], "Constant Variables")}'
        f'{eqns_table}'
        f'<h2>Raw Jaxpr String</h2>'
        f'<div class="raw">{html.escape(p["raw_jaxpr"])}</div>'
        f'</div></body></html>'
    )


# ---------------------------------------------------------------------------
# PDF rendering (WeasyPrint, graceful degradation)
# ---------------------------------------------------------------------------


def render_pdf_from_html(html_str: str, out_path: Path) -> None:
    """Render an HTML string to PDF using WeasyPrint.

    If WeasyPrint is not installed a ``WARNING`` is logged and the PDF is
    silently skipped — the HTML report is unaffected.

    Parameters
    ----------
    html_str : str
        Full HTML document string (output of :func:`render_html_report`).
    out_path : Path
        Destination path for the PDF file.
    """
    try:
        from weasyprint import HTML as WeasyprintHTML  # type: ignore[import]
        WeasyprintHTML(string=html_str).write_pdf(str(out_path))
    except ImportError:
        logger.warning(
            "[jaxpr_reporter]  weasyprint not installed — PDF output skipped for %s",
            out_path.name,
        )
    except Exception as exc:  # pragma: no cover
        logger.warning(
            "[jaxpr_reporter]  PDF generation failed for %s: %s",
            out_path.name, exc,
        )


# ---------------------------------------------------------------------------
# Index page
# ---------------------------------------------------------------------------


def write_index_html(out_dir: Path, all_parsed: dict[str, dict]) -> None:
    """Write a ``jaxpr_index.html`` linking all step reports.

    Parameters
    ----------
    out_dir : Path
        Directory containing the individual step HTML/PDF files.
    all_parsed : dict[str, dict]
        Mapping of ``slug → parsed`` as returned by
        :func:`capture_and_save_jaxpr` (the ``slug`` equals the sanitised
        ``step_label``).
    """
    rows = ""
    for slug, parsed in all_parsed.items():
        rows += (
            f"<tr>"
            f'<td><a href="{html.escape(slug)}.html" style="color:#4f98a3">'
            f'{html.escape(parsed["fn_name"])}</a></td>'
            f'<td>{parsed["n_eqns"]}</td>'
            f'<td>{"&#10003;" if parsed["has_scan"] else "&mdash;"}</td>'
            f'<td>{"&#10003;" if parsed["has_cond"] else "&mdash;"}</td>'
            f'<td>{"&#10003;" if parsed["has_while"] else "&mdash;"}</td>'
            f'<td><a href="{html.escape(slug)}.pdf" style="color:#797876">PDF</a></td>'
            f"</tr>"
        )
    index_html = (
        f'<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">'
        f"<title>Jaxpr Report Index</title>{_CSS}</head>"
        f'<body><div class="page">'
        f"<h1>Jaxpr Report Index</h1>"
        f"<table><thead><tr><th>Function</th><th>Equations</th>"
        f"<th>scan</th><th>cond</th><th>while</th><th>PDF</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>"
        f"</div></body></html>"
    )
    (out_dir / "jaxpr_index.html").write_text(index_html, encoding="utf-8")
    logger.info("[jaxpr_reporter]  index saved  path=%s", out_dir / "jaxpr_index.html")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def capture_and_save_jaxpr(
        fn,
        example_args: tuple,
        step_label: str,
        module_label: str,
        out_dir: Path,
        save_json: bool = True,
) -> dict:
    """Trace ``fn`` with ``example_args``, parse the jaxpr, and save reports.

    This function calls ``jax.make_jaxpr(fn)(*example_args)`` once (a
    Python-level tracing pass — not a JAX-compiled execution).  It must
    **not** be called inside a JIT body or ``jax.lax.while_loop``/``scan``.

    Parameters
    ----------
    fn : callable
        The function to trace.  Must be traceable by ``jax.make_jaxpr``.
    example_args : tuple
        Example inputs matching the runtime shapes and dtypes.  Wrap Python
        scalars as ``jnp.array(value)`` to avoid shape/dtype mismatches.
    step_label : str
        Human-readable label for the step, e.g. ``"mechanistic_residuals_fn"``.
    module_label : str
        Module name for the report header, e.g. ``"phoscrosstalk.fit"``.
    out_dir : Path
        Directory to save reports (created if absent).
    save_json : bool
        When ``True`` (default) also save the parsed dict as JSON.

    Returns
    -------
    dict
        Parsed jaxpr metadata (output of :func:`parse_jaxpr`).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    slug = step_label.replace(" ", "_").replace("/", "-").lower()
    fn_name = getattr(fn, "__name__", str(fn))

    logger.info(
        "[jaxpr_reporter]  tracing  step=%s  fn=%s", step_label, fn_name,
    )

    try:
        closed_jaxpr = jax.make_jaxpr(fn)(*example_args)
    except Exception as exc:
        logger.warning(
            "[jaxpr_reporter]  tracing failed  step=%s  error=%s", step_label, exc,
        )
        return {}

    parsed = parse_jaxpr(closed_jaxpr, fn_name)

    logger.info(
        "[jaxpr_reporter]  parsed  step=%s  n_eqns=%d  n_invars=%d"
        "  has_scan=%s  has_cond=%s  has_while=%s",
        step_label,
        parsed["n_eqns"],
        parsed["n_invars"],
        parsed["has_scan"],
        parsed["has_cond"],
        parsed["has_while"],
    )

    html_str = render_html_report(parsed, step_label, module_label)
    html_path = out_dir / f"{slug}.html"
    pdf_path = out_dir / f"{slug}.pdf"

    html_path.write_text(html_str, encoding="utf-8")
    logger.info("[jaxpr_reporter]  saved HTML  path=%s", html_path)

    render_pdf_from_html(html_str, pdf_path)
    logger.info("[jaxpr_reporter]  saved PDF   path=%s", pdf_path)

    if save_json:
        json_path = out_dir / f"{slug}.json"
        json_path.write_text(json.dumps(parsed, indent=2), encoding="utf-8")

    return parsed
