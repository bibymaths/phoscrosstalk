"""
app.py
PhosCrosstalk Streamlit/Plotly interactive dashboard.

Loads one completed run from a user-provided results directory and provides
interactive analysis from existing saved run artefacts. All plots use Plotly;
no pre-rendered PNG files are displayed.

Usage:
    streamlit run phoscrosstalk/app.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

try:
    import networkx as _nx

    _HAS_NX = True
except ImportError:  # pragma: no cover
    _HAS_NX = False
    _nx = None

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phoscrosstalk.config import ModelDims
from phoscrosstalk.core_mechanisms import decode_theta
from phoscrosstalk.dashboard_io import (
    build_dashboard_cache,
    extract_time_axis,
    get_time_vals,
    load_dashboard_manifest,
    load_derived_rates,
    load_entity_labels,
    load_fit_timeseries,
    load_fitted_params,
    load_internal_states,
    load_knockout_outputs,
    load_preopt_snapshot,
    load_run_config,
    load_sensitivity_outputs,
    load_steadystate_outputs,
    validate_run_directory,
)
from phoscrosstalk.knockouts import run_live_knockout
from phoscrosstalk.simulation import build_full_A0, simulate_p_scipy

# ──────────────────────────────────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PhosCrosstalk Explorer",
    page_icon="\U0001F9EC",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ──────────────────────────────────────────────────────────────────────────
# Cached loaders
# ──────────────────────────────────────────────────────────────────────────


@st.cache_data(show_spinner="Validating run directory…")
def _validate(results_dir: str):
    return validate_run_directory(results_dir)


@st.cache_data(show_spinner="Loading run config…")
def _load_config(results_dir: str):
    return load_run_config(results_dir)


@st.cache_data(show_spinner="Loading entity labels…")
def _load_labels(results_dir: str):
    return load_entity_labels(results_dir)


@st.cache_data(show_spinner="Loading preopt snapshot…")
def _load_snap(results_dir: str):
    return load_preopt_snapshot(results_dir)


@st.cache_data(show_spinner="Loading fitted parameters…")
def _load_params(results_dir: str):
    return load_fitted_params(results_dir)


@st.cache_data(show_spinner="Loading fit timeseries…")
def _load_fit_ts(results_dir: str):
    return load_fit_timeseries(results_dir)


@st.cache_data(show_spinner="Loading internal states…")
def _load_int_states(results_dir: str):
    return load_internal_states(results_dir)


@st.cache_data(show_spinner="Loading derived rates…")
def _load_dr(results_dir: str):
    return load_derived_rates(results_dir)


@st.cache_data(show_spinner="Loading knockout outputs…")
def _load_ko(results_dir: str):
    return load_knockout_outputs(results_dir)


@st.cache_data(show_spinner="Loading sensitivity outputs…")
def _load_sens(results_dir: str):
    return load_sensitivity_outputs(results_dir)


@st.cache_data(show_spinner="Loading steady-state outputs…")
def _load_ss(results_dir: str):
    return load_steadystate_outputs(results_dir)


@st.cache_data(show_spinner="Running forward simulation…")
def _run_simulation(results_dir: str, t_max: float, num_points: int, mechanism: str):
    """Forward simulation from fitted params and preopt snapshot."""
    snap = _load_snap(results_dir)
    params = _load_params(results_dir)
    snap_labels = _load_labels(results_dir)

    if snap is None or params is None:
        return None

    K = len(snap_labels["proteins"])
    M = len(snap_labels["kinases"])
    N = len(snap_labels["sites"])
    ModelDims.set_dims(K, M, N)

    theta = params["theta"]
    t_fine = np.linspace(0.0, t_max, num_points)

    A_scaled = snap.get("A_scaled", np.empty((0, 0)))
    A0 = np.zeros((K, len(t_fine)), dtype=float)
    if A_scaled is not None and A_scaled.size > 0:
        a_prots = snap_labels.get("A_proteins", [])
        prot_map = {p: i for i, p in enumerate(snap_labels["proteins"])}
        for k, aname in enumerate(a_prots):
            if aname in prot_map and k < A_scaled.shape[0] and A_scaled.shape[1] > 0:
                A0[prot_map[aname], 0] = A_scaled[k, 0]

    P_sim, A_sim, S_sim, Kdyn_sim = simulate_p_scipy(
        t_fine,
        snap["P_scaled"],
        A0,
        theta,
        snap["Cg"],
        snap["Cl"],
        snap["site_prot_idx"],
        snap["K_site_kin"],
        snap["R"],
        snap["L_alpha"],
        snap["kin_to_prot_idx"],
        snap["receptor_mask_prot"],
        snap["receptor_mask_kin"],
        mechanism,
        full_output=True,
    )
    return {"t": t_fine, "P_sim": P_sim, "A_sim": A_sim, "S_sim": S_sim, "Kdyn_sim": Kdyn_sim}


# ──────────────────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────────────────
st.sidebar.title("\U0001F9EC PhosCrosstalk Explorer")

results_dir = st.sidebar.text_input(
    "Results directory path",
    value="test_results_dist",  # example; change to your run directory
    help="Path to a completed PhosCrosstalk run directory.",
).strip()

if not os.path.isdir(results_dir):
    st.error(
        f"Directory **{results_dir!r}** not found. "
        "Enter a valid path to a completed run directory."
    )
    st.stop()

val = _validate(results_dir)

if not val["valid"]:
    st.sidebar.error("\u26d4 Run invalid")
    st.error(
        "Missing required artefacts: "
        + ", ".join(val["missing_required"])
        + f"\n\nChecked: `{results_dir}`"
    )
    st.stop()

st.sidebar.success("\u2705 Run valid")

run_config = _load_config(results_dir)
labels = _load_labels(results_dir)
snap = _load_snap(results_dir)
params = _load_params(results_dir)

proteins = labels.get("proteins", [])
kinases = labels.get("kinases", [])
sites = labels.get("sites", [])

meta = snap.get("meta", {}) if snap else {}
mechanism = meta.get("mechanism", run_config.get("mechanism", "dist"))

st.sidebar.markdown(f"**Mechanism:** `{mechanism}`")
st.sidebar.markdown(
    f"K={len(proteins)} proteins · M={len(kinases)} kinases · N={len(sites)} sites"
)

selected_protein = (
    st.sidebar.selectbox("Protein", proteins, index=0) if proteins else None
)

site_prot_idx_arr = snap.get("site_prot_idx", np.array([])) if snap else np.array([])
if selected_protein and len(site_prot_idx_arr) > 0:
    prot_idx_sel = proteins.index(selected_protein)
    sites_for_prot = [
        sites[i]
        for i in np.where(np.asarray(site_prot_idx_arr) == prot_idx_sel)[0]
    ]
else:
    sites_for_prot = []

selected_sites = st.sidebar.multiselect(
    "Phosphosite(s)", sites_for_prot, default=sites_for_prot[:3]
)
selected_kinases = st.sidebar.multiselect(
    "Kinase(s)", kinases, default=kinases[:3] if kinases else []
)

t_orig = get_time_vals(snap)
t_max_data = float(t_orig[-1]) if len(t_orig) > 0 else 240.0

time_range = st.sidebar.slider(
    "Time range (min)",
    0.0, float(t_max_data * 2),
    (0.0, float(t_max_data)),
    step=1.0,
)

show_obs = st.sidebar.toggle("Show observed data", value=True)
use_logx = st.sidebar.toggle("Log x-axis (where applicable)", value=False)

if st.sidebar.button("\U0001F504 Rebuild dashboard cache"):
    with st.spinner("Rebuilding cache…"):
        cache_result = build_dashboard_cache(results_dir, force=True)
    st.sidebar.success(
        f"Cache rebuilt: {len(cache_result['created'])} files created, "
        f"{len(cache_result['errors'])} errors."
    )
    if cache_result["errors"]:
        for e in cache_result["errors"]:
            st.sidebar.warning(e)
    st.rerun()

manifest = load_dashboard_manifest(results_dir)
if manifest is None:
    with st.spinner("Building dashboard cache…"):
        build_dashboard_cache(results_dir, force=False)
    manifest = load_dashboard_manifest(results_dir)


# ──────────────────────────────────────────────────────────────────────────
# Tabs
# ──────────────────────────────────────────────────────────────────────────
(
    tab_overview,
    tab_fit,
    tab_states,
    tab_rates,
    tab_sim,
    tab_ko,
    tab_sens,
    tab_ss,
    tab_net,
) = st.tabs([
    "A \u00b7 Overview",
    "B \u00b7 Fit Explorer",
    "C \u00b7 Internal States",
    "D \u00b7 Derived Rates",
    "E \u00b7 Forward Simulation",
    "F \u00b7 Live Knockout",
    "G \u00b7 Sensitivity",
    "H \u00b7 Steady-State",
    "I \u00b7 Network",
])


# ══════════════════════════════════════════════════════════════════════════
# A · RUN OVERVIEW
# ══════════════════════════════════════════════════════════════════════════
with tab_overview:
    st.header("Run Overview")
    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader("Run configuration")
        if run_config:
            st.json(run_config, expanded=False)
        else:
            st.warning("run_config.json not found or empty.")

        st.subheader("Model dimensions")
        st.markdown(
            f"- **Proteins (K):** {len(proteins)}\n"
            f"- **Kinases (M):** {len(kinases)}\n"
            f"- **Phosphosites (N):** {len(sites)}\n"
            f"- **Time points:** {len(t_orig)}\n"
            f"- **Mechanism:** `{mechanism}`"
        )

        st.subheader("Data governance manifest")
        if manifest:
            st.markdown(
                f"- Created: `{manifest.get('created_at', 'unknown')}`\n"
                f"- Schema version: `{manifest.get('schema_version', 'unknown')}`\n"
                f"- Files created: {len(manifest.get('created_files', []))}\n"
                f"- Errors: {len(manifest.get('errors', []))}"
            )
            if manifest.get("errors"):
                with st.expander("Cache errors"):
                    for e in manifest["errors"]:
                        st.warning(e)
        else:
            st.info("Dashboard cache not yet built.")

    with col_r:
        st.subheader("Available artefacts")
        for name in val["present_optional"]:
            st.markdown(f"\u2705 `{name}`")
        for name in val["missing_optional"]:
            st.markdown(f"\u2b1c `{name}` *(absent)*")

        st.subheader("Parameter summaries")
        for fname in [
            "parameter_summary_proteins.tsv",
            "parameter_summary_kinases.tsv",
            "parameter_summary_sites.tsv",
        ]:
            p = os.path.join(results_dir, fname)
            if os.path.exists(p):
                with st.expander(fname):
                    try:
                        st.dataframe(pd.read_csv(p, sep="\t"), use_container_width=True)
                    except Exception as exc:
                        st.warning(f"Could not read {fname}: {exc}")

        global_txt = os.path.join(results_dir, "parameter_summary_global.txt")
        if os.path.exists(global_txt):
            with st.expander("parameter_summary_global.txt"):
                with open(global_txt, encoding="utf-8") as fh:
                    st.text(fh.read())

        st.subheader("Decoded parameter distributions")
        if snap is not None and params is not None:
            K = len(proteins)
            M = len(kinases)
            N = len(sites)
            if K > 0 and M > 0 and N > 0:
                ModelDims.set_dims(K, M, N)
                try:
                    theta = params["theta"]
                    dec = decode_theta(theta, K, M, N)
                    (k_deact, d_deg, beta_g, beta_l, alpha,
                     kK_act, kK_deact, k_off,
                     gamma_S_p, gamma_A_S, gamma_A_p, gamma_K_net) = dec

                    st.markdown(
                        f"**beta_g** = {beta_g:.4f} &nbsp; "
                        f"**beta_l** = {beta_l:.4f}  \n"
                        f"**\u03b3_S_p** = {gamma_S_p:.3f} &nbsp; "
                        f"**\u03b3_A_S** = {gamma_A_S:.3f} &nbsp; "
                        f"**\u03b3_A_p** = {gamma_A_p:.3f} &nbsp; "
                        f"**\u03b3_K_net** = {gamma_K_net:.3f}"
                    )
                    df_kin_dec = pd.DataFrame({
                        "Kinase": kinases,
                        "alpha": alpha,
                        "kK_act": kK_act,
                        "kK_deact": kK_deact,
                    })
                    fig_dec = px.box(
                        df_kin_dec.melt(id_vars="Kinase"),
                        x="variable", y="value", points="all",
                        title="Kinase parameter distributions",
                    )
                    st.plotly_chart(fig_dec, use_container_width=True)
                except Exception as exc:
                    st.warning(f"Could not decode parameters: {exc}")


# ══════════════════════════════════════════════════════════════════════════
# B · FIT EXPLORER
# ══════════════════════════════════════════════════════════════════════════
with tab_fit:
    st.header("Fit Explorer")
    st.caption("Source: `fit_timeseries.tsv`, optional `mrna_fit_timeseries.tsv`")

    df_ft = _load_fit_ts(results_dir)
    if df_ft is None:
        st.warning("fit_timeseries.tsv not found.")
    else:
        sim_cols = [c for c in df_ft.columns if c.startswith("sim_t")]
        data_cols = [c for c in df_ft.columns if c.startswith("data_t")]
        t_vals = extract_time_axis(df_ft, prefix="sim_t")
        t_axis = t_orig if len(t_vals) == len(t_orig) > 0 else t_vals

        t_mask = (t_axis >= time_range[0]) & (t_axis <= time_range[1]) if len(t_axis) > 0 else np.ones(len(t_vals), dtype=bool)
        sim_cols_f = [c for c, m in zip(sim_cols, t_mask) if m]
        data_cols_f = [c for c, m in zip(data_cols, t_mask) if m]
        t_filt = t_axis[t_mask] if len(t_axis) > 0 else t_vals

        if selected_protein:
            df_prot_row = df_ft[(df_ft["Type"] == "ProteinAbundance") & (df_ft["Protein"] == selected_protein)]
            df_site_rows = df_ft[(df_ft["Type"] == "Phosphosite") & (df_ft["Protein"] == selected_protein)]

            mrna_path = os.path.join(results_dir, "mrna_fit_timeseries.tsv")
            df_mrna = None
            if os.path.exists(mrna_path):
                try:
                    df_mrna_all = pd.read_csv(mrna_path, sep="\t")
                    if "gene" in df_mrna_all.columns:
                        sub = df_mrna_all[df_mrna_all["gene"] == selected_protein]
                        df_mrna = sub if not sub.empty else None
                except Exception:
                    df_mrna = None

            n_panels = 3 if df_mrna is not None else 2
            if n_panels == 3:
                subtitles = [
                    f"{selected_protein} mRNA / R_rna(t)",
                    f"{selected_protein} protein abundance",
                    f"{selected_protein} relative phosphosite signal",
                ]
            else:
                subtitles = [
                    f"{selected_protein} protein abundance",
                    f"{selected_protein} relative phosphosite signal",
                ]

            fig_fit = make_subplots(rows=1, cols=n_panels, subplot_titles=subtitles)
            col_offset = 1

            if n_panels == 3 and df_mrna is not None:
                rna_sub = df_mrna.sort_values("time") if "time" in df_mrna.columns else df_mrna
                t_rna = rna_sub["time"].values if "time" in rna_sub.columns else np.array([])
                rna_fit_col = "fitted" if "fitted" in rna_sub.columns else ("simulated" if "simulated" in rna_sub.columns else None)
                if rna_fit_col:
                    fig_fit.add_trace(
                        go.Scatter(x=t_rna, y=rna_sub[rna_fit_col].values, mode="lines",
                                   name="mRNA model", line=dict(width=2)),
                        row=1, col=col_offset,
                    )
                if show_obs and "observed" in rna_sub.columns:
                    fig_fit.add_trace(
                        go.Scatter(x=t_rna, y=rna_sub["observed"].values, mode="markers",
                                   name="mRNA observed", marker=dict(size=8)),
                        row=1, col=col_offset,
                    )
                fig_fit.update_yaxes(title_text="mRNA / transcriptional activation R_rna(t)", row=1, col=col_offset)
                col_offset += 1

            if not df_prot_row.empty:
                row_p = df_prot_row.iloc[0]
                fig_fit.add_trace(
                    go.Scatter(x=t_filt, y=row_p[sim_cols_f].values.astype(float),
                               mode="lines", name="abundance (model)", line=dict(width=2, color="royalblue")),
                    row=1, col=col_offset,
                )
                if show_obs:
                    fig_fit.add_trace(
                        go.Scatter(x=t_filt, y=row_p[data_cols_f].values.astype(float),
                                   mode="markers", name="abundance (data)",
                                   marker=dict(size=8, symbol="square", color="royalblue")),
                        row=1, col=col_offset,
                    )
            fig_fit.update_yaxes(title_text="Protein abundance", row=1, col=col_offset)
            col_offset += 1

            colors = px.colors.qualitative.Plotly
            for i, (_, row_s) in enumerate(df_site_rows.iterrows()):
                label = f"{row_s['Protein']}_{row_s['Residue']}"
                c = colors[i % len(colors)]
                fig_fit.add_trace(
                    go.Scatter(x=t_filt, y=row_s[sim_cols_f].values.astype(float),
                               mode="lines", name=f"{label} (model)", line=dict(width=2, color=c)),
                    row=1, col=col_offset,
                )
                if show_obs:
                    fig_fit.add_trace(
                        go.Scatter(x=t_filt, y=row_s[data_cols_f].values.astype(float),
                                   mode="markers", name=f"{label} (data)",
                                   marker=dict(size=8, color=c), showlegend=False),
                        row=1, col=col_offset,
                    )
            fig_fit.update_yaxes(title_text="Relative phosphosite signal", row=1, col=col_offset)
            fig_fit.update_xaxes(title_text="Time (min)")
            fig_fit.update_layout(
                height=480, template="plotly_white", hovermode="x unified",
                title=f"Fit trajectories — {selected_protein}",
            )
            st.plotly_chart(fig_fit, use_container_width=True)

        with st.expander("Download fit table"):
            st.dataframe(df_ft, use_container_width=True)
            st.download_button("Download CSV", df_ft.to_csv(index=False).encode(),
                               file_name="fit_timeseries.csv")


# ══════════════════════════════════════════════════════════════════════════
# C · INTERNAL STATES
# ══════════════════════════════════════════════════════════════════════════
with tab_states:
    st.header("Internal States")
    st.caption("Source: `internal_states.tsv`")
    st.markdown(
        "- **S** – protein signalling/activity fraction \u2208 [0, 1]\n"
        "- **Kdyn** – kinase activity fraction \u2208 [0, 1]"
    )

    df_is = _load_int_states(results_dir)
    if df_is is None:
        st.warning("internal_states.tsv not found.")
    else:
        t_cols = [c for c in df_is.columns if c.startswith("t")]
        t_is = t_orig if len(t_orig) == len(t_cols) else np.arange(len(t_cols), dtype=float)
        t_is_mask = (t_is >= time_range[0]) & (t_is <= time_range[1])
        t_cols_f = [c for c, m in zip(t_cols, t_is_mask) if m]
        t_is_f = t_is[t_is_mask]

        df_S = df_is[df_is["Type"] == "S_sim"]
        df_K = df_is[df_is["Type"] == "Kdyn_sim"]

        col_s, col_k = st.columns(2)

        with col_s:
            st.subheader("S — protein signalling/activity fraction")
            fig_s = go.Figure()
            for _, row in df_S[~df_S["ID"].isin([selected_protein])].head(9).iterrows():
                fig_s.add_trace(go.Scatter(x=t_is_f, y=row[t_cols_f].values.astype(float),
                                           mode="lines", opacity=0.4, line=dict(width=1), name=row["ID"]))
            if selected_protein:
                sel_row = df_S[df_S["ID"] == selected_protein]
                for _, row in sel_row.iterrows():
                    fig_s.add_trace(go.Scatter(x=t_is_f, y=row[t_cols_f].values.astype(float),
                                               mode="lines", line=dict(width=3, color="crimson"),
                                               name=f"\u2605 {row['ID']}"))
            fig_s.update_layout(height=380, template="plotly_white", xaxis_title="Time (min)",
                                 yaxis_title="S (activity fraction)",
                                 yaxis=dict(range=[0, 1.05]), title="Protein activity (S)")
            if use_logx:
                fig_s.update_xaxes(type="log")
            st.plotly_chart(fig_s, use_container_width=True)

        with col_k:
            st.subheader("Kdyn — kinase activity fraction")
            fig_k = go.Figure()
            for _, row in df_K[~df_K["ID"].isin(selected_kinases)].head(9).iterrows():
                fig_k.add_trace(go.Scatter(x=t_is_f, y=row[t_cols_f].values.astype(float),
                                           mode="lines", opacity=0.4, line=dict(width=1), name=row["ID"]))
            for kin in selected_kinases:
                sel_rows = df_K[df_K["ID"] == kin]
                for _, row in sel_rows.iterrows():
                    fig_k.add_trace(go.Scatter(x=t_is_f, y=row[t_cols_f].values.astype(float),
                                               mode="lines", line=dict(width=3, color="darkorange"),
                                               name=f"\u2605 {row['ID']}"))
            fig_k.update_layout(height=380, template="plotly_white", xaxis_title="Time (min)",
                                 yaxis_title="Kdyn (kinase activity fraction)",
                                 yaxis=dict(range=[0, 1.05]), title="Kinase activity (Kdyn)")
            if use_logx:
                fig_k.update_xaxes(type="log")
            st.plotly_chart(fig_k, use_container_width=True)

        with st.expander("Download internal states table"):
            st.dataframe(df_is, use_container_width=True)
            st.download_button("Download CSV", df_is.to_csv(index=False).encode(),
                               file_name="internal_states.csv")


# ══════════════════════════════════════════════════════════════════════════
# D · DERIVED RATES
# ══════════════════════════════════════════════════════════════════════════
with tab_rates:
    st.header("Derived Rates")
    st.caption("Source: `derived_rates.npz` / `derived_rates_long.tsv`")
    st.info(
        "**k_act(t)** and **s_prod(t)** are *not* optimised parameters — they are "
        "derived inputs computed from mRNA / kinase-signal data and injected into "
        "the ODE right-hand side as time-varying closures."
    )

    dr = _load_dr(results_dir)
    if dr is None:
        st.warning("Derived rates artefacts not found.")
    else:
        dr_proteins = list(dr.get("proteins", []))
        show_ents = ([selected_protein] if selected_protein and selected_protein in dr_proteins
                     else dr_proteins[:5])

        col_ka, col_sp = st.columns(2)

        with col_ka:
            st.subheader("k_act(t) — transcriptional activation rate")
            if "k_act" in dr and "t_k_act" in dr:
                t_ka = np.asarray(dr["t_k_act"]).ravel()
                k_act_mat = np.asarray(dr["k_act"])
                fig_ka = go.Figure()
                for ename in show_ents:
                    idx = dr_proteins.index(ename) if ename in dr_proteins else -1
                    if 0 <= idx < k_act_mat.shape[0]:
                        fig_ka.add_trace(go.Scatter(x=t_ka, y=k_act_mat[idx],
                                                     mode="lines", name=ename))
                fig_ka.update_layout(height=350, template="plotly_white",
                                     xaxis_title="Time (min)", yaxis_title="k_act(t)",
                                     title="Transcriptional activation rate k_act(t)")
                st.plotly_chart(fig_ka, use_container_width=True)
            else:
                st.info("k_act not available.")

        with col_sp:
            st.subheader("s_prod(t) — kinase-signal synthesis rate")
            if "s_prod" in dr and "t_s_prod" in dr:
                t_sp = np.asarray(dr["t_s_prod"]).ravel()
                s_prod_mat = np.asarray(dr["s_prod"])
                fig_sp = go.Figure()
                for ename in show_ents:
                    idx = dr_proteins.index(ename) if ename in dr_proteins else -1
                    if 0 <= idx < s_prod_mat.shape[0]:
                        fig_sp.add_trace(go.Scatter(x=t_sp, y=s_prod_mat[idx],
                                                     mode="lines", name=ename))
                fig_sp.update_layout(height=350, template="plotly_white",
                                     xaxis_title="Time (min)", yaxis_title="s_prod(t)",
                                     title="Kinase-signal-driven synthesis rate s_prod(t)")
                st.plotly_chart(fig_sp, use_container_width=True)
            else:
                st.info("s_prod not available.")


# ══════════════════════════════════════════════════════════════════════════
# E · FORWARD SIMULATION
# ══════════════════════════════════════════════════════════════════════════
with tab_sim:
    st.header("Forward Simulation")
    st.markdown(
        "Simulate from fitted parameters and preopt snapshot arrays.  "
        "Derived-rate closures default to neutral constants when absent."
    )

    if snap is None or params is None:
        st.error("Cannot simulate: preopt snapshot or fitted_params.npz missing.")
    else:
        c_sim1, c_sim2 = st.columns([1, 3])
        with c_sim1:
            sim_t_max = st.number_input("Simulation horizon (min)", min_value=float(t_max_data),
                                        max_value=float(t_max_data * 20),
                                        value=float(t_max_data), step=float(max(1.0, t_max_data / 10)))
            sim_n_pts = st.slider("Grid points", 50, 500, 200, key="sim_npts")
            run_sim_btn = st.button("\u25b6 Run simulation", type="primary")

        with c_sim2:
            if run_sim_btn:
                with st.spinner("Integrating ODE…"):
                    sim_result = _run_simulation(results_dir, t_max=sim_t_max,
                                                  num_points=sim_n_pts, mechanism=mechanism)
                if sim_result is None:
                    st.error("Simulation failed.")
                else:
                    t_sim = sim_result["t"]
                    P_sim = sim_result["P_sim"]
                    A_sim = sim_result["A_sim"]
                    S_sim = sim_result["S_sim"]
                    Kdyn_sim = sim_result["Kdyn_sim"]

                    if selected_protein and selected_protein in proteins:
                        p_idx = proteins.index(selected_protein)
                        spi = np.asarray(snap.get("site_prot_idx", []))
                        site_idxs = list(np.where(spi == p_idx)[0])
                        kin_to_prot = np.asarray(snap.get("kin_to_prot_idx", []))
                        rel_kins = list(np.where(kin_to_prot == p_idx)[0])

                        fig_esim = make_subplots(rows=2, cols=2, subplot_titles=(
                            "Relative phosphosite signal", "Protein abundance",
                            "S \u2013 activity fraction", "Kdyn \u2013 kinase activity fraction",
                        ))
                        colors = px.colors.qualitative.Plotly

                        for i, si in enumerate(site_idxs[:12]):
                            sname = sites[si] if si < len(sites) else str(si)
                            fig_esim.add_trace(
                                go.Scatter(x=t_sim, y=P_sim[si], mode="lines", name=sname,
                                           line=dict(color=colors[i % len(colors)])),
                                row=1, col=1)

                        fig_esim.add_trace(
                            go.Scatter(x=t_sim, y=A_sim[p_idx], mode="lines", name="A",
                                       line=dict(width=2, color="black")), row=1, col=2)
                        fig_esim.add_trace(
                            go.Scatter(x=t_sim, y=S_sim[p_idx], mode="lines", name="S",
                                       line=dict(width=2, color="purple")), row=2, col=1)

                        for ri, ki in enumerate(rel_kins[:5]):
                            kname = kinases[ki] if ki < len(kinases) else str(ki)
                            fig_esim.add_trace(
                                go.Scatter(x=t_sim, y=Kdyn_sim[ki], mode="lines",
                                           name=f"Kdyn {kname}", line=dict(color=colors[ri % len(colors)])),
                                row=2, col=2)

                        fig_esim.update_xaxes(title_text="Time (min)")
                        fig_esim.update_yaxes(title_text="Relative phosphosite signal", row=1, col=1)
                        fig_esim.update_yaxes(title_text="Protein abundance", row=1, col=2)
                        fig_esim.update_yaxes(title_text="Activity fraction [0-1]", row=2, col=1)
                        fig_esim.update_yaxes(title_text="Activity fraction [0-1]", row=2, col=2)
                        fig_esim.update_layout(height=700, template="plotly_white",
                                               title=f"Forward simulation \u2014 {selected_protein}")
                        if use_logx:
                            fig_esim.update_xaxes(type="log")
                        st.plotly_chart(fig_esim, use_container_width=True)
            else:
                st.info("Configure parameters and press **\u25b6 Run simulation**.")


# ══════════════════════════════════════════════════════════════════════════
# F · LIVE KNOCKOUT
# ══════════════════════════════════════════════════════════════════════════
with tab_ko:
    st.header("Live Knockout Effects")
    st.markdown(
        "Compare wild-type (WT) vs single knockout (KO) using the fitted "
        "parameters and the simulation pipeline (no new biology in the dashboard)."
    )

    with st.expander("\U0001F4C2 Saved knockout outputs"):
        ko_data = _load_ko(results_dir)
        if ko_data is None:
            st.info("No knockout output files found in `knockouts/`.")
        else:
            for key, df_ko in ko_data.items():
                st.subheader(key)
                if df_ko is None or df_ko.empty:
                    st.info("Empty table.")
                    continue
                arr = df_ko.values.astype(float)
                finite_arr = np.where(np.isfinite(arr), arr, np.nan)
                fig_ko_heat = px.imshow(
                    finite_arr, x=list(df_ko.columns), y=list(df_ko.index),
                    color_continuous_scale="RdBu_r", color_continuous_midpoint=1.0,
                    title=f"Fold-change heatmap \u2014 {key}",
                    labels={"color": "FC (KO/WT)"}, aspect="auto",
                )
                fig_ko_heat.update_layout(height=min(600, max(300, 20 * len(df_ko))))
                st.plotly_chart(fig_ko_heat, use_container_width=True)
                st.download_button(f"Download {key}.csv", df_ko.to_csv().encode(),
                                   file_name=f"{key}.csv", key=f"dl_ko_{key}")

    st.divider()
    st.subheader("Live knockout simulation")

    if snap is None or params is None:
        st.error("Cannot run live KO: preopt snapshot or fitted_params.npz missing.")
    else:
        ck1, ck2, ck3 = st.columns(3)
        with ck1:
            ko_type = st.selectbox("Knockout type", ["kinase", "protein", "site"], key="live_ko_type")
        with ck2:
            if ko_type == "kinase":
                target_list = kinases
            elif ko_type == "protein":
                target_list = proteins
            else:
                target_list = sites
            ko_target = st.selectbox("Target", target_list, key="live_ko_target")
        with ck3:
            observe_ko_prot = st.selectbox("Observe on protein", proteins, key="ko_obs_prot")
            ko_t_max = st.slider("Simulation horizon (min)", 50, 2000,
                                  int(t_max_data), key="ko_t_max")

        run_ko_btn = st.button("\u25b6 Run live KO", type="primary")

        if run_ko_btn:
            t_ko_eval = np.linspace(0.0, float(ko_t_max), 200)
            with st.spinner("Simulating WT and KO…"):
                try:
                    ModelDims.set_dims(len(proteins), len(kinases), len(sites))
                    ko_result = run_live_knockout(
                        t_eval=t_ko_eval,
                        theta_opt=params["theta"],
                        ko_type=ko_type,
                        ko_target=ko_target,
                        proteins=proteins,
                        kinases=kinases,
                        sites=sites,
                        snap=snap,
                        a_proteins=labels.get("A_proteins", []),
                    )

                    obs_prot_idx = proteins.index(observe_ko_prot)
                    spi = np.asarray(snap.get("site_prot_idx", []))
                    site_idxs_ko = list(np.where(spi == obs_prot_idx)[0])

                    fig_ko_traj = make_subplots(rows=1, cols=2, subplot_titles=(
                        "Relative phosphosite signal (WT vs KO)",
                        "S \u2013 activity fraction (WT vs KO)",
                    ))
                    colors = px.colors.qualitative.Bold

                    for i, si in enumerate(site_idxs_ko[:8]):
                        sname = sites[si] if si < len(sites) else str(si)
                        c = colors[i % len(colors)]
                        fig_ko_traj.add_trace(
                            go.Scatter(x=t_ko_eval, y=ko_result["wt"]["P_sim"][si],
                                       mode="lines", name=f"WT {sname}", line=dict(color=c, width=2)),
                            row=1, col=1)
                        fig_ko_traj.add_trace(
                            go.Scatter(x=t_ko_eval, y=ko_result["ko"]["P_sim"][si],
                                       mode="lines", name=f"KO {sname}",
                                       line=dict(color=c, width=2, dash="dot"), showlegend=False),
                            row=1, col=1)

                    fig_ko_traj.add_trace(
                        go.Scatter(x=t_ko_eval, y=ko_result["wt"]["S_sim"][obs_prot_idx],
                                   mode="lines", name="WT S", line=dict(color="purple", width=2)),
                        row=1, col=2)
                    fig_ko_traj.add_trace(
                        go.Scatter(x=t_ko_eval, y=ko_result["ko"]["S_sim"][obs_prot_idx],
                                   mode="lines", name="KO S", line=dict(color="purple", width=2, dash="dot")),
                        row=1, col=2)

                    fig_ko_traj.update_xaxes(title_text="Time (min)")
                    fig_ko_traj.update_yaxes(title_text="Relative phosphosite signal", row=1, col=1)
                    fig_ko_traj.update_yaxes(title_text="Activity fraction [0-1]", row=1, col=2, range=[0, 1.05])
                    fig_ko_traj.update_layout(height=450, template="plotly_white",
                                              title=f"WT vs {ko_target} ({ko_type} KO) \u2192 {observe_ko_prot}")
                    st.plotly_chart(fig_ko_traj, use_container_width=True)

                    # Fold-change ranking
                    eps = 1e-9
                    wt_f = ko_result["wt"]["P_sim"][:, -1]
                    ko_f = ko_result["ko"]["P_sim"][:, -1]
                    fc = (ko_f + eps) / (wt_f + eps)
                    df_fc = pd.DataFrame({
                        "Site": sites,
                        "WT_signal": wt_f,
                        "KO_signal": ko_f,
                        "FC_KO_over_WT": fc,
                    })

                    st.subheader("Ranked fold-change (final time point)")
                    sort_abs = st.checkbox("Sort by |FC| (largest absolute change first)",
                                           value=True, key="ko_sort_abs")
                    if sort_abs:
                        df_fc = df_fc.sort_values("FC_KO_over_WT", key=abs, ascending=False)
                    else:
                        df_fc = df_fc.sort_values("FC_KO_over_WT", ascending=False)

                    fig_fc_bar = px.bar(
                        df_fc.head(30), x="Site", y="FC_KO_over_WT",
                        color="FC_KO_over_WT", color_continuous_scale="RdBu_r",
                        color_continuous_midpoint=1.0,
                        title=f"Top 30 affected phosphosites \u2014 {ko_target} KO",
                        labels={"FC_KO_over_WT": "FC (KO/WT)"},
                    )
                    fig_fc_bar.update_layout(height=400, template="plotly_white")
                    st.plotly_chart(fig_fc_bar, use_container_width=True)
                    st.download_button("Download FC table", df_fc.to_csv(index=False).encode(),
                                       file_name=f"ko_fc_{ko_target}.csv")

                except Exception as exc:
                    st.error(f"Knockout simulation failed: {exc}")


# ══════════════════════════════════════════════════════════════════════════
# G · SENSITIVITY
# ══════════════════════════════════════════════════════════════════════════
with tab_sens:
    st.header("Sensitivity Analysis")
    st.caption("Source: `sensitivity/sobol_indices_labeled.tsv`, `perturbation_data.tsv`")

    sens = _load_sens(results_dir)
    if sens is None:
        st.warning("No sensitivity outputs found in `sensitivity/`.")
    else:
        df_sobol = sens["sobol"]
        st.subheader("1. Parameter importance ranking (Sobol indices)")

        cs1, cs2 = st.columns([1, 3])
        with cs1:
            top_n = st.slider("Top N", 5, min(60, len(df_sobol)), 20, key="sobol_top")
            sort_by = st.radio("Sort by", ["Total_Order", "First_Order"], key="sobol_sort")
        with cs2:
            df_ps = df_sobol.sort_values(sort_by, ascending=False).head(top_n)
            fig_sobol = go.Figure()
            param_axis = df_ps.get("Parameter", df_ps.index) if "Parameter" in df_ps.columns else df_ps.index
            if "Total_Order" in df_ps.columns:
                err_y = None
                if "Confidence_Total" in df_ps.columns:
                    err_y = dict(type="data", array=df_ps["Confidence_Total"].values, visible=True)
                fig_sobol.add_trace(go.Bar(x=param_axis, y=df_ps["Total_Order"],
                                           name="Total effect (ST)", marker_color="steelblue", error_y=err_y))
            if "First_Order" in df_ps.columns:
                fig_sobol.add_trace(go.Bar(x=param_axis, y=df_ps["First_Order"],
                                           name="First order (S1)", marker_color="navy", opacity=0.7))
            fig_sobol.update_layout(barmode="overlay", height=450, template="plotly_white",
                                    xaxis_title="Parameter", yaxis_title="Sobol index",
                                    title=f"Top {top_n} sensitive parameters")
            st.plotly_chart(fig_sobol, use_container_width=True)

        with st.expander("Full Sobol table"):
            st.dataframe(df_sobol, use_container_width=True)
            st.download_button("Download CSV", df_sobol.to_csv(index=False).encode(),
                               file_name="sobol_indices.csv")

        st.divider()
        st.subheader("2. Output variance (perturbation envelopes)")
        df_pert = sens.get("perturbation_data")
        if df_pert is None:
            st.info("perturbation_data.tsv not found.")
        else:
            cp1, cp2, cp3 = st.columns(3)
            avail_types = sorted(df_pert["Type"].unique()) if "Type" in df_pert.columns else []
            with cp1:
                sel_type = st.selectbox("Entity type", avail_types, key="pert_type")
            with cp2:
                avail_ents = sorted(df_pert[df_pert["Type"] == sel_type]["Entity"].unique()) if sel_type and "Entity" in df_pert.columns else []
                sel_entity = st.selectbox("Entity", avail_ents, key="pert_entity")
            with cp3:
                max_samp = st.slider("Max samples", 10, 200, 50, key="pert_samp")

            subset = df_pert[(df_pert["Type"] == sel_type) & (df_pert["Entity"] == sel_entity)] if sel_type and sel_entity else pd.DataFrame()
            if not subset.empty:
                shown_ids = subset["Sample_ID"].unique()[:max_samp]
                subset = subset[subset["Sample_ID"].isin(shown_ids)]
                fig_pert = go.Figure()
                for sid, grp in subset.groupby("Sample_ID"):
                    fig_pert.add_trace(go.Scatter(x=grp["Time"], y=grp["Value"],
                                                   mode="lines", line=dict(color="gray", width=1),
                                                   opacity=0.2, showlegend=False))
                stats_df = subset.groupby("Time")["Value"].agg(["mean", "median", "std"]).reset_index()
                fig_pert.add_trace(go.Scatter(x=stats_df["Time"], y=stats_df["median"],
                                              mode="lines", line=dict(color="red", width=2.5), name="Median"))
                fig_pert.add_trace(go.Scatter(x=stats_df["Time"], y=stats_df["mean"] + stats_df["std"],
                                              mode="lines", line=dict(width=0), showlegend=False))
                fig_pert.add_trace(go.Scatter(x=stats_df["Time"], y=stats_df["mean"] - stats_df["std"],
                                              mode="lines", line=dict(width=0), fill="tonexty",
                                              fillcolor="rgba(255,0,0,0.12)", name="Mean \u00b1 1 SD"))
                fig_pert.update_layout(height=450, template="plotly_white",
                                       xaxis_title="Time (min)", yaxis_title="Simulated value",
                                       title=f"Perturbation variance \u2014 {sel_entity}")
                st.plotly_chart(fig_pert, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════
# H · STEADY-STATE
# ══════════════════════════════════════════════════════════════════════════
with tab_ss:
    st.header("Steady-State / Long-Horizon Relaxation")
    st.caption("Source: `steadystate/*.tsv`")
    st.markdown(
        "Relative phosphosite signal is *not* capped at 1 — values above 1 "
        "are valid for this state variable."
    )

    ss_data = _load_ss(results_dir)

    if ss_data is None or len(ss_data) == 0:
        st.warning("No steady-state outputs found in `steadystate/`.")
        if snap is not None and params is not None:
            st.subheader("Live long-horizon simulation")
            css1, css2 = st.columns([1, 3])
            with css1:
                ss_t_end = st.slider("End time (min)", 1000, 50000, 10000, 1000, key="ss_t_end")
                run_ss_btn = st.button("\u25b6 Run steady-state sim", type="primary", key="run_ss")
            with css2:
                if run_ss_btn:
                    from phoscrosstalk.steadystate import build_long_horizon_time_grid
                    t_ss_grid = build_long_horizon_time_grid(
                        t_end=float(ss_t_end), early_end=t_max_data, n_early=100, n_late=200)
                    with st.spinner("Simulating long horizon…"):
                        ss_r = _run_simulation(results_dir, t_max=float(t_ss_grid[-1]),
                                               num_points=len(t_ss_grid), mechanism=mechanism)
                    if ss_r:
                        _plot_ss_results(ss_r["t"], ss_r["P_sim"], ss_r["A_sim"],
                                         ss_r["S_sim"], ss_r["Kdyn_sim"],
                                         proteins, sites, kinases, snap, selected_protein, use_logx)
    else:
        for key, df_ss in ss_data.items():
            if df_ss is None or df_ss.empty:
                continue
            st.subheader(f"steadystate_{key}")
            arr = df_ss.values.astype(float)
            n_fin = int(np.sum(np.isfinite(arr)))
            if n_fin == 0:
                st.warning(f"All values in {key} are non-finite; skipping.")
                continue
            try:
                t_ss_axis = np.array([float(c) for c in df_ss.columns])
            except ValueError:
                t_ss_axis = np.arange(len(df_ss.columns), dtype=float)

            row_names = list(df_ss.index)

            if key in ("S", "Kdyn", "proteins"):
                fig_ss_line = go.Figure()
                for ri, rname in enumerate(row_names[:20]):
                    y = np.where(np.isfinite(arr[ri]), arr[ri], np.nan)
                    fig_ss_line.add_trace(go.Scatter(x=t_ss_axis, y=y, mode="lines", name=str(rname)))
                ylab = ("Activity fraction [0-1]" if key in ("S", "Kdyn") else "Protein abundance")
                fig_ss_line.update_layout(height=400, template="plotly_white",
                                          xaxis_title="Time (min)", yaxis_title=ylab,
                                          title=f"Steady-state \u2014 {key}")
                if use_logx:
                    fig_ss_line.update_xaxes(type="log")
                st.plotly_chart(fig_ss_line, use_container_width=True)

            elif key == "sites":
                finite_arr = np.where(np.isfinite(arr), arr, np.nan)
                fig_ss_h = px.imshow(finite_arr,
                                     x=[str(c) for c in df_ss.columns],
                                     y=[str(r) for r in row_names],
                                     color_continuous_scale="Plasma",
                                     title="Relative phosphosite signal \u2014 steady-state",
                                     labels={"color": "Relative phosphosite signal"}, aspect="auto")
                fig_ss_h.update_layout(height=500)
                st.plotly_chart(fig_ss_h, use_container_width=True)

            n_nan = int(np.sum(~np.isfinite(arr)))
            if n_nan > 0:
                st.caption(f"\u2139\ufe0f {n_nan}/{arr.size} values are NaN/Inf (shown as blank).")

            st.download_button(f"Download steadystate_{key}.csv", df_ss.to_csv().encode(),
                               file_name=f"steadystate_{key}.csv", key=f"dl_ss_{key}")

        if "sites" in ss_data and ss_data["sites"] is not None:
            df_sv = ss_data["sites"]
            arr_sv = df_sv.values.astype(float)
            if arr_sv.shape[1] >= 10 and np.any(np.isfinite(arr_sv)):
                delta = np.nanmean(np.abs(arr_sv[:, -1] - arr_sv[:, -10]))
                if np.isfinite(delta):
                    if delta < 1e-4:
                        st.success(f"System appears converged (\u0394 relative phosphosite signal: {delta:.2e})")
                    else:
                        st.warning(f"System may not be fully converged (\u0394 relative phosphosite signal: {delta:.2e})")


# ══════════════════════════════════════════════════════════════════════════
# I · NETWORK
# ══════════════════════════════════════════════════════════════════════════
with tab_net:
    st.header("Network View")
    st.caption("Source: `network_cytoscape_edges.csv`")

    net_path = os.path.join(results_dir, "network_cytoscape_edges.csv")

    if not os.path.exists(net_path):
        st.warning("`network_cytoscape_edges.csv` not found.")
        if snap is not None:
            st.subheader("Interaction matrices (from preopt snapshot)")
            mat_opt = st.selectbox(
                "Select matrix",
                ["K_site_kin (kinase–substrate)", "Cg (global coupling)",
                 "Cl (local coupling)", "L_alpha (kinase Laplacian)"],
                key="net_mat",
            )
            mat_map = {
                "K_site_kin (kinase–substrate)": ("K_site_kin", kinases, sites),
                "Cg (global coupling)": ("Cg", sites, sites),
                "Cl (local coupling)": ("Cl", sites, sites),
                "L_alpha (kinase Laplacian)": ("L_alpha", kinases, kinases),
            }
            arr_key, x_labels, y_labels = mat_map[mat_opt]
            mat_arr = snap.get(arr_key, np.empty((0, 0)))
            if mat_arr is not None and mat_arr.size > 0:
                xl_lbls = x_labels if len(x_labels) == mat_arr.shape[1] else None
                yl_lbls = y_labels if len(y_labels) == mat_arr.shape[0] else None
                fig_mat = px.imshow(mat_arr, x=xl_lbls, y=yl_lbls,
                                    color_continuous_scale="Viridis", title=mat_opt, aspect="auto")
                fig_mat.update_layout(height=500)
                st.plotly_chart(fig_mat, use_container_width=True)
    else:
        try:
            df_net = pd.read_csv(net_path)
        except Exception as exc:
            st.error(f"Could not load network edges: {exc}")
            st.stop()

        st.markdown(f"Loaded **{len(df_net):,}** edges.")

        if df_net.shape[1] < 2:
            st.error(
                "Network edge file must have at least 2 columns (Source, Target). "
                f"Found {df_net.shape[1]} column(s)."
            )
            st.stop()

        source_col = "Source" if "Source" in df_net.columns else df_net.columns[0]
        target_col = "Target" if "Target" in df_net.columns else df_net.columns[1]

        search_node = st.text_input("Filter by node name", "")
        df_net_f = df_net.copy()
        if search_node:
            mask = (df_net_f[source_col].astype(str).str.contains(search_node, case=False, na=False) |
                    df_net_f[target_col].astype(str).str.contains(search_node, case=False, na=False))
            df_net_f = df_net_f[mask]

        st.dataframe(df_net_f.head(500), use_container_width=True)
        if len(df_net_f) > 500:
            st.caption(f"Showing top 500 of {len(df_net_f):,} edges.")
        st.download_button("Download filtered edges CSV", df_net_f.to_csv(index=False).encode(),
                           file_name="network_edges_filtered.csv")

        st.subheader("Node degree summary")
        all_nodes = pd.concat([df_net[source_col].rename("node"), df_net[target_col].rename("node")])
        degree_counts = all_nodes.value_counts().reset_index()
        degree_counts.columns = ["node", "degree"]
        fig_deg = px.bar(degree_counts.head(30), x="node", y="degree",
                         title="Top 30 nodes by degree",
                         labels={"degree": "Degree", "node": "Node"})
        fig_deg.update_layout(height=350, template="plotly_white")
        st.plotly_chart(fig_deg, use_container_width=True)

        if len(df_net) <= 500:
            st.subheader("Interactive network (Plotly/networkx)")
            _render_plotly_network(df_net, source_col, target_col)
        else:
            st.info(f"Network has {len(df_net):,} edges — too large for Plotly rendering.")


# ──────────────────────────────────────────────────────────────────────────
# Helper functions defined after tabs (Streamlit allows forward references)
# ──────────────────────────────────────────────────────────────────────────


def _render_plotly_network(df_net: pd.DataFrame, src_col: str, tgt_col: str) -> None:
    """Render a small network as Plotly scatter-with-lines."""
    if not _HAS_NX:
        st.info("networkx not installed; cannot render network.")
        return

    G = _nx.DiGraph()
    weight_col = (
        "Weight_Fitted" if "Weight_Fitted" in df_net.columns
        else (df_net.columns[2] if df_net.shape[1] > 2 else None)
    )
    for _, row in df_net.iterrows():
        w = float(row[weight_col]) if weight_col else 1.0
        G.add_edge(str(row[src_col]), str(row[tgt_col]), weight=w)

    pos = _nx.spring_layout(G, seed=42, k=1.5)
    edge_x, edge_y = [], []
    for u, v in G.edges():
        x0, y0 = pos[u]; x1, y1 = pos[v]
        edge_x += [x0, x1, None]; edge_y += [y0, y1, None]

    node_x = [pos[n][0] for n in G.nodes()]
    node_y = [pos[n][1] for n in G.nodes()]
    node_labels = list(G.nodes())
    node_degree = [G.degree(n) for n in G.nodes()]

    fig_net = go.Figure()
    fig_net.add_trace(go.Scatter(x=edge_x, y=edge_y, mode="lines",
                                  line=dict(width=0.8, color="lightgrey"), hoverinfo="none",
                                  showlegend=False))
    fig_net.add_trace(go.Scatter(
        x=node_x, y=node_y, mode="markers+text",
        marker=dict(size=[max(6, d * 2) for d in node_degree], color=node_degree,
                    colorscale="Viridis", showscale=True,
                    colorbar=dict(title="Degree")),
        text=node_labels, textposition="top center",
        hovertext=[f"{n} (deg={d})" for n, d in zip(node_labels, node_degree)],
        hoverinfo="text", name="Nodes",
    ))
    fig_net.update_layout(
        height=600, template="plotly_white", showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        title="Network topology (spring layout)",
    )
    st.plotly_chart(fig_net, use_container_width=True)


def _plot_ss_results(t, P_sim, A_sim, S_sim, Kdyn_sim,
                     proteins, sites, kinases, snap, selected_protein, use_logx):
    """Plot long-horizon simulation results."""
    if selected_protein and selected_protein in proteins:
        p_idx = proteins.index(selected_protein)
        spi = np.asarray(snap.get("site_prot_idx", []))
        site_idxs = list(np.where(spi == p_idx)[0])
        kin_to_prot = np.asarray(snap.get("kin_to_prot_idx", []))
        rel_kins = list(np.where(kin_to_prot == p_idx)[0])
    else:
        p_idx = 0
        site_idxs = list(range(min(5, len(sites))))
        rel_kins = []

    fig_ss = make_subplots(rows=2, cols=2, subplot_titles=(
        "Relative phosphosite signal", "Protein abundance",
        "S \u2013 activity fraction", "Kdyn \u2013 kinase activity fraction",
    ))
    colors = px.colors.qualitative.Plotly

    for i, si in enumerate(site_idxs[:8]):
        sname = sites[si] if si < len(sites) else str(si)
        y = np.where(np.isfinite(P_sim[si]), P_sim[si], np.nan)
        fig_ss.add_trace(go.Scatter(x=t, y=y, mode="lines", name=sname,
                                    line=dict(color=colors[i % len(colors)])), row=1, col=1)

    y_a = np.where(np.isfinite(A_sim[p_idx]), A_sim[p_idx], np.nan)
    fig_ss.add_trace(go.Scatter(x=t, y=y_a, mode="lines", name="A", line=dict(width=2, color="black")),
                     row=1, col=2)
    y_s = np.where(np.isfinite(S_sim[p_idx]), S_sim[p_idx], np.nan)
    fig_ss.add_trace(go.Scatter(x=t, y=y_s, mode="lines", name="S", line=dict(width=2, color="purple")),
                     row=2, col=1)
    for ri, ki in enumerate(rel_kins[:5]):
        kname = kinases[ki] if ki < len(kinases) else str(ki)
        y_k = np.where(np.isfinite(Kdyn_sim[ki]), Kdyn_sim[ki], np.nan)
        fig_ss.add_trace(go.Scatter(x=t, y=y_k, mode="lines", name=f"Kdyn {kname}",
                                    line=dict(color=colors[ri % len(colors)])), row=2, col=2)

    fig_ss.update_xaxes(title_text="Time (min)")
    fig_ss.update_yaxes(title_text="Relative phosphosite signal", row=1, col=1)
    fig_ss.update_yaxes(title_text="Protein abundance", row=1, col=2)
    fig_ss.update_yaxes(title_text="Activity fraction [0-1]", row=2, col=1)
    fig_ss.update_yaxes(title_text="Activity fraction [0-1]", row=2, col=2)
    fig_ss.update_layout(height=700, template="plotly_white",
                         title=f"Long-horizon relaxation \u2014 {selected_protein or (proteins[0] if proteins else '')}")
    if use_logx:
        fig_ss.update_xaxes(type="log")
    st.plotly_chart(fig_ss, use_container_width=True)
