import os
import sys
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

import networkx as nx
import gravis as gv

sys.path.append(os.getcwd())

from phoscrosstalk.config import ModelDims
from phoscrosstalk.simulation import simulate_p_scipy, build_full_A0
from phoscrosstalk.core_mechanisms import decode_theta

# --- Page Config ---
st.set_page_config(
    page_title="PhosCrosstalk Explorer",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)


# --- Helper: Data Loading ---
@st.cache_data
def load_snapshot_data(results_dir):
    """
    Loads all necessary matrices and metadata from the preopt_snapshot directory
    to reconstruct the simulation environment.
    """
    snap_dir = os.path.join(results_dir, "preopt_snapshot")
    fit_path = os.path.join(results_dir, "fitted_params.npz")

    if not os.path.exists(snap_dir) or not os.path.exists(fit_path):
        return None

    # Load Fitted Params
    fit_data = np.load(fit_path, allow_pickle=True)
    theta = fit_data["theta"]

    # Load Metadata
    def load_vec(name, dtype=float):
        return np.loadtxt(os.path.join(snap_dir, f"{name}.tsv"), delimiter="\t", dtype=dtype)

    def load_mat(name, dtype=float):
        # Handle empty matrices
        try:
            return np.loadtxt(os.path.join(snap_dir, f"{name}.tsv"), delimiter="\t", dtype=dtype, ndmin=2)
        except:
            return np.array([])

    def load_txt_list(name):
        with open(os.path.join(snap_dir, f"{name}.txt"), "r") as f:
            return [line.strip() for line in f if line.strip()]

    # Load Data
    data = {
        "theta": theta,
        "t_orig": load_vec("t"),
        "sites": load_txt_list("sites"),
        "proteins": load_txt_list("proteins"),
        "kinases": load_txt_list("kinases"),
        "positions": load_vec("positions"),
        "P_scaled": load_mat("P_scaled"),
        "A_scaled": load_mat("A_scaled"),
        "Y_orig": load_mat("Y"),  # Raw FC data
        "Cg": load_mat("Cg"),
        "Cl": load_mat("Cl"),
        "K_site_kin": load_mat("K_site_kin"),
        "R": load_mat("R"),
        "L_alpha": load_mat("L_alpha"),
        "site_prot_idx": load_vec("site_prot_idx", int).flatten(),
        "kin_to_prot_idx": load_vec("kin_to_prot_idx", int).flatten(),
        "receptor_mask_prot": load_vec("receptor_mask_prot", int).flatten(),
        "receptor_mask_kin": load_vec("receptor_mask_kin", int).flatten(),
    }

    # Load Metadata for Mechanism and Config
    meta_path = os.path.join(snap_dir, "meta.txt")
    meta_dict = {}
    with open(meta_path, "r") as f:
        for line in f:
            if "\t" in line:
                k, v = line.strip().split("\t", 1)
                meta_dict[k] = v
    data["meta"] = meta_dict

    # Reconstruct Prot Index for A (Requires finding mapping again or saving it)
    # In analysis.py we saved A_proteins but not the index map explicitly in simple form.
    # However, A_scaled rows correspond to the proteins in A_proteins.txt.
    # We need to map these back to the main protein index.
    if os.path.exists(os.path.join(snap_dir, "A_proteins.txt")):
        a_prots = load_txt_list("A_proteins")
        prot_map = {p: i for i, p in enumerate(data["proteins"])}
        prot_idx_for_A = np.array([prot_map[p] for p in a_prots if p in prot_map], dtype=int)
        data["prot_idx_for_A"] = prot_idx_for_A
    else:
        data["prot_idx_for_A"] = np.array([], dtype=int)

    return data


@st.cache_data
def run_fine_simulation(data, t_max, num_points=200):
    """
    Runs the simulation using the codebase's engine on a fine time grid.
    """
    # 1. Set Global Dims
    K = len(data["proteins"])
    M = len(data["kinases"])
    N = len(data["sites"])
    ModelDims.set_dims(K, M, N)

    # 2. Time Grid
    t_fine = np.linspace(0, t_max, num_points)

    # 3. Build A0 (Corrected for Fine Resolution)
    # We cannot use build_full_A0 directly because data['A_scaled'] shape (14,)
    # doesn't match t_fine shape (300,).
    # Since simulate_p_scipy ONLY uses column 0 for initial conditions,
    # we manually build a compatible matrix.

    A0_full = np.zeros((K, len(t_fine)), dtype=float)

    if data["A_scaled"].size > 0:
        for k, p_idx in enumerate(data["prot_idx_for_A"]):
            # Only set the initial condition (t=0)
            # We assume the rest evolves via ODEs
            if data["A_scaled"].shape[1] > 0:
                A0_full[p_idx, 0] = data["A_scaled"][k, 0]

    # 4. Simulate
    mechanism = data["meta"].get("mechanism", "dist")

    P_sim, A_sim = simulate_p_scipy(
        t_fine,
        data["P_scaled"],  # Used for Init Cond (P)
        A0_full,  # Used for Init Cond (A)
        data["theta"],
        data["Cg"],
        data["Cl"],
        data["site_prot_idx"],
        data["K_site_kin"],
        data["R"],
        data["L_alpha"],
        data["kin_to_prot_idx"],
        data["receptor_mask_prot"],
        data["receptor_mask_kin"],
        mechanism
    )

    return t_fine, P_sim, A_sim


def calculate_scalers(Y_orig):
    """
    Recompute MinMax scalars to map 0-1 model back to FC.
    """
    baselines = np.min(Y_orig, axis=1)
    ranges = np.max(Y_orig, axis=1) - np.min(Y_orig, axis=1)
    ranges[ranges < 1e-6] = 1.0
    return baselines, ranges


# --- Sidebar ---
st.sidebar.title("PhosCrosstalk Explorer")
results_dir = st.sidebar.text_input("Results Directory Path", value="network_fit")

if not os.path.exists(results_dir):
    st.error(f"Directory '{results_dir}' not found.")
    st.stop()

data = load_snapshot_data(results_dir)

if data is None:
    st.error("Could not load 'fitted_params.npz' or 'preopt_snapshot'. Ensure analysis was run.")
    st.stop()

st.sidebar.success("Data Loaded Successfully")

# --- Global Simulation ---
t_max = data["t_orig"][-1]
t_fine, P_sim_fine, A_sim_fine = run_fine_simulation(data, t_max, num_points=10000)

# Rescaling logic
baselines, amplitudes = calculate_scalers(data["Y_orig"])

# --- Tabs ---
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
    ["Trajectories", "Goodness of Fit", "Parameters", "Network Map", "Steady State", "Knockout Simulator",
     "Network Animation"])

# ==========================================
# TAB 1: Trajectories (Fine Resolution)
# ==========================================
with tab1:
    col1, col2 = st.columns([1, 3])

    with col1:
        st.subheader("Select Target")
        selected_prot = st.selectbox("Select Protein", data["proteins"])

        # Identify sites for this protein
        prot_idx = data["proteins"].index(selected_prot)
        site_indices = np.where(data["site_prot_idx"] == prot_idx)[0]

        # Scaling Toggle
        view_mode = st.radio("View Mode", ["Fold Change (Original)", "Scaled [0-1]"])
        is_fc = view_mode == "Fold Change (Original)"

    with col2:
        # Create Subplots: Left (Protein), Right (Sites)
        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=(f"{selected_prot} Abundance", f"{selected_prot} Phosphosites"))

        # --- Plot Protein Abundance ---
        # Data points
        if data["prot_idx_for_A"].size > 0:
            # Find if this protein has data
            mask_a = np.where(data["prot_idx_for_A"] == prot_idx)[0]
            if len(mask_a) > 0:
                a_idx_data = mask_a[0]
                y_data = data["A_scaled"][a_idx_data]

                # If FC mode, we need to map back?
                # Note: A_data scaling wasn't fully stored in snapshot in simple form,
                # usually A is kept simple. Assuming 0-1 for visualization or raw if available.
                # The codebase usually keeps A normalized. Let's plot what we have.
                fig.add_trace(
                    go.Scatter(x=data["t_orig"], y=y_data, mode='markers',
                               marker=dict(symbol='square', size=10, color='gray'),
                               name='Data (Abundance)'),
                    row=1, col=1
                )

        # Model Line (Fine)
        y_model = A_sim_fine[prot_idx, :]
        fig.add_trace(
            go.Scatter(x=t_fine, y=y_model, mode='lines',
                       line=dict(width=3, color='black'),
                       name='Model (Abundance)'),
            row=1, col=1
        )

        # --- Plot Phosphosites ---
        colors = px.colors.qualitative.Plotly

        if len(site_indices) == 0:
            fig.add_annotation(text="No mapped sites", xref="x2", yref="y2", showarrow=False)
        else:
            for i, s_idx in enumerate(site_indices):
                site_name = data["sites"][s_idx]
                short_name = site_name.split("_")[-1] if "_" in site_name else site_name
                color = colors[i % len(colors)]

                # Prepare Y values
                y_dat = data["P_scaled"][s_idx]
                y_mod = P_sim_fine[s_idx]

                if is_fc:
                    # Rescale to FC
                    b, a = baselines[s_idx], amplitudes[s_idx]
                    y_dat = b + a * y_dat
                    y_mod = b + a * y_mod

                # Data
                fig.add_trace(
                    go.Scatter(x=data["t_orig"], y=y_dat, mode='markers',
                               marker=dict(size=8, color=color, opacity=0.7),
                               name=f"{short_name} (Data)"),
                    row=1, col=2
                )

                # Model
                fig.add_trace(
                    go.Scatter(x=t_fine, y=y_mod, mode='lines',
                               line=dict(width=2, color=color),
                               name=f"{short_name} (Sim)"),
                    row=1, col=2
                )

        fig.update_layout(
            height=500,
            template="plotly_white",
            hovermode="x unified",
            title_text=f"Dynamics for {selected_prot}"
        )
        fig.update_xaxes(title_text="Time (min)")
        if is_fc:
            fig.update_yaxes(title_text="Fold Change")
        else:
            fig.update_yaxes(title_text="Scaled Activity [0-1]")

        st.plotly_chart(fig, use_container_width=True)

# ==========================================
# TAB 2: Goodness of Fit
# ==========================================
with tab2:
    st.subheader("Observed vs Simulated")

    # We need to compute simulation at t_orig points
    # We can rely on simulate_p_scipy again for exact timepoints or interpolation.
    # Exact is better.
    ModelDims.set_dims(len(data["proteins"]), len(data["kinases"]), len(data["sites"]))
    A0_orig = build_full_A0(len(data["proteins"]), len(data["t_orig"]), data["A_scaled"], data["prot_idx_for_A"])

    P_sim_orig, _ = simulate_p_scipy(
        data["t_orig"], data["P_scaled"], A0_orig, data["theta"],
        data["Cg"], data["Cl"], data["site_prot_idx"],
        data["K_site_kin"], data["R"], data["L_alpha"], data["kin_to_prot_idx"],
        data["receptor_mask_prot"], data["receptor_mask_kin"],
        data["meta"].get("mechanism", "dist")
    )

    # Flatten
    y_true = []
    y_pred = []
    labels = []

    # Rescale if needed (Usually GoF is done on scaled data to be fair across dynamic ranges)
    # But user asked for FC. Let's do FC.
    for i in range(len(data["sites"])):
        b, a = baselines[i], amplitudes[i]

        real_dat = b + a * data["P_scaled"][i]
        real_sim = b + a * P_sim_orig[i]

        y_true.extend(real_dat)
        y_pred.extend(real_sim)
        labels.extend([data["sites"][i]] * len(real_dat))

    df_fit = pd.DataFrame({"Observed": y_true, "Simulated": y_pred, "Label": labels})

    # Metrics
    r2 = 1 - np.sum((df_fit["Observed"] - df_fit["Simulated"]) ** 2) / np.sum(
        (df_fit["Observed"] - df_fit["Observed"].mean()) ** 2)

    col_m1, col_m2 = st.columns(2)
    col_m1.metric("R² (Global FC)", f"{r2:.4f}")
    col_m2.metric("Pearson Correlation", f"{df_fit['Observed'].corr(df_fit['Simulated']):.4f}")

    fig_fit = px.scatter(df_fit, x="Observed", y="Simulated", hover_data=["Label"], opacity=0.6)

    # Add identity line
    min_val = min(df_fit["Observed"].min(), df_fit["Simulated"].min())
    max_val = max(df_fit["Observed"].max(), df_fit["Simulated"].max())
    fig_fit.add_shape(type="line", x0=min_val, y0=min_val, x1=max_val, y1=max_val,
                      line=dict(color="Red", dash="dash"))

    fig_fit.update_layout(height=700, title="Goodness of Fit (Fold Change)")
    st.plotly_chart(fig_fit, use_container_width=True)

# ==========================================
# TAB 3: Parameters
# ==========================================
with tab3:
    st.subheader("Parameter Distributions")

    # Decode Theta
    ModelDims.set_dims(len(data["proteins"]), len(data["kinases"]), len(data["sites"]))
    params_decoded = decode_theta(data["theta"], ModelDims.K, ModelDims.M, ModelDims.N)

    # Unpack
    (k_act, k_deact, s_prod, d_deg, beta_g, beta_l, alpha,
     kK_act, kK_deact, k_off, gamma_S_p, gamma_A_S, gamma_A_p, gamma_K_net) = params_decoded

    # Dataframes
    df_prot_params = pd.DataFrame({
        "Protein": data["proteins"],
        "k_act": k_act, "k_deact": k_deact, "s_prod": s_prod, "d_deg": d_deg
    })

    df_kin_params = pd.DataFrame({
        "Kinase": data["kinases"],
        "alpha": alpha, "kK_act": kK_act, "kK_deact": kK_deact
    })

    df_site_params = pd.DataFrame({
        "Site": data["sites"],
        "k_off": k_off
    })

    p_col1, p_col2 = st.columns(2)

    with p_col1:
        st.markdown("#### Global Coupling")
        st.write(f"**Beta Global:** {beta_g:.4f}")
        st.write(f"**Beta Local:** {beta_l:.4f}")
        st.markdown("#### Gammas")
        st.write(f"S_p: {gamma_S_p:.3f}, A_S: {gamma_A_S:.3f}, A_p: {gamma_A_p:.3f}, K_net: {gamma_K_net:.3f}")

    with p_col2:
        param_type = st.selectbox("Visualize Parameter Set", ["Protein Rates", "Kinase Rates", "Site Phosphatase"])

        if param_type == "Protein Rates":
            fig_p = px.box(df_prot_params.melt(id_vars="Protein"), x="variable", y="value", points="all",
                           title="Protein Parameters")
            st.plotly_chart(fig_p)
            st.dataframe(df_prot_params, use_container_width=True)

        elif param_type == "Kinase Rates":
            fig_k = px.box(df_kin_params.melt(id_vars="Kinase"), x="variable", y="value", points="all",
                           title="Kinase Parameters")
            st.plotly_chart(fig_k)
            st.dataframe(df_kin_params, use_container_width=True)

        else:
            fig_s = px.histogram(df_site_params, x="k_off", nbins=30, title="Phosphatase Rate Distribution")
            st.plotly_chart(fig_s)
            st.dataframe(df_site_params, use_container_width=True)

# ==========================================
# TAB 4: Network Map (Matrices)
# ==========================================
with tab4:
    st.subheader("Interaction Matrices")

    mat_opt = st.selectbox("Select Matrix",
                           ["Kinase-Substrate (K_site_kin)", "Global Coupling (Cg)", "Local Coupling (Cl)",
                            "Kinase Network (L_alpha)"])

    if mat_opt == "Kinase-Substrate (K_site_kin)":
        # Sites x Kinases
        mat = data["K_site_kin"]
        x_lab = data["kinases"]
        y_lab = data["sites"]
        title = "Kinase (Col) -> Site (Row) Weights"
    elif mat_opt == "Global Coupling (Cg)":
        mat = data["Cg"]
        x_lab = data["sites"]
        y_lab = data["sites"]
        title = "Site <-> Site Crosstalk (Global)"
    elif mat_opt == "Local Coupling (Cl)":
        mat = data["Cl"]
        x_lab = data["sites"]
        y_lab = data["sites"]
        title = "Site <-> Site Crosstalk (Local)"
    else:
        mat = data["L_alpha"]
        x_lab = data["kinases"]
        y_lab = data["kinases"]
        title = "Kinase <-> Kinase Laplacian"

    # Don't plot huge matrices directly if too big
    if mat.shape[0] > 200:
        st.warning("Matrix is large. Showing top interactions only (Sum > 0).")
        # Filter logic could go here, but for now simple heatmap
        fig_mat = px.imshow(mat, title=title, color_continuous_scale="Viridis")
    else:
        fig_mat = px.imshow(mat, x=x_lab, y=y_lab, title=title, color_continuous_scale="Viridis")

    st.plotly_chart(fig_mat, use_container_width=True)

# ==========================================
# TAB 5: Steady State Analysis
# ==========================================
with tab5:
    st.subheader("Long-term Dynamics")

    col_ss1, col_ss2 = st.columns([1, 3])

    with col_ss1:
        # Simulation Controls
        t_end_ss = st.slider("Simulation Duration (min)", min_value=1000, max_value=20000, value=5000, step=1000)
        ss_prot = st.selectbox("Select Protein to Inspect", data["proteins"], key="ss_prot")

        run_ss = st.button("Run Steady State Sim", type="primary")

    with col_ss2:
        if run_ss:
            with st.spinner(f"Simulating up to {t_end_ss} min..."):
                # 1. Setup Time
                t_log = np.logspace(np.log10(0.1), np.log10(t_end_ss), 300)
                t_ss = np.insert(t_log, 0, 0.0)

                # 2. Setup Dims
                K, M, N = len(data["proteins"]), len(data["kinases"]), len(data["sites"])
                ModelDims.set_dims(K, M, N)

                # 3. Build A0 (Manual Construction for Time Mismatch)
                # Initialize zero matrix of correct simulation shape
                A0_ss = np.zeros((K, len(t_ss)), dtype=float)

                # Set initial conditions from data if available
                if data["A_scaled"].size > 0:
                    for k, p_idx in enumerate(data["prot_idx_for_A"]):
                        if data["A_scaled"].shape[1] > 0:
                            A0_ss[p_idx, 0] = data["A_scaled"][k, 0]

                # 4. Simulate
                P_ss, A_ss = simulate_p_scipy(
                    t_ss,
                    data["P_scaled"],  # Init Cond
                    A0_ss,  # Init Cond
                    data["theta"],
                    data["Cg"], data["Cl"], data["site_prot_idx"],
                    data["K_site_kin"], data["R"], data["L_alpha"], data["kin_to_prot_idx"],
                    data["receptor_mask_prot"], data["receptor_mask_kin"],
                    data["meta"].get("mechanism", "dist")
                )

                # 5. Plotting
                fig_ss = make_subplots(rows=1, cols=2,
                                       subplot_titles=(f"{ss_prot} Abundance", f"{ss_prot} Phosphosites"))

                # Protein Abundance
                p_idx = data["proteins"].index(ss_prot)
                fig_ss.add_trace(
                    go.Scatter(x=t_ss, y=A_ss[p_idx], mode='lines', name="Abundance",
                               line=dict(color='black', width=3)),
                    row=1, col=1
                )

                # Phosphosites (Rescaled)
                site_indices = np.where(data["site_prot_idx"] == p_idx)[0]
                colors = px.colors.qualitative.Plotly

                for i, s_idx in enumerate(site_indices):
                    site_name = data["sites"][s_idx].split("_")[-1]
                    c = colors[i % len(colors)]

                    # Rescale
                    y_scaled = baselines[s_idx] + amplitudes[s_idx] * P_ss[s_idx]

                    fig_ss.add_trace(
                        go.Scatter(x=t_ss, y=y_scaled, mode='lines', name=site_name, line=dict(color=c)),
                        row=1, col=2
                    )

                fig_ss.update_xaxes(type="log", title_text="Time (min) - Log Scale")
                fig_ss.update_yaxes(title_text="Fold Change / Abundance", row=1, col=2)
                fig_ss.update_layout(height=500, template="plotly_white", title=f"Approach to Steady State: {ss_prot}")

                st.plotly_chart(fig_ss, use_container_width=True)

                # Convergence Metric
                final_deriv = np.mean(np.abs(P_ss[:, -1] - P_ss[:, -10]))
                if final_deriv < 1e-4:
                    st.success(f"System reached steady state (Mean delta: {final_deriv:.2e})")
                else:
                    st.warning(f"System may not have fully converged (Mean delta: {final_deriv:.2e})")

# ==========================================
# TAB 6: Knockout Simulator
# ==========================================
with tab6:
    st.subheader("In-Silico Perturbation Analysis")
    st.markdown("Compare **Wild Type (WT)** vs **Knockout (KO)** dynamics.")

    c1, c2, c3 = st.columns(3)

    with c1:
        ko_type = st.radio("Target Type", ["Kinase", "Protein", "Phosphosite"])

    with c2:
        if ko_type == "Kinase":
            target_list = data["kinases"]
        elif ko_type == "Protein":
            target_list = data["proteins"]
        else:
            target_list = data["sites"]

        target = st.selectbox("Select Target to Knockout", target_list)

    with c3:
        observe_prot = st.selectbox("Observe Impact On", data["proteins"], index=0)
        t_ko = st.slider("Time (min)", 100, 1000, 240)

    run_ko = st.button("Simulate Knockout", type="primary")

    if run_ko:
        with st.spinner("Calculating Perturbations..."):
            # Setup
            K, M, N = len(data["proteins"]), len(data["kinases"]), len(data["sites"])
            ModelDims.set_dims(K, M, N)

            # Time vector
            t_eval = np.linspace(0, t_ko, 200)

            # A0 Construction (Manual for Time Mismatch)
            A0_ko = np.zeros((K, len(t_eval)), dtype=float)
            if data["A_scaled"].size > 0:
                for k, p_idx in enumerate(data["prot_idx_for_A"]):
                    if data["A_scaled"].shape[1] > 0:
                        A0_ko[p_idx, 0] = data["A_scaled"][k, 0]

            # --- 1. Run Wild Type ---
            P_wt, A_wt = simulate_p_scipy(
                t_eval, data["P_scaled"], A0_ko, data["theta"],
                data["Cg"], data["Cl"], data["site_prot_idx"],
                data["K_site_kin"], data["R"], data["L_alpha"], data["kin_to_prot_idx"],
                data["receptor_mask_prot"], data["receptor_mask_kin"],
                data["meta"].get("mechanism", "dist")
            )

            # --- 2. Prepare KO Parameters ---
            theta_ko = data["theta"].copy()
            K_site_kin_ko = data["K_site_kin"]  # Default reference (pointer)

            if ko_type == "Kinase":
                # Find index of kinase
                try:
                    k_idx = data["kinases"].index(target)
                    # Alpha indices start at: 4*K + 2
                    idx_alpha = 4 * K + 2 + k_idx
                    theta_ko[idx_alpha] = -20.0  # Log-space effectively 0
                except ValueError:
                    st.error("Kinase mapping error.")
                    st.stop()

            elif ko_type == "Protein":
                # Find index of protein
                try:
                    p_idx = data["proteins"].index(target)
                    # s_prod indices start at: 2*K
                    idx_sprod = 2 * K + p_idx
                    theta_ko[idx_sprod] = -20.0
                except ValueError:
                    st.error("Protein mapping error.")
                    st.stop()

            elif ko_type == "Phosphosite":
                # Modify Matrix, not theta
                try:
                    s_idx = data["sites"].index(target)
                    # Create a COPY of the matrix to modify
                    K_site_kin_ko = data["K_site_kin"].copy()
                    K_site_kin_ko[s_idx, :] = 0.0  # Remove all inputs
                except ValueError:
                    st.error("Site mapping error.")
                    st.stop()

            # --- 3. Run KO Simulation ---
            P_ko, A_ko = simulate_p_scipy(
                t_eval, data["P_scaled"], A0_ko, theta_ko,
                data["Cg"], data["Cl"], data["site_prot_idx"],
                K_site_kin_ko, data["R"], data["L_alpha"], data["kin_to_prot_idx"],
                data["receptor_mask_prot"], data["receptor_mask_kin"],
                data["meta"].get("mechanism", "dist")
            )

            # --- 4. Visualization ---
            obs_idx = data["proteins"].index(observe_prot)
            site_indices = np.where(data["site_prot_idx"] == obs_idx)[0]

            # Plot
            fig_ko = go.Figure()

            # Colors
            colors = px.colors.qualitative.Bold

            for i, s_idx in enumerate(site_indices):
                site_name = data["sites"][s_idx].split("_")[-1]
                c = colors[i % len(colors)]

                # Rescale
                wt_curve = baselines[s_idx] + amplitudes[s_idx] * P_wt[s_idx]
                ko_curve = baselines[s_idx] + amplitudes[s_idx] * P_ko[s_idx]

                # Plot WT (Solid)
                fig_ko.add_trace(go.Scatter(
                    x=t_eval, y=wt_curve, mode='lines',
                    line=dict(color=c, width=2),
                    name=f"{site_name} (WT)",
                    legendgroup=site_name
                ))

                # Plot KO (Dash)
                fig_ko.add_trace(go.Scatter(
                    x=t_eval, y=ko_curve, mode='lines',
                    line=dict(color=c, width=2, dash='dot'),
                    name=f"{site_name} (KO)",
                    legendgroup=site_name,
                    showlegend=False
                ))

            fig_ko.update_layout(
                title=f"Effect of {target} ({ko_type} KO) on {observe_prot} sites",
                xaxis_title="Time (min)",
                yaxis_title="Fold Change",
                template="plotly_white",
                height=600
            )

            st.plotly_chart(fig_ko, use_container_width=True)

            # Abundance check
            delta_A = np.mean(A_wt[obs_idx] - A_ko[obs_idx])
            if abs(delta_A) > 0.1:
                st.info(f"Significant change in protein abundance detected for {observe_prot}.")

# ==========================================
# TAB 7: Network Animation (Gravis)
# ==========================================

with tab7:
    st.subheader("Dynamic Network Visualization")
    st.markdown("Visualize the propagation of signals through the kinase-substrate network over time.")

    # --- Controls ---
    c_anim1, c_anim2 = st.columns([1, 3])

    with c_anim1:
        anim_duration = st.slider("Animation Duration (min)", 10, 1000, 120, key="anim_time")

        # NEW: Edge Threshold Slider
        edge_thresh = st.slider("Edge Visibility Threshold",
                                min_value=0.0, max_value=0.1, value=0.001, step=0.001, format="%.4f",
                                help="Lower this if the network looks disconnected.")

        show_crosstalk = st.checkbox("Show Crosstalk Edges?", value=True,
                                     help="Draw interactions between phosphosites (Cg matrix)")

        # Knockout Context
        use_ko_anim = st.checkbox("Apply Knockout?", value=False)
        ko_target_anim = None
        if use_ko_anim:
            ko_target_anim = st.selectbox("KO Target", data["kinases"] + data["proteins"])

        btn_generate_anim = st.button("Generate Animation", type="primary")

    with c_anim2:
        if btn_generate_anim:
            with st.spinner("Building Network Animation..."):

                # 1. Run Simulation (Fine Grid)
                t_anim = np.linspace(0, anim_duration, 40)  # Keep frame count low (~40) for performance
                K, M, N = len(data["proteins"]), len(data["kinases"]), len(data["sites"])
                ModelDims.set_dims(K, M, N)

                # A0 Construction
                A0_anim = np.zeros((K, len(t_anim)), dtype=float)
                if data["A_scaled"].size > 0 and data["A_scaled"].shape[1] > 0:
                    for k, p_idx in enumerate(data["prot_idx_for_A"]):
                        A0_anim[p_idx, 0] = data["A_scaled"][k, 0]

                # Prepare Theta
                theta_anim = data["theta"].copy()
                K_mat_anim = data["K_site_kin"]

                if use_ko_anim and ko_target_anim:
                    # Apply KO logic
                    if ko_target_anim in data["kinases"]:
                        k_idx = data["kinases"].index(ko_target_anim)
                        theta_anim[4 * K + 2 + k_idx] = -20.0
                    elif ko_target_anim in data["proteins"]:
                        p_idx = data["proteins"].index(ko_target_anim)
                        theta_anim[2 * K + p_idx] = -20.0

                # Simulate
                P_anim, A_anim = simulate_p_scipy(
                    t_anim, data["P_scaled"], A0_anim, theta_anim,
                    data["Cg"], data["Cl"], data["site_prot_idx"],
                    K_mat_anim, data["R"], data["L_alpha"], data["kin_to_prot_idx"],
                    data["receptor_mask_prot"], data["receptor_mask_kin"],
                    data["meta"].get("mechanism", "dist")
                )

                # 2. Build Graph (NetworkX)
                G = nx.DiGraph()

                # --- A. ADD NODES ---

                # Add Kinases
                for m, k_name in enumerate(data["kinases"]):
                    prot_idx = data["kin_to_prot_idx"][m]
                    if prot_idx >= 0:
                        activity_trace = A_anim[prot_idx, :]
                    else:
                        activity_trace = np.ones_like(t_anim) * 0.1

                    G.add_node(k_name, size=25, shape="triangle",
                               group="Kinase", color="red",
                               activity=activity_trace.tolist())

                # Add Phosphosites
                for i, s_name in enumerate(data["sites"]):
                    trace = P_anim[i, :]
                    G.add_node(s_name, size=15, shape="circle",
                               group="Phosphosite", color="blue",
                               activity=trace.tolist())

                # --- B. ADD EDGES (With Slider Threshold) ---

                # 1. Kinase -> Site Edges
                rows, cols = np.where(data["K_site_kin"] > edge_thresh)
                for r, c in zip(rows, cols):
                    site_node = data["sites"][r]
                    kin_node = data["kinases"][c]
                    weight = data["K_site_kin"][r, c]
                    G.add_edge(kin_node, site_node, weight=float(weight), color="gray", opacity=0.5)

                # 2. Site <-> Site Crosstalk (Cg)
                if show_crosstalk:
                    # Cg is N x N
                    rows_cg, cols_cg = np.where(data["Cg"] > edge_thresh)
                    for r, c in zip(rows_cg, cols_cg):
                        if r == c: continue  # Skip self loops
                        s1 = data["sites"][r]
                        s2 = data["sites"][c]
                        w = data["Cg"][r, c]
                        # Use a different color for crosstalk (e.g., orange/dashed)
                        G.add_edge(s1, s2, weight=float(w), color="#FFA500", opacity=0.3)

                # 3. Generate Single Graph (Final State)
                # We pick the last time point to show the "result" of the simulation
                t_idx = -1

                Gt = G.copy()

                for node in Gt.nodes():
                    act_list = G.nodes[node]['activity']
                    val = act_list[t_idx]  # Capture the final value only

                    # Normalize visual intensity
                    intensity = int(255 * val)
                    intensity = max(0, min(255, intensity))

                    # Color Logic
                    if Gt.nodes[node]['group'] == 'Kinase':
                        # Bright Red for high activity, Dark Red for low
                        hex_col = f"#{intensity:02x}0000"
                    else:
                        # Bright Green for high phosphorylation, Dark Blue/Black for low
                        # Mixing Blue (low) to Green (high)
                        inv_intensity = 255 - intensity
                        hex_col = f"#00{intensity:02x}{inv_intensity:02x}"

                    Gt.nodes[node]['color'] = hex_col
                    Gt.nodes[node]['label'] = f"{node}\n({val:.2f})"

                    # Optional: Pulse size
                    base_size = G.nodes[node]['size']
                    Gt.nodes[node]['size'] = base_size + (val * 10)

                # Pass as a single-item list so the rest of the code works unchanged
                graphs = [Gt]

                # 4. Render
                if len(G.edges) == 0:
                    st.warning("No edges found! Try lowering the 'Edge Visibility Threshold'.")
                else:
                    st.caption(
                        f"Rendering {len(graphs)} frames with {len(G.nodes)} nodes and {len(G.edges)} edges...")

                    fig = gv.d3(
                        data=graphs,
                        graph_height=650,

                        # --- Visualization Settings ---
                        zoom_factor=0.6,
                        show_menu=True,  # Enables the bottom menu for Play/Pause/Settings
                        show_menu_toggle_button=True,
                        show_details=True,  # Sidebar for node details on click

                        # --- Node & Edge Appearance ---
                        node_label_data_source='label',  # Uses the 'label' attribute we set (Name + Value)
                        node_size_factor=1.0,
                        node_hover_neighborhood=True,  # Highlight connected nodes on hover
                        edge_size_factor=1.0,
                        edge_curvature=0.2,  # Slight curve looks better for complex networks

                        # --- Physics / Layout Algorithm ---
                        layout_algorithm_active=True,

                        # Repulsion (keeps nodes apart)
                        use_many_body_force=True,
                        many_body_force_strength=-150.0,

                        # Links (pulls connected nodes together)
                        use_links_force=True,  # Correct parameter name
                        links_force_distance=150.0,  # Correct parameter name
                        links_force_strength=0.5,

                        # Centering (keeps graph in view)
                        use_centering_force=True
                    )

                    st.components.v1.html(fig.to_html(), height=700)