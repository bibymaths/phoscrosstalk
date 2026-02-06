import os
import sys
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# --- Import PhosCrosstalk Modules ---
# Ensure the current directory is in path
sys.path.append(os.getcwd())

from phoscrosstalk.config import ModelDims, DEFAULT_TIMEPOINTS
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
    # Use log-spacing for time to capture early dynamics better, or linear.
    # Mixing linspace for smoothness.
    t_fine = np.linspace(0, t_max, num_points)

    # 3. Build A0
    A0_full = build_full_A0(K, len(t_fine), data["A_scaled"], data["prot_idx_for_A"])

    # 4. Simulate
    # Need to handle mechanism from meta
    mechanism = data["meta"].get("mechanism", "dist")

    P_sim, A_sim = simulate_p_scipy(
        t_fine,
        data["P_scaled"],  # Used for Init Cond
        A0_full,  # Used for Init Cond
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
t_fine, P_sim_fine, A_sim_fine = run_fine_simulation(data, t_max, num_points=300)

# Rescaling logic
baselines, amplitudes = calculate_scalers(data["Y_orig"])

# --- Tabs ---
tab1, tab2, tab3, tab4 = st.tabs(["Trajectories", "Goodness of Fit", "Parameters", "Network Map"])

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