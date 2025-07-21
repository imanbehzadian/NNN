
import copy
import math

import numpy as np
import pandas as pd
import pprint

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from transformers import BertTokenizer, BertModel


from captum.attr import IntegratedGradients

from IPython.display import HTML, display


# For Dashboarding
import streamlit as st
from datetime import date, timedelta, time as dt_time
import time


import sys
import NNN_modules as nnn_minimal    # your new module with NNNModel & GeoTimeSeriesDataset
sys.modules['main'] = nnn_minimal  # tell the un‑pickler where to find “main.GeoTimeSeriesDataset”, etc.
sys.modules['nnn_for_mmm_with_scenario_simulator'] = nnn_minimal




# --- Authentication ---
VALID_USERS = ['', 'admin', 'iman', 'hamish', 'shalcky', 'andy']
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

# Sidebar feedback container (persistent)
login_feedback = st.sidebar.empty()

# Sidebar auth inputs (always visible)
username = st.sidebar.text_input('Username', key='username_input')
password = st.sidebar.text_input('Password', type='password', key='password_input')
login = st.sidebar.button('Login', key='login_button')

# Authentication logic (only show errors)
if login:
    if username not in VALID_USERS:
        login_feedback.error('🚫 User not registered')
    elif password != username:
        login_feedback.error('❌ Incorrect password')
    else:
        st.session_state.authenticated = True
        login_feedback.empty()

# Stop execution if not authenticated
if not st.session_state.authenticated:
    st.sidebar.title("Please Log In")
    st.stop()

# --- Page Navigation ---
page = st.sidebar.radio("Go to", [
    "Overview",
    "Data & Model Info",
    "Training Status",
    "Model Performance",
    "Channel Attribution",
    "Impulse-Response Analysis",
    "Actionable Insights",
    "Scenario Planner",
    "Creative-Piece Simulator",
], index=0)

# --- Shared session state placeholders ---
st.session_state.setdefault('data_df', None)
st.session_state.setdefault('hyperparams', {})
st.session_state.setdefault('model_obj', None)

# --- Page: Overview ---
if page == "Overview":
    st.header("Overview")
    st.markdown(
        """
This notebook demonstrates the implementation of the Neural Nested Network (NNN) as proposed in the 2024 paper *Neural Nested Networks for Unified Marketing Mix Modeling*. NNN is a Transformer-based model for learning advertising effects with both direct and indirect pathways (e.g., media → search → conversion). It supports interpretable multi-task modeling of sales and intermediate KPIs like branded search volume, all in an end-to-end neural framework.

We also provide rich visualization and analysis tools to extract actionable insights, plan scenarios, simulate the impact of different creative messages on sales, and attribute performance across channels.

📘 **Paper Summary**
- Models both direct media → sales and indirect media → search → sales paths using a nested architecture.
- Uses factored self-attention to separately capture time and channel effects.
- Learns both sales and search outputs with residual path weighting to model their interaction.
- Enables flexible input design (e.g., embeddings for creative features, seasonality, external signals).
- Unlike Bayesian MMM approaches, NNN allows for transfer learning, non-additive synergies, and rich feature embeddings.
        """
    )
    end_date = date.today()
    start_date = end_date - timedelta(days=30)
    ds = st.date_input("Select analysis window", [start_date, end_date], key='overview_dates')
    cols = st.columns(2)
    cols[0].metric("Baseline Sales", "--")
    cols[1].metric("Media-driven Sales", "--")

# --- Page: Data & Model Info ---
elif page == "Data & Model Info":
    st.header("Data & Model Info")
    uploaded = st.file_uploader(
        "Upload CSV data", type=['csv'], key='data_upload'
    )
    use_sim = st.checkbox(
        "Use simulated data (load pretrained model)", key='use_simulated'
    )
    hyper_file = st.file_uploader(
        "Upload hyperparams.txt (optional)", type=['txt'], key='hyper_upload'
    )
    # Align Go button: cols_go = st.columns([20,2])  # <--- adjust to align
    cols_go = st.columns([20,2])
    with cols_go[1]:
        go = st.button("Go", key='go_button')
    msg_ph = st.empty()


    if go:
        if use_sim:
            file_name = "NNN_vars_3.pkl"
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
            try:
                # now torch.load can resolve your custom classes via sys.modules['main']
                workspace = torch.load(
                    file_name,
                    map_location=device,
                    weights_only=False
                )
                # inject each saved var back into globals()
                for name, obj in workspace.items():
                    globals()[name] = obj
                st.session_state.model_obj = workspace.get('model_obj')

                # stash under Model‑Performance‑specific keys
                st.session_state['mp_model']        = workspace.get('model_obj')      # your trained model
                st.session_state['mp_N_GEOS']       = workspace.get('N_GEOS')
                st.session_state['mp_TIME_STEPS']   = workspace.get('TIME_STEPS')
                st.session_state['mp_NUM_CHANNELS'] = workspace.get('NUM_CHANNELS')
                st.session_state['mp_X_np']         = workspace.get('X_np')
                st.session_state['mp_Y_np']         = workspace.get('Y_np')

                # only mark “ready” if everything’s present
                required_mp = ['mp_model','mp_N_GEOS','mp_TIME_STEPS','mp_NUM_CHANNELS','mp_X_np','mp_Y_np']
                st.session_state['mp_ready'] = all(st.session_state.get(k) is not None for k in required_mp)


                msg_ph.success(f"✅ Loaded pretrained variables from {file_name} onto {device}")
            except FileNotFoundError:
                msg_ph.error(f"❌ File not found: {file_name}")
            except Exception as e:
                msg_ph.error(f"❌ Failed to load {file_name}: {e}")
        elif uploaded is not None:
            df = pd.read_csv(uploaded)
            st.session_state.data_df = df
            msg_ph.success("✅ Data loaded successfully.")
        else:
            st.warning("Please upload a CSV or tick 'Use simulated data'.")


    # Handle hyperparams override
    if hyper_file is not None:
        content = hyper_file.read().decode()
        params = {k.strip(): v.strip() for k, v in (
            line.split('=', 1) for line in content.splitlines() if '=' in line
        )}
        st.session_state.hyperparams = params
        st.success("✅ Hyperparameters overridden.")
    df = st.session_state.data_df
    if df is not None and not df.empty:
        st.subheader("Data Preview")
        st.dataframe(df.head())



elif page == "Training Status":
    st.header("Training Status")

    # Track if user has loaded results this session
    if 'load_results_done' not in st.session_state:
        st.session_state.load_results_done = False

    # --- Action buttons ---
    action_cols = st.columns([1, 1])
    with action_cols[0]:
        start_training = st.button("Start Training", key='start_training')
    with action_cols[1]:
        load_button = st.button(
            "Load Results", key='load_results_button',
            help="Load a .pkl of an already trained model to skip training."
        )
        if load_button:
            uploaded_pkl = st.file_uploader(
                "Select .pkl to load", type=["pkl"], key="load_results_file"
            )
            if uploaded_pkl is not None:
                # Load workspace dict via torch
                buf = io.BytesIO(uploaded_pkl.read())
                device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
                workspace = torch.load(buf, map_location=device, weights_only=False)
                # Inject variables into globals
                for name, obj in workspace.items():
                    globals()[name] = obj
                # Pick up the model object
                st.session_state.model_obj = workspace.get('model') or workspace.get('model_obj')
                st.session_state.load_results_done = True
                st.success("✅ Results loaded into session")

    # # If we have a pretrained model, skip training UI
    # if st.session_state.get('model_obj') is not None:
    #     st.success("✅ Pretrained model is ready")
    #     return  # or just skip the training section

    # --- Simulated training UI ---
    steps = ["Loading data", "Preprocessing Data", "Create Embeddings", "Fitting neural network"]
    errors = [
        "Epoch 050 | Sales RMSE: 0.10, MAE: 0.08, R²: 0.829 | Search RMSE: 0.4, MAE: 0.4, R²: -7.956",
        "Epoch 100 | Sales RMSE: 0.08, MAE: 0.06, R²: 0.896 | Search RMSE: 0.4, MAE: 0.4, R²: -5.325",
        "Epoch 150 | Sales RMSE: 0.07, MAE: 0.06, R²: 0.910 | Search RMSE: 0.3, MAE: 0.3, R²: -3.319",
        "Epoch 200 | Sales RMSE: 0.08, MAE: 0.06, R²: 0.907 | Search RMSE: 0.3, MAE: 0.3, R²: -2.681",
        "Epoch 250 | Sales RMSE: 0.07, MAE: 0.06, R²: 0.908 | Search RMSE: 0.2, MAE: 0.2, R²: -1.483",
        "Epoch 300 | Sales RMSE: 0.07, MAE: 0.06, R²: 0.913 | Search RMSE: 0.4, MAE: 0.4, R²: -5.609",
        "Epoch 350 | Sales RMSE: 0.07, MAE: 0.06, R²: 0.915 | Search RMSE: 0.3, MAE: 0.3, R²: -4.356",
        "Epoch 400 | Sales RMSE: 0.07, MAE: 0.06, R²: 0.917 | Search RMSE: 0.1, MAE: 0.1, R²: 0.171",
        "Epoch 450 | Sales RMSE: 0.07, MAE: 0.06, R²: 0.917 | Search RMSE: 0.3, MAE: 0.3, R²: -2.411",
        "Epoch 500 | Sales RMSE: 0.07, MAE: 0.06, R²: 0.917 | Search RMSE: 0.2, MAE: 0.2, R²: -1.104",
    ]
    step_ph = [st.empty() for _ in steps]
    error_ph = [st.empty() for _ in errors]

    def render_steps(done_idx):
        for i, s in enumerate(steps):
            icon = "✅" if i <= done_idx else "⚪"
            step_ph[i].markdown(f"{icon} {s}")

    def render_errors():
        for ph, msg in zip(error_ph, errors):
            ph.text(msg)
            time.sleep(0.5)

    if start_training:
        for i in range(len(steps)):
            render_steps(i)
            time.sleep(1)
        render_errors()
        render_steps(len(steps) - 1)
        st.success("✅ Model is ready")


        # Only show Save Results if training (not sim data) and not already loaded
        if (not st.session_state.get('use_simulated', False)
            and not st.session_state.load_results_done):
            if st.button("Save Results", key="save_results_button"):
                import io
                import pickle

                picklable = []
                to_save = globals()   #TODO to narrow down the list of saved variables change this line
                for name in to_save:
                    if name in globals():
                        try:
                            pickle.dumps(globals()[name])
                            picklable.append(name)
                        except Exception:
                            pass

                # Build workspace dict and serialize via torch
                workspace = {n: globals()[n] for n in picklable}
                buf = io.BytesIO()
                torch.save(workspace, buf)
                buf.seek(0)

                st.download_button(
                    "Download NNN_vars_3.pkl",
                    data=buf.getvalue(),
                    file_name="NNN_vars_3.pkl",
                    mime="application/octet-stream",
                    key="download_results_button"
                )
     

elif page == "Model Performance":
    st.header("Model Performance")

    if st.session_state.get('mp_ready', False):
        # pull out your page‑specific vars
        model    = st.session_state['mp_model']
        N        = st.session_state['mp_N_GEOS']
        T        = st.session_state['mp_TIME_STEPS']
        C        = st.session_state['mp_NUM_CHANNELS']
        X_np     = st.session_state['mp_X_np']
        Y_np     = st.session_state['mp_Y_np']

        model.eval()
        import numpy as np
        import matplotlib.pyplot as plt
        DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        geo_idx = torch.arange(N, device=DEVICE)

        Xf = torch.tensor(X_np, dtype=torch.float32, device=DEVICE)
        Xb = Xf.clone(); Xb[..., :C, :] = 0.0

        with torch.no_grad():
            full = model(Xf, geo_idx).cpu().numpy()
            base = model(Xb, geo_idx).cpu().numpy()

        sales_full   = np.expm1(full[..., 0]).sum(axis=0)
        sales_base   = np.expm1(base[..., 0]).sum(axis=0)
        sales_actual = np.expm1(Y_np[..., 0]).sum(axis=0)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(sales_actual, label="Actual",      linewidth=2)
        ax.plot(sales_full,   label="Predicted",   linewidth=2)
        ax.plot(sales_base,   label="No‑Media BL", linestyle="--", linewidth=2)
        ax.set_title("Weekly Sales: Actual vs Predicted", fontsize=16)
        ax.set_xlabel("Week", fontsize=14)
        ax.set_ylabel("Sales", fontsize=14)
        ax.legend(frameon=True, fontsize=12)
        ax.grid(True, linestyle="--", alpha=0.5)
        fig.tight_layout()
        st.pyplot(fig)

    else:
        st.info(
            "✅ First go to Data & Model Info, tick “Use simulated data” and click Go "
            "to load the pretrained model before viewing performance."
        )


# --- Page: Channel Attribution ---
elif page == "Channel Attribution":
    st.header("Channel Attribution")
    st.write("Media vs. baseline attribution by channel.")

# --- Page: Impulse-Response Analysis ---
elif page == "Impulse-Response Analysis":
    st.header("Impulse-Response Analysis")
    st.write("One-time spend shock propagation.")

# --- Page: Actionable Insights ---
elif page == "Actionable Insights":
    st.header("Actionable Insights")
    st.write("Identify key embedding dimensions and aligned terms.")

# --- Page: Scenario Planner ---
elif page == "Scenario Planner":
    st.header("Scenario Planner")
    st.write("Simulate spend multipliers across channels.")

# --- Page: Creative-Piece Simulator ---
elif page == "Creative-Piece Simulator":
    st.header("Creative-Piece Simulator")
    st.write("Batch file or single-entry creative simulation.")
