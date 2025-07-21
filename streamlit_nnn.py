
import copy
import math

import numpy as np
import pandas as pd


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
import modules.NNN_modules as nnn_minimal    # your new module with NNNModel & GeoTimeSeriesDataset
sys.modules['__main__'] = nnn_minimal
sys.modules['main']     = nnn_minimal
sys.modules['nnn_for_mmm_with_scenario_simulator'] = nnn_minimal

st.session_state['DEVICE'] = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")




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
            device = st.session_state['DEVICE']
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
                st.session_state.model_obj = workspace.get('model')

                # Stash under Model‑Performance‑specific keys:
                st.session_state['mp_best_wts']     = workspace.get('best_wts')
                st.session_state['mp_N_GEOS']       = workspace.get('N_GEOS')
                st.session_state['mp_NUM_CHANNELS'] = workspace.get('NUM_CHANNELS')
                st.session_state['mp_TIME_STEPS']   = workspace.get('TIME_STEPS')
                st.session_state['mp_X']            = workspace.get('X')
                st.session_state['mp_Y']            = workspace.get('Y')
                st.session_state['mp_sales']        = workspace.get('sales')
                required_mp = [
                    'mp_best_wts',
                    'mp_N_GEOS',
                    'mp_NUM_CHANNELS',
                    'mp_TIME_STEPS',
                    'mp_X',
                    'mp_Y',
                    'mp_sales',
                ]
                st.session_state['mp_ready'] = all(
                    st.session_state.get(k) is not None for k in required_mp
                )

                # Stash under Creative‑Scenarios‑specific keys:
                st.session_state['cs_best_wts']            = workspace.get('best_wts')
                st.session_state['cs_N_GEOS']              = workspace.get('N_GEOS')
                st.session_state['cs_TIME_STEPS']          = workspace.get('TIME_STEPS')
                st.session_state['cs_NUM_CHANNELS']        = workspace.get('NUM_CHANNELS')
                st.session_state['cs_X']                   = workspace.get('X')
                st.session_state['cs_creative_embeddings'] = workspace.get('creative_embeddings')
                st.session_state['cs_id_worst']            = workspace.get('id_worst')
                st.session_state['cs_id_best']             = workspace.get('id_best')
                st.session_state['cs_pred_total_sales']    = workspace.get('pred_total_sales')
                required_cs = [
                    'cs_best_wts',
                    'cs_N_GEOS',
                    'cs_TIME_STEPS',
                    'cs_NUM_CHANNELS',
                    'cs_X',
                    'cs_creative_embeddings',
                    'cs_id_worst',
                    'cs_id_best',
                    'cs_pred_total_sales',
                ]
                st.session_state['cs_ready'] = all(
                    st.session_state.get(k) is not None for k in required_cs
                )

                # Stash under Scenario‑Planner‑specific keys:
                st.session_state['sp_X']              = workspace.get('X')
                st.session_state['sp_DEVICE']         = st.session_state['DEVICE']
                st.session_state['sp_N_GEOS']         = workspace.get('N_GEOS')
                st.session_state['sp_TIME_STEPS']     = workspace.get('TIME_STEPS')
                st.session_state['sp_NUM_CHANNELS']   = workspace.get('NUM_CHANNELS')
                st.session_state['sp_MEDIA_CHANNELS'] = workspace.get('MEDIA_CHANNELS')
                required_sp = [
                    'sp_X',
                    'sp_DEVICE',
                    'sp_N_GEOS',
                    'sp_TIME_STEPS',
                    'sp_NUM_CHANNELS',
                    'sp_MEDIA_CHANNELS',
                ]
                st.session_state['sp_ready'] = all(
                    st.session_state.get(k) is not None for k in required_sp
                )

                # Stash under Creative‑Scenario‑Planner‑specific keys:
                st.session_state['csp_X']         = workspace.get('X')
                st.session_state['csp_DEVICE']    = st.session_state['sp_DEVICE']
                st.session_state['csp_tokeniser'] = workspace.get('tokeniser')
                required_csp = [
                    'csp_X',
                    'csp_DEVICE',
                    'csp_tokeniser',
                ]
                st.session_state['csp_ready'] = all(
                    st.session_state.get(k) is not None for k in required_csp
                )

                # Stash under Attribution‑specific keys:
                st.session_state['attr_X']              = workspace.get('X')
                st.session_state['attr_NUM_CHANNELS']   = workspace.get('NUM_CHANNELS')
                st.session_state['attr_DEVICE']         = st.session_state['sp_DEVICE']
                st.session_state['attr_MEDIA_CHANNELS'] = workspace.get('MEDIA_CHANNELS')
                required_attr = [
                    'attr_X',
                    'attr_NUM_CHANNELS',
                    'attr_DEVICE',
                    'attr_MEDIA_CHANNELS',
                ]
                st.session_state['attr_ready'] = all(
                    st.session_state.get(k) is not None for k in required_attr
                )

                # Stash under Insights‑on‑Creatives‑specific keys:
                st.session_state['ioc_X']           = workspace.get('X')
                st.session_state['ioc_DEVICE']      = st.session_state['sp_DEVICE']
                st.session_state['ioc_E_np']        = workspace.get('E_np')
                st.session_state['ioc_tokenizer']   = workspace.get('tokenizer')
                st.session_state['ioc_token2id']    = workspace.get('token2id')
                st.session_state['ioc_messages']    = workspace.get('messages')
                required_ioc = [
                    'ioc_X',
                    'ioc_DEVICE',
                    'ioc_E_np',
                    'ioc_tokenizer',
                    'ioc_token2id',
                    'ioc_messages',
                ]
                st.session_state['ioc_ready'] = all(
                    st.session_state.get(k) is not None for k in required_ioc
                )

                # Stash under Impulse‑Response‑specific keys:
                st.session_state['ir_X']              = workspace.get('X')
                st.session_state['ir_media_spend']    = workspace.get('media_spend')
                st.session_state['ir_DEVICE']         = st.session_state['sp_DEVICE']
                st.session_state['ir_MEDIA_CHANNELS'] = workspace.get('MEDIA_CHANNELS')
                required_ir = [
                    'ir_X',
                    'ir_media_spend',
                    'ir_DEVICE',
                    'ir_MEDIA_CHANNELS',
                ]
                st.session_state['ir_ready'] = all(
                    st.session_state.get(k) is not None for k in required_ir
                )

                # Stash under Impulse‑Response‑Animated‑specific keys:
                st.session_state['ira_X']               = workspace.get('X')
                st.session_state['ira_media_spend']     = workspace.get('media_spend')
                st.session_state['ira_DEVICE']          = st.session_state['sp_DEVICE']
                st.session_state['ira_MEDIA_CHANNELS']  = workspace.get('MEDIA_CHANNELS')
                st.session_state['ira_EMBED_DIM']       = workspace.get('EMBED_DIM')
                required_ira = [
                    'ira_X',
                    'ira_media_spend',
                    'ira_DEVICE',
                    'ira_MEDIA_CHANNELS',
                    'ira_EMBED_DIM',
                ]
                st.session_state['ira_ready'] = all(
                    st.session_state.get(k) is not None for k in required_ira
                )




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
     

# --- Page: Model Performance ---
elif page == "Model Performance":
    st.header("Model Performance")

    # only show if model performance vars are ready and a pretrained model is loaded
    if st.session_state.get('mp_ready', False) and st.session_state.get('model_obj') is not None:
        # pull out page-specific vars
        model    = st.session_state.model_obj
        N        = st.session_state['mp_N_GEOS']
        T        = st.session_state['mp_TIME_STEPS']
        C        = st.session_state['mp_NUM_CHANNELS']
        X        = st.session_state['mp_X']
        Y        = st.session_state['mp_Y']

        model.eval()

        DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        geo_idx = torch.arange(N, device=DEVICE)

        # prepare inputs
        Xf = torch.tensor(X, dtype=torch.float32, device=DEVICE)
        Xb = Xf.clone()
        Xb[..., :C, :] = 0.0

        # get full and baseline outputs
        with torch.no_grad():
            full = model(Xf, geo_idx).cpu().numpy()
            base = model(Xb, geo_idx).cpu().numpy()

        # aggregate sales
        sales_full   = np.expm1(full[..., 0]).sum(axis=0)
        sales_base   = np.expm1(base[..., 0]).sum(axis=0)
        sales_actual = np.expm1(Y[..., 0]).sum(axis=0)

        # static plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(sales_actual, label="Actual Sales", linewidth=2)
        ax.plot(sales_full,   label="Model Prediction", linewidth=2)
        ax.plot(sales_base,   linestyle="--", label="No-Media Baseline", linewidth=2)
        ax.set_title("Weekly Aggregate Sales: Actual vs. Predicted vs. No-Media Baseline", fontsize=16)
        ax.set_xlabel("Week", fontsize=14)
        ax.set_ylabel("Sales", fontsize=14)
        ax.legend(frameon=True, fontsize=12)
        ax.grid(True, linestyle="--", alpha=0.5)
        fig.tight_layout()
        st.pyplot(fig)

    else:
        st.info(
            "✅ First go to Data & Model Info, tick “Use simulated data” "
            "and click Go to load the pretrained model before viewing Model Performance."
        )



# --- Page: Channel Attribution ---
elif page == "Channel Attribution":
    st.header("Channel Attribution")

    if st.session_state.get('attr_ready', False) and st.session_state.get('model_obj') is not None:
        import numpy as np
        import matplotlib.pyplot as plt
        from modules.NNN_modules import scenario_planner_per_channel

        # pull out session‐state vars
        model           = st.session_state.model_obj
        X               = st.session_state['attr_X']
        DEVICE          = st.session_state['attr_DEVICE']
        NUM_CHANNELS    = st.session_state['attr_NUM_CHANNELS']
        MEDIA_CHANNELS  = st.session_state['attr_MEDIA_CHANNELS']

        # define permutation importance
        def permutation_importance(model, X, geo_idx, time_step, channel_idx):
            # baseline prediction
            df_base = scenario_planner_per_channel(
                model, X, [1.0]*NUM_CHANNELS,
                geo_idx=geo_idx, time_step=time_step, device=DEVICE
            )
            base_sales = df_base["total_sales"].iat[0]

            # permute one channel across geos
            X_perm = X.copy()
            np.random.shuffle(X_perm[:, :, channel_idx, :])

            # re-predict
            df_perm = scenario_planner_per_channel(
                model, X_perm, [1.0]*NUM_CHANNELS,
                geo_idx=geo_idx, time_step=time_step, device=DEVICE
            )
            perm_sales = df_perm["total_sales"].iat[0]

            return base_sales - perm_sales

        # parameters for attribution
        geo_idx   = 3    # which geo to analyze
        time_step = 10   # which week to analyze

        # compute importances
        importances = [
            permutation_importance(model, X, geo_idx, time_step, j)
            for j in range(NUM_CHANNELS)
        ]

        # plot
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(MEDIA_CHANNELS, importances)
        ax.set_xticks(range(len(MEDIA_CHANNELS)))
        ax.set_xticklabels(MEDIA_CHANNELS, rotation=45, ha='right')
        ax.set_ylabel("Sales drop when permuted")
        ax.set_title(f"Permutation Importance: Geo {geo_idx}, Week {time_step}")
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        fig.tight_layout()
        st.pyplot(fig)

    else:
        st.info(
            "✅ First go to Data & Model Info, tick “Use simulated data” "
            "and click Go to load the pretrained model before viewing Channel Attribution."
        )


# --- Page: Impulse-Response Analysis ---
elif page == "Impulse-Response Analysis":
    st.header("Impulse-Response Analysis")

    # Only show static & animated if both sets of data are ready and a model is loaded
    if (
        st.session_state.get('ir_ready', False)
        and st.session_state.get('ira_ready', False)
        and st.session_state.get('model_obj') is not None
    ):
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
        import torch

        # Pull out static vars
        model         = st.session_state.model_obj
        X_static      = st.session_state['ir_X']
        media_spend   = st.session_state['ir_media_spend']
        DEVICE        = st.session_state['ir_DEVICE']
        MEDIA_CHANNELS= st.session_state['ir_MEDIA_CHANNELS']

        # --- Static impulse-response ---
        delta_pct_dict = impulse_response_analysis(
            model, X_static, media_spend, device=DEVICE
        )
        st.subheader("Static Impulse Response")
        st.dataframe(pd.DataFrame(delta_pct_dict))

        # Pull out animated vars (EMBED_DIM)
        EMBED_DIM = st.session_state['ira_EMBED_DIM']

        # --- Animated impulse-response ---
        st.subheader("Animated Impulse Response")
        # Prepare tensors
        X_t         = torch.tensor(X_static, dtype=torch.float32, device=DEVICE)
        full_geo_ids= torch.arange(X_t.shape[0], device=DEVICE)
        model.eval()
        eps = 1e-9
        # Baseline
        with torch.no_grad():
            Yb = model(X_t, full_geo_ids)
        sales_base = torch.expm1(Yb[..., 0]).cpu().numpy().sum(axis=0)
        std_rg      = np.std(media_spend, axis=1)

        # Set up figure
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
        axes = axes.flatten()
        lines = []
        for ax in axes:
            ln, = ax.plot([], [], lw=2, marker='o')
            lines.append(ln)
            ax.set_xlim(1, 13)
            ax.set_ylim(-0.7, 1.5)
            ax.set_xlabel("Week")
            ax.set_ylabel("Δ Sales (%)")
            ax.set_title("")
            ax.grid(axis='y')

        def init():
            for ln in lines:
                ln.set_data([], [])
            return lines

        def update(t):
            for j, ln in enumerate(lines):
                X_scen = X_t.clone()
                s = media_spend[:, t, j]
                inc = std_rg[:, j]
                s_prime = s + inc
                for g in range(X_scen.shape[0]):
                    v = torch.log1p(torch.tensor(s_prime[g], device=DEVICE))
                    V = v.repeat(EMBED_DIM)
                    norm = torch.norm(V)
                    E = (V / (norm + eps)) * torch.log1p(norm)
                    X_scen[g, t, j, :] = E
                with torch.no_grad():
                    Yi = model(X_scen, full_geo_ids)
                sales_imp = torch.expm1(Yi[..., 0]).cpu().numpy().sum(axis=0)
                delta_pct = (sales_imp - sales_base) / (sales_base + eps) * 100
                weeks = np.arange(1, 14)
                ln.set_data(weeks, delta_pct[1:14])
            for idx, ax in enumerate(axes):
                ax.set_title(f"Impulse: {MEDIA_CHANNELS[idx]} at t={t}")
            return lines

        ani = FuncAnimation(fig, update, frames=range(1, 14), init_func=init, blit=False, repeat=False)
        plt.tight_layout()
        html = ani.to_jshtml()
        plt.close(fig)
        st.components.v1.html(html, height=600)

    else:
        st.info(
            "✅ First go to Data & Model Info, tick “Use simulated data” "
            "and click Go to load the pretrained model before viewing Impulse-Response Analysis."
        )


# --- Page: Actionable Insights ---
elif page == "Actionable Insights":
    st.header("Actionable Insights")
    if st.session_state.get('ioc_ready', False) and st.session_state.get('model_obj') is not None:
        model      = st.session_state.model_obj
        X          = st.session_state['ioc_X']
        DEVICE     = st.session_state['ioc_DEVICE']
        E_np       = st.session_state['ioc_E_np']
        tokenizer  = st.session_state['ioc_tokenizer']
        token2id   = st.session_state['ioc_token2id']
        messages   = st.session_state['ioc_messages']
        # … your actionable insights code goes here …
    else:
        st.info(
            "✅ First go to Data & Model Info, tick “Use simulated data” "
            "and click Go to load the pretrained model before viewing Actionable Insights."
        )

# --- Page: Scenario Planner ---
elif page == "Scenario Planner":
    st.header("Scenario Planner")
    if st.session_state.get('sp_ready', False) and st.session_state.get('model_obj') is not None:
        model           = st.session_state.model_obj
        X               = st.session_state['sp_X']
        DEVICE          = st.session_state['sp_DEVICE']
        N_GEOS          = st.session_state['sp_N_GEOS']
        TIME_STEPS      = st.session_state['sp_TIME_STEPS']
        NUM_CHANNELS    = st.session_state['sp_NUM_CHANNELS']
        MEDIA_CHANNELS  = st.session_state['sp_MEDIA_CHANNELS']
        # … your scenario-planner code goes here …
    else:
        st.info(
            "✅ First go to Data & Model Info, tick “Use simulated data” "
            "and click Go to load the pretrained model before viewing Scenario Planner."
        )

# --- Page: Creative-Piece Simulator ---
elif page == "Creative-Piece Simulator":
    st.header("Creative-Piece Simulator")
    if st.session_state.get('csp_ready', False) and st.session_state.get('model_obj') is not None:
        model      = st.session_state.model_obj
        X          = st.session_state['csp_X']
        DEVICE     = st.session_state['csp_DEVICE']
        tokeniser  = st.session_state['csp_tokeniser']
        # … your creative-piece simulator code goes here …
    else:
        st.info(
            "✅ First go to Data & Model Info, tick “Use simulated data” "
            "and click Go to load the pretrained model before viewing Creative-Piece Simulator."
        )

