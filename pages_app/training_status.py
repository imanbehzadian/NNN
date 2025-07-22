# pages/training_status.py

import streamlit as st
import time
import io
import torch
import pickle

def render():
    st.header("Training Status")

    # Initialize load‐results flag
    if 'load_results_done' not in st.session_state:
        st.session_state.load_results_done = False

    # Action buttons
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
                buf = io.BytesIO(uploaded_pkl.read())
                device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
                workspace = torch.load(buf, map_location=device, weights_only=False)
                # Inject into session_state
                for name, obj in workspace.items():
                    st.session_state[name] = obj
                st.session_state.model_obj = workspace.get('model_obj')
                st.session_state.load_results_done = True
                st.success("✅ Results loaded into session")

    # If a pretrained model is already in session, skip training UI
    if st.session_state.get('model_obj') is not None:
        st.success("✅ Pretrained model is ready")
        return

    # Simulated training progress
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
    step_ph  = [st.empty() for _ in steps]
    error_ph = [st.empty() for _ in errors]

    def render_steps(done_idx):
        for i, s in enumerate(steps):
            icon = "✅" if i <= done_idx else "⚪"
            step_ph[i].markdown(f"{icon}  {s}")

    def render_errors():
        for ph, msg in zip(error_ph, errors):
            ph.text(msg)
            time.sleep(0.5)

    if start_training:
        # Simulate progress
        for i in range(len(steps)):
            render_steps(i)
            time.sleep(1)
        render_errors()
        render_steps(len(steps) - 1)
        st.success("✅ Model is ready")

        # Show Save Results only if not using simulated data and not already loaded
        if (not st.session_state.get('use_simulated', False)
                and not st.session_state.load_results_done):
            if st.button("Save Results", key="save_results_button"):
                # Collect picklable variables
                picklable = []
                for name, obj in globals().items():
                    try:
                        pickle.dumps(obj)
                        picklable.append(name)
                    except Exception:
                        pass

                # Build and save workspace dict
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
