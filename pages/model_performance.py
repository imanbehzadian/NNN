import streamlit as st
import numpy as np
import torch
import matplotlib.pyplot as plt

def render():
    st.header("Model Performance")

    # guard: only render if all needed vars are loaded
    if not st.session_state.get('mp_ready', False) or st.session_state.model_obj is None:
        st.info(
            "✅ First go to Data & Model Info, tick “Use simulated data” and click Go "
            "to load the pretrained model before viewing Model Performance."
        )
        return

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
