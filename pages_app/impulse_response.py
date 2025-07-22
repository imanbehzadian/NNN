# pages/impulse_response.py

import streamlit as st
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from modules.utils import impulse_response_analysis

def render():
    st.header("Impulse‑Response Analysis")

    ready = (
        st.session_state.get('ir_ready', False)
        and st.session_state.get('ira_ready', False)
        and st.session_state.get('model_obj') is not None
    )
    if not ready:
        st.info(
            "✅ First go to Data & Model Info, tick “Use simulated data” and click Go "
            "to load the pretrained model before viewing Impulse‑Response Analysis."
        )
        return

    model         = st.session_state.model_obj
    X_static      = st.session_state['ir_X']
    media_spend   = st.session_state['ir_media_spend']
    DEVICE        = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    MEDIA_CHANNELS= st.session_state['ir_MEDIA_CHANNELS']
    EMBED_DIM     = st.session_state['ira_EMBED_DIM']

    # ─── Static impulse response ───────────────────────────────────────────
    st.subheader("Static Impulse Response (1σ bump at t=0)")
    delta_pct_dict = impulse_response_analysis(
        model, X_static, media_spend, std_multiplier=1.0, device=DEVICE
    )
    st.dataframe(pd.DataFrame(delta_pct_dict))

    # ─── Animated impulse response ────────────────────────────────────────
    st.subheader("Animated Impulse Response (1σ bump)")
    # tensor prep
    X_t         = torch.tensor(X_static, dtype=torch.float32, device=DEVICE)
    full_geo_ids= torch.arange(X_t.shape[0], device=DEVICE)
    model.eval()
    eps = 1e-9
    with torch.no_grad():
        Yb = model(X_t, full_geo_ids)
    sales_base = torch.expm1(Yb[..., 0]).cpu().numpy().sum(axis=0)
    std_rg      = np.std(media_spend, axis=1)

    # figure setup
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    axes = axes.flatten()
    lines = []
    for ax in axes:
        ln, = ax.plot([], [], lw=2, marker='o')
        lines.append(ln)
        ax.set_xlim(1, 13)
        ax.set_ylim(-70, 150)
        ax.set_xlabel("Week")
        ax.set_ylabel("Δ Sales (%)")

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

            axes[j].set_title(f"{MEDIA_CHANNELS[j]} at t={t}")

        return lines

    ani = FuncAnimation(fig, update, frames=range(1, 14), init_func=init, blit=False, repeat=False)
    plt.tight_layout()
    html = ani.to_jshtml()
    plt.close(fig)
    st.components.v1.html(html, height=600)
