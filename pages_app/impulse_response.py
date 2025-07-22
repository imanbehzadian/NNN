# pages/impulse_response.py

import streamlit as st
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from modules.utils import impulse_response_analysis

def render():
    st.subheader("Impulse Response (1σ bump at t=0)")

    # ─── Top tabs for Static vs. Animation ────────────────────────
    tab_static, tab_anim = st.tabs(["🔵 Static", "🎞️ Animation"])

    model = st.session_state.model_obj
    std_rg = np.std(st.session_state['ir_media_spend'], axis=1)

    delta_pct_dict, plot_data = impulse_response_analysis(
        model,
        st.session_state['ir_X'],
        st.session_state['ir_media_spend'],
        std_rg=std_rg,
        std_multiplier=1.0,
        device=st.session_state['ir_DEVICE'],
        media_channels=st.session_state['ir_MEDIA_CHANNELS'],
    )

    # ─── STATIC VIEW ───────────────────────────────────────────────
    with tab_static:
        st.caption("Impulse Response analysis shows, for each media channel, how a one unit (1 std) spend bump at week 0 alters percentage sales over the following weeks .")
        fig, axes = plt.subplots(nrows=len(plot_data), figsize=(8*.8, 4 * .8  * len(plot_data)))
        if len(plot_data) == 1:
            axes = [axes]

        for ax, d in zip(axes, plot_data):
            ax.plot(d["weeks"], d["delta_pct"], marker='o')
            ax.set_title(f"Impulse response of the channel: {d['channel']}")
            ax.set_xlim(0, 13)
            ax.set_xlabel("Week")
            ax.set_ylabel("Δ Sales (%)")
            ax.grid(True)

            # annotation
            first_val = d["delta_pct"][0]
            xpos, ypos, va, ha = (0.98, 0.02, 'bottom', 'right') if first_val < 0 else (0.98, 0.98, 'top', 'right')
            txt = (
                f"{d['label']}\n"
                f"Δ% Sales (t→t+3): {d['summary_3w']:.1f}%\n"
                f"Δ% Sales (t→t+13): {d['summary_13w']:.1f}%"
            )
            ax.text(xpos, ypos, txt,
                    transform=ax.transAxes, fontsize=9,
                    va=va, ha=ha,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.8))

        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)


    # ─── ANIMATED VIEW ─────────────────────────────────────────────
    with tab_anim:
        st.caption("Animated mode shows, for each media channel, how a one unit (1 std) spend bump at week 0 to 13 alters percentage sales over the following weeks in a dynamic 2×2 grid.")
        # prep tensors
        X_t = torch.tensor(st.session_state['ir_X'],
                        dtype=torch.float32,
                        device=st.session_state['ir_DEVICE'])
        full_geo_ids = torch.arange(X_t.shape[0], device=st.session_state['ir_DEVICE'])
        model.eval()
        eps = 1e-9
        with torch.no_grad():
            Yb = model(X_t, full_geo_ids)
        sales_base = torch.expm1(Yb[..., 0]).cpu().numpy().sum(axis=0)

        # set up canvas with proper spacing
        fig, axes = plt.subplots(2, 2, figsize=(12 * 0.6, 8 * 0.6), sharex=True, sharey=True)
        axes = axes.flatten()
        
        # Adjust subplot spacing - this creates space above titles and prevents right-side cutoff
        plt.subplots_adjust(top=0.85, bottom=0.15, left=0.15, right=0.95, hspace=0.4, wspace=0.3)
        
        lines = []
        for ax in axes:
            ln, = ax.plot([], [], lw=2, marker='o')
            lines.append(ln)
            ax.set_xlim(1, 13)
            ax.set_ylim(-5, 10)
            ax.set_xlabel("Week", fontsize=8)
            ax.set_ylabel("Δ Sales (%)", fontsize=8, labelpad=1)  # Smaller font, closer to axis
            ax.tick_params(axis='y', labelsize=8)  # Reduced tick label font size
            ax.tick_params(axis='x', labelsize=8)  # Reduced tick label font size
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x)}'))  # No decimals
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x)}'))  # No decimals
            plt.subplots_adjust(top=0.85, bottom=0.15, left=0.05, right=0.95, hspace=0.4, wspace=0.3)

        def init():
            for ln in lines:
                ln.set_data([], [])
            return lines

        def update(t):
            for j, ln in enumerate(lines):
                X_scen = X_t.clone()
                s = st.session_state['ir_media_spend'][:, t, j]
                inc = std_rg[:, j]
                s_prime = s + inc
                for g in range(X_scen.shape[0]):
                    v = torch.log1p(torch.tensor(s_prime[g], device=st.session_state['ir_DEVICE']))
                    V = v.repeat(st.session_state['ira_EMBED_DIM'])
                    norm = torch.norm(V)
                    E = (V / (norm + eps)) * torch.log1p(norm)
                    X_scen[g, t, j, :] = E

                with torch.no_grad():  # Fixed: was torch.no_d()
                    Yi = model(X_scen, full_geo_ids)
                sales_imp = torch.expm1(Yi[..., 0]).cpu().numpy().sum(axis=0)
                delta_pct = (sales_imp - sales_base) / (sales_base + eps) * 100

                weeks = np.arange(1, 14)
                ln.set_data(weeks, delta_pct[1:14])
                # Set title with padding for space above
                axes[j].set_title(f"{st.session_state['ira_MEDIA_CHANNELS'][j]} at t={t}", pad=10, fontsize=9)
            return lines

        ani = FuncAnimation(fig, update, frames=range(1, 14),
                            init_func=init, blit=False, repeat=False)
        
        # Remove tight_layout as it conflicts with subplots_adjust
        html = ani.to_jshtml()
        plt.close(fig)
        st.components.v1.html(html, height=600, scrolling=True)