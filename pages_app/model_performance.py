import streamlit as st
import numpy as np
import torch
import matplotlib.pyplot as plt

def render():
    st.header("Model Performance")    
    st.subheader("Actual vs. Predicted Sales Performance")
    st.write("""
    Discover how accurately our AI model predicts weekly sales performance across all locations. 
    This analysis helps validate the model's reliability in forecasting business outcomes and its potential for strategic decision-making.
    Also the baseline reveals the true ROI of your media strategy and highlights potential optimization opportunities.
    """)
    
    with st.spinner("Analyzing sales performance data..."):
        if not st.session_state.get('mp_ready', False) or st.session_state.model_obj is None:
            st.info(
                "✅ First go to Data & Model Info, tick “Use simulated data” "
                "to load the pretrained model before viewing Model Performance."
            )
            return

        model = st.session_state.model_obj
        N = st.session_state['mp_N_GEOS']
        T = st.session_state['mp_TIME_STEPS']
        C = st.session_state['mp_NUM_CHANNELS']
        X = st.session_state['mp_X']
        Y = st.session_state['mp_Y']

        model.eval()

        DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        geo_idx = torch.arange(N, device=DEVICE)

        Xf = torch.tensor(X, dtype=torch.float32, device=DEVICE)
        Xb = Xf.clone()
        Xb[..., :C, :] = 0.0

        with torch.no_grad():
            full = model(Xf, geo_idx).cpu().numpy()
            base = model(Xb, geo_idx).cpu().numpy()

        sales_full = np.expm1(full[..., 0]).sum(axis=0)
        sales_base = np.expm1(base[..., 0]).sum(axis=0)
        sales_actual = np.expm1(Y[..., 0]).sum(axis=0)

        weeks = np.arange(len(sales_actual))

        def update_weeks():
            st.session_state.start_week = st.session_state.week_slider[0]
            st.session_state.end_week = st.session_state.week_slider[1]

        if 'start_week' not in st.session_state:
            st.session_state.start_week = 0
        if 'end_week' not in st.session_state:
            st.session_state.end_week = weeks.shape[0] - 1

        def display_sales_plot(start_week, end_week):
                fig, ax = plt.subplots(figsize=(12, 7))

                base_color = '#FF7043'
                ax.plot(weeks[start_week:end_week+1],
                        sales_actual[start_week:end_week+1],
                        label="Actual Sales",
                    color='#546E7A',
                        linewidth=2.5)

                ax.plot(weeks[start_week:end_week+1],
                        sales_full[start_week:end_week+1],
                        label="Model Prediction",
                    color='#7986CB',
                        linewidth=2.5)

                ax.plot(weeks[start_week:end_week+1],
                        sales_base[start_week:end_week+1],
                        label="No-Media Scenario",
                        color=base_color,
                        linewidth=2.5)

                ax.fill_between(weeks[start_week:end_week+1],
                            sales_base[start_week:end_week+1],
                            alpha=0.1,
                            color=base_color)

                plt.style.use('seaborn-v0_8-whitegrid')
                ax.set_xlabel("Week", fontsize=12, fontfamily='sans-serif')
                ax.set_ylabel("Sales ($M)", fontsize=12, fontfamily='sans-serif')

                ax.legend(frameon=True, fontsize=11, loc='upper right')

                ax.grid(True, linestyle=":", alpha=0.3)

                for spine in ax.spines.values():
                    spine.set_linewidth(0.5)

                plt.tight_layout()
                return fig

        figure = display_sales_plot(st.session_state.start_week, st.session_state.end_week)
        st.pyplot(figure)

        left_spacer, slider_col, right_spacer = st.columns([0.9, 10, 0.5])

        with slider_col:
            with st.container():
                st.select_slider(
                    label=" ",
                    options=weeks,
                    value=(st.session_state.start_week, st.session_state.end_week),
                    format_func=lambda x: f"Week {x+1}",
                    key='week_slider',
                    on_change=update_weeks
                )
