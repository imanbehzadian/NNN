# pages/channel_attribution.py

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from modules.NNN_modules import scenario_planner_per_channel

def render():
    st.header("Channel Attribution")

    # Only render if the workspace is loaded and model is present
    if st.session_state.get('attr_ready', False) and st.session_state.get('model_obj') is not None:
        # Pull variables from session state
        model          = st.session_state.model_obj
        X              = st.session_state['attr_X']
        DEVICE         = st.session_state['attr_DEVICE']
        NUM_CHANNELS   = st.session_state['attr_NUM_CHANNELS']
        MEDIA_CHANNELS = st.session_state['attr_MEDIA_CHANNELS']

        # Define permutation‐importance helper
        def permutation_importance(model, X, geo_idx, time_step, channel_idx):
            # baseline prediction
            df_base = scenario_planner_per_channel(
                model,
                X,
                [1.0] * NUM_CHANNELS,
                geo_idx=geo_idx,
                time_step=time_step,
                device=DEVICE,
            )
            base_sales = df_base["total_sales"].iat[0]

            # permute that channel across geos
            X_perm = X.copy()
            np.random.shuffle(X_perm[:, :, channel_idx, :])

            # re‐predict
            df_perm = scenario_planner_per_channel(
                model,
                X_perm,
                [1.0] * NUM_CHANNELS,
                geo_idx=geo_idx,
                time_step=time_step,
                device=DEVICE,
            )
            perm_sales = df_perm["total_sales"].iat[0]

            # importance = drop in sales
            return base_sales - perm_sales

        # Let user choose geo and week
        geo_idx   = st.number_input("Geo index", min_value=0, max_value=NUM_CHANNELS-1, value=0, step=1)
        time_step = st.number_input("Week index", min_value=0, max_value=st.session_state['attr_X'].shape[1]-1, value=0, step=1)

        # Compute importances
        importances = [
            permutation_importance(model, X, geo_idx, time_step, j)
            for j in range(NUM_CHANNELS)
        ]

        # Plot bar chart
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(MEDIA_CHANNELS, importances)
        ax.set_xticks(range(len(MEDIA_CHANNELS)))
        ax.set_xticklabels(MEDIA_CHANNELS, rotation=45, ha='right')
        ax.set_ylabel("Δ Sales when permuted")
        ax.set_title(f"Permutation Importance: Geo {geo_idx}, Week {time_step}")
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        fig.tight_layout()

        st.pyplot(fig)

    else:
        st.info(
            "✅ First go to Data & Model Info, tick “Use simulated data” "
            "and click Go to load the pretrained model before viewing Channel Attribution."
        )
