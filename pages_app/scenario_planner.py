# pages/scenario_planner.py

import streamlit as st
import torch
from modules.utils import scenario_planner_per_channel

def render():
    st.header("Scenario Planner")

    ready = st.session_state.get('sp_ready', False) and st.session_state.get('model_obj') is not None
    if not ready:
        st.info(
            "✅ First go to Data & Model Info, tick “Use simulated data” and click Go "
            "to load the pretrained model before viewing Scenario Planner."
        )
        return

    model           = st.session_state.model_obj
    X               = st.session_state['sp_X']
    DEVICE          = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    N_GEOS          = st.session_state['sp_N_GEOS']
    TIME_STEPS      = st.session_state['sp_TIME_STEPS']
    NUM_CHANNELS    = st.session_state['sp_NUM_CHANNELS']
    MEDIA_CHANNELS  = st.session_state['sp_MEDIA_CHANNELS']

    # User inputs
    geo_idx   = st.number_input("Geo index (or leave at –1 for all)", value=-1, step=1)
    time_step = st.number_input("Week index (or –1 for all weeks)", value=-1, step=1)
    mults_str = st.text_input(
        f"Enter {NUM_CHANNELS} multipliers, comma‑separated",
        value=",".join("1.0" for _ in range(NUM_CHANNELS))
    )

    try:
        channel_multipliers = [float(x) for x in mults_str.split(",")]
    except:
        st.error("Invalid multipliers format")
        return

    df = scenario_planner_per_channel(
        model,
        X,
        channel_multipliers,
        geo_idx=None if geo_idx < 0 else geo_idx,
        time_step=None if time_step < 0 else time_step,
        device=DEVICE,
    )
    st.dataframe(df)
