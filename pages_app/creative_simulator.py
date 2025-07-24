# pages/creative_simulator.py

import streamlit as st
import torch
#from modules.nnn_modules import scenario_planner_creative

def render():

    ready = st.session_state.get('csp_ready', False) and st.session_state.get('model_obj') is not None
    if not ready:
        st.info(
             "✅ First go to Data & Model Info, tick “Use simulated data” "
            "to load the pretrained model before viewing Creative‑Piece Simulator."
        )
        return

    model     = st.session_state.model_obj
    X         = st.session_state['csp_X']
    DEVICE    = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    tokeniser = st.session_state['csp_tokeniser']

    st.title("Creative Simulator")
    st.markdown("""
        This simulator predicts how well your new creative would have performed across different locations and time periods.
        Select a range of geos and weeks to see the projected performance based on the trained model.
    """)

    creative_text = st.text_area("Enter your creative message", height=100)

    col1, col2 = st.columns(2)
    with col1:
        geo_range = st.select_slider(
            "Geo range",
            options=range(X.shape[0]),
            value=(0, X.shape[0]//2),
            key="geo_range"
        )

    with col2:
        time_range = st.select_slider(
            "Week range",
            options=range(X.shape[1]),
            value=(0, X.shape[1]//2),
            key="time_range"
        )

    if st.button("Simulate Creative"):
        out = scenario_planner_creative(
            model=model,
            X=X,
            creative_piece=creative_text,
            tokeniser=tokeniser,
            geo_idx=None if geo_range[0]<0 else geo_range,
            time_step=None if time_range[0]<0 else time_range,
            device=DEVICE,
        )
        st.write("Simulation result embedding:", out)
