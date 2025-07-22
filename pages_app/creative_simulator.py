# pages/creative_simulator.py

import streamlit as st
import torch
#from modules.nnn_modules import scenario_planner_creative


def render():
    st.header("Creative‑Piece Simulator")

    ready = st.session_state.get('csp_ready', False) and st.session_state.get('model_obj') is not None
    if not ready:
        st.info(
            "✅ First go to Data & Model Info, tick “Use simulated data” and click Go "
            "to load the pretrained model before viewing Creative‑Piece Simulator."
        )
        return

    model     = st.session_state.model_obj
    X         = st.session_state['csp_X']
    DEVICE    = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    tokeniser = st.session_state['csp_tokeniser']

    creative_text = st.text_area("Enter your creative message", height=100)
    geo_idx   = st.number_input("Geo index (or –1 for all)", value=-1, step=1)
    time_step = st.number_input("Week index (or –1 for all)", value=-1, step=1)

    if st.button("Simulate Creative"):
        out = scenario_planner_creative(
            model=model,
            X=X,
            creative_piece=creative_text,
            tokeniser=tokeniser,
            geo_idx=None if geo_idx<0 else geo_idx,
            time_step=None if time_step<0 else time_step,
            device=DEVICE,
        )
        st.write("Simulation result embedding:", out)
