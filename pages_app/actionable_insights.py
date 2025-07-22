# pages/actionable_insights.py

import streamlit as st
import numpy as np
import torch
from captum.attr import IntegratedGradients
#from modules.NNN_modules import E_np  # cell→vocab projection matrix in NNN_modules

def render():
    import streamlit as st

    # Custom CSS for button group
    st.markdown("""
        <style>
        .tab-buttons {
            display: flex;
            margin-top: -10px;
            justify-content: flex-end;
        }
        .tab-button {
            background-color: #f6f8fa;
            border: 1px solid #d0d7de;
            border-bottom: none;
            padding: 6px 12px;
            cursor: pointer;
            font-size: 14px;
            margin-right: -1px;
            font-weight: 500;
            color: #24292f;
        }
        .tab-button.active {
            background-color: white;
            border-top: 2px solid #0969da;
            border-bottom: 1px solid white;
        }
        </style>
    """, unsafe_allow_html=True)

    # Tabs logic
    tab = st.session_state.get("tab", "Local")
    clicked_tab = st.radio("Choose Mode", ["Local", "Codespaces"], horizontal=True, label_visibility="collapsed")
    st.session_state.tab = clicked_tab

    # Show buttons styled like GitHub
    st.markdown(f"""
    <div class="tab-buttons">
        <div class="tab-button {'active' if clicked_tab=='Local' else ''}">Local</div>
        <div class="tab-button {'active' if clicked_tab=='Codespaces' else ''}">Codespaces</div>
    </div>
    """, unsafe_allow_html=True)

    # Example output
    if clicked_tab == "Local":
        st.success("Local mode content")
    else:
        st.info("Codespaces mode content")
