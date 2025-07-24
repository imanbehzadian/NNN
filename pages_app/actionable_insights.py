import streamlit as st
import numpy as np
import torch
import plotly.graph_objects as go
import streamlit.components.v1 as components
from modules.utils import scenario_planner_per_channel

def render_scenario_planner():
    st.subheader("Custom Channel Investment Simulator")
    
    st.markdown("""
        Adjust individual channel investments using the controls below. The colored band shows 
        the ±50% range around current investment levels. You can adjust from 0% to 200% of 
        current spend. We recommend to stay within ±50% of current spend to maintain model accuracy.
    """)

    if not (st.session_state.get('sp_ready', False) and st.session_state.get('model_obj') is not None):
        st.info("⚠️ Please load the model in Data & Model Info tab first")
        return
        
    model = st.session_state.model_obj
    X = st.session_state['sp_X']
    DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    channels = st.session_state['sp_MEDIA_CHANNELS']

    base_result = scenario_planner_per_channel(model, X, [1.0] * len(channels), device=DEVICE)
    base_sales = base_result['total_sales'].sum() 

    cols = st.columns(len(channels))
    multipliers = []
    base_spend = [0] * len(channels)
    multipliers = [1.0] * len(channels)

    for idx, (col, channel) in enumerate(zip(cols, channels)):
        with col:
            st.markdown(f"**{channel}**")
            base_spend[idx] = np.sum(X[:, :, idx, 1])
            
            tooltip = f"Recommendation: Stay within ±50% of current spend (${base_spend[idx]/1e3:.1f}M) to maintain model accuracy"
            
            channel_spend = multipliers[idx]

            mult = st.slider(
                f"${channel_spend/1e3:.1f}M",
                min_value=0.0,
                max_value=2.0,
                value=1.0,
                step=0.1 ,
                key=f"slider_{channel}",
                label_visibility="collapsed",
                format=f"x%1.1f",
                help=tooltip,
            )
            multipliers[idx] = mult * base_spend[idx]
            status1 = "🟢 " if 0.5 * base_spend[idx] <= multipliers[idx] <= 1.5 * base_spend[idx] else "🟡 "
            status2 = "Within Range" if 0.5 * base_spend[idx] <= multipliers[idx] <= 1.5 * base_spend[idx] else "Out of Conf Range"
            st.markdown(f"{status1} ${multipliers[idx]/1e3:.1f}M", unsafe_allow_html=True)
            st.markdown(f"<div style='margin-top: -15px;'>{status2}</div>", unsafe_allow_html=True)
    st.markdown("---")

    result = scenario_planner_per_channel(model, X, multipliers, device=DEVICE)
    total_sales = result['total_sales'].sum()
    total_spend = np.sum(multipliers,0)

    incremental_sales = total_sales - base_sales
    incremental_spend = total_spend - np.sum(base_spend[:])
    spend_growth_pct = (incremental_spend / np.sum(base_spend[:])) * 100
    sales_growth_pct = (incremental_sales / base_sales) * 100
    roi = total_sales / total_spend if abs(total_spend) > 0.001 else 0
    metrics_cols = st.columns(4)
    
    card_height = "350px"
    with metrics_cols[0]:
        st.markdown("""
    <div style="background: rgba(240, 147, 251, 0.25); backdrop-filter: blur(20px);
                border: 1px solid rgba(255, 255, 255, 0.3); box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
                padding: 25px; border-radius: 20px; text-align: center; margin: 5px; height: {};">
        <div style="font-size: 32px; margin-bottom: 12px;">💸</div>
        <div style="font-family: 'Segoe UI', 'Roboto', sans-serif; font-size: 16px; font-weight: 600;
                    color: rgba(80, 80, 80, 0.9); margin: 5px 0;">Incremental Spend</div>
        <div style="font-family: 'Segoe UI', 'Roboto', sans-serif; font-size: 36px; font-weight: 700;
                    color: rgba(60, 60, 60, 1); margin: 10px 0;">${:,.1f}<br><span style="font-size: 12px;">million</span></div>
        <div style="font-family: 'Segoe UI', 'Roboto', sans-serif; font-size: 13px; font-weight: 500;
                    color: rgba(100, 100, 100, 0.8);">{:+.1f}% growth</div>
        </div>
                """.format(card_height, incremental_spend/1e3, spend_growth_pct), unsafe_allow_html=True)
    
    with metrics_cols[1]:
        st.markdown("""
    <div style="background: rgba(102, 126, 234, 0.25); backdrop-filter: blur(20px);
                border: 1px solid rgba(255, 255, 255, 0.3); box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
                padding: 25px; border-radius: 20px; text-align: center; margin: 5px; height: {};">
        <div style="font-size: 32px; margin-bottom: 12px;">💰</div>
        <div style="font-family: 'Segoe UI', 'Roboto', sans-serif; font-size: 16px; font-weight: 600;
                    color: rgba(80, 80, 80, 0.9); margin: 5px 0;">Incremental Sales</div>
        <div style="font-family: 'Segoe UI', 'Roboto', sans-serif; font-size: 36px; font-weight: 700;
                    color: rgba(60, 60, 60, 1); margin: 10px 0;">${:,.1f}<br><span style="font-size: 12px;">million</span></div>
        <div style="font-family: 'Segoe UI', 'Roboto', sans-serif; font-size: 13px; font-weight: 500;
                    color: rgba(100, 100, 100, 0.8);">{:+.1f}% growth</div>
        </div>
                """.format(card_height, incremental_sales/1e3, sales_growth_pct), unsafe_allow_html=True)
        
    with metrics_cols[2]:
        st.markdown("""
    <div style="background: rgba(79, 172, 254, 0.25); backdrop-filter: blur(20px);
                border: 1px solid rgba(255, 255, 255, 0.3); box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
                padding: 25px; border-radius: 20px; text-align: center; margin: 5px; height: {};">
        <div style="font-size: 32px; margin-bottom: 12px;">📊</div>
        <div style="font-family: 'Segoe UI', 'Roboto', sans-serif; font-size: 16px; font-weight: 600;
                    color: rgba(80, 80, 80, 0.9); margin: 5px 0;">Overall<br>Sales</div>
        <div style="font-family: 'Segoe UI', 'Roboto', sans-serif; font-size: 36px; font-weight: 700;
                    color: rgba(60, 60, 60, 1); margin: 10px 0;">${:,.1f}<br><span style="font-size: 12px;">million</span></div>
        <div style="font-family: 'Segoe UI', 'Roboto', sans-serif; font-size: 13px; font-weight: 500;
                    color: rgba(100, 100, 100, 0.8);">Current projection</div>
        </div>
                """.format(card_height, total_sales/1e3), unsafe_allow_html=True)
        
    with metrics_cols[3]:
        base_roi = base_sales / np.sum(base_spend[:])
        st.markdown("""
    <div style="background: rgba(250, 112, 154, 0.25); backdrop-filter: blur(20px);
                border: 1px solid rgba(255, 255, 255, 0.3); box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
                padding: 25px; border-radius: 20px; text-align: center; margin: 5px; height: {};">
        <div style="font-size: 32px; margin-bottom: 12px;">🎯</div>
        <div style="font-family: 'Segoe UI', 'Roboto', sans-serif; font-size: 16px; font-weight: 600;
                    color: rgba(80, 80, 80, 0.9); margin: 5px 0;">Marketing ROI</div>
        <div style="font-family: 'Segoe UI', 'Roboto', sans-serif; font-size: 36px; font-weight: 700;
                    color: rgba(60, 60, 60, 1); margin: 10px 0;">{:.2f}x</div>
        <div style="font-family: 'Segoe UI', 'Roboto', sans-serif; font-size: 13px; font-weight: 500;
                    color: rgba(100, 100, 100, 0.8);">{}</div>
        </div>
                """.format(card_height, roi, "less than baseline ROI" if roi < base_roi else "Great!"), unsafe_allow_html=True)

def render():
        render_scenario_planner()
