import streamlit as st
import torch
import plotly.graph_objects as go
from modules.utils import (
    plot_overall_multiplier_impact,
    plot_channel_specific_impact
)

def render():
    st.header("Scenario Planner")
    st.subheader("Overall Media Investment Impact")
    st.markdown("""
        This analysis shows how changes to your total marketing budget affect overall sales performance.
        By simulating different spending levels (from -90% to +200% of current investment), you can see
        the relationship between total marketing investment and sales outcomes. This helps identify the
        optimal overall budget level and potential diminishing returns, enabling smarter decisions about
        total marketing spend.
    """)
    with st.spinner("⏳ Computing Overall Budget Scenario Planner, please wait…"):
             
        ready = st.session_state.get('sp_ready', False) and st.session_state.get('model_obj') is not None
        if not ready:
            st.info(
                "✅ First go to Data & Model Info, tick “Use simulated data” "
                "and click Go to load the pretrained model before viewing Channel Attribution."
            )
            return

        model = st.session_state.model_obj
        X = st.session_state['sp_X']
        DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        MEDIA_CHANNELS = st.session_state['sp_MEDIA_CHANNELS']

        df_overall = plot_overall_multiplier_impact(model, X, DEVICE)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_overall['multiplier'], 
            y=df_overall['total_sales'],
            mode='lines+markers',
            name='Total Sales'
        ))
        fig.update_layout(
            title='Impact of Overall Media Investment on Total Sales',
            xaxis_title='Media Investment Multiplier',
            yaxis_title='Total Sales'
        )
        st.plotly_chart(fig)

    st.subheader("Channel-Specific Impact Analysis")
    st.markdown("""
        Dive deep into how each marketing channel responds to different investment levels. The analysis
        reveals both incremental sales and ROI curves for each channel, helping you identify optimal
        spending points and diminishing returns thresholds. This granular view enables strategic
        budget allocation across channels, maximizing the impact of every marketing dollar while
        avoiding over-investment in any single channel.
    """)
    
    for idx, channel in enumerate(MEDIA_CHANNELS):

        with st.spinner(f"⏳ Computing Scenario Planner for {channel}, please wait…"):
        
            df_channel = plot_channel_specific_impact(model, X, DEVICE, idx)
            col1, col2 = st.columns(2)

            with col1:
                negative_transition = None
                for i in range(len(df_channel) - 1):
                    if df_channel['incremental_sales'].iloc[i] >= 0 and \
                    all(df_channel['incremental_sales'].iloc[i+1:] < 0):
                        negative_transition = i
                        break

                if negative_transition is not None:
                    cutoff_idx = min(negative_transition + 1, len(df_channel) - 1)
                    plot_data = df_channel.iloc[:cutoff_idx + 1]
                else:
                    plot_data = df_channel
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=plot_data['multiplier'],
                    y=plot_data['incremental_sales'],
                    name=f'{channel} Incremental Sales',
                    mode='lines+markers'
                ))
                fig.update_layout(
                    title=f'{channel} Investment Impact on Incremental Sales',
                    xaxis_title='Channel Investment Multiplier',
                    yaxis_title='Incremental Sales'
                )
                st.plotly_chart(fig)
                
            with col2:
                negative_transition = None
                for i in range(len(df_channel) - 1):
                    if df_channel['incremental_sales'].iloc[i] >= 0 and \
                    all(df_channel['incremental_sales'].iloc[i+1:] < 0):
                        negative_transition = i
                        break

                if negative_transition is not None:
                    cutoff_idx = min(negative_transition + 1, len(df_channel) - 1)
                    plot_data = df_channel.iloc[:cutoff_idx + 1]
                else:
                    plot_data = df_channel
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=plot_data['multiplier'],
                    y=plot_data['roi'],
                    name=f'{channel} ROI',
                    mode='lines+markers'
                ))
                fig.update_layout(
                    title=f'{channel} Investment ROI',
                    xaxis_title='Channel Investment Multiplier',
                    yaxis_title='incremental ROI'
                )
                st.plotly_chart(fig)