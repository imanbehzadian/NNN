import streamlit as st
import torch
import numpy as np
import torch.nn as nn
from modules.utils import compute_token_recommendations, calculate_creative_impact, creative_simulator
from transformers import  BertModel

def display_token_recommendations():
    if not all(key in st.session_state for key in [ 'ioc_E_np', 'ioc_tokenizer', 'ioc_token2id', 'ioc_messages']):
        st.info("🔍 Token analysis data not available. Please ensure all required components are loaded.")
        return
    
    try:
        E_np = st.session_state.get('ioc_E_np')
        tokenizer = st.session_state.get('ioc_tokenizer')
        token2id = st.session_state.get('ioc_token2id')
        messages = st.session_state.get('ioc_messages')

        top_k_corpus, top_k_global = compute_token_recommendations(
            st.session_state['impact_per_dim'],
            E_np,
            tokenizer,
            token2id,
            messages
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🎯 Top Key Words")
            st.markdown("*From your existing creatives*")
            
            with st.container():
                for i, (token, score) in enumerate(top_k_corpus, 1):
                    st.markdown(f"""<div style="background-color: #f0f9ff;border-left: 4px solid #0ea5e9;padding: 8px 12px;margin: 4px 0;border-radius: 4px;"><strong>{i}. {token}</strong><span style="color: #64748b; font-size: 0.9em;">(Score: {score:.4f})</span></div>""", unsafe_allow_html=True)
        with col2:
            st.markdown("### 🎉 Fresh Word Suggestions")
            st.markdown("*From the full dictionary vocabulary*")
            
            with st.container():
                for i, (token, score) in enumerate(top_k_global, 1):
                    st.markdown(f"""<div style="background-color: #f0fdf4;border-left: 4px solid #22c55e;padding: 8px 12px;margin: 4px 0;border-radius: 4px;"><strong>{i}. {token}</strong><span style="color: #64748b; font-size: 0.9em;">(Score: {score:.4f})</span></div>""", unsafe_allow_html=True)

        st.markdown("""<div style="background-color: #fefce8;border: 1px solid #facc15;border-radius: 8px;padding: 12px;margin: 16px 0;"><strong>💡 How to use these recommendations:</strong><ul style="margin: 8px 0; padding-left: 20px;"><li><strong>Sales-Driving Words:</strong> These are high-performing words already in your creative portfolio</li><li><strong>Fresh Suggestions:</strong> New vocabulary that could boost your creative performance</li><li>Higher scores indicate stronger alignment with successful sales outcomes</li></ul></div>""", unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error computing token recommendations: {str(e)}")

def tokeniser(texts: list[str],global_mean,global_std) -> np.ndarray:
                    """
                    texts: list of strings
                    returns: (len(texts), 256)-array of normalized embeddings
                    """
                    DEVICE = st.session_state['ioc_DEVICE']
                    sequence_length = 50
                    bert_model  = BertModel.from_pretrained('bert-base-uncased').to(DEVICE).eval()
                    layer_norm  = nn.LayerNorm(bert_model.config.hidden_size).to(DEVICE)
                    project256  = nn.Linear(bert_model.config.hidden_size, 256).to(DEVICE)
                    tokenizer = st.session_state['ioc_tokenizer']
                    def _embed_batch(texts: list[str]) -> np.ndarray:
                        DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
                        enc = tokenizer(
                            texts,
                            padding='max_length',
                            truncation=True,
                            max_length=sequence_length,
                            return_tensors='pt'
                        ).to(DEVICE)
                        with torch.no_grad():
                            out      = bert_model(**enc)
                            cls_vec  = out.last_hidden_state[:,0,:]         
                            normed   = layer_norm(cls_vec)                
                            emb256   = project256(normed)                   
                        return emb256.cpu().numpy()

                    emb_raw = _embed_batch(texts)                     # (n,256)
                    return (emb_raw - global_mean) / global_std           

def render():
    ready = st.session_state.get('csp_ready', False) and st.session_state.get('model_obj') is not None
    if not ready:
        st.info( "✅ First go to Data & Model Info, tick “Use simulated data” to load the pretrained model before viewing Creative‑Piece Simulator.")
        return
    
    st.title("Creative Simulator")
    st.markdown("This simulator predicts how well your new creative would have performed across different locations and time periods. Select a range of geos and weeks to see the projected performance based on the trained model.")

    X = st.session_state['csp_X']
    DEVICE = st.session_state['ioc_DEVICE']
    model = st.session_state.get('model_obj')

    if 'impact_per_dim' not in st.session_state or len(st.session_state.get('impact_per_dim', [])) == 0:
        st.session_state['impact_per_dim'] = []
        impact_per_dim = calculate_creative_impact(model=model,  X=st.session_state['ioc_X'], device=st.session_state['ioc_DEVICE'], geo_id=3 )
        st.session_state['impact_per_dim'] = impact_per_dim

    display_token_recommendations()
    
    st.markdown("---")
    
    st.markdown("### 📝 Creative Performance Simulation")
    
    creative_text = st.text_area("Enter your creative message", height=100)

    col1, col2 = st.columns(2)
    with col1:
        geo_range = st.select_slider("Geo range", options=range(X.shape[0]), value=(0, X.shape[0]//2), key="geo_range")
    with col2:
        time_range = st.select_slider("Week range", options=range(X.shape[1]), value=(0, X.shape[1]//2), key="time_range")
    if st.button("Simulate Creative", type="primary"):
        with st.spinner("Running simulation..."):
            try:
                                    
                sim_results = creative_simulator(model=model, X=X, creative_piece=creative_text, tokeniser=tokeniser, geo_idx=None if geo_range[0]<0 else geo_range, time_step=None if time_range[0]<0 else time_range,global_mean =st.session_state['ioc_global_mean'],global_std = st.session_state['ioc_global_std'], device=DEVICE)
                st.success("Simulation completed!")
                
                st.markdown("### 📊 Simulation Results")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(
                        "Base Sales",
                        f"{sim_results['base_total']:,.2f}",
                        help="Total sales under baseline scenario"
                    )
                with col2:
                    st.metric(
                        "Scenario Sales", 
                        f"{sim_results['scn_total']:,.2f}",
                        help="Total sales with new creative"
                    )
                with col3:
                    st.metric(
                        "Sales Uplift",
                        f"{sim_results['uplift']:+.2f}%",
                        delta=f"{sim_results['uplift']:+.2f}%",
                        help="Percentage change in sales from baseline"
                    )
                    
            except Exception as e:
                st.error(f"Simulation failed: {str(e)}")