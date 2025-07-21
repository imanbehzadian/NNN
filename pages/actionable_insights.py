# pages/actionable_insights.py

import streamlit as st
import numpy as np
import torch
from captum.attr import IntegratedGradients
#from modules.NNN_modules import E_np  # cell→vocab projection matrix in NNN_modules

def render():
    st.header("Actionable Insights")

    ready = st.session_state.get('ioc_ready', False) and st.session_state.get('model_obj') is not None
    if not ready:
        st.info(
            "✅ First go to Data & Model Info, tick “Use simulated data” and click Go "
            "to load the pretrained model before viewing Actionable Insights."
        )
        return

    model      = st.session_state.model_obj
    X          = st.session_state['ioc_X']
    DEVICE     = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    tokenizer  = st.session_state['ioc_tokenizer']
    token2id   = st.session_state['ioc_token2id']
    messages   = st.session_state['ioc_messages']

    # 1) Pick one geo
    geo_id = st.number_input("Geo index", min_value=0, max_value=X.shape[0]-1, value=0)
    time_step = st.number_input("Week index", min_value=0, max_value=X.shape[1]-1, value=X.shape[1]-1)

    X_single = torch.tensor(X[geo_id:geo_id+1], dtype=torch.float32, device=DEVICE)
    geo_idx_single = torch.tensor([geo_id], dtype=torch.long, device=DEVICE)

    # 2) IG wrapper
    def forward_fn(x, g):
        out = model(x, g)
        return out[..., 0].sum(dim=1)

    ig = IntegratedGradients(forward_fn)
    attr, delta = ig.attribute(
        inputs=X_single,
        additional_forward_args=(geo_idx_single,),
        target=None,
        n_steps=50,
        return_convergence_delta=True
    )
    impact_per_dim = (
        attr[0, time_step, -2, :].abs().cpu().numpy()
    )

    # 3) Top‑k dims
    top_k = st.slider("Top K embedding dims", 1, 20, 5)
    top_dims = np.argsort(-impact_per_dim)[:top_k]
    st.write("Top dims:", top_dims.tolist())

    # 4) Token scores
    scores = impact_per_dim @ E_np  # (V,)
    # corpus tokens
    corpus_subtokens = set(st.session_state['ioc_tokenizer'].tokenize(" ".join(messages)))
    corpus_tokens = [t.lstrip("##") for t in corpus_subtokens if t in token2id]
    corpus_scores = sorted(
        ((t, scores[token2id[t]]) for t in corpus_tokens),
        key=lambda x: -x[1]
    )[:top_k]
    st.write("Top sales‑driving words in your creatives:", [t for t,_ in corpus_scores])

    # 5) Global vocab
    global_tokens = [t for t in token2id if "##" not in t]
    global_scores = sorted(
        ((t, scores[token2id[t]]) for t in global_tokens),
        key=lambda x: -x[1]
    )[:top_k]
    st.write("Top fresh word ideas from BERT vocab:", [t for t,_ in global_scores])
