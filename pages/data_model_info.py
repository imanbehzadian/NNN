# pages/data_model_info.py

import streamlit as st
import pandas as pd
import torch

st.session_state['DEVICE'] = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def render():
    st.header("Data & Model Info")
    st.session_state['DEVICE'] = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # --- Inputs ---
    uploaded     = st.file_uploader("Upload CSV data", type=['csv'], key='data_upload')
    use_sim      = st.checkbox("Use simulated data (load pretrained model)", key='use_simulated')
    hyper_file   = st.file_uploader("Upload hyperparams.txt (optional)", type=['txt'], key='hyper_upload')

    cols_go = st.columns([20, 2])
    with cols_go[1]:
        go = st.button("Go", key='go_button')
    msg_ph = st.empty()

    if go:
        if use_sim:
            file_name = "simulation_data/NNN_vars_3.pkl"
            device = st.session_state['DEVICE']
            try:
                # now torch.load can resolve your custom classes via sys.modules['main']
                workspace = torch.load(
                    file_name,
                    map_location=device,
                    weights_only=False
                )
                # inject each saved var back into globals()
                for name, obj in workspace.items():
                    globals()[name] = obj
                st.session_state.model_obj = workspace.get('model')

                # Stash under Model‑Performance‑specific keys:
                st.session_state['mp_best_wts']     = workspace.get('best_wts')
                st.session_state['mp_N_GEOS']       = workspace.get('N_GEOS')
                st.session_state['mp_NUM_CHANNELS'] = workspace.get('NUM_CHANNELS')
                st.session_state['mp_TIME_STEPS']   = workspace.get('TIME_STEPS')
                st.session_state['mp_X']            = workspace.get('X')
                st.session_state['mp_Y']            = workspace.get('Y')
                st.session_state['mp_sales']        = workspace.get('sales')
                required_mp = [
                    'mp_best_wts',
                    'mp_N_GEOS',
                    'mp_NUM_CHANNELS',
                    'mp_TIME_STEPS',
                    'mp_X',
                    'mp_Y',
                    'mp_sales',
                ]
                st.session_state['mp_ready'] = all(
                    st.session_state.get(k) is not None for k in required_mp
                )

                # Stash under Creative‑Scenarios‑specific keys:
                st.session_state['cs_best_wts']            = workspace.get('best_wts')
                st.session_state['cs_N_GEOS']              = workspace.get('N_GEOS')
                st.session_state['cs_TIME_STEPS']          = workspace.get('TIME_STEPS')
                st.session_state['cs_NUM_CHANNELS']        = workspace.get('NUM_CHANNELS')
                st.session_state['cs_X']                   = workspace.get('X')
                st.session_state['cs_creative_embeddings'] = workspace.get('creative_embeddings')
                st.session_state['cs_id_worst']            = workspace.get('id_worst')
                st.session_state['cs_id_best']             = workspace.get('id_best')
                st.session_state['cs_pred_total_sales']    = workspace.get('pred_total_sales')
                required_cs = [
                    'cs_best_wts',
                    'cs_N_GEOS',
                    'cs_TIME_STEPS',
                    'cs_NUM_CHANNELS',
                    'cs_X',
                    'cs_creative_embeddings',
                    'cs_id_worst',
                    'cs_id_best',
                    'cs_pred_total_sales',
                ]
                st.session_state['cs_ready'] = all(
                    st.session_state.get(k) is not None for k in required_cs
                )

                # Stash under Scenario‑Planner‑specific keys:
                st.session_state['sp_X']              = workspace.get('X')
                st.session_state['sp_DEVICE']         = st.session_state['DEVICE']
                st.session_state['sp_N_GEOS']         = workspace.get('N_GEOS')
                st.session_state['sp_TIME_STEPS']     = workspace.get('TIME_STEPS')
                st.session_state['sp_NUM_CHANNELS']   = workspace.get('NUM_CHANNELS')
                st.session_state['sp_MEDIA_CHANNELS'] = workspace.get('MEDIA_CHANNELS')
                required_sp = [
                    'sp_X',
                    'sp_DEVICE',
                    'sp_N_GEOS',
                    'sp_TIME_STEPS',
                    'sp_NUM_CHANNELS',
                    'sp_MEDIA_CHANNELS',
                ]
                st.session_state['sp_ready'] = all(
                    st.session_state.get(k) is not None for k in required_sp
                )

                # Stash under Creative‑Scenario‑Planner‑specific keys:
                st.session_state['csp_X']         = workspace.get('X')
                st.session_state['csp_DEVICE']    = st.session_state['sp_DEVICE']
                st.session_state['csp_tokeniser'] = workspace.get('tokeniser')
                required_csp = [
                    'csp_X',
                    'csp_DEVICE',
                    'csp_tokeniser',
                ]
                st.session_state['csp_ready'] = all(
                    st.session_state.get(k) is not None for k in required_csp
                )

                # Stash under Attribution‑specific keys:
                st.session_state['attr_X']              = workspace.get('X')
                st.session_state['attr_NUM_CHANNELS']   = workspace.get('NUM_CHANNELS')
                st.session_state['attr_DEVICE']         = st.session_state['sp_DEVICE']
                st.session_state['attr_MEDIA_CHANNELS'] = workspace.get('MEDIA_CHANNELS')
                required_attr = [
                    'attr_X',
                    'attr_NUM_CHANNELS',
                    'attr_DEVICE',
                    'attr_MEDIA_CHANNELS',
                ]
                st.session_state['attr_ready'] = all(
                    st.session_state.get(k) is not None for k in required_attr
                )

                # Stash under Insights‑on‑Creatives‑specific keys:
                st.session_state['ioc_X']           = workspace.get('X')
                st.session_state['ioc_DEVICE']      = st.session_state['sp_DEVICE']
                st.session_state['ioc_E_np']        = workspace.get('E_np')
                st.session_state['ioc_tokenizer']   = workspace.get('tokenizer')
                st.session_state['ioc_token2id']    = workspace.get('token2id')
                st.session_state['ioc_messages']    = workspace.get('messages')
                required_ioc = [
                    'ioc_X',
                    'ioc_DEVICE',
                    'ioc_E_np',
                    'ioc_tokenizer',
                    'ioc_token2id',
                    'ioc_messages',
                ]
                st.session_state['ioc_ready'] = all(
                    st.session_state.get(k) is not None for k in required_ioc
                )

                # Stash under Impulse‑Response‑specific keys:
                st.session_state['ir_X']              = workspace.get('X')
                st.session_state['ir_media_spend']    = workspace.get('media_spend')
                st.session_state['ir_DEVICE']         = st.session_state['sp_DEVICE']
                st.session_state['ir_MEDIA_CHANNELS'] = workspace.get('MEDIA_CHANNELS')
                required_ir = [
                    'ir_X',
                    'ir_media_spend',
                    'ir_DEVICE',
                    'ir_MEDIA_CHANNELS',
                ]
                st.session_state['ir_ready'] = all(
                    st.session_state.get(k) is not None for k in required_ir
                )

                # Stash under Impulse‑Response‑Animated‑specific keys:
                st.session_state['ira_X']               = workspace.get('X')
                st.session_state['ira_media_spend']     = workspace.get('media_spend')
                st.session_state['ira_DEVICE']          = st.session_state['sp_DEVICE']
                st.session_state['ira_MEDIA_CHANNELS']  = workspace.get('MEDIA_CHANNELS')
                st.session_state['ira_EMBED_DIM']       = workspace.get('EMBED_DIM')
                required_ira = [
                    'ira_X',
                    'ira_media_spend',
                    'ira_DEVICE',
                    'ira_MEDIA_CHANNELS',
                    'ira_EMBED_DIM',
                ]
                st.session_state['ira_ready'] = all(
                    st.session_state.get(k) is not None for k in required_ira
                )



                msg_ph.success(f"✅ Loaded pretrained variables from {file_name} onto {device}")
            except FileNotFoundError:
                msg_ph.error(f"❌ File not found: {file_name}")
            except Exception as e:
                msg_ph.error(f"❌ Failed to load {file_name}: {e}")
        elif uploaded is not None:
            df = pd.read_csv(uploaded)
            st.session_state.data_df = df
            msg_ph.success("✅ Data loaded successfully.")
        else:
            st.warning("Please upload a CSV or tick 'Use simulated data'.")


    # Handle hyperparams override
    if hyper_file is not None:
        content = hyper_file.read().decode()
        params = {k.strip(): v.strip() for k, v in (
            line.split('=', 1) for line in content.splitlines() if '=' in line
        )}
        st.session_state.hyperparams = params
        st.success("✅ Hyperparameters overridden.")

    if st.session_state.data_df is not None and not st.session_state.data_df.empty:
        st.subheader("Data Preview")
        st.dataframe(df.head())
