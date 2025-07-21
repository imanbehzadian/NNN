
import streamlit as st
import sys

# —————————————————————————————————————————————————————————————
# 1) Alias your slim NNN modules so torch.load finds them
import modules.NNN_modules as nnn_minimal
sys.modules['__main__']                          = nnn_minimal
sys.modules['main']                              = nnn_minimal
sys.modules['nnn_for_mmm_with_scenario_simulator'] = nnn_minimal
# —————————————————————————————————————————————————————————————

# 2) Authentication (sidebar)
VALID_USERS = ['', 'admin', 'iman', 'hamish', 'shalcky', 'andy']
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

login_feedback = st.sidebar.empty()
username      = st.sidebar.text_input('Username', key='username_input')
password      = st.sidebar.text_input('Password', type='password', key='password_input')
login_button  = st.sidebar.button('Login')

# Initialize any session‐state keys you’ll read later
for key, default in [
    ('data_df',      None),
    ('hyperparams',  {}),
    ('model_obj',    None),
    ('load_results_done', False),
]:
    st.session_state.setdefault(key, default)

if login_button:
    if username not in VALID_USERS:
        login_feedback.error('🚫 User not registered')
    elif password != username:
        login_feedback.error('❌ Incorrect password')
    else:
        st.session_state.authenticated = True
        login_feedback.empty()

if not st.session_state.authenticated:
    st.sidebar.title("🔒 Please Log In")
    st.stop()

# 3) Page imports
from pages.overview             import render as render_overview
from pages.data_model_info      import render as render_data_model_info
from pages.training_status      import render as render_training_status
from pages.model_performance    import render as render_model_performance
from pages.channel_attribution  import render as render_channel_attribution
from pages.impulse_response     import render as render_impulse_response
from pages.actionable_insights  import render as render_actionable_insights
from pages.scenario_planner     import render as render_scenario_planner
from pages.creative_simulator   import render as render_creative_simulator

PAGES = {
    "Overview":              render_overview,
    "Data & Model Info":     render_data_model_info,
    "Training Status":       render_training_status,
    "Model Performance":     render_model_performance,
    "Channel Attribution":   render_channel_attribution,
    "Impulse-Response Analysis": render_impulse_response,
    "Actionable Insights":   render_actionable_insights,
    "Scenario Planner":      render_scenario_planner,
    "Creative-Piece Simulator": render_creative_simulator,
}

# 4) Sidebar navigation
st.sidebar.title("Navigation")
selection = st.sidebar.radio("Go to", list(PAGES.keys()), index=0)

# 5) Dispatch
PAGES[selection]()
