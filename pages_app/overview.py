import streamlit as st
from datetime import date, timedelta

def render():
    st.header("Overview")
    st.markdown(
        """
This app demonstrates the **Neural Nested Network (NNN)** for unified Marketing Mix Modeling:

- A Transformer‑based model capturing **direct** (media → sales) and **indirect** (media → search → sales) effects  
- Factored self‑attention to disentangle time vs. channel interactions  
- End‑to‑end learning of sales & search with residual path weighting  
- Flexible inputs (creative embeddings, seasonality, external signals)  
- Transfer learning, non‑additive synergies, and rich feature embeddings  

📘 **Paper Summary**  
- Nested architecture for direct/indirect pathways  
- Separate attention for time and channel factors  
- Joint sales & search prediction with interpretable weights  
- Supports creative simulation, impulse analysis, and scenario planning  
        """
    )

    # Date‐window selector (last 30 days by default)
    end_date = date.today()
    start_date = end_date - timedelta(days=30)
    _ = st.date_input(
        "Select analysis window",
        value=[start_date, end_date],
        key="overview_dates"
    )

    # Metrics placeholders
    col1, col2 = st.columns(2)
    col1.metric("Baseline Sales", "--")
    col2.metric("Media-driven Sales", "--")
