import streamlit as st
from logic.basic_code import get_categorical_columns
from logic.descriptive_stats_page_logic import (
    get_frequency_table, get_barplot)
from gui.components import show_code

def render_descriptive_categorical_page():
    # --- 1. Initial Validation ---
    if "df" not in st.session_state or st.session_state.df is None:
        st.warning("⚠️ Please upload a dataset first to see descriptive statistics.")
        return

    df = st.session_state.df
    st.title("📈 Descriptive Statistics: Categorical Variables")

    # Get categorical columns
    categorical_cols = get_categorical_columns(df)
    if not categorical_cols:
        st.error("The dataset does not contain any categorical columns.")
        return
    
    # --- 2. Variable Selection ---
    st.subheader("Configuration")
    selected_column = st.selectbox("Select column to analyze", categorical_cols, key="cat_stats_sel")

    # --- 3. Context ID and State Management ---
    # We invalidate the cache if the user switches the column
    current_context_id = f"cat_stats_{selected_column}"

    if ("cat_stats_state" not in st.session_state or 
        st.session_state.get("cat_stats_id") != current_context_id):
        
        st.session_state.cat_stats_state = {}  # Isolated results dictionary
        st.session_state.cat_stats_id = current_context_id

    # Shortcut to our result dictionary
    state = st.session_state.cat_stats_state

    st.divider()

    # --- 4. Granular Execution Sections ---

    # SECTION: Frequency Table
    with st.expander("📊 1. Frequency Table", expanded=not state.get("table")):
        if st.button("Generate Frequency Table", key="btn_freq_table"):
            with st.spinner("Processing categories..."):
                freq_table, code_t = get_frequency_table(df, selected_column)
                state["table"] = {"data": freq_table, "code": code_t}

        if "table" in state:
            res_t = state["table"]
            show_code(res_t["code"])
            st.write("### Frequency Results:")
            st.dataframe(res_t["data"], use_container_width=True)

    # SECTION: Bar Plot
    with st.expander("📉 2. Visualization (Bar Plot)", expanded=False):
        if st.button("Generate Bar Plot", key="btn_bar_plot"):
            with st.spinner("Generating visualization..."):
                fig, code_p = get_barplot(df, selected_column)
                state["plot"] = {"fig": fig, "code": code_p}

        if "plot" in state:
            res_p = state["plot"]
            show_code(res_p["code"])
            st.pyplot(res_p["fig"])