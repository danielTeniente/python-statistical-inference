import streamlit as st
from logic.basic_code import get_categorical_columns
from logic.descriptive_stats_page_logic import (
    get_frequency_table, get_barplot)

from gui.components import show_code

def render_descriptive_categorical_page():
    if st.session_state.df is None:
        st.warning("⚠️ Please upload a dataset first to see descriptive statistics.")
        return

    df = st.session_state.df
    st.title("📈 Descriptive Statistics: Categorical Variables")

    # Get categorical columns
    categorical_cols = get_categorical_columns(df)
    if not categorical_cols:
        st.error("The dataset does not contain any categorical columns.")
        return
    
    st.subheader("DESCRIPTIVE STATISTICS FOR EACH COLUMN")
    st.markdown("#### Frequency Table")
    selected_column = st.selectbox("Select column", categorical_cols)
    freq_table, code = get_frequency_table(df, selected_column)
    show_code(code)
    st.dataframe(freq_table, width="stretch")

    st.markdown("#### Bar Plot")
    fig, plot_code = get_barplot(df, selected_column)
    show_code(plot_code)
    st.pyplot(fig)
