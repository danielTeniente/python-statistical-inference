import pandas as pd
import streamlit as st
from gui.components import show_code
from logic.basic_code import get_numeric_columns
from logic.kpop_logic import perform_bootstrap_pairwise_median, perform_krustall_wallis

def render_kpop_medians_page():
    st.title("k-Sample Median Tests")
    
    if "df" not in st.session_state or st.session_state.df is None:
        st.warning("⚠️ No data found. Please upload a file in the 'Upload Dataset' section first.")
        return
        
    df = st.session_state.df
    numeric_cols = get_numeric_columns(df)
    
    if not numeric_cols:
        st.error("The dataset does not contain any numeric columns.")
        return
    
    selected_cols = st.multiselect("Select variables for the test", numeric_cols, default=numeric_cols[:3])
    
    if len(selected_cols) < 2:
        st.warning("Please select at least two variables to perform the tests.")
        return

    confidence = st.slider("Confidence level", 0.80, 0.99, 0.95, 0.01)

    st.divider()
    
    with st.expander("Kruskal-Wallis H-test for equality of medians", expanded=True):
        st.markdown("### Kruskal-Wallis H-test for equality of medians")
        stat, p_value, code = perform_krustall_wallis(df, selected_cols)
        
        st.metric("Kruskal-Wallis H statistic", f"{stat:.4f}")
        st.metric("P-value", f"{p_value:.4f}")
        show_code(code)

    with st.expander("Bootstrap Pairwise Confidence Intervals", expanded=False):
        st.markdown("### Bootstrap Pairwise Confidence Intervals")
        results_df, fig, code = perform_bootstrap_pairwise_median(df, selected_cols, confidence)
        
        st.markdown("**Statistical Results:**")
        st.dataframe(results_df)
        
        st.markdown("**Confidence Interval Plot:**")
        st.pyplot(fig)
        show_code(code)