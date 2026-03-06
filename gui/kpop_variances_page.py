import streamlit as st
from gui.components import show_code
from logic.basic_code import get_numeric_columns
from logic.kpop_logic import perform_bartlett, perform_levene

def render_kpop_variances_page():
    st.title("K population variances tests")
    
    # 1. Verificación de datos
    if "df" not in st.session_state or st.session_state.df is None:
        st.warning("⚠️ No data found. Please upload a file in the 'Upload Dataset' section first.")
        return
        
    df = st.session_state.df
    numeric_cols = get_numeric_columns(df)
    
    if not numeric_cols:
        st.error("The dataset does not contain any numeric columns.")
        return
    
    selected_cols = st.multiselect("Select variables for the test", numeric_cols, default=numeric_cols[:3])

    st.divider()
    with st.expander("Bartlett's test for equal variances", expanded=True):
        
        st.markdown("### Bartlett's test for equal variances if all populations are normally distributed")
        stat, p_value, code = perform_bartlett(df, selected_cols)
        show_code(code)
        st.metric("Bartlett statistic", f"{stat:.4f}")
        st.metric("P-value", f"{p_value:.4f}")

    with st.expander("Levene's test for equal variances", expanded=False):
        st.markdown("### Levene's test for equal variances if at least one population is not normally distributed")
        stat, p_value, code = perform_levene(df, selected_cols)
        show_code(code)
        st.metric("Levene statistic", f"{stat:.4f}")
        st.metric("P-value", f"{p_value:.4f}")