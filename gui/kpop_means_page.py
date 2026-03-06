import pandas as pd
import streamlit as st
from gui.components import show_code
from logic.basic_code import get_numeric_columns
from logic.kpop_logic import perform_oneway_anova, perform_pairwise_tukeyhsd, perform_pairwise_gameshowell

def render_kpop_means_page():
    st.title("k-Sample Mean Tests")
    
    if "df" not in st.session_state or st.session_state.df is None:
        st.warning("⚠️ No data found. Please upload a file in the 'Upload Dataset' section first.")
        return
        
    df = st.session_state.df
    numeric_cols = get_numeric_columns(df)
    
    if not numeric_cols:
        st.error("The dataset does not contain any numeric columns.")
        return
    
    selected_cols = st.multiselect("Select variables for the test", numeric_cols, default=numeric_cols[:3])
    
    # Validar que al menos haya dos variables para comparar
    if len(selected_cols) < 2:
        st.warning("Please select at least two variables to perform the tests.")
        return

    confidence = st.slider("Confidence level", 0.80, 0.99, 0.95, 0.01)
    equal_var = st.checkbox("Assume equal variances", value=True, key="equal_var")

    st.divider()
    
    with st.expander("One-way ANOVA for equality of means", expanded=True):
        st.markdown("### One-way ANOVA for equality of means")
        stat, p_value, code = perform_oneway_anova(df, selected_cols, equal_var)
        
        st.metric("One-way ANOVA statistic", f"{stat:.4f}")
        st.metric("P-value", f"{p_value:.4f}")
        show_code(code)

    if equal_var:
        with st.expander("Pairwise Tukey HSD test (Equal Variances Assumed)", expanded=False):
            st.markdown("### Pairwise Tukey HSD test")
            st.info("💡 **Tukey HSD** is used here because you assume equal variances'.")
            
            tukey_result, fig, code = perform_pairwise_tukeyhsd(df, selected_cols, confidence)
            
            st.markdown("**Statistical Results:**")
            st.text(str(tukey_result))
            
            st.markdown("**Confidence Interval Plot:**")
            st.pyplot(fig)
            show_code(code)
            
    else:
        with st.expander("Pairwise Games-Howell test (Equal Variances NOT Assumed)", expanded=False):
            st.markdown("### Pairwise Games-Howell test")
            st.info("💡 **Games-Howell** is used here because you don't assume equal variances'.")
            
            gh_result, fig, code = perform_pairwise_gameshowell(df, selected_cols, confidence)
            st.markdown("**Statistical Results:**")
            st.dataframe(gh_result, use_container_width=True)
            st.markdown("**Confidence Interval Plot:**")
            st.pyplot(fig)
            show_code(code)