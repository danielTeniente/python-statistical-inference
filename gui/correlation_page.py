import streamlit as st
from gui.components import show_code
from logic.basic_code import get_numeric_columns

from logic.correlation_logic import (
    get_scatterplot, 
    perform_pearson_correlation, 
    perform_spearman_correlation, 
    perform_kendall_correlation
)

def render_correlation_page():
    st.title("Correlation Analysis")
    
    if "df" not in st.session_state or st.session_state.df is None:
        st.warning("⚠️ No data found. Please upload a file in the 'Upload Dataset' section first.")
        return
        
    df = st.session_state.df
    numeric_cols = get_numeric_columns(df)
    
    # Validation: We need at least 2 numerical columns for correlation
    if len(numeric_cols) < 2:
        st.error("Error: The dataset must contain at least two numerical columns to perform correlation analysis.")
        st.info("Please review your dataset or use a different test.")
        return
    
    st.markdown("### Select Variables")
    col1, col2 = st.columns(2)
    with col1:
        # Default to the first numeric column
        selected_x = st.selectbox("Select X variable (Independent)", numeric_cols, index=0, key="corr_x")
    with col2:
        # Default to the second numeric column if available
        default_y_index = 1 if len(numeric_cols) > 1 else 0
        selected_y = st.selectbox("Select Y variable (Dependent)", numeric_cols, index=default_y_index, key="corr_y")

    if selected_x == selected_y:
        st.warning("⚠️ You have selected the same variable for both X and Y. The correlation will be exactly 1.0")

    st.divider()
    
    # --- Correlation Measures ---
    st.markdown("### Correlation Measures")
    
    # 1. Pearson Expander
    with st.expander("Pearson Correlation (Linear Relationship)", expanded=True):
        st.markdown("Measures the linear relationship between two continuous variables.")
        corr_p, p_val_p, ci_low_p, ci_high_p, code_p = perform_pearson_correlation(df, selected_x, selected_y)
        
        show_code(code_p)
        
        res1, res2, res3 = st.columns(3)
        res1.metric("Pearson's r", f"{corr_p:.4f}")
        res2.metric("P-value", f"{p_val_p:.4f}")
        res3.metric("95% Confidence Interval", f"[{ci_low_p:.4f}, {ci_high_p:.4f}]")

    # 2. Spearman Expander
    with st.expander("Spearman Rank Correlation (Monotonic Relationship)", expanded=False):
        st.markdown("Non-parametric test that measures monotonic relationships. Uses Bootstrap for Confidence Intervals.")
        corr_s, p_val_s, ci_low_s, ci_high_s, code_s = perform_spearman_correlation(df, selected_x, selected_y)
        
        show_code(code_s)
        
        res1, res2, res3 = st.columns(3)
        res1.metric("Spearman's rho", f"{corr_s:.4f}")
        res2.metric("P-value", f"{p_val_s:.4f}")
        res3.metric("95% Bootstrap CI", f"[{ci_low_s:.4f}, {ci_high_s:.4f}]")

    # 3. Kendall Expander
    with st.expander("Kendall Tau Correlation (Ordinal Association)", expanded=False):
        st.markdown("Non-parametric test based on concordant and discordant pairs. Uses Bootstrap for Confidence Intervals.")
        corr_k, p_val_k, ci_low_k, ci_high_k, code_k = perform_kendall_correlation(df, selected_x, selected_y)
        
        show_code(code_k)
        
        res1, res2, res3 = st.columns(3)
        res1.metric("Kendall's tau", f"{corr_k:.4f}")
        res2.metric("P-value", f"{p_val_k:.4f}")
        res3.metric("95% Bootstrap CI", f"[{ci_low_k:.4f}, {ci_high_k:.4f}]")

    # --- Visual Exploration: Scatterplot ---
    st.markdown("### Visual Exploration")
    show_line = st.checkbox("Show linear regression line", value=False, key="show_reg_line")
    
    fig, code_scatter = get_scatterplot(df, selected_x, selected_y, show_line=show_line)
    st.pyplot(fig)
    
    show_code(code_scatter)

    st.divider()
