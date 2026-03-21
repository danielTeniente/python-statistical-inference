import streamlit as st
import pandas as pd
from gui.components import show_code
from logic.basic_code import get_numeric_columns, get_categorical_columns
from logic.twopop_logic import perform_ftest, perform_levene, plot_confidence_interval

def render_twopop_variances_page():
    st.title("Two Population Variances Tests")
    
    if "df" not in st.session_state or st.session_state.df is None:
        st.warning("⚠️ No data found. Please upload a file in the 'Upload Dataset' section first.")
        return
        
    df = st.session_state.df
    
    numeric_cols = get_numeric_columns(df)
    all_categorical_cols = get_categorical_columns(df)
    
    if not numeric_cols:
        st.error("The dataset does not contain any numeric columns.")
        return
        
    # 2. Filter categorical columns to find those with exactly 2 unique values
    valid_categorical_cols = [col for col in all_categorical_cols if df[col].nunique() == 2]
                
    # 3. Handle case where no valid categorical column exists
    if not valid_categorical_cols:
        st.error("Error: The dataset must contain at least one categorical column with exactly two categories (e.g., Male/Female) to perform this test.")
        st.info("Please review your dataset or use a different test.")
        return

    st.markdown("### Test Setup")
    col1, col2 = st.columns(2)
    with col1:
        # Select the numeric variable to test
        selected_num_col = st.selectbox("Select numerical variable to analyze", numeric_cols, key="num_col")
    with col2:
        # Select the categorical variable to group by
        selected_cat_col = st.selectbox("Select grouping variable (2 populations)", valid_categorical_cols, key="cat_col")
    
    groups = df[selected_cat_col].dropna().unique()
    st.caption(f"Comparing groups: **{groups[0]}** vs **{groups[1]}**")
    
    col3, col4 = st.columns(2)
    with col3:
        alternative = st.selectbox("Alternative hypothesis", ["two-sided", "less", "greater"], key="alternative")
    with col4:
        confidence = st.slider("Confidence level", 0.80, 0.99, 0.95, 0.01)

    st.divider() 
    
    with st.expander("F-test for equality of variances", expanded=True):
        st.markdown("### F-test to compare variances if both populations are normally distributed")
        f_stat, p_value, ci, code = perform_ftest(
            df, selected_num_col, selected_cat_col, alternative, confidence
        )
        show_code(code)
        res1, res2, res3 = st.columns(3)
        res1.metric("F-statistic", f"{f_stat:.4f}")
        res2.metric(f"P-value ({alternative})", f"{p_value:.4f}")
        res3.metric("Confidence Interval", f"({ci[0]:.4f}, {ci[1]:.4f})")
        
    with st.expander("Levene's test for equality of variances", expanded=False):
        st.markdown("### Levene's test for equal variances if the populations are not normally distributed")
        stat, p_value_levene, ci_levene, code_levene = perform_levene(
            df, selected_num_col, selected_cat_col, confidence
        )
        
        show_code(code_levene)
        
        res1, res2, res3 = st.columns(3)
        res1.metric("Levene statistic", f"{stat:.4f}")
        res2.metric("P-value", f"{p_value_levene:.4f}")
        res3.metric("Confidence Interval", f"({ci_levene[0]:.4f}, {ci_levene[1]:.4f})")

    with st.expander("Plot of the confidence interval", expanded=False):
        st.markdown("### Plot of the confidence interval for the ratio of variances")
        fig, code_plot = plot_confidence_interval(
            ci[0], ci[1], f_stat, 
            title="Confidence Interval for the Variance Ratio", 
            x_label="Ratio", 
            y_label="Variance Test", 
            H0=1
        )
        show_code(code_plot)
        st.pyplot(fig)        