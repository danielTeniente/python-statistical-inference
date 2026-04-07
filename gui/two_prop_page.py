import streamlit as st
import pandas as pd
from logic.proportions_logic import (
    perform_two_proportion_ztest, 
    get_two_proportion_confint
)
from logic.independence_logic import perform_fisher_exact_test, get_contingency_table
from gui.components import show_code

def render_twoprop_test_page():
    st.title("Two Proportions Test")

    if "df" not in st.session_state or st.session_state.df is None:
        st.warning("⚠️ No data found. Please upload a file in the 'Upload Dataset' section first.")
        return
    
    df = st.session_state.df

    valid_cols = []
    for col in df.columns:
        if df[col].dropna().nunique() == 2:
            valid_cols.append(col)

    if len(valid_cols) < 2:
        st.error("The dataset must contain at least two binary columns (e.g., one for Groups, one for Outcomes).")
        return

    col1, col2 = st.columns(2)
    
    with col1:
        group_col = st.selectbox("Select Grouping Variable", valid_cols, index=0, 
                                 help="The variable that splits your data into two distinct groups.")
    
    remaining_cols = [c for c in valid_cols if c != group_col]
    
    with col2:
        if remaining_cols:
            outcome_col = st.selectbox("Select Outcome Variable", remaining_cols, index=0)
        else:
            outcome_col = st.selectbox("Select Outcome Variable", valid_cols, index=0)

    if group_col == outcome_col:
        st.warning("⚠️ Grouping variable and Outcome variable should be different.")
        return

    st.markdown("### Test Parameters")
    
    unique_vals = df[outcome_col].dropna().unique()
    success_term = st.selectbox(
        "Select the value that represents 'Success' in Outcome", 
        options=unique_vals,
        help="This value will be counted as a success to calculate the proportions in both groups."
    )

    col3, col4 = st.columns(2)
    with col3:
        confidence = st.slider("Confidence level", 0.80, 0.99, 0.95, 0.01)
    with col4:
        alternative = st.selectbox("Select alternative hypothesis", ["two-sided", "larger", "smaller"])

    st.divider()

    # contingency table
    with st.expander("Contingency Table", expanded=True):
        st.markdown("### Contingency Table (Crosstab) of the selected variables")
        contingency_table, code = get_contingency_table(df, group_col, outcome_col)
        
        show_code(code)
        
        st.write("### Contingency Table:")
        st.dataframe(contingency_table)

    with st.expander("Fisher's Exact Test", expanded=True):
        st.markdown("### Exact test for two proportions (Recommended for small samples)")
        stat, p_val, code = perform_fisher_exact_test(
            df=df, 
            var1_col=group_col, 
            var2_col=outcome_col, 
            alternative=alternative
        )
        
        show_code(code)
        
        st.write("### Results")
        if stat is not None:
            st.metric("**Odds Ratio (Statistic):**", f"{stat:.4f}")
        st.metric("**p-value:**", f"{p_val:.4f}")

    # --- Pruebas de Hipótesis ---
    with st.expander("Z-Test for Two Proportions", expanded=False):
        st.markdown("### Z-test for the difference in proportions")
        stat, p_val, code = perform_two_proportion_ztest(
            df=df, 
            group_col=group_col, 
            outcome_col=outcome_col, 
            success_term=success_term,
            alternative=alternative
        )
        
        show_code(code)
        
        st.write("### Results")
        st.metric("**Statistic (Z):**", f"{stat:.4f}")
        st.metric("**p-value:**", f"{p_val:.4f}")


    # --- Intervalos de Confianza ---
    with st.expander("Newcombe Confidence Interval", expanded=True):
        st.markdown("### Newcombe Confidence Interval for the difference in proportions")
        (lower, upper), code = get_two_proportion_confint(
            df=df, 
            group_col=group_col, 
            outcome_col=outcome_col,
            success_term=success_term,
            confidence=confidence,    
            method='newcomb'
        )
        
        show_code(code)
        
        st.write("### Results")
        st.metric(f"**Difference CI ({confidence*100:.0f}%):**", f"({lower:.4f}, {upper:.4f})")

    with st.expander("Wald Confidence Interval", expanded=False):
        st.markdown("### Wald Confidence Interval for the difference in proportions")
        (lower, upper), code = get_two_proportion_confint(
            df=df, 
            group_col=group_col, 
            outcome_col=outcome_col,
            success_term=success_term,
            confidence=confidence,    
            method='wald'
        )
        
        show_code(code)
        
        st.write("### Results")
        st.metric(f"**Difference CI ({confidence*100:.0f}%):**", f"({lower:.4f}, {upper:.4f})")