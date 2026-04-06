import streamlit as st
import pandas as pd
from logic.proportions_logic import perform_one_proportion_binomial_test, perform_one_proportion_ztest, get_one_proportion_interval
from gui.components import show_code

def render_oneprop_test_page():
    st.title("One Proportion Test")

    if "df" not in st.session_state or st.session_state.df is None:
        st.warning("⚠️ No data found. Please upload a file in the 'Upload Dataset' section first.")
        return
    
    df = st.session_state.df

    valid_cols = []
    for col in df.columns:
        if df[col].isnull().all():
            continue
        is_numeric = pd.api.types.is_numeric_dtype(df[col])
        is_bool = pd.api.types.is_bool_dtype(df[col])
        if is_bool:
            valid_cols.append(col)
            continue
        if is_numeric:
            col_min = df[col].min()
            col_max = df[col].max()
            if col_min >= 0 and col_max <= 1:
                if df[col].dropna().isin([0, 1]).all():
                    valid_cols.append(col)
                    continue 

        if df[col].nunique() == 2:
            if col not in valid_cols:
                valid_cols.append(col)


    if not valid_cols:
        st.error("The dataset does not contain any valid binary columns (0/1 or exactly two categories).")
        return

    selected_col = st.selectbox("Select variable", valid_cols)
    confidence = st.slider("Confidence level", 0.80, 0.99, 0.95, 0.01)
    
    unique_vals = df[selected_col].dropna().unique()
    is_numeric = pd.api.types.is_numeric_dtype(df[selected_col])
    is_0_1_only = set(unique_vals).issubset({0, 1, 0.0, 1.0})
    
    success_term = None
    if not (is_numeric and is_0_1_only):
        success_term = st.selectbox(
            "Select the value that represents 'Success'", 
            options=unique_vals,
            help="This value will be counted to calculate the proportion."
        )

    p0 = st.number_input(
        label="Hypothesized Proportion (H₀)", 
        value=0.50, 
        min_value=0.01,
        max_value=0.99,
        step=0.05,
        help="Enter the theoretical proportion you want to test against (e.g., 0.5 for 50%)."
    )    
    
    alternative = st.selectbox("Select alternative hypothesis", ["two-sided", "greater", "less"])

    with st.expander("Binomial Test", expanded=True):
        st.markdown("### Exact binomial test for the proportion")
        stat, p_val, code = perform_one_proportion_binomial_test(
            df=df, 
            selected_column=selected_col, 
            p0=p0, 
            alternative=alternative, 
            success_term=success_term
        )
        
        show_code(code)
        
        st.write("### Results")
        st.metric(f"**Proportion of Successes (Statistic):**", f"{stat:.4f} ({stat*100:.2f}%)")
        st.metric(f"**p-value:**", f"{p_val:.4f}")

    with st.expander("Z-Test", expanded=False):
        st.markdown("### Z-test for the proportion")
        stat, p_val, code = perform_one_proportion_ztest(
            df=df, 
            selected_column=selected_col, 
            p0=p0, 
            alternative=alternative, 
            success_term=success_term
        )
        
        show_code(code)
        
        st.write("### Results")
        st.metric(f"**Statistic:**", f"{stat:.4f} ({stat*100:.2f}%)")
        st.metric(f"**p-value:**", f"{p_val:.4f}")

    with st.expander("Clopper-Pearson Interval", expanded=True):
        st.markdown("### Clopper-Pearson Confidence Interval for the proportion")
        (lower, upper), code = get_one_proportion_interval(
            df=df, 
            selected_column=selected_col, 
            confidence=confidence,    
            success_term=success_term,
            method='beta'
        )
        st.write("### Results")
        st.metric(f"**Confidence Interval ({confidence*100:.0f}%):**", f"({lower:.4f}, {upper:.4f})")
        
        show_code(code)

    with st.expander("Wilson Interval", expanded=False):
        st.markdown("### Wilson Confidence Interval for the proportion")
        (lower, upper), code = get_one_proportion_interval(
            df=df, 
            selected_column=selected_col, 
            confidence=confidence,    
            success_term=success_term,
            method='wilson'
        )
        st.write("### Results")
        st.metric(f"**Confidence Interval ({confidence*100:.0f}%):**", f"({lower:.4f}, {upper:.4f})")
        
        show_code(code)        
        

        