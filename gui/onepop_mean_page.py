import streamlit as st
from logic.basic_code import get_numeric_columns
from gui.components import show_code
from logic.onepop_mean_logic import perform_ttest

def render_onepop_mean_page():
    st.title("One population mean test")
    st.markdown("### One sample t-test if the population is normally distributed")
    
    if "df" not in st.session_state or st.session_state.df is None:
        st.warning("⚠️ No data found. Please upload a file in the 'Upload Dataset' section first.")
        return
    df = st.session_state.df
    numeric_cols = get_numeric_columns(df)
    if not numeric_cols:
        st.error("The dataset does not contain any numeric columns.")
        return
    
    selected_col = st.selectbox("Select variable", numeric_cols)
    popmean = st.number_input(
        label="Hypothesized Population Mean (H₀)", 
        value=df[selected_col].mean(), 
        step=0.1,
        help="Enter the population mean value you want to test your sample against (Null Hypothesis)."
    )    
    
    alternative = st.selectbox("Select alternative hypothesis", 
        ["two-sided", "greater", "less"])
    confidence = st.slider("Confidence level", 0.80, 0.99, 0.95, 0.01)
    res, ci, code = perform_ttest(df, selected_col, popmean, alternative, confidence)
    show_code(code)
    st.write(f"t-statistic: {res.statistic:.4f}")
    st.write(f"p-value: {res.pvalue:.4f}")
    st.write(f"Confidence Interval: ({ci[0]:.4f}, {ci[1]:.4f})")