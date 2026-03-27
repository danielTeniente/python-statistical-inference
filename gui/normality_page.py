import streamlit as st
from logic.basic_code import get_numeric_columns
from gui.components import show_code
from logic.normality_page_logic import run_normality_test, get_qqplot

def render_normality_test_page():
    st.title("📊 Normality Tests")
    
    if "df" not in st.session_state or st.session_state.df is None:
        st.warning("⚠️ No data found. Please upload a file in the 'Upload Dataset' section first.")
        return

    df = st.session_state.df
    numeric_cols = get_numeric_columns(df)
    
    if not numeric_cols:
        st.error("The dataset does not contain any numeric columns.")
        return

    # User Selection
    col1, col2 = st.columns(2)
    with col1:
        selected_col = st.selectbox("Select variable", numeric_cols)
    with col2:
        test_name = st.selectbox("Select test", 
            ["Shapiro–Wilk", "D’Agostino–Pearson", 
             "Kolmogorov–Smirnov", "Anderson-Darling"])
    
    # Call Logic
    stat, p, code = run_normality_test(df, selected_col, test_name)
    # Show code
    show_code(code)    
    # Display Results
    st.divider()
    st.subheader(f"Results for {test_name}")
    
    res_c1, res_c2 = st.columns(2)
    res_c1.metric("Statistic", f"{stat:.4f}")
    res_c2.metric("p-value", f"{p:.4f}")
    
    # Show QQ-Plot
    st.divider()
    st.subheader("QQ-Plot")
    fig, qq_code = get_qqplot(df, selected_col)
    show_code(qq_code)
    st.pyplot(fig)