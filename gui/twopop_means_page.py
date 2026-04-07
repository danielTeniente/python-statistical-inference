import streamlit as st
from gui.components import show_code
from logic.basic_code import get_numeric_columns, get_categorical_columns
from logic.twopop_logic import perform_ttest, plot_confidence_interval, get_sample_difference_in_means

def render_twopop_means_page():
    st.title("Two population means tests")
    
    if "df" not in st.session_state or st.session_state.df is None:
        st.warning("⚠️ No data found. Please upload a file in the 'Upload Dataset' section first.")
        return
        
    df = st.session_state.df
    numeric_cols = get_numeric_columns(df)
    all_categorical_cols = get_categorical_columns(df)
    
    if not numeric_cols:
        st.error("The dataset does not contain any numeric columns.")
        return

    valid_categorical_cols = [col for col in all_categorical_cols if df[col].nunique() == 2]
    if not valid_categorical_cols:
        st.error("Error: The dataset must contain at least one categorical column with exactly two categories (e.g., Male/Female) to perform this test.")
        st.info("Please review your dataset or use a different test.")
        return
    
    col1, col2 = st.columns(2)
    with col1:
        selected_num_col = st.selectbox("Select numerical variable to analyze", numeric_cols, key="num_col")
    with col2:
        selected_cat_col = st.selectbox("Select grouping variable (2 populations)", valid_categorical_cols, key="cat_col")

    groups = df[selected_cat_col].dropna().unique()
    st.caption(f"Comparing groups: **{groups[0]}** vs **{groups[1]}**")

    col3, col4 = st.columns(2)
    with col3:
        alternative = st.selectbox("Alternative hypothesis", ["two-sided", "less", "greater"], key="alternative")
    with col4:
        confidence = st.slider("Confidence level", 0.80, 0.99, 0.95, 0.01)
    # equal variances assumption
    equal_var = st.checkbox("Assume equal variances", value=True, key="equal_var")

    st.divider() 
        
    st.markdown("### T-test to compare means")
    t_stat, p_value, ci, code = perform_ttest(
        df, selected_num_col, selected_cat_col, alternative, confidence, equal_var
    )
    show_code(code)
    res1, res2, res3 = st.columns(3)
    res1.metric("T-statistic", f"{t_stat:.4f}")
    res2.metric(f"P-value ({alternative})", f"{p_value:.4f}")
    res3.metric("Confidence Interval", f"({ci[0]:.4f}, {ci[1]:.4f})")
        
    with st.expander("Plot of the confidence interval", expanded=False):
        st.markdown("### Plot of the confidence interval for the difference in means")
        st.markdown('**Get the sample difference in means**')
        sample_diff, code_diff = get_sample_difference_in_means(df, selected_num_col, selected_cat_col)
        st.metric("Sample Difference in Means", f"{sample_diff:.4f}")

        show_code(code_diff)
        
        fig, code_plot = plot_confidence_interval(ci[0], ci[1], 
            sample_diff, title="Confidence Interval for the Difference in Means", 
            x_label="Difference in Means", y_label="Means Test")
        show_code(code_plot)
        st.pyplot(fig)
