import streamlit as st
from gui.components import show_code
from logic.basic_code import get_numeric_columns, get_categorical_columns
from logic.twopop_logic import perform_mannwhitney, plot_confidence_interval, get_sample_difference_in_medians

def render_twopop_medians_page():
    st.title("Two population medians tests")
    
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
    st.divider() 
    
    with st.expander("", expanded=True):
        
        st.markdown("### Mann-Whitney U test to compare medians")
        u_stat, p_value, ci, code = perform_mannwhitney(
            df, selected_num_col, selected_cat_col, alternative, confidence
        )
        show_code(code)
        res1, res2, res3 = st.columns(3)
        res1.metric("U-statistic", f"{u_stat:.4f}")
        res2.metric(f"P-value ({alternative})", f"{p_value:.4f}")
        res3.metric("Confidence Interval", f"({ci[0]:.4f}, {ci[1]:.4f})")
        
    with st.expander("Plot of the confidence interval", expanded=False):
        st.markdown("### Plot of the confidence interval for the ratio of variances")
        st.markdown('**Get the sample difference in medians**')
        dataset_diff, code_diff = get_sample_difference_in_medians(df, selected_num_col, selected_cat_col)
        st.markdown(f'{dataset_diff:.4f}')
        show_code(code_diff)
        fig, code_plot = plot_confidence_interval(ci[0], ci[1], 
            dataset_diff, title="Confidence Interval for the Difference in Medians", 
            x_label="Difference in Medians", y_label="Medians Test")
        show_code(code_plot)
        st.pyplot(fig)
