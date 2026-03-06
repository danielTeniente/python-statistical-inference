import streamlit as st
from gui.components import show_code
from logic.basic_code import get_numeric_columns
from logic.twopop_logic import perform_mannwhitney, plot_confidence_interval
from logic.descriptive_stats_page_logic import get_median

def render_twopop_medians_page():
    st.title("Two population medians tests")
    
    if "df" not in st.session_state or st.session_state.df is None:
        st.warning("⚠️ No data found. Please upload a file in the 'Upload Dataset' section first.")
        return
        
    df = st.session_state.df
    numeric_cols = get_numeric_columns(df)
    
    if not numeric_cols:
        st.error("The dataset does not contain any numeric columns.")
        return
    
    col1, col2 = st.columns(2)
    with col1:
        selected_col1 = st.selectbox("Select variable for population 1", numeric_cols, key="pop1")
    with col2:
        selected_col2 = st.selectbox("Select variable for population 2", numeric_cols, key="pop2")
    
    col3, col4 = st.columns(2)
    with col3:
        alternative = st.selectbox("Alternative hypothesis", ["two-sided", "less", "greater"], key="alternative")
    with col4:
        confidence = st.slider("Confidence level", 0.80, 0.99, 0.95, 0.01)
    # equal variances assumption
    st.divider() 
    
    with st.expander("", expanded=True):
        
        st.markdown("### Mann-Whitney U test to compare medians")
        u_stat, p_value, ci, code = perform_mannwhitney(
            df, selected_col1, selected_col2, alternative, confidence
        )
        show_code(code)
        res1, res2, res3 = st.columns(3)
        res1.metric("U-statistic", f"{u_stat:.4f}")
        res2.metric(f"P-value ({alternative})", f"{p_value:.4f}")
        res3.metric("Confidence Interval", f"({ci[0]:.4f}, {ci[1]:.4f})")
        
    with st.expander("Plot of the confidence interval", expanded=False):
        st.markdown("### Plot of the confidence interval for the ratio of variances")
        dataset_median1,_ = get_median(df,selected_col1)
        dataset_median2,_ = get_median(df,selected_col2)
        dataset_diff = dataset_median1 - dataset_median2

        fig, code_plot = plot_confidence_interval(ci[0], ci[1], 
            dataset_diff, title="Confidence Interval for the Difference in Medians", 
            x_label="Difference in Medians", y_label="Medians Test")
        show_code(code_plot)
        st.pyplot(fig)
