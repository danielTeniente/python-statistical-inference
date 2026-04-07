import streamlit as st
from gui.components import show_code
from logic.basic_code import get_numeric_columns, get_categorical_columns
from logic.ovr_logic import perform_ttest_ovr, get_sample_difference_in_means_ovr
from logic.twopop_logic import plot_confidence_interval

def render_ovr_means_page():
    st.title("One-vs-Rest: Means Tests")
    
    if "df" not in st.session_state or st.session_state.df is None:
        st.warning("⚠️ No data found. Please upload a file in the 'Upload Dataset' section first.")
        return
        
    df = st.session_state.df
    numeric_cols = get_numeric_columns(df)
    all_categorical_cols = get_categorical_columns(df)
    
    if not numeric_cols:
        st.error("The dataset does not contain any numeric columns.")
        return

    valid_categorical_cols = [col for col in all_categorical_cols if df[col].nunique() > 2]
    
    if not valid_categorical_cols:
        st.error("Error: The dataset must contain at least one categorical column with 3 or more categories (e.g., North/South/East) to perform a One-vs-Rest test.")
        st.info("For columns with exactly 2 categories, please use the standard 'Group vs. Group' tests.")
        return
    
    st.markdown("### Test Setup")
    col1, col2, col3 = st.columns(3)
    with col1:
        selected_num_col = st.selectbox("Select numerical variable", numeric_cols, key="ovr_mean_num")
    with col2:
        selected_cat_col = st.selectbox("Select grouping variable", valid_categorical_cols, key="ovr_mean_cat")
    with col3:
        available_categories = df[selected_cat_col].dropna().unique().tolist()
        target_cat = st.selectbox("Select Target Population ('One')", available_categories, key="ovr_mean_target")

    st.caption(f"Comparing: **{target_cat}** vs **The Rest**")

    st.markdown("#### Test Parameters")
    col4, col5 = st.columns(2)
    with col4:
        alternative = st.selectbox("Alternative hypothesis", ["two-sided", "less", "greater"], key="ovr_mean_alt")
    with col5:
        confidence = st.slider("Confidence level", 0.80, 0.99, 0.95, 0.01, key="ovr_mean_conf")
        
    equal_var = st.checkbox("Assume equal variances", value=True, key="ovr_equal_var")

    st.divider() 
        
    st.markdown("### T-test to compare means")
    t_stat, p_value, ci, code = perform_ttest_ovr(
        df, selected_num_col, selected_cat_col, target_cat, alternative, confidence, equal_var
    )
    
    show_code(code)
    
    res1, res2, res3 = st.columns(3)
    res1.metric("T-statistic", f"{t_stat:.4f}")
    res2.metric(f"P-value ({alternative})", f"{p_value:.4f}")
    res3.metric("Confidence Interval", f"({ci[0]:.4f}, {ci[1]:.4f})")
        
    # 6. Gráfico del Intervalo de Confianza
    with st.expander("Plot of the confidence interval", expanded=False):
        st.markdown("### Plot of the confidence interval for the difference in means")
        st.markdown('**Get the sample difference in means**')
        
        # Llamar a la versión _ovr que calcula la diferencia entre target y el resto
        sample_diff, code_diff = get_sample_difference_in_means_ovr(
            df, selected_num_col, selected_cat_col, target_cat
        )
        
        st.metric("Sample Difference in Means", f"{sample_diff:.4f}")
        show_code(code_diff)
        
        fig, code_plot = plot_confidence_interval(
            ci[0], ci[1], sample_diff, 
            title=f"Confidence Interval for Difference in Means ({target_cat} - Rest)", 
            x_label="Difference in Means", 
            y_label="Means Test"
        )
        
        show_code(code_plot)
        st.pyplot(fig)