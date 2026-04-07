import streamlit as st
from gui.components import show_code
from logic.basic_code import get_numeric_columns, get_categorical_columns
# Asegúrate de tener estas versiones _ovr en tu logic.twopop_logic
from logic.ovr_logic import perform_mannwhitney_ovr, get_sample_difference_in_medians_ovr
from logic.twopop_logic import plot_confidence_interval

def render_ovr_medians_page():
    st.title("One-vs-Rest: Medians Tests")
    
    # 1. Verificación de datos
    if "df" not in st.session_state or st.session_state.df is None:
        st.warning("⚠️ No data found. Please upload a file in the 'Upload Dataset' section first.")
        return
        
    df = st.session_state.df
    numeric_cols = get_numeric_columns(df)
    all_categorical_cols = get_categorical_columns(df)
    
    if not numeric_cols:
        st.error("The dataset does not contain any numeric columns.")
        return
    
    # 2. Filter categorical columns for OvR (needs > 2 categories)
    valid_categorical_cols = [col for col in all_categorical_cols if df[col].nunique() > 2]
    
    if not valid_categorical_cols:
        st.error("Error: The dataset must contain at least one categorical column with 3 or more categories to perform a One-vs-Rest test.")
        st.info("For columns with exactly 2 categories, please use the standard 'Group vs. Group' tests.")
        return
    
    # 3. User Inputs (3 columns for Target Category)
    st.markdown("### Test Setup")
    col1, col2, col3 = st.columns(3)
    with col1:
        selected_num_col = st.selectbox("Select numerical variable to analyze", numeric_cols, key="ovr_median_num")
    with col2:
        selected_cat_col = st.selectbox("Select grouping variable", valid_categorical_cols, key="ovr_median_cat")
    with col3:
        available_categories = df[selected_cat_col].dropna().unique().tolist()
        target_cat = st.selectbox("Select Target Population ('One')", available_categories, key="ovr_median_target")

    st.caption(f"Comparing: **{target_cat}** vs **The Rest**")

    st.markdown("#### Test Parameters")
    col4, col5 = st.columns(2)
    with col4:
        alternative = st.selectbox("Alternative hypothesis", ["two-sided", "less", "greater"], key="ovr_median_alt")
    with col5:
        confidence = st.slider("Confidence level", 0.80, 0.99, 0.95, 0.01, key="ovr_median_conf")
    
    st.divider() 
    
    with st.expander("Mann-Whitney U test to compare medians", expanded=True):
        st.markdown("### Mann-Whitney U test to compare medians")
        u_stat, p_value, ci, code = perform_mannwhitney_ovr(
            df, selected_num_col, selected_cat_col, target_cat, alternative, confidence
        )
        show_code(code)
        
        res1, res2, res3 = st.columns(3)
        res1.metric("U-statistic", f"{u_stat:.4f}")
        res2.metric(f"P-value ({alternative})", f"{p_value:.4f}")
        res3.metric("Confidence Interval", f"({ci[0]:.4f}, {ci[1]:.4f})")
        
    with st.expander("Plot of the confidence interval", expanded=False):
        st.markdown("### Plot of the confidence interval for the difference in medians")
        st.markdown('**Get the sample difference in medians**')
        
        dataset_diff, code_diff = get_sample_difference_in_medians_ovr(
            df, selected_num_col, selected_cat_col, target_cat
        )
        
        st.markdown(f'{dataset_diff:.4f}')
        show_code(code_diff)
        
        fig, code_plot = plot_confidence_interval(
            ci[0], ci[1], dataset_diff, 
            title=f"Confidence Interval for Difference in Medians ({target_cat} - Rest)", 
            x_label="Difference in Medians", 
            y_label="Medians Test"
        )
        
        show_code(code_plot)
        st.pyplot(fig)