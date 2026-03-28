import streamlit as st
import pandas as pd
from gui.components import show_code
from logic.basic_code import get_numeric_columns, get_categorical_columns
from logic.ovr_logic import perform_ftest_ovr, perform_levene_ovr
from logic.twopop_logic import plot_confidence_interval

def render_ovr_variances_page():
    st.title("One-vs-Rest: Variances Tests")
    
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
        
    # 2. Filter categorical columns to find those with MORE THAN 2 unique values for OvR
    valid_categorical_cols = [col for col in all_categorical_cols if df[col].nunique() > 2]
                
    # 3. Handle case where no valid categorical column exists
    if not valid_categorical_cols:
        st.error("Error: The dataset must contain at least one categorical column with 3 or more categories (e.g., North/South/East) to perform a One-vs-Rest test.")
        st.info("For columns with exactly 2 categories, please use the standard 'Group vs. Group' tests.")
        return

    # 4. User Inputs
    st.markdown("### Test Setup")
    col1, col2, col3 = st.columns(3)
    with col1:
        selected_num_col = st.selectbox("Select numerical variable", numeric_cols, key="ovr_num_col")
    with col2:
        selected_cat_col = st.selectbox("Select grouping variable", valid_categorical_cols, key="ovr_cat_col")
    with col3:
        # Dynamically get categories based on selected column
        available_categories = df[selected_cat_col].dropna().unique().tolist()
        target_cat = st.selectbox("Select Target Population ('One')", available_categories, key="ovr_target_cat")
    
    # Feedback visual
    st.caption(f"Comparing: **{target_cat}** vs **The Rest**")
    
    st.markdown("#### Test Parameters")
    col4, col5 = st.columns(2)
    with col4:
        alternative = st.selectbox("Alternative hypothesis", ["two-sided", "less", "greater"], key="ovr_alternative")
    with col5:
        confidence = st.slider("Confidence level", 0.80, 0.99, 0.95, 0.01, key="ovr_confidence")

    st.divider() 
    
    # 5. Execute Tests
    with st.expander("F-test for equality of variances", expanded=True):
        st.markdown("### F-test to compare variances if both populations are normally distributed")
        # Llamar a la función OvR pasando target_cat
        f_stat, p_value, ci, code = perform_ftest_ovr(
            df, selected_num_col, selected_cat_col, target_cat, alternative, confidence
        )
        show_code(code)
        
        res1, res2, res3 = st.columns(3)
        res1.metric("F-statistic", f"{f_stat:.4f}")
        res2.metric(f"P-value ({alternative})", f"{p_value:.4f}")
        res3.metric("Confidence Interval", f"({ci[0]:.4f}, {ci[1]:.4f})")
        
    with st.expander("Levene's test for equality of variances", expanded=False):
        st.markdown("### Levene's test for equal variances if the populations are not normally distributed")
        stat_levene, p_value_levene, ci_levene, code_levene = perform_levene_ovr(
            df, selected_num_col, selected_cat_col, target_cat, confidence
        )
        
        show_code(code_levene)
        
        res1, res2, res3 = st.columns(3)
        res1.metric("Levene statistic", f"{stat_levene:.4f}")
        res2.metric("P-value", f"{p_value_levene:.4f}")
        res3.metric("Confidence Interval", f"({ci_levene[0]:.4f}, {ci_levene[1]:.4f})")

    with st.expander("Plot of the confidence interval", expanded=False):
        st.markdown("### Plot of the confidence interval for the ratio of variances")
        # El plot function no cambia, ya que solo recibe los intervalos pre-calculados
        fig, code_plot = plot_confidence_interval(
            ci[0], ci[1], f_stat, 
            title=f"Variance Ratio CI ({target_cat} / Rest)", 
            x_label="Ratio", 
            y_label="Variance Test", 
            H0=1
        )
        show_code(code_plot)
        st.pyplot(fig)