import pandas as pd
import streamlit as st
from gui.components import show_code
from logic.basic_code import get_numeric_columns, get_categorical_columns
from logic.kpop_logic import perform_oneway_anova, perform_pairwise_tukeyhsd, perform_pairwise_gameshowell

def render_kpop_means_page():
    st.title("k-Sample Mean Tests")
    
    if "df" not in st.session_state or st.session_state.df is None:
        st.warning("⚠️ No data found. Please upload a file in the 'Upload Dataset' section first.")
        return
        
    df = st.session_state.df
    numeric_cols = get_numeric_columns(df)
    all_categorical_cols = get_categorical_columns(df)
    
    if not numeric_cols:
        st.error("The dataset does not contain any numeric columns.")
        return
    
    valid_categorical_cols = [col for col in all_categorical_cols if df[col].nunique() >= 3]
            
    if not valid_categorical_cols:
        st.error("Error: The dataset must contain at least one categorical column with 3 or more categories (e.g., North/South/East) to perform these tests.")
        st.info("Please review your dataset or use a two-population test instead.")
        return
    
    st.markdown("### Test Setup")
    col1, col2 = st.columns(2)
    with col1:
        selected_num_col = st.selectbox("Select numerical variable to analyze", numeric_cols, key="kpop_num")
    with col2:
        selected_cat_col = st.selectbox("Select grouping variable (≥ 3 categories)", valid_categorical_cols, key="kpop_cat")

    available_categories = df[selected_cat_col].dropna().unique().tolist()

    st.markdown("#### Filter Populations")
    selected_categories = st.multiselect(
        "Select the specific categories you want to compare (minimum 3 required):", 
        available_categories, 
        default=available_categories, # By default, all are selected
        key="kpop_categories"
    )

    # Enforce minimum 3 categories rule
    if len(selected_categories) < 3:
        st.warning("⚠️ Please select at least 3 categories to perform K population tests.")
        return # Stop execution until the user selects enough categories
    filtered_df = df[df[selected_cat_col].isin(selected_categories)].copy()

    confidence = st.slider("Confidence level", 0.80, 0.99, 0.95, 0.01)
    equal_var = st.checkbox("Assume equal variances", value=True, key="equal_var")

    st.divider()
    
    with st.expander("One-way ANOVA for equality of means", expanded=True):
        st.markdown("### One-way ANOVA for equality of means")
        stat, p_value, code = perform_oneway_anova(filtered_df, selected_num_col, selected_cat_col)
        
        st.metric("One-way ANOVA statistic", f"{stat:.4f}")
        st.metric("P-value", f"{p_value:.4f}")
        show_code(code)

    if equal_var:
        with st.expander("Pairwise Tukey HSD test (Equal Variances Assumed)", expanded=False):
            st.markdown("### Pairwise Tukey HSD test")
            st.info("💡 **Tukey HSD** is used here because you assume equal variances'.")
            
            tukey_result, fig, code = perform_pairwise_tukeyhsd(filtered_df, selected_num_col, selected_cat_col, confidence)
            
            st.markdown("**Statistical Results:**")
            st.text(str(tukey_result))
            
            st.markdown("**Confidence Interval Plot:**")
            st.pyplot(fig)
            show_code(code)
            
    else:
        with st.expander("Pairwise Games-Howell test (Equal Variances NOT Assumed)", expanded=False):
            st.markdown("### Pairwise Games-Howell test")
            st.info("💡 **Games-Howell** is used here because you don't assume equal variances'.")
            
            gh_result, fig, code = perform_pairwise_gameshowell(filtered_df, selected_num_col, selected_cat_col, confidence)
            st.markdown("**Statistical Results:**")
            st.dataframe(gh_result, use_container_width=True)
            st.markdown("**Confidence Interval Plot:**")
            st.pyplot(fig)
            show_code(code)