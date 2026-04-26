import pandas as pd
import streamlit as st
from gui.components import show_code
from logic.basic_code import get_numeric_columns, get_categorical_columns
from logic.kpop_logic import (
    perform_oneway_anova, 
    perform_pairwise_tukeyhsd, 
    perform_pairwise_gameshowell
)

def render_kpop_means_page():
    st.title("k-Sample Mean Tests")
    
    # --- 1. Data Validation ---
    if "df" not in st.session_state or st.session_state.df is None:
        st.warning("⚠️ No data found. Please upload a file in the 'Upload Dataset' section first.")
        return
        
    df = st.session_state.df
    numeric_cols = get_numeric_columns(df)
    all_categorical_cols = get_categorical_columns(df)
    
    if not numeric_cols:
        st.error("The dataset does not contain any numeric columns.")
        return
    
    # Lightweight logic to find valid categories
    valid_categorical_cols = [col for col in all_categorical_cols if df[col].nunique() >= 3]
            
    if not valid_categorical_cols:
        st.error("Error: The dataset must contain at least one categorical column with 3 or more categories.")
        return
    
    # --- 2. Test Setup (Inputs) ---
    st.markdown("### Test Setup")
    col1, col2 = st.columns(2)
    with col1:
        selected_num_col = st.selectbox("Select numerical variable", numeric_cols, key="kpop_num")
    with col2:
        selected_cat_col = st.selectbox("Select grouping variable (≥ 3 categories)", valid_categorical_cols, key="kpop_cat")

    available_categories = df[selected_cat_col].dropna().unique().tolist()

    selected_categories = st.multiselect(
        "Select the specific categories to compare:", 
        available_categories, 
        default=available_categories,
        key="kpop_categories"
    )

    # UI Rule: Minimum 3 categories
    if len(selected_categories) < 3:
        st.warning("⚠️ Please select at least 3 categories to perform K population tests.")
        return 

    confidence = st.slider("Confidence level", 0.80, 0.99, 0.95, 0.01)
    equal_var = st.checkbox("Assume equal variances", value=True, key="equal_var")

    # --- 3. Context ID and Cache Invalidation ---
    # Create a unique ID based on all parameters that affect the results
    # We sort categories to ensure the same selection always produces the same ID
    context_id = f"{selected_num_col}_{selected_cat_col}_{sorted(selected_categories)}_{confidence}_{equal_var}"
    
    if ("kpop_state" not in st.session_state or 
        st.session_state.get("kpop_context_id") != context_id):
        
        # Clear specific page state when inputs change
        st.session_state.kpop_state = {}
        st.session_state.kpop_context_id = context_id

    state = st.session_state.kpop_state

    st.divider()

    # --- 4. Granular Analysis Sections ---

    # SECTION: One-way ANOVA
    with st.expander("One-way ANOVA for Equality of Means", expanded=not state.get("anova")):
        if st.button("Calculate ANOVA", key="btn_anova"):
            with st.spinner("Computing ANOVA..."):
                # Lazy filtering only when calculation is requested
                filtered_df = df[df[selected_cat_col].isin(selected_categories)].copy()
                stat, p_value, code = perform_oneway_anova(filtered_df, selected_num_col, selected_cat_col)
                state["anova"] = {"stat": stat, "p": p_value, "code": code}
        
        if "anova" in state:
            res = state["anova"]
            c1, c2 = st.columns(2)
            c1.metric("F-Statistic", f"{res['stat']:.4f}")
            c2.metric("P-value", f"{res['p']:.4f}")
            show_code(res["code"])

    # SECTION: Post-hoc Tests (Tukey or Games-Howell)
    section_title = "Pairwise Tukey HSD (Equal Variances)" if equal_var else "Pairwise Games-Howell (Unequal Variances)"
    
    with st.expander(section_title, expanded=False):
        if equal_var:
            st.info("💡 **Tukey HSD** is selected based on the equal variances assumption.")
        else:
            st.info("💡 **Games-Howell** is selected for populations with unequal variances.")

        if st.button("Run Post-hoc Comparison", key="btn_posthoc"):
            with st.spinner("Performing pairwise comparisons..."):
                filtered_df = df[df[selected_cat_col].isin(selected_categories)].copy()
                
                if equal_var:
                    res_stat, fig, code = perform_pairwise_tukeyhsd(filtered_df, selected_num_col, selected_cat_col, confidence)
                else:
                    res_stat, fig, code = perform_pairwise_gameshowell(filtered_df, selected_num_col, selected_cat_col, confidence)
                
                state["post_hoc"] = {"data": res_stat, "fig": fig, "code": code}

        if "post_hoc" in state:
            res = state["post_hoc"]
            st.markdown("**Statistical Results:**")
            
            # Render dataframe or text depending on the test type
            if isinstance(res["data"], pd.DataFrame):
                st.dataframe(res["data"], use_container_width=True)
            else:
                st.text(str(res["data"]))
            
            st.markdown("**Confidence Interval Plot:**")
            st.pyplot(res["fig"])
            show_code(res["code"])