import pandas as pd
import streamlit as st
from gui.components import show_code
from logic.basic_code import get_numeric_columns, get_categorical_columns
from logic.kpop_logic import perform_bartlett, perform_levene

def render_kpop_variances_page():
    st.title("K Population Variances Tests")
    
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

    # Lightweight filtering for the UI
    valid_categorical_cols = [col for col in all_categorical_cols if df[col].nunique() >= 3]
            
    if not valid_categorical_cols:
        st.error("Error: The dataset must contain at least one categorical column with 3 or more categories.")
        return

    # --- 2. Test Setup (Input Selection) ---
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

    # UI Requirement: Minimum 3 categories
    if len(selected_categories) < 3:
        st.warning("⚠️ Please select at least 3 categories to perform K population tests.")
        return

    # --- 3. Context ID and State Management ---
    # Generate unique ID based on selections to detect changes
    current_context_id = f"{selected_num_col}_{selected_cat_col}_{sorted(selected_categories)}"
    
    # Reset page state if parameters change
    if ("variances_state" not in st.session_state or 
        st.session_state.get("variances_context_id") != current_context_id):
        
        st.session_state.variances_state = {} # Clear stored results
        st.session_state.variances_context_id = current_context_id

    state = st.session_state.variances_state

    st.divider()

    # --- 4. Granular Analysis Sections ---

    # SECTION: Bartlett's Test
    with st.expander("Bartlett's Test for Equal Variances", expanded=not state.get("bartlett")):
        if st.button("Run Bartlett's Test", key="btn_bartlett"):
            with st.spinner("Calculating Bartlett statistic..."):
                # Filtering logic is deferred until execution
                filtered_df = df[df[selected_cat_col].isin(selected_categories)].copy()
                stat, p_val, code = perform_bartlett(filtered_df, selected_num_col, selected_cat_col)
                state["bartlett"] = {"stat": stat, "p": p_val, "code": code}

        if "bartlett" in state:
            res = state["bartlett"]
            show_code(res["code"])
            r1, r2 = st.columns(2)
            r1.metric("Bartlett Statistic", f"{res['stat']:.4f}")
            r2.metric("P-value", f"{res['p']:.4f}")

    # SECTION: Levene's Test
    with st.expander("Levene's Test for Equal Variances", expanded=False):
        if st.button("Run Levene's Test", key="btn_levene"):
            with st.spinner("Calculating Levene statistic..."):
                filtered_df = df[df[selected_cat_col].isin(selected_categories)].copy()
                stat_l, p_val_l, code_l = perform_levene(filtered_df, selected_num_col, selected_cat_col)
                state["levene"] = {"stat": stat_l, "p": p_val_l, "code": code_l}

        if "levene" in state:
            res = state["levene"]
            show_code(res["code"])
            rl1, rl2 = st.columns(2)
            rl1.metric("Levene Statistic", f"{res['stat']:.4f}")
            rl2.metric("P-value", f"{res['p']:.4f}")