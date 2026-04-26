import streamlit as st
import pandas as pd
from gui.components import show_code
from logic.basic_code import get_numeric_columns, get_categorical_columns
from logic.ovr_logic import perform_ftest_ovr, perform_levene_ovr
from logic.twopop_logic import plot_confidence_interval

def render_ovr_variances_page():
    st.title("One-vs-Rest: Variances Tests")
    
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
        
    # Lightweight check for valid categorical columns (> 2 categories for OvR)
    valid_categorical_cols = [col for col in all_categorical_cols if df[col].nunique() > 2]
                
    if not valid_categorical_cols:
        st.error("Error: The dataset must contain at least one categorical column with 3 or more categories.")
        return

    # --- 2. Test Setup (UI Selection) ---
    st.markdown("### Configuration")
    col1, col2, col3 = st.columns(3)
    with col1:
        selected_num_col = st.selectbox("Select numerical variable", numeric_cols, key="ovr_var_num")
    with col2:
        selected_cat_col = st.selectbox("Select grouping variable", valid_categorical_cols, key="ovr_var_cat")
    with col3:
        available_categories = df[selected_cat_col].dropna().unique().tolist()
        target_cat = st.selectbox("Select Target Population ('One')", available_categories, key="ovr_var_target")
    
    st.caption(f"Comparing: **{target_cat}** vs **The Rest**")
    
    st.markdown("#### Test Parameters")
    col4, col5 = st.columns(2)
    with col4:
        alternative = st.selectbox("Alternative hypothesis", ["two-sided", "less", "greater"], key="ovr_var_alt")
    with col5:
        confidence = st.slider("Confidence level", 0.80, 0.99, 0.95, 0.01, key="ovr_var_conf")

    # --- 3. Context ID and Cache Management ---
    # Construct an ID that uniquely identifies the current test configuration
    current_context_id = f"{selected_num_col}_{selected_cat_col}_{target_cat}_{alternative}_{confidence}"

    # Reset page state if the context changed
    if ("ovr_var_state" not in st.session_state or 
        st.session_state.get("ovr_var_id") != current_context_id):
        
        st.session_state.ovr_var_state = {}  # Clean dictionary for results
        st.session_state.ovr_var_id = current_context_id

    # Reference to the isolated state dictionary
    state = st.session_state.ovr_var_state

    st.divider()

    # --- 4. Granular Execution (On-Demand) ---

    # SECTION: F-Test
    with st.expander("🧪 1. F-Test for Equality of Variances (One-vs-Rest)", expanded=not state.get("ftest")):
        if st.button("Run F-Test", key="btn_run_ovr_ftest"):
            with st.spinner("Computing F-test statistics..."):
                f_stat, p_value, ci, code = perform_ftest_ovr(
                    df, selected_num_col, selected_cat_col, target_cat, alternative, confidence
                )
                state["ftest"] = {
                    "f_stat": f_stat,
                    "p_value": p_value,
                    "ci": ci,
                    "code": code
                }

        if "ftest" in state:
            res_f = state["ftest"]
            show_code(res_f["code"])
            f1, f2, f3 = st.columns(3)
            f1.metric("F-statistic", f"{res_f['f_stat']:.4f}")
            f2.metric(f"P-value ({alternative})", f"{res_f['p_value']:.4f}")
            f3.metric("Confidence Interval", f"({res_f['ci'][0]:.4f}, {res_f['ci'][1]:.4f})")

    # SECTION: Levene's Test
    with st.expander("🧪 2. Levene's Test for Equality of Variances (One-vs-Rest)", expanded=False):
        if st.button("Run Levene's Test", key="btn_run_ovr_levene"):
            with st.spinner("Computing Levene statistics..."):
                stat_levene, p_value_levene, ci_levene, code_levene = perform_levene_ovr(
                    df, selected_num_col, selected_cat_col, target_cat, confidence
                )
                state["levene"] = {
                    "stat": stat_levene,
                    "p_value": p_value_levene,
                    "ci": ci_levene,
                    "code": code_levene
                }

        if "levene" in state:
            res_l = state["levene"]
            show_code(res_l["code"])
            l1, l2, l3 = st.columns(3)
            l1.metric("Levene Statistic", f"{res_l['stat']:.4f}")
            l2.metric("P-value", f"{res_l['p_value']:.4f}")
            l3.metric("Confidence Interval", f"({res_l['ci'][0]:.4f}, {res_l['ci'][1]:.4f})")

    # SECTION: Visual Analysis (Plot)
    with st.expander("📊 3. Visual Analysis (Confidence Interval Plot)", expanded=False):
        if st.button("Generate CI Plot", key="btn_gen_ovr_var_plot"):
            with st.spinner("Generating visualization..."):
                # Ensure we have F-test data for the plot (calculate if not in state)
                if "ftest" not in state:
                    f_stat, p_value, ci, _ = perform_ftest_ovr(
                        df, selected_num_col, selected_cat_col, target_cat, alternative, confidence
                    )
                else:
                    f_stat = state["ftest"]["f_stat"]
                    ci = state["ftest"]["ci"]

                fig, code_plot = plot_confidence_interval(
                    ci[0], ci[1], f_stat, 
                    title=f"Variance Ratio CI ({target_cat} / Rest)", 
                    x_label="Ratio", 
                    y_label="Variance Test", 
                    H0=1
                )
                state["plot_data"] = {"fig": fig, "code": code_plot}

        if "plot_data" in state:
            res_p = state["plot_data"]
            st.pyplot(res_p["fig"])
            show_code(res_p["code"])