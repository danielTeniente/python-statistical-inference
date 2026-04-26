import streamlit as st
import pandas as pd
from gui.components import show_code
from logic.basic_code import get_numeric_columns, get_categorical_columns
from logic.twopop_logic import perform_ftest, perform_levene, plot_confidence_interval

def render_twopop_variances_page():
    st.title("Two Population Variances Tests")
    
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
        
    # Lightweight filter for binary categories
    valid_categorical_cols = [col for col in all_categorical_cols if df[col].nunique() == 2]
                
    if not valid_categorical_cols:
        st.error("Error: The dataset must contain at least one categorical column with exactly two categories.")
        return

    # --- 2. Test Configuration ---
    st.markdown("### Test Setup")
    col1, col2 = st.columns(2)
    with col1:
        selected_num_col = st.selectbox("Select numerical variable", numeric_cols, key="tpv_num")
    with col2:
        selected_cat_col = st.selectbox("Select grouping variable (2 populations)", valid_categorical_cols, key="tpv_cat")
    
    groups = df[selected_cat_col].dropna().unique()
    st.caption(f"Comparing variances of: **{groups[0]}** vs **{groups[1]}**")
    
    col3, col4 = st.columns(2)
    with col3:
        alternative = st.selectbox("Alternative hypothesis", ["two-sided", "less", "greater"], key="tpv_alt")
    with col4:
        confidence = st.slider("Confidence level", 0.80, 0.99, 0.95, 0.01, key="tpv_conf")

    # --- 3. Context ID and Cache Management ---
    # Unique ID based on all input parameters
    current_context_id = f"{selected_num_col}_{selected_cat_col}_{alternative}_{confidence}"
    
    # Reset page state if the context changed
    if ("twopop_var_state" not in st.session_state or 
        st.session_state.get("twopop_var_id") != current_context_id):
        
        st.session_state.twopop_var_state = {}  # Isolated results dictionary
        st.session_state.twopop_var_id = current_context_id

    # Reference to the isolated state
    state = st.session_state.twopop_var_state

    st.divider()

    # --- 4. Granular Execution (On-Demand) ---

    # SECTION: F-Test
    with st.expander("🧪 1. F-Test for Equality of Variances", expanded=not state.get("ftest")):
        if st.button("Run F-Test", key="btn_run_ftest"):
            with st.spinner("Computing F-test statistics..."):
                f_stat, p_val, ci, code = perform_ftest(
                    df, selected_num_col, selected_cat_col, alternative, confidence
                )
                state["ftest"] = {
                    "f_stat": f_stat,
                    "p_value": p_val,
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
    with st.expander("🧪 2. Levene's Test for Equality of Variances", expanded=False):
        if st.button("Run Levene's Test", key="btn_run_levene"):
            with st.spinner("Computing Levene statistics..."):
                stat, p_val_l, ci_l, code_l = perform_levene(
                    df, selected_num_col, selected_cat_col, confidence
                )
                state["levene"] = {
                    "stat": stat,
                    "p_value": p_val_l,
                    "ci": ci_l,
                    "code": code_l
                }

        if "levene" in state:
            res_l = state["levene"]
            show_code(res_l["code"])
            l1, l2, l3 = st.columns(3)
            l1.metric("Levene Statistic", f"{res_l['stat']:.4f}")
            l2.metric("P-value", f"{res_l['p_value']:.4f}")
            l3.metric("Confidence Interval", f"({res_l['ci'][0]:.4f}, {res_l['ci'][1]:.4f})")

    # SECTION: Plot
    with st.expander("📊 3. Visual Analysis (Confidence Interval Plot)", expanded=False):
        if st.button("Generate Variance Ratio Plot", key="btn_gen_plot"):
            with st.spinner("Generating visualization..."):
                # Ensure we have F-test results for the plot (calculate if missing)
                if "ftest" not in state:
                    f_stat, p_val, ci, _ = perform_ftest(
                        df, selected_num_col, selected_cat_col, alternative, confidence
                    )
                else:
                    f_stat = state["ftest"]["f_stat"]
                    ci = state["ftest"]["ci"]

                fig, code_plot = plot_confidence_interval(
                    ci[0], ci[1], f_stat, 
                    title="Confidence Interval for the Variance Ratio", 
                    x_label="Ratio", 
                    y_label="Variance Test", 
                    H0=1
                )
                state["plot"] = {"fig": fig, "code": code_plot}

        if "plot" in state:
            res_p = state["plot"]
            st.pyplot(res_p["fig"])
            show_code(res_p["code"])