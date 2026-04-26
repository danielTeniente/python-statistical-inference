import streamlit as st
import pandas as pd
from logic.proportions_logic import (
    perform_two_proportion_ztest, 
    get_two_proportion_confint
)
from logic.independence_logic import perform_fisher_exact_test, get_contingency_table
from gui.components import show_code

def render_twoprop_test_page():
    st.title("Two Proportions Test")

    # --- 1. Data Validation ---
    if "df" not in st.session_state or st.session_state.df is None:
        st.warning("⚠️ No data found. Please upload a file in the 'Upload Dataset' section first.")
        return
    
    df = st.session_state.df

    # Quick check for binary columns
    valid_cols = [col for col in df.columns if df[col].dropna().nunique() == 2]

    if len(valid_cols) < 2:
        st.error("The dataset must contain at least two binary columns (e.g., Groups and Outcomes).")
        return

    # --- 2. Test Configuration ---
    st.markdown("### Configuration")
    col1, col2 = st.columns(2)
    
    with col1:
        group_col = st.selectbox("Select Grouping Variable", valid_cols, index=0, 
                                 help="The variable that splits your data into two groups.",
                                 key="twoprop_group")
        
        # Determine success term based on outcome variable
        remaining_cols = [c for c in valid_cols if c != group_col]
        outcome_col = st.selectbox("Select Outcome Variable", 
                                   remaining_cols if remaining_cols else valid_cols, 
                                   index=0, key="twoprop_outcome")

    with col2:
        unique_vals = df[outcome_col].dropna().unique()
        success_term = st.selectbox("Value for 'Success'", options=unique_vals, key="twoprop_success")
        confidence = st.slider("Confidence level", 0.80, 0.99, 0.95, 0.01, key="twoprop_conf")
    
    alternative = st.selectbox("Alternative hypothesis", ["two-sided", "larger", "smaller"], key="twoprop_alt")

    # --- 3. Context ID and Cache Management ---
    # Create unique ID based on all parameters
    current_id = f"{group_col}_{outcome_col}_{success_term}_{confidence}_{alternative}"
    
    # Invalidate results if parameters changed
    if ("twoprop_state" not in st.session_state or 
        st.session_state.get("twoprop_id") != current_id):
        
        st.session_state.twoprop_state = {}  # Isolated results dictionary
        st.session_state.twoprop_id = current_id

    state = st.session_state.twoprop_state

    st.divider()

    # --- 4. Granular Analysis Sections ---

    # SECTION: Contingency Table
    with st.expander("📊 1. Contingency Table", expanded=not state.get("table")):
        if st.button("Generate Contingency Table", key="btn_twoprop_table"):
            with st.spinner("Calculating table..."):
                table, code_t = get_contingency_table(df, group_col, outcome_col)
                state["table"] = {"data": table, "code": code_t}
        
        if "table" in state:
            res_t = state["table"]
            show_code(res_t["code"])
            st.dataframe(res_t["data"])

    # SECTION: Fisher's Exact Test
    with st.expander("🧪 2. Fisher's Exact Test", expanded=False):
        st.info("Recommended for small sample sizes.")
        if st.button("Run Fisher's Exact Test", key="btn_twoprop_fisher"):
            with st.spinner("Computing exact p-value..."):
                stat, p_val, code_f = perform_fisher_exact_test(
                    df=df, var1_col=group_col, var2_col=outcome_col, alternative=alternative
                )
                state["fisher"] = {"stat": stat, "p": p_val, "code": code_f}

        if "fisher" in state:
            res_f = state["fisher"]
            show_code(res_f["code"])
            f_c1, f_c2 = st.columns(2)
            if res_f["stat"] is not None:
                f_c1.metric("Odds Ratio", f"{res_f['stat']:.4f}")
            f_c2.metric("p-value", f"{res_f['p']:.4f}")

    # SECTION: Z-Test
    with st.expander("📏 3. Z-Test for Two Proportions", expanded=False):
        if st.button("Run Z-Test", key="btn_twoprop_z"):
            with st.spinner("Computing Z-statistic..."):
                stat_z, p_z, code_z = perform_two_proportion_ztest(
                    df=df, group_col=group_col, outcome_col=outcome_col, 
                    success_term=success_term, alternative=alternative
                )
                state["ztest"] = {"stat": stat_z, "p": p_z, "code": code_z}

        if "ztest" in state:
            res_z = state["ztest"]
            show_code(res_z["code"])
            z_c1, z_c2 = st.columns(2)
            z_c1.metric("Z-Statistic", f"{res_z['stat']:.4f}")
            z_c2.metric("p-value", f"{res_z['p']:.4f}")

    # SECTION: Confidence Intervals
    with st.expander("🔒 4. Confidence Intervals for Difference", expanded=False):
        col_ci1, col_ci2 = st.columns(2)
        
        # Newcombe
        with col_ci1:
            st.markdown("**Newcombe Interval**")
            if st.button("Calculate Newcombe", key="btn_twoprop_newcombe"):
                with st.spinner("Calculating..."):
                    (l, u), code_n = get_two_proportion_confint(
                        df=df, group_col=group_col, outcome_col=outcome_col,
                        success_term=success_term, confidence=confidence, method='newcomb'
                    )
                    state["newcombe"] = {"ci": (l, u), "code": code_n}
            
            if "newcombe" in state:
                res_n = state["newcombe"]
                st.metric(f"CI ({confidence*100:.0f}%)", f"({res_n['ci'][0]:.4f}, {res_n['ci'][1]:.4f})")
                show_code(res_n["code"])

        # Wald
        with col_ci2:
            st.markdown("**Wald Interval**")
            if st.button("Calculate Wald", key="btn_twoprop_wald"):
                with st.spinner("Calculating..."):
                    (l, u), code_w = get_two_proportion_confint(
                        df=df, group_col=group_col, outcome_col=outcome_col,
                        success_term=success_term, confidence=confidence, method='wald'
                    )
                    state["wald"] = {"ci": (l, u), "code": code_w}
            
            if "wald" in state:
                res_w = state["wald"]
                st.metric(f"CI ({confidence*100:.0f}%)", f"({res_w['ci'][0]:.4f}, {res_w['ci'][1]:.4f})")
                show_code(res_w["code"])