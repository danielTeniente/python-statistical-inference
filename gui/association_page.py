import streamlit as st
from logic.association_logic import (
    perform_cramers_v_test,
    perform_pearsons_c_test,
    perform_phi_coefficient_test,
    perform_odds_ratio_test
)
from logic.independence_logic import get_contingency_table 
from gui.components import show_code

def render_association_measures_page():
    st.title("Measures of Association (Effect Size)")

    # --- Data Validation ---
    if "df" not in st.session_state or st.session_state.df is None:
        st.warning("⚠️ No data found. Please upload a file in the 'Upload Dataset' section first.")
        return
    
    df = st.session_state.df
    valid_cols = [col for col in df.columns if 2 <= df[col].dropna().nunique() <= 30]

    if len(valid_cols) < 2:
        st.error("The dataset must contain at least two categorical columns.")
        return

    # --- Variable Selection ---
    col1, col2 = st.columns(2)
    with col1:
        var1_col = st.selectbox("Select Variable 1", valid_cols, key="assoc_var1")
    
    remaining_cols = [c for c in valid_cols if c != var1_col]
    with col2:
        var2_col = st.selectbox("Select Variable 2", remaining_cols if remaining_cols else valid_cols, key="assoc_var2")

    # Unique identifier to clear results if variables change
    analysis_id = f"{var1_col}_{var2_col}"

    # Initialize the state for individual results
    if "assoc_state_id" not in st.session_state or st.session_state.assoc_state_id != analysis_id:
        st.session_state.assoc_state_id = analysis_id
        st.session_state.assoc_results = {}  # We will store each calculation separately here

    st.divider()

    # Quick check for 2x2 without building the full table first
    is_2x2 = (df[var1_col].dropna().nunique() == 2 and df[var2_col].dropna().nunique() == 2)

    # --- Individual Rendering (Lazy Execution) ---

    # 1. Contingency Table
    with st.expander("📊 Contingency Table", expanded=True):
        if st.button("Generate Contingency Table", key="btn_ct"):
            with st.spinner("Processing table..."):
                ct, code_ct = get_contingency_table(df, var1_col, var2_col)
                st.session_state.assoc_results["table"] = {"ct": ct, "code": code_ct}

        if "table" in st.session_state.assoc_results:
            res = st.session_state.assoc_results["table"]
            show_code(res["code"])
            st.dataframe(res["ct"], use_container_width=True)

    # 2. Cramér's V
    with st.expander("1. Cramér's V", expanded=True):
        if st.button("Calculate Cramér's V", key="btn_cramer"):
            with st.spinner("Calculating Cramér's V..."):
                st.session_state.assoc_results["cramers"] = perform_cramers_v_test(df, var1_col, var2_col)
        
        if "cramers" in st.session_state.assoc_results:
            val, p, code = st.session_state.assoc_results["cramers"]
            show_code(code)
            c1, c2 = st.columns(2)
            c1.metric("**Cramér's V:**", f"{val:.4f}")
            c2.metric("**p-value:**", f"{p:.4f}")

    # 3. Pearson's C
    with st.expander("2. Pearson's C", expanded=True):
        if st.button("Calculate Pearson's C", key="btn_pearson"):
            with st.spinner("Calculating Pearson's C..."):
                st.session_state.assoc_results["pearson"] = perform_pearsons_c_test(df, var1_col, var2_col)
                
        if "pearson" in st.session_state.assoc_results:
            val, p, code = st.session_state.assoc_results["pearson"]
            show_code(code)
            c1, c2 = st.columns(2)
            c1.metric("**Pearson's C:**", f"{val:.4f}")
            c2.metric("**p-value:**", f"{p:.4f}")

    # 4. Phi & Odds Ratio (Only if 2x2)
    if is_2x2:
        with st.expander("3. Phi Coefficient (φ)", expanded=True):
            if st.button("Calculate Phi", key="btn_phi"):
                with st.spinner("Calculating Phi..."):
                    st.session_state.assoc_results["phi"] = perform_phi_coefficient_test(df, var1_col, var2_col)
                    
            if "phi" in st.session_state.assoc_results:
                val, p, code = st.session_state.assoc_results["phi"]
                show_code(code)
                c1, c2 = st.columns(2)
                c1.metric("**Phi:**", f"{val:.4f}")
                c2.metric("**p-value:**", f"{p:.4f}")

        with st.expander("4. Odds Ratio (OR)", expanded=True):
            if st.button("Calculate Odds Ratio", key="btn_odds"):
                with st.spinner("Calculating Odds Ratio..."):
                    st.session_state.assoc_results["odds"] = perform_odds_ratio_test(df, var1_col, var2_col)
                    
            if "odds" in st.session_state.assoc_results:
                val, low, high, p, code = st.session_state.assoc_results["odds"]
                show_code(code)
                c1, c2, c3 = st.columns(3)
                c1.metric("**OR:**", f"{val:.4f}")
                c2.metric("**CI Lower:**", f"{low:.4f}")
                c3.metric("**CI Upper:**", f"{high:.4f}")
    else:
        st.info("ℹ️ Phi and Odds Ratio are skipped (only for 2x2 tables).")