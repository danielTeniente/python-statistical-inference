import streamlit as st
from logic.basic_code import get_numeric_columns
from gui.components import show_code
# Assuming you saved the three new functions in onepop_mean_logic.py
from logic.onepop_mean_logic import perform_wilcoxon, get_bootstrap_ci, get_exact_median_ci

def render_onepop_median_page():
    st.title("One Population Median Test")
    st.markdown("### Wilcoxon Signed-Rank Test & Confidence Intervals")

    # --- 1. Data Validation ---
    if "df" not in st.session_state or st.session_state.df is None:
        st.warning("⚠️ No data found. Please upload a file in the 'Upload Dataset' section first.")
        return
    
    df = st.session_state.df
    numeric_cols = get_numeric_columns(df)
    
    if not numeric_cols:
        st.error("The dataset does not contain any numeric columns.")
        return
    
    # --- 2. Test Configuration ---
    st.markdown("### Test Parameters")
    col1, col2 = st.columns(2)
    
    with col1:
        selected_col = st.selectbox("Select variable", numeric_cols, key="one_med_var")
        alternative = st.selectbox(
            "Select alternative hypothesis", 
            ["two-sided", "greater", "less"],
            key="one_med_alt"
        )
    
    with col2:
        # Lightweight calculation for a default value
        current_median = float(df[selected_col].dropna().median())
        popmedian = st.number_input(
            label="Hypothesized Population Median (H₀)", 
            value=current_median, 
            step=0.1,
            help="Value for the Null Hypothesis.",
            key="one_med_h0"
        )
        confidence = st.slider("Confidence level", 0.80, 0.99, 0.95, 0.01, key="one_med_conf")

    # --- 3. Context ID and Cache Management ---
    # Create a unique ID based on the inputs to detect changes
    current_context_id = f"{selected_col}_{popmedian}_{alternative}_{confidence}"
    
    # Reset results if the input parameters change
    if ("onepop_median_state" not in st.session_state or 
        st.session_state.get("onepop_median_id") != current_context_id):
        
        st.session_state.onepop_median_state = {}  # Isolated state dictionary
        st.session_state.onepop_median_id = current_context_id

    # Shortcut to the isolated state
    state = st.session_state.onepop_median_state

    st.divider()

    # --- 4. Granular Execution (On-Demand) ---
    
    # --- MODULE 1: Wilcoxon Test ---
    with st.expander("1. Wilcoxon Signed-Rank Test", expanded=not state.get("wilcoxon")):
        st.markdown("Calculates the test statistic and p-value for the specified median hypothesis.")
        
        if st.button("Run Wilcoxon Test", key="btn_run_wilcoxon"):
            with st.spinner("Computing statistical results..."):
                res, code = perform_wilcoxon(df, selected_col, popmedian, alternative)
                
                state["wilcoxon"] = {
                    "statistic": res.statistic,
                    "pvalue": res.pvalue,
                    "code": code
                }

        if "wilcoxon" in state:
            r = state["wilcoxon"]
            res_c1, res_c2 = st.columns(2)
            res_c1.metric("Wilcoxon Statistic", f"{r['statistic']:.4f}")
            res_c2.metric("p-value", f"{r['pvalue']:.4f}")
            show_code(r["code"])

    # --- MODULE 2: Bootstrap Confidence Interval ---
    with st.expander("2. Bootstrap Confidence Interval", expanded=not state.get("bootstrap")):
        st.markdown("Estimates the confidence interval through resampling. *Automatically subsamples if records exceed 5,000 for server stability.*")
        
        if st.button("Run Bootstrap CI", key="btn_run_bootstrap"):
            with st.spinner("Running bootstrap resamples..."):
                ci, code = get_bootstrap_ci(df, selected_col, confidence, threshold=5000)
                
                state["bootstrap"] = {
                    "low": ci.low,
                    "high": ci.high,
                    "code": code
                }

        if "bootstrap" in state:
            r = state["bootstrap"]
            st.metric("Bootstrap Confidence Interval", f"({r['low']:.4f}, {r['high']:.4f})")
            show_code(r["code"])

    # --- MODULE 3: Exact Median Confidence Interval ---
    with st.expander("3. Exact Median CI (Order Statistics)", expanded=not state.get("exact_ci")):
        st.markdown("Calculates the mathematical confidence interval using Order Statistics. Highly recommended for datasets over 5,000 records.")
        
        if st.button("Run Exact Median CI", key="btn_run_exact_ci"):
            with st.spinner("Calculating exact order statistics..."):
                ci, code = get_exact_median_ci(df, selected_col, confidence)
                
                state["exact_ci"] = {
                    "low": ci.low,
                    "high": ci.high,
                    "code": code
                }

        if "exact_ci" in state:
            r = state["exact_ci"]
            st.metric("Exact Confidence Interval", f"({r['low']:.4f}, {r['high']:.4f})")
            show_code(r["code"])