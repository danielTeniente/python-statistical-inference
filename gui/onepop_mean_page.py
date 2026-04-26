import streamlit as st
from logic.basic_code import get_numeric_columns
from gui.components import show_code
from logic.onepop_mean_logic import perform_ttest

def render_onepop_mean_page():
    st.title("One Population Mean Test")
    
    # --- 1. Data Validations ---
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
        selected_col = st.selectbox("Select variable", numeric_cols, key="one_pop_var")
        alternative = st.selectbox(
            "Select alternative hypothesis", 
            ["two-sided", "greater", "less"],
            key="one_pop_alt"
        )
    
    with col2:
        # Lightweight logic to provide a sensible default
        sample_mean = float(df[selected_col].mean())
        popmean = st.number_input(
            label="Hypothesized Population Mean (H₀)", 
            value=sample_mean, 
            step=0.1,
            help="Value for the Null Hypothesis.",
            key="one_pop_h0"
        )
        confidence = st.slider("Confidence level", 0.80, 0.99, 0.95, 0.01, key="one_pop_conf")

    # --- 3. Context ID and Cache Management ---
    # Create a unique ID to detect input changes
    current_context_id = f"{selected_col}_{popmean}_{alternative}_{confidence}"
    
    # If the setup changes, reset the results for this specific page
    if ("onepop_mean_state" not in st.session_state or 
        st.session_state.get("onepop_mean_id") != current_context_id):
        
        st.session_state.onepop_mean_state = {} # Isolated dictionary for this page
        st.session_state.onepop_mean_id = current_context_id

    # Reference to the isolated state
    state = st.session_state.onepop_mean_state

    st.divider()

    # --- 4. Granular Execution (On-Demand) ---
    with st.expander("T-Test Analysis", expanded=not state.get("analysis")):
        # Only perform heavy logic when this specific button is clicked
        if st.button("Run One-Sample T-Test", key="btn_run_ttest"):
            with st.spinner("Computing statistical results..."):
                res, ci, code = perform_ttest(df, selected_col, popmean, alternative, confidence)
                
                # Store results in state to survive Streamlit reruns
                state["analysis"] = {
                    "statistic": res.statistic,
                    "pvalue": res.pvalue,
                    "ci": ci,
                    "code": code
                }

        # Render results if they exist in the page state
        if "analysis" in state:
            r = state["analysis"]
            
            res_c1, res_c2, res_c3 = st.columns(3)
            res_c1.metric("t-statistic", f"{r['statistic']:.4f}")
            res_c2.metric("p-value", f"{r['pvalue']:.4f}")
            res_c3.metric("Confidence Interval", f"({r['ci'][0]:.4f}, {r['ci'][1]:.4f})")
            
            show_code(r["code"])