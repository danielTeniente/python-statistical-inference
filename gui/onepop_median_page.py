import streamlit as st
from logic.basic_code import get_numeric_columns
from gui.components import show_code
from logic.onepop_mean_logic import perform_wilcoxon

def render_onepop_median_page():
    st.title("One Population Median Test")
    st.markdown("### Wilcoxon Signed-Rank Test")

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
        current_median = float(df[selected_col].median())
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
    with st.expander("Wilcoxon Test Analysis", expanded=not state.get("analysis")):
        # Calculation is only triggered when the user clicks the button
        if st.button("Run Wilcoxon Signed-Rank Test", key="btn_run_wilcoxon"):
            with st.spinner("Computing statistical results..."):
                res, ci, code = perform_wilcoxon(df, selected_col, popmedian, alternative, confidence)
                
                # Store the result in the state dictionary
                state["analysis"] = {
                    "statistic": res.statistic,
                    "pvalue": res.pvalue,
                    "ci": ci,
                    "code": code
                }

        # Render results from the state (ensures they persist during app interactions)
        if "analysis" in state:
            r = state["analysis"]
            
            res_c1, res_c2, res_c3 = st.columns(3)
            res_c1.metric("Wilcoxon Statistic", f"{r['statistic']:.4f}")
            res_c2.metric("p-value", f"{r['pvalue']:.4f}")
            res_c3.metric("Confidence Interval", f"({r['ci'][0]:.4f}, {r['ci'][1]:.4f})")
            
            show_code(r["code"])