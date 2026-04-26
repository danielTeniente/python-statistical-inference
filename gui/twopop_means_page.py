import streamlit as st
from gui.components import show_code
from logic.basic_code import get_numeric_columns, get_categorical_columns
from logic.twopop_logic import (
    perform_ttest, 
    plot_confidence_interval, 
    get_sample_difference_in_means
)

def render_twopop_means_page():
    st.title("Two Population Means Tests")
    
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

    # Lightweight filtering for binary categories
    valid_categorical_cols = [col for col in all_categorical_cols if df[col].nunique() == 2]
    if not valid_categorical_cols:
        st.error("Error: The dataset must contain at least one categorical column with exactly two categories.")
        return
    
    # --- 2. Test Configuration ---
    st.markdown("### Test Setup")
    col1, col2 = st.columns(2)
    with col1:
        selected_num_col = st.selectbox("Select numerical variable", numeric_cols, key="tp_num")
    with col2:
        selected_cat_col = st.selectbox("Select grouping variable (2 populations)", valid_categorical_cols, key="tp_cat")

    groups = df[selected_cat_col].dropna().unique()
    st.caption(f"Comparing groups: **{groups[0]}** vs **{groups[1]}**")

    col3, col4 = st.columns(2)
    with col3:
        alternative = st.selectbox("Alternative hypothesis", ["two-sided", "less", "greater"], key="tp_alt")
    with col4:
        confidence = st.slider("Confidence level", 0.80, 0.99, 0.95, 0.01, key="tp_conf")
    
    equal_var = st.checkbox("Assume equal variances", value=True, key="tp_eq_var")

    # --- 3. Context ID and Cache Management ---
    # Create a unique ID for the current parameter state
    current_context_id = f"{selected_num_col}_{selected_cat_col}_{alternative}_{confidence}_{equal_var}"
    
    # Reset results if context changes
    if ("twopop_means_state" not in st.session_state or 
        st.session_state.get("twopop_means_id") != current_context_id):
        
        st.session_state.twopop_means_state = {}  # Isolated results dictionary
        st.session_state.twopop_means_id = current_context_id

    # Reference to the isolated state
    state = st.session_state.twopop_means_state

    st.divider()

    # --- 4. Granular Execution (On-Demand) ---

    # SECTION: T-Test
    with st.expander("🧪 1. T-Test for Means Comparison", expanded=not state.get("ttest")):
        if st.button("Run T-Test", key="btn_run_ttest"):
            with st.spinner("Computing T-test statistics..."):
                t_stat, p_value, ci, code = perform_ttest(
                    df, selected_num_col, selected_cat_col, alternative, confidence, equal_var
                )
                state["ttest"] = {
                    "stat": t_stat, 
                    "p": p_value, 
                    "ci": ci, 
                    "code": code
                }

        if "ttest" in state:
            res = state["ttest"]
            show_code(res["code"])
            m1, m2, m3 = st.columns(3)
            m1.metric("T-statistic", f"{res['stat']:.4f}")
            m2.metric(f"P-value ({alternative})", f"{res['p']:.4f}")
            m3.metric("Confidence Interval", f"({res['ci'][0]:.4f}, {res['ci'][1]:.4f})")

    # SECTION: Confidence Interval Plot
    with st.expander("📊 2. Confidence Interval Plot", expanded=False):
        if st.button("Generate Plot", key="btn_gen_plot"):
            with st.spinner("Generating visualization..."):
                # Check if we need to calculate CI first or if it's already in state
                if "ttest" not in state:
                    t_stat, p_value, ci, code_tt = perform_ttest(
                        df, selected_num_col, selected_cat_col, alternative, confidence, equal_var
                    )
                    # We don't necessarily save ttest here to keep the "Run" logic isolated
                else:
                    ci = state["ttest"]["ci"]

                # Calculate sample difference
                sample_diff, code_diff = get_sample_difference_in_means(df, selected_num_col, selected_cat_col)
                
                # Create plot
                fig, code_plot = plot_confidence_interval(
                    ci[0], ci[1], sample_diff, 
                    title="Confidence Interval for the Difference in Means", 
                    x_label="Difference in Means", 
                    y_label="Means Test"
                )
                
                state["plot"] = {
                    "fig": fig, 
                    "diff": sample_diff, 
                    "code_diff": code_diff, 
                    "code_plot": code_plot
                }

        if "plot" in state:
            res_p = state["plot"]
            st.metric("Sample Difference in Means", f"{res_p['diff']:.4f}")
            
            st.markdown("**Sample Difference Logic:**")
            show_code(res_p["code_diff"])
            
            st.pyplot(res_p["fig"])
            
            st.markdown("**Plotting Logic:**")
            show_code(res_p["code_plot"])