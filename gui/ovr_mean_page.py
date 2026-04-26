import streamlit as st
from gui.components import show_code
from logic.basic_code import get_numeric_columns, get_categorical_columns
from logic.ovr_logic import perform_ttest_ovr, get_sample_difference_in_means_ovr
from logic.twopop_logic import plot_confidence_interval

def render_ovr_means_page():
    st.title("One-vs-Rest: Means Tests")
    
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

    # Lightweight check for valid categorical columns
    valid_categorical_cols = [col for col in all_categorical_cols if df[col].nunique() > 2]
    
    if not valid_categorical_cols:
        st.error("Error: The dataset must contain at least one categorical column with 3 or more categories.")
        return
    
    # --- 2. Test Setup (UI Selection) ---
    st.markdown("### Configuration")
    col1, col2, col3 = st.columns(3)
    with col1:
        selected_num_col = st.selectbox("Select numerical variable", numeric_cols, key="ovr_mean_num")
    with col2:
        selected_cat_col = st.selectbox("Select grouping variable", valid_categorical_cols, key="ovr_mean_cat")
    with col3:
        available_categories = df[selected_cat_col].dropna().unique().tolist()
        target_cat = st.selectbox("Select Target Population ('One')", available_categories, key="ovr_mean_target")

    st.caption(f"Comparing: **{target_cat}** vs **The Rest**")

    st.markdown("#### Test Parameters")
    col4, col5 = st.columns(2)
    with col4:
        alternative = st.selectbox("Alternative hypothesis", ["two-sided", "less", "greater"], key="ovr_mean_alt")
    with col5:
        confidence = st.slider("Confidence level", 0.80, 0.99, 0.95, 0.01, key="ovr_mean_conf")
        
    equal_var = st.checkbox("Assume equal variances", value=True, key="ovr_equal_var")

    # --- 3. Context ID and Cache Management ---
    # Construct an ID that uniquely identifies the current test configuration
    current_context_id = f"{selected_num_col}_{selected_cat_col}_{target_cat}_{alternative}_{confidence}_{equal_var}"

    # If parameters change, we clear the saved results for this specific page
    if ("ovr_means_state" not in st.session_state or 
        st.session_state.get("ovr_means_id") != current_context_id):
        
        st.session_state.ovr_means_state = {}  # Clean dictionary for results
        st.session_state.ovr_means_id = current_context_id

    # Reference to the isolated state dictionary
    state = st.session_state.ovr_means_state

    st.divider()

    # --- 4. Granular Execution (On-Demand) ---

    # SECTION: Statistical T-test
    with st.expander("🧪 1. T-test to Compare Means (One-vs-Rest)", expanded=not state.get("ttest")):
        if st.button("Run Means Comparison", key="btn_ovr_ttest"):
            with st.spinner("Computing statistics..."):
                t_stat, p_value, ci, code = perform_ttest_ovr(
                    df, selected_num_col, selected_cat_col, target_cat, alternative, confidence, equal_var
                )
                state["ttest"] = {
                    "t_stat": t_stat,
                    "p_value": p_value,
                    "ci": ci,
                    "code": code
                }

        if "ttest" in state:
            res = state["ttest"]
            show_code(res["code"])
            m1, m2, m3 = st.columns(3)
            m1.metric("T-statistic", f"{res['t_stat']:.4f}")
            m2.metric(f"P-value ({alternative})", f"{res['p_value']:.4f}")
            m3.metric("Confidence Interval", f"({res['ci'][0]:.4f}, {res['ci'][1]:.4f})")

    # SECTION: Confidence Interval Plot
    with st.expander("📊 2. Visual Analysis (Confidence Interval Plot)", expanded=False):
        if st.button("Generate CI Plot", key="btn_ovr_plot"):
            with st.spinner("Generating visualization..."):
                # Ensure the CI is available from state or calculate if missing
                if "ttest" not in state:
                    t_stat, p_value, ci, _ = perform_ttest_ovr(
                        df, selected_num_col, selected_cat_col, target_cat, alternative, confidence, equal_var
                    )
                else:
                    ci = state["ttest"]["ci"]

                # Calculate sample difference for OVR
                sample_diff, code_diff = get_sample_difference_in_means_ovr(
                    df, selected_num_col, selected_cat_col, target_cat
                )
                
                # Generate plot
                fig, code_plot = plot_confidence_interval(
                    ci[0], ci[1], sample_diff, 
                    title=f"CI for Difference in Means ({target_cat} - Rest)", 
                    x_label="Difference in Means", 
                    y_label="Means Test"
                )
                
                state["plot_data"] = {
                    "diff": sample_diff,
                    "code_diff": code_diff,
                    "fig": fig,
                    "code_plot": code_plot
                }

        if "plot_data" in state:
            p_res = state["plot_data"]
            st.metric("Sample Difference in Means", f"{p_res['diff']:.4f}")
            
            st.markdown("**Sample Difference Logic:**")
            show_code(p_res["code_diff"])
            
            st.pyplot(p_res["fig"])
            
            st.markdown("**Plotting Logic:**")
            show_code(p_res["code_plot"])