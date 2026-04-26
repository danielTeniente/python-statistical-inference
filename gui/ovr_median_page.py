import streamlit as st
from gui.components import show_code
from logic.basic_code import get_numeric_columns, get_categorical_columns
from logic.ovr_logic import perform_mannwhitney_ovr, get_sample_difference_in_medians_ovr
from logic.twopop_logic import plot_confidence_interval

def render_ovr_medians_page():
    st.title("One-vs-Rest: Medians Tests")
    
    # --- 1. Data Verification ---
    if "df" not in st.session_state or st.session_state.df is None:
        st.warning("⚠️ No data found. Please upload a file in the 'Upload Dataset' section first.")
        return
        
    df = st.session_state.df
    numeric_cols = get_numeric_columns(df)
    all_categorical_cols = get_categorical_columns(df)
    
    if not numeric_cols:
        st.error("The dataset does not contain any numeric columns.")
        return
    
    # Filter categorical columns for OvR (needs > 2 categories)
    valid_categorical_cols = [col for col in all_categorical_cols if df[col].nunique() > 2]
    
    if not valid_categorical_cols:
        st.error("Error: The dataset must contain at least one categorical column with 3 or more categories.")
        return
    
    # --- 2. Test Setup (UI) ---
    st.markdown("### Configuration")
    col1, col2, col3 = st.columns(3)
    with col1:
        selected_num_col = st.selectbox("Select numerical variable", numeric_cols, key="ovr_med_num")
    with col2:
        selected_cat_col = st.selectbox("Select grouping variable", valid_categorical_cols, key="ovr_med_cat")
    with col3:
        available_categories = df[selected_cat_col].dropna().unique().tolist()
        target_cat = st.selectbox("Select Target Population ('One')", available_categories, key="ovr_med_target")

    st.caption(f"Comparing: **{target_cat}** vs **The Rest**")

    st.markdown("#### Test Parameters")
    col4, col5 = st.columns(2)
    with col4:
        alternative = st.selectbox("Alternative hypothesis", ["two-sided", "less", "greater"], key="ovr_med_alt")
    with col5:
        confidence = st.slider("Confidence level", 0.80, 0.99, 0.95, 0.01, key="ovr_med_conf")

    # --- 3. Context ID and Cache Management ---
    # Create a unique ID for the current parameter configuration
    current_context_id = f"{selected_num_col}_{selected_cat_col}_{target_cat}_{alternative}_{confidence}"

    # Reset results if the configuration changes
    if ("ovr_medians_state" not in st.session_state or 
        st.session_state.get("ovr_medians_id") != current_context_id):
        
        st.session_state.ovr_medians_state = {}  # Isolated results dictionary
        st.session_state.ovr_medians_id = current_context_id

    # Reference to the isolated state
    state = st.session_state.ovr_medians_state

    st.divider()

    # --- 4. Granular Execution (On-Demand) ---

    # SECTION: Mann-Whitney U Test
    with st.expander("🧪 1. Mann-Whitney U Test (One-vs-Rest)", expanded=not state.get("mw_test")):
        if st.button("Run Median Comparison", key="btn_run_ovr_mw"):
            with st.spinner("Computing Mann-Whitney statistics..."):
                u_stat, p_value, ci, code = perform_mannwhitney_ovr(
                    df, selected_num_col, selected_cat_col, target_cat, alternative, confidence
                )
                state["mw_test"] = {
                    "u_stat": u_stat,
                    "p_value": p_value,
                    "ci": ci,
                    "code": code
                }

        if "mw_test" in state:
            res = state["mw_test"]
            show_code(res["code"])
            m1, m2, m3 = st.columns(3)
            m1.metric("U-statistic", f"{res['u_stat']:.4f}")
            m2.metric(f"P-value ({alternative})", f"{res['p_value']:.4f}")
            m3.metric("Confidence Interval", f"({res['ci'][0]:.4f}, {res['ci'][1]:.4f})")

    # SECTION: CI Plot
    with st.expander("📊 2. Visual Analysis (Confidence Interval Plot)", expanded=False):
        if st.button("Generate CI Plot", key="btn_run_ovr_mw_plot"):
            with st.spinner("Generating plot..."):
                # Fetch CI from state or calculate if not already present
                if "mw_test" not in state:
                    u_stat, p_value, ci, _ = perform_mannwhitney_ovr(
                        df, selected_num_col, selected_cat_col, target_cat, alternative, confidence
                    )
                else:
                    ci = state["mw_test"]["ci"]

                # Calculate sample difference for OVR
                dataset_diff, code_diff = get_sample_difference_in_medians_ovr(
                    df, selected_num_col, selected_cat_col, target_cat
                )
                
                # Generate visual plot
                fig, code_plot = plot_confidence_interval(
                    ci[0], ci[1], dataset_diff, 
                    title=f"CI for Difference in Medians ({target_cat} - Rest)", 
                    x_label="Difference in Medians", 
                    y_label="Medians Test"
                )
                
                state["plot_data"] = {
                    "diff": dataset_diff,
                    "code_diff": code_diff,
                    "fig": fig,
                    "code_plot": code_plot
                }

        if "plot_data" in state:
            p_res = state["plot_data"]
            st.metric("Sample Difference in Medians", f"{p_res['diff']:.4f}")
            
            st.markdown("**Sample Difference Logic:**")
            show_code(p_res["code_diff"])
            
            st.pyplot(p_res["fig"])
            
            st.markdown("**Plotting Logic:**")
            show_code(p_res["code_plot"])