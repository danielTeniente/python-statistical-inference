import streamlit as st
from gui.components import show_code
from logic.basic_code import get_numeric_columns, get_categorical_columns
from logic.twopop_logic import (
    perform_mannwhitney, 
    plot_confidence_interval, 
    get_sample_difference_in_medians
)

def render_twopop_medians_page():
    st.title("Two Population Medians Tests")
    
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
    
    # Lightweight filter for binary categorical columns
    valid_categorical_cols = [col for col in all_categorical_cols if df[col].nunique() == 2]
    if not valid_categorical_cols:
        st.error("Error: The dataset must contain at least one categorical column with exactly two categories.")
        return
    
    # --- 2. Test Configuration ---
    st.markdown("### Test Setup")
    col1, col2 = st.columns(2)
    with col1:
        selected_num_col = st.selectbox("Select numerical variable", numeric_cols, key="tpm_num")
    with col2:
        selected_cat_col = st.selectbox("Select grouping variable (2 populations)", valid_categorical_cols, key="tpm_cat")

    groups = df[selected_cat_col].dropna().unique()
    st.caption(f"Comparing groups: **{groups[0]}** vs **{groups[1]}**")

    col3, col4 = st.columns(2)
    with col3:
        alternative = st.selectbox("Alternative hypothesis", ["two-sided", "less", "greater"], key="tpm_alt")
    with col4:
        confidence = st.slider("Confidence level", 0.80, 0.99, 0.95, 0.01, key="tpm_conf")

    # --- 3. Context ID and Cache Management ---
    # Create a unique ID to detect if any parameters have changed
    current_context_id = f"{selected_num_col}_{selected_cat_col}_{alternative}_{confidence}"
    
    # Reset page state if the context changed
    if ("twopop_medians_state" not in st.session_state or 
        st.session_state.get("twopop_medians_id") != current_context_id):
        
        st.session_state.twopop_medians_state = {}  # Isolated dictionary for results
        st.session_state.twopop_medians_id = current_context_id

    # Reference to the isolated state dictionary
    state = st.session_state.twopop_medians_state

    st.divider()

    # --- 4. Granular Execution (On-Demand) ---

    # SECTION: Mann-Whitney U Test
    with st.expander("🧪 1. Mann-Whitney U Test (Medians Comparison)", expanded=not state.get("mann_whitney")):
        if st.button("Run Mann-Whitney Test", key="btn_run_mw"):
            with st.spinner("Computing statistics..."):
                u_stat, p_value, ci, code = perform_mannwhitney(
                    df, selected_num_col, selected_cat_col, alternative, confidence
                )
                # Store everything in the state dictionary
                state["mann_whitney"] = {
                    "u_stat": u_stat,
                    "p_value": p_value,
                    "ci": ci,
                    "code": code
                }

        # Render results if they exist in state
        if "mann_whitney" in state:
            res = state["mann_whitney"]
            show_code(res["code"])
            m1, m2, m3 = st.columns(3)
            m1.metric("U-statistic", f"{res['u_stat']:.4f}")
            m2.metric(f"P-value ({alternative})", f"{res['p_value']:.4f}")
            m3.metric("Confidence Interval", f"({res['ci'][0]:.4f}, {res['ci'][1]:.4f})")

    # SECTION: Confidence Interval Plot
    with st.expander("📊 2. Visual Analysis (Confidence Interval Plot)", expanded=False):
        if st.button("Generate Plot", key="btn_gen_mw_plot"):
            with st.spinner("Generating visualization..."):
                # Ensure the CI is available (either calculate it now or fetch from existing state)
                if "mann_whitney" not in state:
                    u_stat, p_value, ci, code_mw = perform_mannwhitney(
                        df, selected_num_col, selected_cat_col, alternative, confidence
                    )
                else:
                    ci = state["mann_whitney"]["ci"]

                # Get sample medians difference
                dataset_diff, code_diff = get_sample_difference_in_medians(df, selected_num_col, selected_cat_col)
                
                # Generate plot
                fig, code_plot = plot_confidence_interval(
                    ci[0], ci[1], dataset_diff, 
                    title="Confidence Interval for the Difference in Medians", 
                    x_label="Difference in Medians", 
                    y_label="Medians Test"
                )
                
                # Save to plot state
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