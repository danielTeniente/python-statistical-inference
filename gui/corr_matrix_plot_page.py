import streamlit as st
from gui.components import show_code
from logic.basic_code import get_numeric_columns
from logic.correlation_logic import get_correlation_heatmap

def render_correlation_heatmap_page():
    st.title("Correlation Matrix Heatmap")
    
    # --- 1. Data Validation ---
    if "df" not in st.session_state or st.session_state.df is None:
        st.warning("⚠️ No data found. Please upload a file in the 'Upload Dataset' section first.")
        return
        
    df = st.session_state.df
    numeric_cols = get_numeric_columns(df)
    
    if len(numeric_cols) < 2:
        st.error("Error: The dataset must contain at least two numerical columns to generate a correlation heatmap.")
        return

    # --- 2. Heatmap Configuration (UI Inputs) ---
    st.markdown("### Heatmap Configuration")
    
    # Default selection logic
    default_selected_cols = numeric_cols[:5] if len(numeric_cols) >= 5 else numeric_cols
    
    selected_columns = st.multiselect(
        "Select numerical variables to include (minimum 2 required):",
        options=numeric_cols,
        default=default_selected_cols,
        key="heatmap_cols"
    )
    
    if len(selected_columns) < 2:
        st.warning("⚠️ Please select at least 2 variables to generate the correlation matrix.")
        return
        
    col1, col2 = st.columns(2)
    with col1:
        selected_method = st.selectbox(
            "Select Correlation Method", 
            ["pearson", "spearman", "kendall"], 
            key="heatmap_method",
            help="Pearson: linear. Spearman: monotonic. Kendall: ordinal."
        )
    with col2:
        selected_shape = st.selectbox(
            "Select Heatmap Shape", 
            ["triangle", "square"], 
            key="heatmap_shape",
            help="Triangle hides the redundant upper half and main diagonal."
        )

    # --- 3. Context ID and Cache Management ---
    # We sort the column list so that the ID remains consistent regardless of selection order
    current_context_id = f"{sorted(selected_columns)}_{selected_method}_{selected_shape}"

    # If parameters change, we clear the results dictionary for this page
    if ("heatmap_page_state" not in st.session_state or 
        st.session_state.get("heatmap_page_id") != current_context_id):
        
        st.session_state.heatmap_page_state = {}  # Isolated results dictionary
        st.session_state.heatmap_page_id = current_context_id

    # Shortcut to the isolated state
    state = st.session_state.heatmap_page_state

    st.divider()

    # --- 4. Granular Execution (On-Demand) ---
    with st.expander("📊 Correlation Matrix Visualization", expanded=not state.get("viz")):
        # Only trigger the heavy plotting logic when the button is pressed
        if st.button("Generate Heatmap", key="btn_run_heatmap"):
            with st.spinner("Generating correlation heatmap..."):
                fig, code_heatmap = get_correlation_heatmap(
                    df, 
                    columns=selected_columns, 
                    method=selected_method, 
                    shape=selected_shape
                )
                
                # Store results in the page state dictionary
                state["viz"] = {
                    "fig": fig,
                    "code": code_heatmap
                }

        # Render from state if results exist (ensures persistence during app reruns)
        if "viz" in state:
            res = state["viz"]
            st.pyplot(res["fig"])
            
            st.divider()
            st.markdown("#### Reproducible Code")
            st.info("Use this code to reproduce the heatmap in your own environment.")
            show_code(res["code"])