import streamlit as st
from gui.components import show_code
from logic.basic_code import get_numeric_columns
from logic.correlation_logic import get_correlation_heatmap

def render_correlation_heatmap_page():
    st.title("Correlation Matrix Heatmap")
    
    if "df" not in st.session_state or st.session_state.df is None:
        st.warning("⚠️ No data found. Please upload a file in the 'Upload Dataset' section first.")
        return
        
    df = st.session_state.df
    numeric_cols = get_numeric_columns(df)
    
    # Validation: Need at least 2 numeric columns for a matrix
    if len(numeric_cols) < 2:
        st.error("Error: The dataset must contain at least two numerical columns to generate a correlation heatmap.")
        st.info("Please review your dataset.")
        return

    st.markdown("### Heatmap Configuration")
    
    # Determine default columns (first 5 numeric columns max)
    default_selected_cols = numeric_cols[:5] if len(numeric_cols) >= 5 else numeric_cols
    
    selected_columns = st.multiselect(
        "Select numerical variables to include in the heatmap (minimum 2 required):",
        options=numeric_cols,
        default=default_selected_cols,
        key="heatmap_cols"
    )
    
    # Enforce minimum 2 columns rule to draw a matrix
    if len(selected_columns) < 2:
        st.warning("⚠️ Please select at least 2 variables to generate the correlation matrix.")
        return
        
    # Layout for method and shape selection
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
        
    st.divider()
    
    # --- Visualization Section ---
    st.markdown("### Heatmap Visualization")
    
    # Generate the heatmap using the backend function
    fig, code_heatmap = get_correlation_heatmap(
        df, 
        columns=selected_columns, 
        method=selected_method, 
        shape=selected_shape
    )
    
    st.pyplot(fig)
    
    # --- Reproducible Code Section ---
    with st.expander("View Heatmap Code", expanded=False):
        st.markdown("Use this code to reproduce the heatmap in your own environment.")
        show_code(code_heatmap)