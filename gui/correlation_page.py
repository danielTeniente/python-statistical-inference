import streamlit as st
from gui.components import show_code
from logic.basic_code import get_numeric_columns

from logic.correlation_logic import (
    get_scatterplot, 
    perform_pearson_correlation, 
    perform_spearman_correlation, 
    perform_kendall_correlation
)

def render_correlation_page():
    st.title("Correlation Analysis")
    
    # --- Data Validation ---
    if "df" not in st.session_state or st.session_state.df is None:
        st.warning("⚠️ No data found. Please upload a file in the 'Upload Dataset' section first.")
        return
        
    df = st.session_state.df
    numeric_cols = get_numeric_columns(df)
    
    if len(numeric_cols) < 2:
        st.error("Error: The dataset must contain at least two numerical columns to perform correlation analysis.")
        st.info("Please review your dataset or use a different test.")
        return
    
    # --- Variable & Method Selection ---
    st.markdown("### Configuration")
    col1, col2 = st.columns(2)
    with col1:
        selected_x = st.selectbox("Select X variable (Independent)", numeric_cols, index=0, key="corr_x")
    with col2:
        default_y_index = 1 if len(numeric_cols) > 1 else 0
        selected_y = st.selectbox("Select Y variable (Dependent)", numeric_cols, index=default_y_index, key="corr_y")

    if selected_x == selected_y:
        st.warning("⚠️ You have selected the same variable for both X and Y. The correlation will be exactly 1.0")
        
    st.markdown("### Methodology Setup")
    col3, col4 = st.columns(2)
    with col3:
        # Correlation Method Selection
        selected_method = st.selectbox(
            "Select Correlation Method", 
            ["Pearson", "Spearman", "Kendall"], 
            index=0, 
            key="corr_method"
        )
    with col4:
        # Confidence Level Slider
        confidence = st.slider(
            "Confidence level", 0.80, 0.99, 0.95, 0.01, key="corr_conf"
        )

    # Calculate clean valid rows to trigger dynamic warnings
    clean_len = len(df[[selected_x, selected_y]].dropna())

    # Unique identifier to clear results if variables, method, or confidence change
    analysis_id = f"{selected_x}_{selected_y}_{selected_method}_{confidence}"

    # Initialize the state for individual results
    if "corr_state_id" not in st.session_state or st.session_state.corr_state_id != analysis_id:
        st.session_state.corr_state_id = analysis_id
        st.session_state.corr_results = {}

    st.divider()
    
    # --- Correlation Measures ---
    st.markdown(f"### {selected_method} Correlation Results")
    
    # Dynamic Descriptions
    if selected_method == "Pearson":
        st.markdown("Measures the **linear relationship** between two continuous variables.")
    elif selected_method == "Spearman":
        st.markdown("Non-parametric test that measures **monotonic relationships**. Uses Bootstrap for Confidence Intervals.")
    elif selected_method == "Kendall":
        st.markdown("Non-parametric test based on **concordant and discordant pairs** (Ordinal Association). Uses Bootstrap for Confidence Intervals.")

    # Single dynamic calculation button
    with st.expander(f"🧪 Execution: {selected_method}", expanded=True):
        if st.button(f"Calculate {selected_method}", key="btn_calc"):
            with st.spinner(f"Calculating {selected_method} correlation..."):
                if selected_method == "Pearson":
                    st.session_state.corr_results["calc"] = perform_pearson_correlation(df, selected_x, selected_y, confidence_level=confidence)
                elif selected_method == "Spearman":
                    st.session_state.corr_results["calc"] = perform_spearman_correlation(df, selected_x, selected_y, confidence_level=confidence)
                elif selected_method == "Kendall":
                    st.session_state.corr_results["calc"] = perform_kendall_correlation(df, selected_x, selected_y, confidence_level=confidence)
                    
        # Display Results if calculated
        if "calc" in st.session_state.corr_results:
            # Display cloud limits for heavy computations
            if selected_method in ["Spearman", "Kendall"] and clean_len > 5000:
                st.info(f"ℹ️ **Cloud Limit Note:** To prevent the app from freezing, the Bootstrap Confidence Interval was calculated using a random sample of 5,000 rows. The base correlation uses your full data. The code provided below will run the full bootstrap on your local machine.")
            
            # Unpack results
            corr_val, p_val, ci_low, ci_high, code_str = st.session_state.corr_results["calc"]
            
            # Display metrics dynamically based on chosen confidence
            res1, res2, res3 = st.columns(3)
            ci_label_percent = f"{confidence*100:.0f}%"
            
            if selected_method == "Pearson":
                metric_name, ci_name = "Pearson's r", f"{ci_label_percent} Confidence Interval"
            elif selected_method == "Spearman":
                metric_name, ci_name = "Spearman's rho", f"{ci_label_percent} Bootstrap CI"
            else:
                metric_name, ci_name = "Kendall's tau", f"{ci_label_percent} Bootstrap CI"
                
            res1.metric(metric_name, f"{corr_val:.4f}")
            res2.metric("P-value", f"{p_val:.4f}")
            res3.metric(ci_name, f"[{ci_low:.4f}, {ci_high:.4f}]")
            
            show_code(code_str)

    # --- Visual Exploration ---
    st.divider()
    with st.expander("📊 Visual Exploration (Scatterplot)", expanded=False):
        show_line = st.checkbox("Show linear regression line", value=False, key="show_reg_line")
        
        # Execute on-demand to prevent double rendering
        if st.button("Generate Scatterplot", key="btn_scatter"):
            with st.spinner("Rendering plot..."):
                fig, code_scatter = get_scatterplot(df, selected_x, selected_y, show_line=show_line)
                st.session_state.corr_results["scatter"] = {
                    "fig": fig, 
                    "code": code_scatter, 
                    "show_line": show_line # Track state to notify user to refresh
                }
                
        if "scatter" in st.session_state.corr_results:
            res_scatter = st.session_state.corr_results["scatter"]
            
            if clean_len > 3000:
                st.info(f"🎨 **Rendering Note:** Plotting {clean_len:,} points would freeze your browser. We are visualizing a representative sample of 3,000 points. The linear regression line (if enabled) and the code below still use your complete dataset.")
            
            # UX warning if they toggled the line but haven't clicked generate
            if res_scatter["show_line"] != show_line:
                st.warning("⚠️ You toggled the regression line. Click 'Generate Scatterplot' again to update the plot.")
                
            st.pyplot(res_scatter["fig"])
            show_code(res_scatter["code"])