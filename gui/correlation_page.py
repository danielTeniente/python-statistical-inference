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
    
    # --- Variable Selection ---
    st.markdown("### Select Variables")
    col1, col2 = st.columns(2)
    with col1:
        selected_x = st.selectbox("Select X variable (Independent)", numeric_cols, index=0, key="corr_x")
    with col2:
        default_y_index = 1 if len(numeric_cols) > 1 else 0
        selected_y = st.selectbox("Select Y variable (Dependent)", numeric_cols, index=default_y_index, key="corr_y")

    if selected_x == selected_y:
        st.warning("⚠️ You have selected the same variable for both X and Y. The correlation will be exactly 1.0")

    # Calculate clean valid rows to trigger dynamic warnings
    clean_len = len(df[[selected_x, selected_y]].dropna())

    # Unique identifier to clear results if variables change
    analysis_id = f"{selected_x}_{selected_y}"

    # Initialize the state for individual results
    if "corr_state_id" not in st.session_state or st.session_state.corr_state_id != analysis_id:
        st.session_state.corr_state_id = analysis_id
        st.session_state.corr_results = {}

    st.divider()
    
    # --- Correlation Measures (Lazy Loading) ---
    st.markdown("### Correlation Measures")
    
    # 1. Pearson (Fast, no limits needed)
    with st.expander("Pearson Correlation (Linear Relationship)", expanded=True):
        st.markdown("Measures the linear relationship between two continuous variables.")
        if st.button("Calculate Pearson", key="btn_pearson"):
            with st.spinner("Calculating Pearson correlation..."):
                st.session_state.corr_results["pearson"] = perform_pearson_correlation(df, selected_x, selected_y)
                
        if "pearson" in st.session_state.corr_results:
            corr_p, p_val_p, ci_low_p, ci_high_p, code_p = st.session_state.corr_results["pearson"]
            show_code(code_p)
            res1, res2, res3 = st.columns(3)
            res1.metric("Pearson's r", f"{corr_p:.4f}")
            res2.metric("P-value", f"{p_val_p:.4f}")
            res3.metric("95% Confidence Interval", f"[{ci_low_p:.4f}, {ci_high_p:.4f}]")

    # 2. Spearman (Bootstrap Limit: 5000)
    with st.expander("Spearman Rank Correlation (Monotonic Relationship)", expanded=False):
        st.markdown("Non-parametric test that measures monotonic relationships. Uses Bootstrap for Confidence Intervals.")
        
        if st.button("Calculate Spearman", key="btn_spearman"):
            with st.spinner("Calculating Spearman & Bootstrap CI... (This may take a moment)"):
                st.session_state.corr_results["spearman"] = perform_spearman_correlation(df, selected_x, selected_y)
                
        if "spearman" in st.session_state.corr_results:
            if clean_len > 5000:
                st.info(f"ℹ️ **Cloud Limit Note:** To prevent the app from freezing, the Bootstrap Confidence Interval was calculated using a random sample of 5,000 rows. The base correlation uses your full data. The code provided below will run the full bootstrap on your local machine.")
            
            corr_s, p_val_s, ci_low_s, ci_high_s, code_s = st.session_state.corr_results["spearman"]
            show_code(code_s)
            res1, res2, res3 = st.columns(3)
            res1.metric("Spearman's rho", f"{corr_s:.4f}")
            res2.metric("P-value", f"{p_val_s:.4f}")
            res3.metric("95% Bootstrap CI", f"[{ci_low_s:.4f}, {ci_high_s:.4f}]")

    # 3. Kendall (Bootstrap Limit: 5000)
    with st.expander("Kendall Tau Correlation (Ordinal Association)", expanded=False):
        st.markdown("Non-parametric test based on concordant and discordant pairs. Uses Bootstrap for Confidence Intervals.")
        
        if st.button("Calculate Kendall", key="btn_kendall"):
            with st.spinner("Calculating Kendall & Bootstrap CI... (This may take a moment)"):
                st.session_state.corr_results["kendall"] = perform_kendall_correlation(df, selected_x, selected_y)
                
        if "kendall" in st.session_state.corr_results:
            if clean_len > 5000:
                st.info(f"ℹ️ **Cloud Limit Note:** To prevent the app from freezing, the Bootstrap Confidence Interval was calculated using a random sample of 5,000 rows. The base correlation uses your full data. The code provided below will run the full bootstrap on your local machine.")

            corr_k, p_val_k, ci_low_k, ci_high_k, code_k = st.session_state.corr_results["kendall"]
            show_code(code_k)
            res1, res2, res3 = st.columns(3)
            res1.metric("Kendall's tau", f"{corr_k:.4f}")
            res2.metric("P-value", f"{p_val_k:.4f}")
            res3.metric("95% Bootstrap CI", f"[{ci_low_k:.4f}, {ci_high_k:.4f}]")

    # --- Visual Exploration ---
    st.divider()
    st.markdown("### Visual Exploration")
    
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