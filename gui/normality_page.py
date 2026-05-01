import streamlit as st
from logic.basic_code import get_numeric_columns
from gui.components import show_code
from logic.normality_page_logic import run_normality_test, get_qqplot

def render_normality_test_page():
    st.title("📊 Normality Tests")
    
    # --- 1. Data Validation ---
    if "df" not in st.session_state or st.session_state.df is None:
        st.warning("⚠️ No data found. Please upload a file in the 'Upload Dataset' section first.")
        return

    df = st.session_state.df
    numeric_cols = get_numeric_columns(df)
    
    if not numeric_cols:
        st.error("The dataset does not contain any numeric columns.")
        return

    # --- 2. User Selection ---
    st.markdown("### Configuration")
    col1, col2 = st.columns(2)
    with col1:
        selected_col = st.selectbox("Select variable", numeric_cols, key="norm_sel_col")
    with col2:
        test_name = st.selectbox("Select test", 
            ["Shapiro–Wilk", "D’Agostino–Pearson", 
             "Kolmogorov–Smirnov", "Anderson-Darling"], key="norm_sel_test")

    # --- 3. Context ID and State Management ---
    # Create a unique ID for the current selection
    current_context_id = f"{selected_col}_{test_name}"

    # If the variables change, reset the results dictionary for this page
    if ("normality_page_state" not in st.session_state or 
        st.session_state.get("normality_page_id") != current_context_id):
        
        st.session_state.normality_page_state = {}  # Dictionary for individual results
        st.session_state.normality_page_id = current_context_id

    # Shortcut reference to the state dictionary
    state = st.session_state.normality_page_state

    st.divider()

    # --- 4. Granular Analysis Sections ---

    # SECTION: Statistical Normality Test
    with st.expander(f"Statistical Test: {test_name}", expanded=not state.get("stats")):
        if st.button("Run Statistical Test", key="btn_run_stats"):
            with st.spinner("Computing test statistics..."):
                # Unpacking the 4 variables, including the sampled_flag
                stat, p, code, is_sampled = run_normality_test(df, selected_col, test_name)
                state["stats"] = {"stat": stat, "p": p, "code": code, "is_sampled": is_sampled}

        if "stats" in state:
            res = state["stats"]
            
            # Rule 3: Transparency regarding Undersampling
            if res.get("is_sampled"):
                st.info("ℹ️ **Note:** Due to statistical constraints and cloud performance limits for large datasets, the data was randomly sampled to 5,000 rows. The generated code below reflects this process.")
                
            res_c1, res_c2 = st.columns(2)
            res_c1.metric("Statistic", f"{res['stat']:.4f}")
            res_c2.metric("p-value", f"{res['p']:.4f}")
            show_code(res["code"])

    # SECTION: Visual Analysis (QQ-Plot)
    with st.expander("Visual Analysis (QQ-Plot)", expanded=False):
        if st.button("Generate QQ-Plot", key="btn_run_qq"):
            with st.spinner("Generating plot..."):
                # Unpacking the 3 variables, including the sampled_flag
                fig, qq_code, is_sampled = get_qqplot(df, selected_col)
                state["qq_plot"] = {"fig": fig, "code": qq_code, "is_sampled": is_sampled}

        if "qq_plot" in state:
            res_qq = state["qq_plot"]
            
            # Rule 3: Transparency regarding Undersampling for plotting
            if res_qq.get("is_sampled"):
                st.info("ℹ️ **Note:** To prevent browser memory overload during rendering, the data was randomly sampled to 5,000 points. The generated code includes this step.")
                
            st.pyplot(res_qq["fig"])
            show_code(res_qq["code"])