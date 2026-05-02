import streamlit as st
from gui.components import show_code
from logic.basic_code import get_numeric_columns, get_categorical_columns   
from logic.ovr_logic import run_normality_test_ovr

# --- Rule 4: Cache heavy DataFrame scans for UI elements ---
@st.cache_data(show_spinner=False)
def get_valid_ovr_categoricals(df, categorical_cols):
    """Caches the identification of columns with >2 categories to avoid lag."""
    return [col for col in categorical_cols if df[col].nunique() > 2]

@st.cache_data(show_spinner=False)
def get_unique_categories(df, cat_col):
    """Caches the unique categories extraction."""
    return df[cat_col].dropna().unique().tolist()

def render_ovr_normality_test_page():
    st.title("📊 One-vs-Rest: Normality Tests")
    
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
        
    # Lightweight check using cached function
    valid_categorical_cols = get_valid_ovr_categoricals(df, all_categorical_cols)
    
    if not valid_categorical_cols:
        st.error("Error: The dataset must contain at least one categorical column with 3 or more categories.")
        return

    # --- 2. Test Configuration (UI Inputs) ---
    st.markdown("### Test Setup")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        selected_num = st.selectbox("Select numerical variable", numeric_cols, key="ovr_norm_num")
    with col2:
        selected_cat = st.selectbox("Select grouping variable", valid_categorical_cols, key="ovr_norm_cat")
    with col3:
        test_name = st.selectbox("Select test", 
            ["Shapiro–Wilk", "D’Agostino–Pearson", 
             "Kolmogorov–Smirnov", "Anderson-Darling"], key="ovr_norm_test")
             
    # Use cached function for unique categories
    available_categories = get_unique_categories(df, selected_cat)
    target_cat = st.selectbox("Select Target Population ('One')", available_categories, key="ovr_norm_target")
    
    st.caption(f"Testing normality for: **{target_cat}** vs **The Rest**")

    # --- 3. Context ID and Cache Management ---
    # Create unique ID to detect input changes
    current_context_id = f"{selected_num}_{selected_cat}_{test_name}_{target_cat}"
    
    # Reset page state if the context changed
    if ("ovr_norm_state" not in st.session_state or 
        st.session_state.get("ovr_norm_id") != current_context_id):
        
        st.session_state.ovr_norm_state = {}  # Isolated results dictionary
        st.session_state.ovr_norm_id = current_context_id

    state = st.session_state.ovr_norm_state

    st.divider()

    # --- 4. Granular Execution (On-Demand) ---

    with st.expander(f"Normality Analysis: {test_name}", expanded=not state.get("results")):
        # Only perform the test logic when the button is clicked
        if st.button(f"Run {test_name} Analysis", key="btn_run_ovr_norm"):
            with st.spinner(f"Computing {test_name} for target and rest..."):
                
                # Unpacking the 3 variables returned by the optimized backend
                results_df, code, is_sampled = run_normality_test_ovr(
                    df, selected_num, selected_cat, target_cat, test_name
                )
                
                # Store results in the state dictionary
                state["results"] = {"data": results_df, "code": code, "is_sampled": is_sampled}

        # Render results if they exist in state
        if "results" in state:
            res = state["results"]
            
            st.markdown(f"### Results for {test_name}")
            
            # Rule 3: Transparency regarding Undersampling (Specifically for Shapiro-Wilk)
            if res.get("is_sampled"):
                st.info("ℹ️ **Note:** The Shapiro-Wilk test is mathematically constrained and loses accuracy with large datasets (N > 5000). To ensure test validity and prevent cloud memory issues, the data was randomly sampled. The generated code reflects this adjustment.")
                
            st.dataframe(
                res["data"].style.format({
                    'Statistic': '{:.4f}', 
                    'p-value': '{:.4f}'
                }), 
                use_container_width=True,
                hide_index=True
            )
            
            show_code(res["code"])