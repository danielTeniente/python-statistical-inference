import streamlit as st
from gui.components import show_code
from logic.basic_code import get_numeric_columns, get_categorical_columns   
from logic.normality_page_logic import run_normality_test_by_group

# Rule 4: Cache the filtering process to avoid re-scanning massive datasets
@st.cache_data(show_spinner=False)
def filter_dataframe_by_categories(df, cat_col, selected_cats):
    """Efficiently filters the dataframe and caches the result."""
    return df[df[cat_col].isin(selected_cats)].copy()

def render_normality_test_by_group_page():
    st.title("📊 Normality Tests by Group")
    
    # --- 1. Data Validation ---
    if "df" not in st.session_state or st.session_state.df is None:
        st.warning("⚠️ No data found. Please upload a file in the 'Upload Dataset' section first.")
        return

    df = st.session_state.df
    numeric_cols = get_numeric_columns(df)
    categorical_cols = get_categorical_columns(df)
    
    if not numeric_cols:
        st.error("The dataset does not contain any numeric columns.")
        return
        
    if not categorical_cols:
        st.error("The dataset does not contain any categorical columns to group by.")
        return

    # --- 2. Test Setup (UI Inputs) ---
    st.markdown("### Test Setup")
    col1, col2, col3 = st.columns(3)
    with col1:
        selected_num = st.selectbox("Select numerical variable", numeric_cols, key="norm_num")
    with col2:
        selected_cat = st.selectbox("Select grouping variable", categorical_cols, key="norm_cat")
    with col3:
        test_name = st.selectbox("Select test", 
            ["Shapiro–Wilk", "D’Agostino–Pearson", 
             "Kolmogorov–Smirnov", "Anderson-Darling"], key="norm_test_type")
             
    available_categories = df[selected_cat].dropna().unique().tolist()
    
    # Rule 2: Limit max_selections to prevent combinatorial explosion / OOM errors
    selected_categories = st.multiselect(
        "Select populations to test (maximum 15 allowed):",
        available_categories,
        default=available_categories[:15], # Safe default limit
        max_selections=15, 
        key="norm_populations"
    )
    
    if len(selected_categories) == 0:
        st.warning("⚠️ Please select at least one population to run the test.")
        return

    # --- 3. Context ID and State Management ---
    # Generate unique ID based on selections (sorting categories to ensure consistency)
    current_context_id = f"{selected_num}_{selected_cat}_{test_name}_{sorted(selected_categories)}"
    
    # Invalidate cache if parameters changed
    if ("norm_group_state" not in st.session_state or 
        st.session_state.get("norm_group_context_id") != current_context_id):
        
        st.session_state.norm_group_state = {}  # Isolated results dictionary
        st.session_state.norm_group_context_id = current_context_id

    state = st.session_state.norm_group_state

    st.divider()

    # --- 4. Granular Execution (On-Demand) ---

    with st.expander(f"Analysis Results: {test_name}", expanded=not state.get("results")):
        # Button to trigger the calculation
        if st.button(f"Run {test_name} Test", key="btn_run_normality"):
            with st.spinner(f"Performing {test_name} for selected groups..."):
                
                # Rule 4 implemented: Using the cached filtering function
                filtered_df = filter_dataframe_by_categories(df, selected_cat, selected_categories)
                
                # Unpacking the 3 variables, including is_sampled
                results_df, code, is_sampled = run_normality_test_by_group(
                    filtered_df, selected_num, selected_cat, test_name
                )
                
                # Rule 5: Inject filtering step into the educational code snippet
                filter_code = "# First, filter the dataset for the selected populations\n"
                filter_code += f"selected_pops = {selected_categories}\n"
                filter_code += f"df = df[df['{selected_cat}'].isin(selected_pops)]\n\n"
                final_code = filter_code + code
                
                # Store results in the isolated state dictionary
                state["results"] = {"data": results_df, "code": final_code, "is_sampled": is_sampled}

        # Render results if they exist in state (persists across reruns)
        if "results" in state:
            res = state["results"]
            
            st.markdown(f"### Statistical Results ({test_name})")
            
            # Rule 3: Transparency regarding Undersampling by group
            if res.get("is_sampled"):
                st.info("ℹ️ **Note:** Due to statistical constraints and performance limits for large datasets, groups exceeding 5,000 rows were randomly sampled. The provided code reflects this adjustment.")
            
            st.dataframe(
                res["data"].style.format({
                    'Statistic': '{:.4f}', 
                    'p-value': '{:.4f}'
                }), 
                use_container_width=True,
                hide_index=True
            )
            
            show_code(res["code"])