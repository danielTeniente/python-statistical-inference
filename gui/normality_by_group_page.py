import streamlit as st
from gui.components import show_code
from logic.basic_code import get_numeric_columns, get_categorical_columns   
from logic.normality_page_logic import run_normality_test_by_group

def render_normality_test_by_group_page():
    st.title("📊 Normality Tests by Group")
    
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

    st.markdown("### Test Setup")
    col1, col2, col3 = st.columns(3)
    with col1:
        selected_num = st.selectbox("Select numerical variable", numeric_cols)
    with col2:
        selected_cat = st.selectbox("Select grouping variable", categorical_cols)
    with col3:
        test_name = st.selectbox("Select test", 
            ["Shapiro–Wilk", "D’Agostino–Pearson", 
             "Kolmogorov–Smirnov", "Anderson-Darling"])
             
    st.markdown("#### Filter Populations")
    available_categories = df[selected_cat].dropna().unique().tolist()
    
    selected_categories = st.multiselect(
        "Select populations to test (minimum 1 required):",
        available_categories,
        default=available_categories # Seleccionadas todas por defecto
    )
    
    if len(selected_categories) == 0:
        st.warning("⚠️ Please select at least one population to run the test.")
        return

    filtered_df = df[df[selected_cat].isin(selected_categories)].copy()

    st.divider()
    
    st.subheader(f"Results for {test_name}")
    results_df, code = run_normality_test_by_group(filtered_df, selected_num, selected_cat, test_name)
    
    
    # Mostrar resultados en una tabla
    st.dataframe(
        results_df.style.format({
            'Statistic': '{:.4f}', 
            'p-value': '{:.4f}'
        }), 
        use_container_width=True,
        hide_index=True
    )
    
    show_code(code)    