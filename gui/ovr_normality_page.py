import streamlit as st
from gui.components import show_code
from logic.basic_code import get_numeric_columns, get_categorical_columns   
from logic.ovr_logic import run_normality_test_ovr

def render_ovr_normality_test_page():
    st.title("📊 One-vs-Rest: Normality Tests")
    
    # 1. Verificación de datos
    if "df" not in st.session_state or st.session_state.df is None:
        st.warning("⚠️ No data found. Please upload a file in the 'Upload Dataset' section first.")
        return

    df = st.session_state.df
    numeric_cols = get_numeric_columns(df)
    all_categorical_cols = get_categorical_columns(df)
    
    if not numeric_cols:
        st.error("The dataset does not contain any numeric columns.")
        return
        
    valid_categorical_cols = [col for col in all_categorical_cols if df[col].nunique() > 2]
    
    if not valid_categorical_cols:
        st.error("Error: The dataset must contain at least one categorical column with 3 or more categories to perform a One-vs-Rest test.")
        return

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
             
    available_categories = df[selected_cat].dropna().unique().tolist()
    target_cat = st.selectbox("Select Target Population ('One')", available_categories, key="ovr_norm_target")
    
    st.caption(f"Testing normality for: **{target_cat}** and **The Rest**")

    st.divider()
    
    st.subheader(f"Results for {test_name}")
    results_df, code = run_normality_test_ovr(df, selected_num, selected_cat, target_cat, test_name)    
    st.dataframe(
        results_df.style.format({
            'Statistic': '{:.4f}', 
            'p-value': '{:.4f}'
        }), 
        use_container_width=True,
        hide_index=True
    )
    show_code(code)    
