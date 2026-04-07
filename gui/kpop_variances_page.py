import streamlit as st
from gui.components import show_code
from logic.basic_code import get_numeric_columns, get_categorical_columns
from logic.kpop_logic import perform_bartlett, perform_levene

def render_kpop_variances_page():
    st.title("K Population Variances Tests")
    
    if "df" not in st.session_state or st.session_state.df is None:
        st.warning("⚠️ No data found. Please upload a file in the 'Upload Dataset' section first.")
        return
        
    df = st.session_state.df
    numeric_cols = get_numeric_columns(df)
    all_categorical_cols = get_categorical_columns(df)
    
    if not numeric_cols:
        st.error("The dataset does not contain any numeric columns.")
        return

    valid_categorical_cols = [col for col in all_categorical_cols if df[col].nunique() >= 3]
            
    if not valid_categorical_cols:
        st.error("Error: The dataset must contain at least one categorical column with 3 or more categories to perform these tests.")
        st.info("Please review your dataset or use a two-population test instead.")
        return

    st.markdown("### Test Setup")
    col1, col2 = st.columns(2)
    with col1:
        selected_num_col = st.selectbox("Select numerical variable to analyze", numeric_cols, key="kpop_num")
    with col2:
        selected_cat_col = st.selectbox("Select grouping variable (≥ 3 categories)", valid_categorical_cols, key="kpop_cat")

    available_categories = df[selected_cat_col].dropna().unique().tolist()
    
    st.markdown("#### Filter Populations")
    selected_categories = st.multiselect(
        "Select the specific categories you want to compare (minimum 3 required):", 
        available_categories, 
        default=available_categories, # By default, all are selected
        key="kpop_categories"
    )

    # Enforce minimum 3 categories rule
    if len(selected_categories) < 3:
        st.warning("⚠️ Please select at least 3 categories to perform K population tests.")
        return # Stop execution until the user selects enough categories

    # 5. Filter the dataframe based on user selection
    filtered_df = df[df[selected_cat_col].isin(selected_categories)].copy()

    st.divider()
    with st.expander("Bartlett's test for equal variances", expanded=True):
        st.markdown("### Bartlett's test for equal variances")
        stat, p_value, code = perform_bartlett(filtered_df, selected_num_col, selected_cat_col)
        show_code(code)
        
        res1, res2 = st.columns(2)
        res1.metric("Bartlett statistic", f"{stat:.4f}")
        res2.metric("P-value", f"{p_value:.4f}")

    with st.expander("Levene's test for equal variances", expanded=False):
        st.markdown("### Levene's test for equal variances")
        # Assuming perform_levene in kpop_logic takes (df, num_col, cat_col) just like Bartlett
        stat_l, p_value_l, code_l = perform_levene(filtered_df, selected_num_col, selected_cat_col)
        show_code(code_l)
        
        res1, res2 = st.columns(2)
        res1.metric("Levene statistic", f"{stat_l:.4f}")
        res2.metric("P-value", f"{p_value_l:.4f}")