import streamlit as st
from logic.basic_code import get_numeric_columns
from logic.descriptive_stats_page_logic import (
    describe_dataset, get_histogram, get_boxplot, 
    get_sample_size, get_dataset_size, get_mean, 
    get_median, get_mode, get_std, get_variance,
    get_min, get_max, get_range, get_quartiles,
    get_iqr, get_skewness)
from gui.components import show_code

def render_descriptive_st_page():
    if st.session_state.df is None:
        st.warning("⚠️ Please upload a dataset first to see descriptive statistics.")
        return

    df = st.session_state.df
    st.title("📈 Descriptive Analysis")
    
    # Get numeric columns for both sections
    numeric_cols = get_numeric_columns(df)
    
    if not numeric_cols:
        st.error("The dataset does not contain any numeric columns.")
        return

    # --- SECTION 1: STATISTICAL SUMMARY ---
    st.subheader("1. Statistical Summary")

    st.markdown("**Sample size of the dataset**")
    sample_size, code = get_sample_size(df)
    show_code(code)
    st.markdown(f"Sample size: **{sample_size}**")

    st.markdown("**Shape of the dataset (rows, columns)**")
    dataset_size, code = get_dataset_size(df)
    show_code(code)
    st.markdown(f"Dataset size: **{dataset_size}**")

    st.markdown("**Summary of descriptive statistics for numeric columns**")
    desc_df, code = describe_dataset(df)
    show_code(code)
    st.dataframe(desc_df, width="stretch")

    # --- SECTION 2: DESCRIPTIVE STATISTICS FOR EACH COLUMN ---
    st.divider()
    st.subheader("2. DESCRIPTIVE STATISTICS FOR EACH COLUMN")
    selected_column = st.selectbox("Select column", numeric_cols)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        mean, code = get_mean(df, selected_column)
        show_code(code)
        st.markdown(f"Mean: **{mean}**")

    with col2:
        median, code = get_median(df, selected_column)
        show_code(code)
        st.markdown(f"Median: **{median}**")

    with col3:
        mode, code = get_mode(df, selected_column)
        show_code(code)
        st.markdown(f"Mode: **{mode}**")

    col1, col2 = st.columns(2)
    with col1:
        std_dev, code = get_std(df, selected_column)
        show_code(code)
        st.markdown(f"Standard Deviation: **{std_dev}**")

    with col2:
        variance, code = get_variance(df, selected_column)
        show_code(code)
        st.markdown(f"Variance: **{variance}**")



    # --- SECTION ?: DATA VISUALIZATION ---
    st.divider()
    st.subheader("2. DATA VISUALIZATION")
    selected_column = st.selectbox("Select column to plot", numeric_cols)

    st.markdown("#### Histogram")
    bins_count = st.slider("Number of bins", min_value=5, max_value=50, value=20)
    if st.button("Generate Histogram"):
        # Call the logic layer
        fig, plot_code = get_histogram(df, selected_column, bins=bins_count)
        show_code(plot_code)
        # Display the result in the UI
        st.pyplot(fig)
        
    #-------------------------------------------    
    st.markdown("#### Boxplot")
    if st.button("Generate Boxplot"):
        # Call the logic layer
        fig, plot_code = get_boxplot(df, selected_column)
        show_code(plot_code)
        # Display the result in the UI
        st.pyplot(fig)
        