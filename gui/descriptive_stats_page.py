import streamlit as st
from logic.basic_code import get_numeric_columns
from logic.descriptive_stats_page_logic import describe_dataset, get_histogram, get_boxplot
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
    st.write("Click the button below to calculate the descriptive statistics (mean, std, etc.).")
    
    if st.button("Generate Summary Table"):
        desc_df, code = describe_dataset(df)
        show_code(code)
        st.dataframe(desc_df, width="stretch")

    st.divider()

    # --- SECTION 2: DATA VISUALIZATION ---
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
        