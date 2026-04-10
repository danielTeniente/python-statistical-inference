import pandas as pd
import numpy as np
import streamlit as st
from gui.components import show_code
from logic.basic_code import get_numeric_columns, create_categorical_column, generate_save_code

def render_create_categorical_page():
    st.title("Create Categorical Variable from Numerical")
    
    if "df" not in st.session_state or st.session_state.df is None:
        st.warning("⚠️ No data found. Please upload a file in the 'Upload Dataset' section first.")
        return
        
    df = st.session_state.df
    numeric_cols = get_numeric_columns(df)
    
    if not numeric_cols:
        st.error("The dataset does not contain any numeric columns to transform.")
        return
    
    st.markdown("### Basic Configuration")
    col1, col2 = st.columns(2)
    with col1:
        source_col = st.selectbox("Select Numeric Column", numeric_cols)
    with col2:
        new_col_name = st.text_input("New Column Name", value=f"{source_col}_cat")

    col_min = float(df[source_col].min())
    col_max = float(df[source_col].max())
    st.info(f"💡 **Tip:** The values in '{source_col}' range from **{col_min:.2f}** to **{col_max:.2f}**.")

    st.markdown("### Define Categories and Cutoffs")
    
    num_categories = st.number_input("How many categories do you want to create?", min_value=2, max_value=10, value=2)
    right_inclusive = st.checkbox("Include right extreme in each interval (Right Inclusive)", value=True, 
                                  help="If checked, the cutoff value belongs to the lower category (e.g., value <= cutoff). If unchecked, it belongs to the upper category (e.g., value < cutoff).")
    
    st.write("Define the **upper limit (cutoff)** and the **name** for each category:")
    
    labels = []
    cutoffs = []
    
    for i in range(num_categories):
        c1, c2, c3 = st.columns([2, 2, 3])
        
        with c1:
            label = st.text_input(f"Name (Category {i+1})", value=f"Cat_{i+1}", key=f"label_{i}")
            labels.append(label)
            
        with c2:
            if i < num_categories - 1:
                # Para todas las categorías excepto la última, pedimos el límite superior
                default_val = col_min + ((col_max - col_min) / num_categories) * (i + 1)
                cutoff = st.number_input(f"Upper Limit", value=float(default_val), key=f"cutoff_{i}", step=1.0)
                cutoffs.append(cutoff)
            else:
                st.text_input("Upper Limit", value="+Infinity (Max)", disabled=True, key=f"cutoff_{i}")

    bins = [-np.inf] + cutoffs + [np.inf]
    
    st.markdown("#### Resulting Intervals Preview")
    for i in range(num_categories):
        lower = bins[i]
        upper = bins[i+1]
        
        low_str = "-∞" if lower == -np.inf else str(lower)
        up_str = "+∞" if upper == np.inf else str(upper)

        if right_inclusive:
            notation = f"]{low_str}, {up_str}]"
            explanation = f"x > {low_str} and x ≤ {up_str}" if lower != -np.inf else f"x ≤ {up_str}"
        else:
            notation = f"[{low_str}, {up_str}["
            explanation = f"x ≥ {low_str} and x < {up_str}" if upper != np.inf else f"x ≥ {low_str}"

        st.write(f"- **{labels[i]}**: `{notation}` *(Means: {explanation})*")

    st.divider()

    if st.button("Transform Variable", type="primary"):
        if cutoffs != sorted(cutoffs) or len(set(cutoffs)) != len(cutoffs):
            st.error("⚠️ The upper limits must be strictly increasing. Please adjust your cutoffs.")
            return
            
        if new_col_name in df.columns:
            st.warning(f"⚠️ Column '{new_col_name}' already exists and will be overwritten.")
            
        try:
            df_updated, code = create_categorical_column(
                df=df, 
                source_col=source_col, 
                new_col_name=new_col_name, 
                bins=bins, 
                labels=labels, 
                right_inclusive=right_inclusive
            )
            
            st.session_state.df = df_updated
            
            st.success(f"✅ Categorical variable '{new_col_name}' successfully created!")
            
            show_code(code)
            
            st.markdown("### Data Preview")
            st.dataframe(df_updated[[source_col, new_col_name]].head(10))


            
        except Exception as e:
            st.error(f"An error occurred while creating the variable: {e}")
    
    st.divider()

    st.markdown("### Download Updated Dataset")
    st.write("Download your dataset to your local machine with the new categorical variable included.")
    
    col_name, col_ext = st.columns([3, 1])
    
    with col_name:
        filename_base = st.text_input("Enter filename (without extension)", value="updated_dataset")
        
    with col_ext:
        #file_extension = st.selectbox("Format", options=[".csv", ".xlsx (Coming Soon)"])
        file_extension = st.selectbox("Format", options=[".csv"])
    if file_extension == ".csv":
        final_filename = f"{filename_base}.csv"
        csv_data = st.session_state.df.to_csv(index=False).encode('utf-8')
        
        if st.download_button(
            label="📥 Download CSV File",
            data=csv_data,
            file_name=final_filename,
            mime="text/csv",
            type="primary"
        ):
            st.success(f"✅ File `{final_filename}` downloaded successfully!")
            
            save_code = generate_save_code(final_filename)
            show_code(save_code)
            
    elif file_extension == ".xlsx (Coming Soon)":
        st.download_button(label="📥 Download Excel File", data="", disabled=True)
        st.warning("⚠️ Excel export is not yet implemented. Please select .csv format for now.")