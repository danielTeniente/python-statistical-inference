import pandas as pd
import streamlit as st
import gc # Importado para la gestión de memoria
from gui.components import show_code
from logic.basic_code import generate_save_code

# Import the backend functions
from logic.data_cleaning import (
    replace_substring,
    trim_whitespace,
    standardize_case
)

def render_data_cleaning_page():
    st.title("Text Data Cleaning & Transformation")
    
    if "df" not in st.session_state or st.session_state.df is None:
        st.warning("⚠️ No data found. Please upload a file in the 'Upload Dataset' section first.")
        return
        
    df = st.session_state.df
    all_columns = df.columns.tolist()
    
    if not all_columns:
        st.error("The dataset does not contain any columns.")
        return
    
    st.markdown("### 1. Select Operation")
    operation = st.selectbox(
        "Choose a data cleaning procedure:",
        [
            "Find and Replace Substring",
            "Trim Whitespace",
            "Standardize Case (Upper/Lower/Title)"
        ]
    )
    
    st.markdown("### 2. Basic Configuration")
    col1, col2 = st.columns(2)
    with col1:
        source_col = st.selectbox("Select Target Column", all_columns)
        
        # --- NUEVO: Preview de la columna seleccionada ---
        st.caption(f"👀 Quick preview of `{source_col}` (Top non-null values):")
        # Mostramos los primeros 5 valores no nulos para que el usuario vea el texto real
        preview_df = df[[source_col]].dropna().head(5)
        if preview_df.empty:
            st.info("This column is completely empty (all nulls).")
        else:
            st.dataframe(preview_df, use_container_width=True)
        # ------------------------------------------------
            
    with col2:
        new_col_name = st.text_input("New Column Name", value=f"{source_col}_cleaned")

    # Dynamic UI elements based on the selected operation
    st.markdown("### 3. Operation Settings")
    
    to_replace = ""
    replace_value = ""
    case_type = "lower"
    
    if operation == "Find and Replace Substring":
        st.info("💡 **Tip:** This replaces exact character matches. Useful for changing decimal separators (e.g., ',' to '.') or fixing specific typos.")
        r_col1, r_col2 = st.columns(2)
        with r_col1:
            to_replace = st.text_input("Text to find", value="")
        with r_col2:
            replace_value = st.text_input("Replace with", value="")
            
    elif operation == "Trim Whitespace":
        st.info("💡 **Tip:** This removes accidental spaces at the very beginning and the very end of the text. E.g., '  Ecuador ' becomes 'Ecuador'.")
        
    elif operation == "Standardize Case (Upper/Lower/Title)":
        st.info("💡 **Tip:** Normalizing text casing is vital to prevent categories like 'Apple' and 'apple' from being treated as different items.")
        case_option = st.radio(
            "Select casing format:",
            ["Lowercase (e.g., apple)", "Uppercase (e.g., APPLE)", "Title Case (e.g., Apple)"]
        )
        if "Lowercase" in case_option:
            case_type = "lower"
        elif "Uppercase" in case_option:
            case_type = "upper"
        else:
            case_type = "title"

    st.divider()

    # Execution Block
    if st.button("Transform Variable", type="primary"):
        if operation == "Find and Replace Substring" and not to_replace:
            st.error("⚠️ 'Text to find' cannot be empty for this operation.")
            return

        if new_col_name in df.columns and new_col_name != source_col:
            st.warning(f"⚠️ Column '{new_col_name}' already exists and will be overwritten.")
            
        try:
            if operation == "Find and Replace Substring":
                df_updated, code = replace_substring(df, source_col, new_col_name, to_replace, replace_value)
                
            elif operation == "Trim Whitespace":
                df_updated, code = trim_whitespace(df, source_col, new_col_name)
                
            elif operation == "Standardize Case (Upper/Lower/Title)":
                df_updated, code = standardize_case(df, source_col, new_col_name, case_type)

            # Update Session State (aunque sea el mismo objeto, asegura que Streamlit detecte el cambio)
            st.session_state.df = df_updated
            
            st.success(f"✅ Cleaning operation successfully applied to '{new_col_name}'!")
            
            show_code(code)
            
            st.markdown("### Result Preview")
            st.dataframe(df_updated[[source_col, new_col_name]].head(10))
            
            # --- Forzar liberación de memoria ---
            gc.collect()
            
        except Exception as e:
            st.error(f"An error occurred while cleaning the variable: {e}")
    
    st.divider()

    # Download Block 
    st.markdown("### Download Updated Dataset")
    st.write("Download your dataset to your local machine with the newly cleaned variable included.")
    
    col_name, col_ext = st.columns([3, 1])
    
    with col_name:
        filename_base = st.text_input("Enter filename (without extension)", value="cleaned_dataset")
        
    with col_ext:
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
            
            # Limpieza tras crear el binario del CSV
            gc.collect()