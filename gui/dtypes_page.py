import pandas as pd
import streamlit as st
from gui.components import show_code
from logic.basic_code import generate_save_code

def render_change_dtype_page():
    st.title("🔄 Change Column Data Type")
    
    # 1. Check if dataset exists
    if "df" not in st.session_state or st.session_state.df is None:
        st.warning("⚠️ No data found. Please upload a file in the 'Upload Dataset' section first.")
        return
        
    df = st.session_state.df
    columns = df.columns.tolist()
    
    if not columns:
        st.error("The dataset does not contain any columns.")
        return

    st.markdown("### Select Column and Target Type")
    
    # 2. UI for selection
    col1, col2, col3 = st.columns([2, 1, 2])
    
    with col1:
        selected_col = st.selectbox("Select Column to Transform", columns)
        
    # Get current data type
    current_dtype = str(df[selected_col].dtype)
    
    with col2:
        # Display current data type in a disabled input for clean UI
        st.text_input("Current Data Type", value=current_dtype, disabled=True)
        
    with col3:
        # Mapping friendly names to pandas dtypes
        dtype_options = {
            "String / Text (str)": "str",
            "Integer (int64)": "int64",
            "Float / Decimal (float64)": "float64",
            "Boolean (bool)": "bool",
            "Datetime (datetime64)": "datetime64[ns]",
            "Categorical (category)": "category"
        }
        
        target_dtype_label = st.selectbox("Select New Data Type", list(dtype_options.keys()))
        target_dtype = dtype_options[target_dtype_label]

    st.divider()

    # 3. Action Button and Transformation Logic
    if st.button("Change Data Type", type="primary"):
        if current_dtype == target_dtype:
            st.info(f"💡 The column '{selected_col}' is already of type `{current_dtype}`. No changes needed.")
        else:
            try:
                # We copy the dataframe to avoid modifying the state if an error occurs mid-way
                df_updated = df.copy()
                
                # Code generation for the UI display
                if target_dtype == "datetime64[ns]":
                    df_updated[selected_col] = pd.to_datetime(df_updated[selected_col])
                    generated_code = f"df['{selected_col}'] = pd.to_datetime(df['{selected_col}'])"
                else:
                    df_updated[selected_col] = df_updated[selected_col].astype(target_dtype)
                    generated_code = f"df['{selected_col}'] = df['{selected_col}'].astype('{target_dtype}')"

                # Update session state only if successful
                st.session_state.df = df_updated
                
                st.success(f"✅ Successfully changed '{selected_col}' from `{current_dtype}` to `{target_dtype}`!")
                show_code(generated_code)
                
                st.markdown("### Updated Data Preview")
                st.dataframe(df_updated[[selected_col]].head(10))
                
            # --- ERROR HANDLING ---
            except ValueError as e:
                st.error("❌ **Value Error:** Could not convert the data.")
                st.error(f"**Details:** {e}")
                
                # Provide helpful hints based on the target type
                if target_dtype in ["int64", "float64"]:
                    st.warning("💡 **Hint:** This usually happens when you try to convert text containing letters or special characters into numbers. Check your column for invalid entries like 'N/A', spaces, or words.")
                elif target_dtype == "datetime64[ns]":
                    st.warning("💡 **Hint:** The dates might be in an unrecognized format, or there might be invalid text in the column.")
                    
            except TypeError as e:
                st.error("❌ **Type Error:** This data type cannot be converted directly.")
                st.error(f"**Details:** {e}")
                st.warning("💡 **Hint:** Some data types are fundamentally incompatible. You might need to clean or extract the data first.")
                
            except OverflowError as e:
                st.error("❌ **Overflow Error:** The numbers are too large or too small.")
                st.error(f"**Details:** {e}")
                st.warning("💡 **Hint:** This happens when trying to fit extremely large numbers into an integer format. Try using 'Float' instead.")
                
            except Exception as e:
                st.error("❌ **An unexpected error occurred.**")
                st.error(f"**Details:** {e}")

    st.divider()

    # 4. Download Section
    st.markdown("### Download Updated Dataset")
    st.write("Download your dataset to your local machine with the updated data types.")
    
    col_name, col_ext = st.columns([3, 1])
    
    with col_name:
        filename_base = st.text_input("Enter filename (without extension)", value="updated_dataset")
        
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