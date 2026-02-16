import streamlit as st
from logic.basic_code import load_dataset
from gui.components import show_code

def render_upload_page():
    st.title("📂 Load the data")
    st.info("Upload your CSV file to begin the analysis.")
    
    file = st.file_uploader("Choose a CSV file", type=["csv"])
    
    if file is not None:
        df, code = load_dataset(file)
        show_code(code)

        if df is not None:
            st.session_state.df = df
            st.success("File uploaded successfully!")
            st.markdown("### Dataset Preview")
            st.dataframe(df.head())
        else:
            st.error("There was an error processing the file.")
