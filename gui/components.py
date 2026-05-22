import streamlit as st

def show_code(code: str):
    """Utility function to display code in Streamlit."""
    header_comment = "#Use this code to reproduce the function in your own environment."
    code_with_header = f"{header_comment}\n\n{code}"
    st.code(code_with_header, language="python")