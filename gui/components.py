import streamlit as st

def show_code(code: str):
    """Utility function to display code in Streamlit."""
    st.info("Use this code to reproduce the function in your own environment.")
    st.code(code, language="python")