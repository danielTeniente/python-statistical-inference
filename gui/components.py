import streamlit as st

def show_code(code: str):
    """Utility function to display code in Streamlit."""
    st.markdown("Python code you can copy and run in your own environment:")
    st.code(code, language="python")