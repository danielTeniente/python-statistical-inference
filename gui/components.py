import streamlit as st

def show_code(code):
    """Utility function to display code in Streamlit."""
    st.markdown("Python code")
    st.code(code, language="python")