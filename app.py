import streamlit as st
from gui.load_dataset_page import render_upload_page
from gui.normality_page import render_normality_test_page
from gui.descriptive_stats_page import render_descriptive_st_page

# Global Configuration
st.set_page_config(page_title="Statistics in Python", layout="wide")

# Initialize Session State
if "df" not in st.session_state:
    st.session_state.df = None

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("Navigation")
selection = st.sidebar.radio(
    "Go to:",
    ["Upload Dataset",
     "Descriptive Statistics", 
     "Normality Tests"]
)

# --- PAGE ROUTING ---
if selection == "Upload Dataset":
    render_upload_page()
elif selection == "Descriptive Statistics":
    render_descriptive_st_page()
elif selection == "Normality Tests":
    render_normality_test_page()