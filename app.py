import streamlit as st
from gui.load_dataset_page import render_upload_page
from gui.normality_page import render_normality_test_page
from gui.descriptive_stats_page import render_descriptive_numerical_page
from gui.descriptive_stats_categorical_page import render_descriptive_categorical_page
from gui.onepop_mean_page import render_onepop_mean_page
from gui.onepop_median_page import render_onepop_median_page

# Global Configuration
st.set_page_config(page_title="Statistics in Python", layout="wide")

# --- INITIALIZE SESSION STATE ---
if "df" not in st.session_state:
    st.session_state.df = None

if "current_page" not in st.session_state:
    st.session_state.current_page = "Upload Dataset"

# Función auxiliar para cambiar de página fácilmente
def change_page(page_name):
    st.session_state.current_page = page_name

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("Navigation")

if st.sidebar.button("Upload Dataset", use_container_width=True):
    change_page("Upload Dataset")

with st.sidebar.expander("Descriptive Statistics", expanded=True):
    # Subsección: Variables Numéricas
    if st.button("Numerical Variables", use_container_width=True):
        change_page("Descriptive - Numerical")
        
    # Subsección: Variables Categóricas (labels a, b, c)
    if st.button("Categorical Variables", use_container_width=True):
        change_page("Descriptive - Categorical")

if st.sidebar.button("Normality Tests", use_container_width=True):
    change_page("Normality Tests")

with st.sidebar.expander("One Population tests", expanded=True):
    if st.button("One Population Mean Test", use_container_width=True):
        change_page("One Population Mean Test")
    if st.button("One Population Median Test", use_container_width=True):
        change_page("One Population Median Test")

# --- PAGE ROUTING ---
# Leemos la página actual desde el session_state y renderizamos
page = st.session_state.current_page

if page == "Upload Dataset":
    render_upload_page()
    
elif page == "Descriptive - Numerical":
    # Aquí puedes llamar a una función específica, por ejemplo:
    render_descriptive_numerical_page()
    
elif page == "Descriptive - Categorical":
    # Aquí puedes llamar a una función específica, por ejemplo:
    render_descriptive_categorical_page()
    
elif page == "Normality Tests":
    render_normality_test_page()

elif page == "One Population Mean Test":
    render_onepop_mean_page()

elif page == "One Population Median Test":
    render_onepop_median_page()