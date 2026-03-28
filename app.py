import streamlit as st
from gui.load_dataset_page import render_upload_page
from gui.normality_page import render_normality_test_page
from gui.normality_by_group_page import render_normality_test_by_group_page
from gui.descriptive_stats_page import render_descriptive_numerical_page
from gui.descriptive_stats_categorical_page import render_descriptive_categorical_page
from gui.onepop_mean_page import render_onepop_mean_page
from gui.onepop_median_page import render_onepop_median_page
from gui.twopop_variances_page import render_twopop_variances_page
from gui.twopop_means_page import render_twopop_means_page
from gui.twopop_median_page import render_twopop_medians_page
from gui.kpop_variances_page import render_kpop_variances_page
from gui.kpop_means_page import render_kpop_means_page
from gui.kpop_medians_page import render_kpop_medians_page
from gui.ovr_normality_page import render_ovr_normality_test_page
from gui.ovr_variances_page import render_ovr_variances_page
from gui.ovr_mean_page import render_ovr_means_page
from gui.ovr_median_page import render_ovr_medians_page
from gui.oneprop_page import render_oneprop_test_page

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

with st.sidebar.expander("Descriptive Statistics", expanded=False):
    # Subsección: Variables Numéricas
    if st.button("Numerical Variables", use_container_width=True):
        change_page("Descriptive - Numerical")
        
    # Subsección: Variables Categóricas (labels a, b, c)
    if st.button("Categorical Variables", use_container_width=True):
        change_page("Descriptive - Categorical")

with st.sidebar.expander("Normality Tests", expanded=False):
    if st.button("Whole Sample Normality", use_container_width=True):
        change_page("Whole Sample Normality")
    if st.button("Normality Tests by Group", use_container_width=True):
        change_page("Normality Tests by Group")

with st.sidebar.expander("One-Sample Tests", expanded=False):
    if st.button("One-Sample Mean Test", use_container_width=True):
        change_page("One-Sample Mean Test")
    if st.button("One-Sample Median Test", use_container_width=True):
        change_page("One-Sample Median Test")

with st.sidebar.expander("Two-Sample Tests", expanded=False):
    if st.button("Two-Sample Variance Tests", use_container_width=True):
        change_page("Two-Sample Variance Tests")
    if st.button("Two-Sample Mean Tests", use_container_width=True):
        change_page("Two-Sample Mean Tests")
    if st.button("Two-Sample Median Tests", use_container_width=True):
        change_page("Two-Sample Median Tests")

with st.sidebar.expander("k-Sample Tests", expanded=False):
    if st.button("k-Sample Variance Tests", use_container_width=True):
        change_page("k-Sample Variance Tests")
    if st.button("k-Sample Mean Tests", use_container_width=True):
        change_page("k-Sample Mean Tests")
    if st.button("k-Sample Median Tests", use_container_width=True):
        change_page("k-Sample Median Tests")

with st.sidebar.expander("One-vs-Rest Tests", expanded=False):
    if st.button("One-vs-Rest Normality Tests", use_container_width=True):
        change_page("One-vs-Rest Normality Tests")
    if st.button("One-vs-Rest Variance Tests", use_container_width=True):
        change_page("One-vs-Rest Variance Tests")
    if st.button("One-vs-Rest Mean Tests", use_container_width=True):
        change_page("One-vs-Rest Mean Tests")
    if st.button("One-vs-Rest Median Tests", use_container_width=True):
        change_page("One-vs-Rest Median Tests")

with st.sidebar.expander("Proportion Tests", expanded=False):
    if st.button("One-Proportion Test", use_container_width=True):
        change_page("One-Proportion Test")


# --- PAGE ROUTING ---
# Leemos la página actual desde el session_state y renderizamos
page = st.session_state.current_page

if page == "Upload Dataset":
    render_upload_page()
    
elif page == "Descriptive - Numerical":
    render_descriptive_numerical_page()
    
elif page == "Descriptive - Categorical":
    render_descriptive_categorical_page()
    
elif page == "Whole Sample Normality":
    render_normality_test_page()

elif page == "Normality Tests by Group":
    render_normality_test_by_group_page()    

elif page == "One-Sample Mean Test":
    render_onepop_mean_page()

elif page == "One-Sample Median Test":
    render_onepop_median_page()

elif page == "Two-Sample Variance Tests":
    render_twopop_variances_page()

elif page == "Two-Sample Mean Tests":
    render_twopop_means_page()

elif page == "Two-Sample Median Tests":
    render_twopop_medians_page()

elif page == "k-Sample Variance Tests":
    render_kpop_variances_page()

elif page == "k-Sample Mean Tests":
    render_kpop_means_page()

elif page == "k-Sample Median Tests":
    render_kpop_medians_page()

elif page == "One-vs-Rest Normality Tests":
    render_ovr_normality_test_page()

elif page == "One-vs-Rest Variance Tests":
    render_ovr_variances_page()

elif page == "One-vs-Rest Mean Tests":
    render_ovr_means_page()

elif page == "One-vs-Rest Median Tests":
    render_ovr_medians_page()

elif page == "One-Proportion Test":
    render_oneprop_test_page()