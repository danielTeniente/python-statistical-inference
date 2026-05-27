import streamlit as st
from gui.load_dataset_page import render_upload_page
from gui.dtypes_page import render_change_dtype_page
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
from gui.two_prop_page import render_twoprop_test_page
from gui.about_page import render_about_page
from gui.create_cat_variables import render_create_categorical_page
from gui.kprop_page import render_kprop_test_page
from gui.independence_page import render_independence_test_page
from gui.association_page import render_association_measures_page
from gui.corr_matrix_plot_page import render_correlation_heatmap_page
from gui.correlation_page import render_correlation_page
from gui.data_cleaning_page import render_data_cleaning_page

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

if st.sidebar.button("Upload Dataset", width='stretch'):
    change_page("Upload Dataset")

with st.sidebar.expander("Data Transformation", expanded=False):
    if st.button("Change Data Types", width='stretch'):
        change_page('Change Data Types')

    if st.button("Create Categorical Variable", width='stretch'):
        change_page("Create Categorical Variable")

    if st.button("Clean Data", width='stretch'):
        change_page("Clean Data")

with st.sidebar.expander("Descriptive Statistics", expanded=False):
    # Subsección: Variables Numéricas
    if st.button("Numerical Variables", width='stretch'):
        change_page("Descriptive - Numerical")
        
    # Subsección: Variables Categóricas (labels a, b, c)
    if st.button("Categorical Variables", width='stretch'):
        change_page("Descriptive - Categorical")
        
with st.sidebar.expander("Normality Tests", expanded=False):
    if st.button("Whole Sample Normality", width='stretch'):
        change_page("Whole Sample Normality")
    if st.button("Normality Tests by Group", width='stretch'):
        change_page("Normality Tests by Group")

with st.sidebar.expander("One-Sample Tests", expanded=False):
    if st.button("One-Sample Mean Test", width='stretch'):
        change_page("One-Sample Mean Test")
    if st.button("One-Sample Median Test", width='stretch'):
        change_page("One-Sample Median Test")

with st.sidebar.expander("Two-Sample Tests", expanded=False):
    if st.button("Two-Sample Variance Tests", width='stretch'):
        change_page("Two-Sample Variance Tests")
    if st.button("Two-Sample Mean Tests", width='stretch'):
        change_page("Two-Sample Mean Tests")
    if st.button("Two-Sample Median Tests", width='stretch'):
        change_page("Two-Sample Median Tests")

with st.sidebar.expander("k-Sample Tests", expanded=False):
    if st.button("k-Sample Variance Tests", width='stretch'):
        change_page("k-Sample Variance Tests")
    if st.button("k-Sample Mean Tests", width='stretch'):
        change_page("k-Sample Mean Tests")
    if st.button("k-Sample Median Tests", width='stretch'):
        change_page("k-Sample Median Tests")

with st.sidebar.expander("One-vs-Rest Tests", expanded=False):
    if st.button("One-vs-Rest Normality Tests", width='stretch'):
        change_page("One-vs-Rest Normality Tests")
    if st.button("One-vs-Rest Variance Tests", width='stretch'):
        change_page("One-vs-Rest Variance Tests")
    if st.button("One-vs-Rest Mean Tests", width='stretch'):
        change_page("One-vs-Rest Mean Tests")
    if st.button("One-vs-Rest Median Tests", width='stretch'):
        change_page("One-vs-Rest Median Tests")

with st.sidebar.expander("Proportion Tests", expanded=False):
    if st.button("One-Proportion Test", width='stretch'):
        change_page("One-Proportion Test")
    if st.button("Two-Proportions Test", width='stretch'):
        change_page("Two-Proportions Test")
    if st.button("K Proportions Test", width='stretch'):
        change_page("K Proportions Test")

with st.sidebar.expander("Association and Independence Tests", expanded=False):
    if st.button("Tests of Independence", width='stretch'):
        change_page("Tests of Independence")
    if st.button('Measures of Association', width='stretch'):
        change_page("Measures of Association")

with st.sidebar.expander("Correlation Analysis", expanded=False):
    if st.button("Correlation Analysis", width='stretch'):
        change_page("Correlation Analysis")
    if st.button("Correlation Matrix Heatmap", width='stretch'):
        change_page("Correlation Matrix Heatmap")        

if st.sidebar.button("About", width='stretch'):
    change_page("About")


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
elif page == "Two-Proportions Test":
    render_twoprop_test_page()
elif page == "K Proportions Test":
    render_kprop_test_page()

elif page == "Tests of Independence":
    render_independence_test_page()
elif page == "Measures of Association":
    render_association_measures_page()

elif page == "Correlation Analysis":
    render_correlation_page()
elif page == "Correlation Matrix Heatmap":
    render_correlation_heatmap_page()    

elif page == "About":
    render_about_page()

elif page == "Create Categorical Variable":
    render_create_categorical_page()
elif page == "Change Data Types":
    render_change_dtype_page()
elif page == "Clean Data":
    render_data_cleaning_page()