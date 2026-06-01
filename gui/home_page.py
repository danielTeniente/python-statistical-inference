import streamlit as st

def render_home_page():
    # Title & Hero Section
    st.title("🔬 Inferential Statistics with Python")
    st.markdown("### *Learning Statistics Through Interactive Python Tools*")
    st.markdown("Daniel Díaz Bedoya | Data Science Master's Programme | Politécnico de Leiria")
    st.markdown("---")
    
    # Core Value Proposition
    st.markdown("""
    This platform is designed to help students, 
    researchers, and data enthusiasts perform robust inferential statistics without getting bogged down by syntax.
    
    **💡 The Dual-Output Idea:** Every time you run a test, the application delivers both the **statistical insights** and the **reproducible Python code** used behind the scenes. Just copy, paste, and integrate it into your own Python environment, Notebooks or production workflows.
    """)
    
    st.markdown("---")
    
    # How to Get Started Workflow
    st.markdown("### 🚀 How to Get Started")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 1. Upload your Data")
        st.info("Go to the **'Upload Dataset'** section in the menu to load your data in **CSV format**.")
        
    with col2:
        st.markdown("#### 2. Navigate & Select")
        st.info("Explore the sidebar menu to choose your data transformations, descriptive metrics, or hypothesis tests.")
        
    with col3:
        st.markdown("#### 3. Analyze & Learn")
        st.info("Review your statistical results and copy the automatically generated code to replicate it on your machine.")

    st.markdown("---")
    
    # Toolkit Overview (Summarized from your menu structure)
    st.markdown("### 🛠️ What can you do with this tool?")
    
    feat_col1, feat_col2 = st.columns(2)
    
    with feat_col1:
        with st.expander("📊 Data Preparation & Diagnostics", expanded=True):
            st.markdown("""
            * **Data Transformation:** Adjust data types, create categorical variables, and clean text strings.
            * **Descriptive Statistics:** Get instant summaries for both numerical and categorical distributions.
            * **Normality Testing:** Validate assumptions with whole sample or grouped normality checks.
            """)
            
    with feat_col2:
        with st.expander("🧪 Inferential Hypothesis Testing", expanded=True):
            st.markdown("""
            * **Parametric & Non-Parametric Suites:** Run One-Sample, Two-Sample, and k-Sample tests for Means, Medians, and Variances.
            * **Categorical & Relationships:** Execute Proportion tests, Independence tests, Association measures, and dynamic Correlation Heatmaps.
            """)

    # Academic Footer
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown(
        "<center style='color: #64748b;'><small>Developed as a Graduation Project for the "
        "<b>Data Science Master's Programme</b> at the <b>Politécnico de Leiria</b>, Portugal.</small></center>", 
        unsafe_allow_html=True
    )