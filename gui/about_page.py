import streamlit as st

def render_about_page():
    st.title("About This Project")
    
    # Etiqueta de versión usando un contenedor de información de Streamlit
    st.info("**Current Version:** v0.1.0-alpha (Work in Progress / Under Development)")

    st.markdown("### Background & Motivation")
    st.markdown("""
    This application was developed as a final graduation project for the Data Science Master’s Programme at the 
    **[Politécnico de Leiria](https://www.ipleiria.pt/)**, Portugal.

    The inspiration for this tool comes from GUI-driven statistical software available in the R ecosystem, 
    such as *R Commander (Rcmdr)*. The primary objective is to bridge the gap between graphical statistical 
    analysis and Python programming. 
    
    Designed as an educational platform, it allows students and researchers to upload datasets and perform 
    inferential statistics tests through an intuitive interface. Upon execution, the application displays both 
    the statistical results and the exact, reproducible Python code used behind the scenes. This dual-output 
    approach empowers users to conduct robust analyses while simultaneously learning how to implement them in Python.
    """)

    st.markdown("---")

    st.markdown("### Resources & Open Source")
    st.markdown("""
    This project is open-source and actively maintained. You can view the source code, track the progress, 
    and review the documentation on the official GitHub repository:
    
    * 💻 **GitHub Repository:** [python-statistical-inference](https://github.com/danielTeniente/python-statistical-inference)
    """)

    st.markdown("---")

    st.markdown("### Contact")
    st.markdown("""
    For inquiries, feedback, or technical support regarding this project, please feel free to reach out via email:
    
    * 📧 **Email:** [danieldiazworkcolab@gmail.com](mailto:danieldiazworkcolab@gmail.com)
    """)
    
    # Opcional: Un pequeño footer
    st.markdown("<br><hr><center><i>Developed for educational purposes.</i></center>", unsafe_allow_html=True)