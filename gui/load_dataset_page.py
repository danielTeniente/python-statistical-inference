import streamlit as st
import pandas as pd
from logic.basic_code import load_dataset
from gui.components import show_code

def render_upload_page():
    st.title("📂 Load the data")
    st.info("Upload your CSV file to begin the analysis. If you already uploaded one, it will be kept in memory.")
    
    separators = {
        "Comma ( , )": ",",
        "Semicolon ( ; )": ";",
        "Tab ( \\t )": "\t",
        "Pipe ( | )": "|"
    }
    encodings = ["Auto", "utf-8", "latin-1", "cp1252", "iso-8859-1"]
    
    col1, col2 = st.columns(2)
    with col1:
        selected_encoding = st.selectbox("Encoding", encodings)
    with col2:
        selected_sep_label = st.selectbox("Separator", list(separators.keys()))
        selected_sep = separators[selected_sep_label]
        
    # 1. LÓGICA DE CARGA: El uploader sigue siempre disponible
    file = st.file_uploader("Choose a CSV file to upload or replace the current one", type=["csv"])
    
    if file is not None:
        df, code, error_msg, used_enc = load_dataset(file, selected_encoding, selected_sep)
        
        show_code(code)

        if df is not None:
            # Guardamos el nuevo dataframe en el estado de la sesión
            st.session_state.df = df
            st.success(f"File uploaded successfully! (Encoding: `{used_enc}` | Separator: `{repr(selected_sep)}`)")
        else:
            st.error("There was an error processing the file.")
            st.error(f"Details: {error_msg}")
            st.warning("Try changing the **Separator** or **Encoding** above.")

    # 2. LÓGICA DE VISUALIZACIÓN: Revisamos si ya hay un df en memoria
    if 'df' in st.session_state and st.session_state.df is not None:
        current_df = st.session_state.df
        
        st.markdown("### Dataset Preview")
        show_code('df.head()')
        st.dataframe(current_df.head())
        
        # Mostrar tipos de datos de forma opcional (colapsado por defecto)
        with st.expander("🔍 Show Column Data Types"):
            show_code('df.dtypes')
            
            # Formateamos los dtypes a un DataFrame limpio para que Streamlit lo renderice bonito
            dtypes_df = current_df.dtypes.astype(str).reset_index()
            dtypes_df.columns = ["Column", "Data Type"]
            
            # hide_index=True oculta el índice numérico para que se vea más profesional
            st.dataframe(dtypes_df, hide_index=True)