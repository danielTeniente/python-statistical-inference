import streamlit as st
from logic.basic_code import load_dataset
from gui.components import show_code

def render_upload_page():
    st.title("📂 Load the data")
    st.info("Upload your CSV file to begin the analysis.")
    
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
        
    file = st.file_uploader("Choose a CSV file", type=["csv"])
    
    if file is not None:
        df, code, error_msg, used_enc = load_dataset(file, selected_encoding, selected_sep)
        
        show_code(code)

        if df is not None:
            st.session_state.df = df
            st.success(f"File uploaded successfully! (Encoding: `{used_enc}` | Separator: `{repr(selected_sep)}`)")
            st.markdown("### Dataset Preview")
            
            show_code('df.head()')
            st.dataframe(df.head())
        else:
            st.error("There was an error processing the file.")
            st.error(f"Details: {error_msg}")
            st.warning("Try changing the **Separator** or **Encoding** above.")