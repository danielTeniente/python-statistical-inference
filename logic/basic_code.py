import pandas as pd
import numpy as np

def load_dataset(uploaded_file):
    """Reads the uploaded CSV file and returns a DataFrame."""
    code = f"import pandas as pd\n"
    code += f"df = pd.read_csv('{uploaded_file.name}')\n"
    try:
        df = pd.read_csv(uploaded_file)
    except Exception:
        df = None
    return df, code
        

def get_numeric_columns(df):
    """Returns a list of column names that contain numeric data."""
    return df.select_dtypes(include=np.number).columns.tolist()

def get_categorical_columns(df):
    """Returns a list of column names that contain categorical data."""
    return df.select_dtypes(include=['object', 'category']).columns.tolist()
