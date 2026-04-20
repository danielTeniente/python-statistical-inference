import pandas as pd
import numpy as np
import chardet

def detect_encoding(uploaded_file):
    """Lee los primeros 10KB del archivo para inferir la codificación."""
    raw_data = uploaded_file.read(10000)
    uploaded_file.seek(0)
    result = chardet.detect(raw_data)
    return result['encoding']

def load_dataset(uploaded_file, user_encoding="Auto", separator=","):
    """Carga el dataset manejando codificación y separador."""
    error_msg = None
    
    if user_encoding == "Auto":
        guessed_enc = detect_encoding(uploaded_file)
        enc_to_use = guessed_enc if guessed_enc else 'utf-8' 
    else:
        enc_to_use = user_encoding

    code = f"import pandas as pd\n"
    code += f"df = pd.read_csv('{uploaded_file.name}', encoding='{enc_to_use}', sep={repr(separator)})\n"

    try:
        df = pd.read_csv(uploaded_file, encoding=enc_to_use, sep=separator)
    except Exception as e:
        df = None
        error_msg = str(e)
        uploaded_file.seek(0)

    return df, code, error_msg, enc_to_use

def get_numeric_columns(df):
    """Returns a list of column names that contain numeric data."""
    return df.select_dtypes(include=np.number).columns.tolist()

def get_categorical_columns(df):
    """Returns a list of column names that contain categorical data."""
    return df.select_dtypes(include=['object', 'category']).columns.tolist()

def create_categorical_column(df, source_col, new_col_name, bins, labels, right_inclusive=True):
    """
    Creates a new categorical column by binning a continuous/numeric column.
    
    Parameters:
    - df: The pandas DataFrame.
    - source_col: The name of the numeric column to be binned.
    - new_col_name: The name of the new categorical column to be created.
    - bins: A list of numerical edges (e.g., [0, 60, 100] or [-np.inf, 18, 65, np.inf]).
    - labels: A list of string labels for the bins (length must be len(bins) - 1).
    - right_inclusive: Boolean, indicates whether the bins include the rightmost edge.
    
    Returns:
    - df_updated: The DataFrame with the new column added.
    - code: String containing the Python code to reproduce the transformation.
    """
    # Creamos una copia para evitar modificar el df original inadvertidamente 
    # y evitar el SettingWithCopyWarning de pandas
    df_updated = df.copy()
    
    # Aplicamos pd.cut para crear las categorías
    df_updated[new_col_name] = pd.cut(df_updated[source_col], bins=bins, labels=labels, right=right_inclusive)
    
    # Generamos el código reproducible para el usuario
    code = f"# Create a new categorical variable '{new_col_name}' based on '{source_col}'\n"
    code += f"import numpy as np\n"
    code += f"import pandas as pd\n\n"
    code += f"bins = {bins}\n"
    code += f"labels = {labels}\n"
    code += f"df['{new_col_name}'] = pd.cut(df['{source_col}'], bins=bins, labels=labels, right={right_inclusive})\n"
    
    return df_updated, code

def generate_save_code(filename):
    """Generates the reproducible Python code to save a DataFrame."""
    code = f"import pandas as pd\n\n"
    code += f"# Save the dataframe to your local machine\n"
    code += f"df.to_csv('{filename}', index=False)\n"
    
    return code