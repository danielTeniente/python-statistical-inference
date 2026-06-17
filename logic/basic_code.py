import pandas as pd
import numpy as np
import charset_normalizer

def detect_encoding(uploaded_file, sample_size=50000):
    """Lee un fragmento del archivo para inferir la codificación usando charset_normalizer."""
    raw_data = uploaded_file.read(sample_size)
    uploaded_file.seek(0)
    
    result = charset_normalizer.detect(raw_data)
    return result['encoding']

def load_dataset(uploaded_file, user_encoding="Auto", separator=","):
    """Carga el dataset manejando codificación y separador."""
    error_msg = None
    df = None
    successful_enc = None

    if user_encoding == "Auto":
        guessed_enc = detect_encoding(uploaded_file)
        primary_enc = guessed_enc if guessed_enc else 'utf-8'
        encodings_to_try = [primary_enc, 'utf-8', 'latin-1', 'cp1252']
        encodings_to_try = list(dict.fromkeys(encodings_to_try))
    else:
        encodings_to_try = [user_encoding]

    for current_enc in encodings_to_try:
        try:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, encoding=current_enc, sep=separator)
            successful_enc = current_enc
            error_msg = None
            break 
            
        except UnicodeDecodeError:
            continue
            
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            break

    if df is None:
        code = ""
        if not error_msg: 
            error_msg = "It was not possible to decode the file with the provided encodings. Please try a different encoding or check the file format."
        successful_enc = user_encoding 
    else:
        code = f"import pandas as pd\n"
        code += f"df = pd.read_csv('{uploaded_file.name}', encoding='{successful_enc}', sep={repr(separator)})\n"

    uploaded_file.seek(0) # Siempre dejamos el puntero en 0 por buenas prácticas

    return df, code, error_msg, successful_enc

def get_numeric_columns(df):
    """Returns a list of column names that contain numeric data."""
    return df.select_dtypes(include=np.number).columns.tolist()

def get_categorical_columns(df):
    """Returns a list of column names that contain categorical data."""
    return df.select_dtypes(include=['object', 'category']).columns.tolist()

def create_categorical_column(df, source_col, new_col_name, bins, labels, right_inclusive=True, is_ordinal=True):
    """
    Creates a new categorical column by binning a continuous/numeric column.
    
    Parameters:
    * df: The pandas DataFrame.
    * source_col: The name of the numeric column to be binned.
    * new_col_name: The name of the new categorical column to be created.
    * bins: A list of numerical edges (e.g., [0, 60, 100] or [-np.inf, 18, 65, np.inf]).
    * labels: A list of string labels for the bins (length must be len(bins) - 1).
    * right_inclusive: Boolean, indicates whether the bins include the rightmost edge.
    * is_ordinal: Boolean, indicates whether the resulting categories have a logical order.
    
    Returns:
    * df_updated: The DataFrame with the new column added.
    * code: String containing the Python code to reproduce the transformation.
    """
    # Creamos una copia para evitar modificar el df original inadvertidamente 
    # y evitar el SettingWithCopyWarning de pandas
    df_updated = df.copy()
    
    # Aplicamos pd.cut para crear las categorías, haciendo explícito el orden
    df_updated[new_col_name] = pd.cut(
        df_updated[source_col], 
        bins=bins, 
        labels=labels, 
        right=right_inclusive,
        ordered=is_ordinal
    )
    
    code = f"# Create a new categorical variable '{new_col_name}' based on '{source_col}'\n"
    code += "import numpy as np\n"
    code += "import pandas as pd\n\n"
    code += f"bins = {bins}\n"
    code += f"labels = {labels}\n"
    code += f"df['{new_col_name}'] = pd.cut(df['{source_col}'], bins=bins, labels=labels, right={right_inclusive}, ordered={is_ordinal})\n"
    
    return df_updated, code

def generate_save_code(filename):
    """Generates the reproducible Python code to save a DataFrame."""
    code = f"import pandas as pd\n\n"
    code += f"# Save the dataframe to your local machine\n"
    code += f"df.to_csv('{filename}', index=False)\n"
    
    return code