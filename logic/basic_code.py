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

import pandas as pd
import numpy as np

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