def replace_substring(df, source_col, new_col_name, to_replace, value):
    
    df[new_col_name] = df[source_col].astype("string").str.replace(to_replace, value, regex=False)
    
    code = f"import pandas as pd\n\n"
    code += f"df['{new_col_name}'] = df['{source_col}'].astype('string').str.replace('{to_replace}', '{value}', regex=False)\n"
    
    return df, code

def trim_whitespace(df, source_col, new_col_name):
    
    df[new_col_name] = df[source_col].astype("string").str.strip()
    
    code = f"import pandas as pd\n\n"
    code += f"df['{new_col_name}'] = df['{source_col}'].astype('string').str.strip()\n"
    
    return df, code

def standardize_case(df, source_col, new_col_name, case_type="lower"):
    
    text_col = df[source_col].astype("string")
    
    if case_type == "lower":
        df[new_col_name] = text_col.str.lower()
    elif case_type == "upper":
        df[new_col_name] = text_col.str.upper()
    elif case_type == "title":
        df[new_col_name] = text_col.str.title()
    else:
        raise ValueError("case_type should be 'lower', 'upper' o 'title'")
        
    code = f"import pandas as pd\n\n"
    code += f"text_col = df['{source_col}'].astype('string')\n"
    if case_type == "lower":
        code += f"df['{new_col_name}'] = text_col.str.lower()\n"
    elif case_type == "upper":
        code += f"df['{new_col_name}'] = text_col.str.upper()\n"
    elif case_type == "title":
        code += f"df['{new_col_name}'] = text_col.str.title()\n"
    
    return df, code