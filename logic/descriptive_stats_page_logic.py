import matplotlib.pyplot as plt

# descriptive statistics logic
def describe_dataset(df):
    """Returns a DataFrame with descriptive statistics for numeric columns."""
    code = f"df.describe()\n"
    return df.describe(), code

def get_sample_size(df):
    """Returns the sample size of the dataset."""
    code = f"len(df)\n"
    return len(df), code

def get_dataset_size(df):
    """Returns the size of the dataset in terms of rows and columns."""
    code = f"df.shape\n"
    return df.shape, code

def get_mean(df, column):
    """Returns the mean of a specified column."""
    code = f"df['{column}'].mean()\n"
    return df[column].mean(), code

def get_median(df, column):
    """Returns the median of a specified column."""
    code = f"df['{column}'].median()\n"
    return df[column].median(), code

def get_mode(df, column):
    """Returns the mode of a specified column."""
    code = f"df['{column}'].mode()[0]\n"
    return df[column].mode()[0], code

def get_std(df, column):
    """Returns the standard deviation of a specified column."""
    code = f"df['{column}'].std()\n"
    return df[column].std(), code

def get_variance(df, column):
    """Returns the variance of a specified column."""
    code = f"df['{column}'].var()\n"
    return df[column].var(), code

def get_min(df, column):
    """Returns the minimum value of a specified column."""
    code = f"df['{column}'].min()\n"
    return df[column].min(), code

def get_max(df, column):
    """Returns the maximum value of a specified column."""
    code = f"df['{column}'].max()\n"
    return df[column].max(), code   

def get_range(df, column):
    """Returns the range of a specified column."""
    code = f"df['{column}'].max() - df['{column}'].min()\n"
    return df[column].max() - df[column].min(), code

def get_quartiles(df, column):
    """Returns the quartiles of a specified column."""
    code = f"df['{column}'].quantile([0.25, 0.5, 0.75])\n"
    return df[column].quantile([0.25, 0.5, 0.75]), code

def get_iqr(df, column):
    """Returns the interquartile range of a specified column."""
    code = f"df['{column}'].quantile(0.75) - df['{column}'].quantile(0.25)\n"
    return df[column].quantile(0.75) - df[column].quantile(0.25), code

def get_skewness(df, column):
    """Returns the skewness of a specified column."""
    code = f"df['{column}'].skew()\n"
    return df[column].skew(), code

def get_histogram(df, column, bins=20, color='skyblue'):
    """
    Processes the data and returns the figure and the 
    equivalent Python code as a string.
    """
    # Create the figure
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(df[column].dropna(), bins=bins, color=color, edgecolor='black')
    ax.set_title(f'Histogram of {column}')
    ax.set_xlabel(column)
    ax.set_ylabel('Frequency')
    
    # Generate the string code for the user
    code = f"import matplotlib.pyplot as plt \n"
    code += f"plt.figure(figsize=(10, 6))\n"
    code += f"plt.hist(df['{column}'].dropna(), bins={bins}, color='{color}', edgecolor='black')\n"
    code += f"plt.title('Histogram of {column}')\n"
    code += f"plt.xlabel('{column}')\n"
    code += f"plt.ylabel('Frequency')\n"
    code += f"plt.show()"
    return fig, code

def get_boxplot(df, column):
    """
    Processes the data and returns the figure and the 
    equivalent Python code as a string.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.boxplot(df[column].dropna())
    ax.set_title(f'Boxplot of {column}')
    ax.set_xlabel(column)
    ax.set_ylabel('Values')
    
    code = f"import matplotlib.pyplot as plt \n"
    code += f"plt.figure(figsize=(10, 6))\n"
    code += f"plt.boxplot(df['{column}'].dropna())\n"
    code += f"plt.title('Boxplot of {column}')\n"
    code += f"plt.xlabel('{column}')\n"
    code += f"plt.ylabel('Values')\n"
    code += f"plt.show()"
    
    return fig, code






