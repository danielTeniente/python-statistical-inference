from scipy import stats
import matplotlib.pyplot as plt

def run_normality_test(df, column, test_type):
    """
    Executes the selected normality test.
    Returns: (statistic, p_value, generated_code_string)
    """
    if test_type == "Shapiro–Wilk":
        stat, p = stats.shapiro(df[column])
        code = f"from scipy import stats\nstat, p = stats.shapiro(df[column])\nprint(f'Statistic: {{stat:.4f}}, p-value: {{p:.4f}}')"
    elif test_type == "D’Agostino–Pearson":
        stat, p = stats.normaltest(df[column])
        code = f"from scipy import stats\nstat, p = stats.normaltest(df[column])\nprint(f'Statistic: {{stat:.4f}}, p-value: {{p:.4f}}')"
    elif test_type == "Kolmogorov–Smirnov":
        # Standardize the df before testing
        z_df = (df[column] - df[column].mean()) / df[column].std(ddof=0) # teoretical
        stat, p = stats.kstest(z_df, 'norm')
        code = f"#Standardize the df before testing \n"
        code += f"z_df = (df[column] - df[column].mean()) / df[column].std(ddof=0)\n"
        code += f"from scipy import stats\nstat, p = stats.kstest(z_df, 'norm')\n"
        code += f"print(f'Statistic: {{stat:.4f}}, p-value: {{p:.4f}}')"
    else: # Anderson-Darling
        # Using method='interpolate' to avoid FutureWarning and get exact p-value
        result = stats.anderson(df[column], dist='norm', method='interpolate')
        stat = result.statistic
        p = result.pvalue
        
        code = "# Run Anderson-Darling test with interpolation for p-value\n"
        code += "from scipy import stats\n"
        code += f"result = stats.anderson(df['{column}'], dist='norm', method='interpolate')\n"
        code += "stat, p = result.statistic, result.pvalue\n"
        code += "print(f'Statistic: {stat:.4f}, p-value: {p:.4f}')"        
        
    return stat, p, code

def get_qqplot(df, column):
    """
    Processes the data to generate a QQ-plot and returns the figure 
    and the equivalent Python code as a string.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    stats.probplot(df[column].dropna(), dist="norm", plot=ax)
    ax.set_title(f'QQ-Plot of {column}')
    ax.grid(alpha=0.3)
    
    # 4. Generar el código en formato string
    code = "import matplotlib.pyplot as plt\n"
    code += "from scipy import stats\n"
    code += "fig, ax = plt.subplots(figsize=(10, 6))\n"
    code += f"stats.probplot(df['{column}'].dropna(), dist='norm', plot=ax)\n"
    code += f"ax.set_title('QQ-Plot of {column}')\n"
    code += "ax.grid(alpha=0.3)\n"
    code += "plt.show()"
    
    # 5. Retornar ambos elementos
    return fig, code