from scipy import stats
import matplotlib.pyplot as plt
import pandas as pd

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
    
    code = "import matplotlib.pyplot as plt\n"
    code += "from scipy import stats\n"
    code += "fig, ax = plt.subplots(figsize=(10, 6))\n"
    code += f"stats.probplot(df['{column}'].dropna(), dist='norm', plot=ax)\n"
    code += f"ax.set_title('QQ-Plot of {column}')\n"
    code += "ax.grid(alpha=0.3)\n"
    code += "plt.show()"
    
    return fig, code

def run_normality_test_by_group(df, num_col, cat_col, test_type):
    """
    Executes the selected normality test for each group defined by a categorical column.
    Returns: (results_df, generated_code_string)
    """
    categories = df[cat_col].dropna().unique()
    results = []

    for cat in categories:
        data = df[df[cat_col] == cat][num_col].dropna()
        n_size = len(data)
        
        # Guardrails: minimum sample size requirements for each test
        if n_size < 3 and test_type == "Shapiro–Wilk":
            results.append({'Group': cat, 'Statistic': None, 'p-value': None, 'N': n_size, 'Status': ' (error) N < 3'})
            continue
        if n_size < 8 and test_type == "D’Agostino–Pearson":
            results.append({'Group': cat, 'Statistic': None, 'p-value': None, 'N': n_size, 'Status': ' (error) N < 8'})
            continue
        if n_size == 0:
            continue

        stat, p, status = None, None, "test performed"
        
        if test_type == "Shapiro–Wilk":
            stat, p = stats.shapiro(data)
        elif test_type == "D’Agostino–Pearson":
            stat, p = stats.normaltest(data)
        elif test_type == "Kolmogorov–Smirnov":
            # Estandarizar antes de probar
            z_data = (data - data.mean()) / data.std(ddof=0)
            stat, p = stats.kstest(z_data, 'norm')
        else: # Anderson-Darling
            result = stats.anderson(data, dist='norm', method='interpolate')
            stat = result.statistic
            p = result.pvalue
            
        results.append({
            'Group': cat, 
            'Statistic': stat, 
            'p-value': p, 
            'N': n_size, 
            'Status': status
        })

    results_df = pd.DataFrame(results)

    code = "import pandas as pd\n"
    code += "from scipy import stats\n\n"
    code += f"categories = df['{cat_col}'].dropna().unique()\n"
    code += "results = []\n\n"
    code += "for cat in categories:\n"
    code += f"    data = df[df['{cat_col}'] == cat]['{num_col}'].dropna()\n"
    
    if test_type == "Shapiro–Wilk":
        code += "    if len(data) >= 3:\n"
        code += "        stat, p = stats.shapiro(data)\n"
        code += "        results.append({'Group': cat, 'Statistic': stat, 'p-value': p})\n"
    elif test_type == "D’Agostino–Pearson":
        code += "    if len(data) >= 8:\n"
        code += "        stat, p = stats.normaltest(data)\n"
        code += "        results.append({'Group': cat, 'Statistic': stat, 'p-value': p})\n"
    elif test_type == "Kolmogorov–Smirnov":
        code += "    if len(data) > 0:\n"
        code += "        z_data = (data - data.mean()) / data.std(ddof=0)\n"
        code += "        stat, p = stats.kstest(z_data, 'norm')\n"
        code += "        results.append({'Group': cat, 'Statistic': stat, 'p-value': p})\n"
    else:
        code += "    if len(data) > 0:\n"
        code += "        res = stats.anderson(data, dist='norm', method='interpolate')\n"
        code += "        results.append({'Group': cat, 'Statistic': res.statistic, 'p-value': res.pvalue})\n"
        
    code += "\nresults_df = pd.DataFrame(results)\n"
    code += "print(results_df)\n"

    return results_df, code