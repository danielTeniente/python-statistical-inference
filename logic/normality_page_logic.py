import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

def run_normality_test(df, column, test_type):
    """
    Executes the selected normality test.
    Returns: (statistic, p_value, generated_code_string, sampled_flag)
    """
    data = df[column].dropna()
    n_total = len(data)
    sampled_flag = False
    
    # 1. Building the educational snippet
    code = "import pandas as pd\nfrom scipy import stats\n\n"
    code += "# Assuming 'df' is your loaded pandas DataFrame\n"
    code += f"data = df['{column}'].dropna()\n"

    # Rule 3: Undersampling for Shapiro-Wilk due to Scipy constraints (N > 5000)
    if test_type == "Shapiro–Wilk" and n_total > 5000:
        data = data.sample(n=5000, random_state=42)
        sampled_flag = True
        code += "\n# Note: Shapiro-Wilk p-values may not be accurate for N > 5000.\n"
        code += "# We apply random sampling to maintain statistical validity and performance.\n"
        code += "data = data.sample(n=5000, random_state=42)\n"

    code += "\n"
    
    # 2. Executing logic and finishing snippet
    if test_type == "Shapiro–Wilk":
        stat, p = stats.shapiro(data)
        code += "stat, p = stats.shapiro(data)\n"
    elif test_type == "D’Agostino–Pearson":
        stat, p = stats.normaltest(data)
        code += "stat, p = stats.normaltest(data)\n"
    elif test_type == "Kolmogorov–Smirnov":
        z_data = (data - data.mean()) / data.std(ddof=0)
        stat, p = stats.kstest(z_data, 'norm')
        code += "# Kolmogorov-Smirnov requires the data to be standardized (Z-scores)\n"
        code += "z_data = (data - data.mean()) / data.std(ddof=0)\n"
        code += "stat, p = stats.kstest(z_data, 'norm')\n"
    else: # Anderson-Darling
        result = stats.anderson(data, dist='norm', method='interpolate')
        stat, p = result.statistic, result.pvalue
        code += "# Using 'interpolate' to get the exact p-value instead of critical values\n"
        code += "result = stats.anderson(data, dist='norm', method='interpolate')\n"
        code += "stat, p = result.statistic, result.pvalue\n"
        
    code += "print(f'Statistic: {stat:.4f}, p-value: {p:.4f}')"

    return stat, p, code, sampled_flag


def get_qqplot(df, column):
    """
    Processes the data to generate a QQ-plot. 
    Undersamples if N > 5000 to prevent browser/rendering freezing.
    Returns: (fig, generated_code_string, sampled_flag)
    """
    data = df[column].dropna()
    sampled_flag = False
    
    code = "import matplotlib.pyplot as plt\nfrom scipy import stats\n\n"
    code += "# Assuming 'df' is your loaded pandas DataFrame\n"
    code += f"data = df['{column}'].dropna()\n"

    if len(data) > 5000:
        data = data.sample(n=5000, random_state=42)
        sampled_flag = True
        code += "\n# Undersampling to 5000 points to prevent high memory consumption during plotting\n"
        code += "data = data.sample(n=5000, random_state=42)\n"

    code += "\nfig, ax = plt.subplots(figsize=(8, 5))\n"
    code += "stats.probplot(data, dist='norm', plot=ax)\n"
    code += f"ax.set_title('QQ-Plot of {column}')\n"
    code += "ax.grid(alpha=0.3)\n"
    code += "plt.show()"

    fig, ax = plt.subplots(figsize=(8, 5))
    stats.probplot(data, dist="norm", plot=ax)
    ax.set_title(f'QQ-Plot of {column}')
    ax.grid(alpha=0.3)
    
    return fig, code, sampled_flag


def run_normality_test_by_group(df, num_col, cat_col, test_type):
    """
    Executes the normality test using efficient groupby operations.
    Returns: (results_df, generated_code_string, sampled_flag)
    """
    results = []
    sampled_flag = False
    
    code = "import pandas as pd\nfrom scipy import stats\n\n"
    code += "# Assuming 'df' is your loaded pandas DataFrame\n"
    code += "results = []\n\n"
    code += f"for cat, group_data in df.groupby('{cat_col}'):\n"
    code += f"    data = group_data['{num_col}'].dropna()\n"
    code += "    n_total = len(data)\n\n"
    code += "    if n_total == 0:\n"
    code += "        continue\n\n"

    # Iterate efficiently over groups avoiding boolean masks
    for cat, group_data in df.groupby(cat_col):
        data = group_data[num_col].dropna()
        n_total = len(data)
        
        if n_total == 0:
            continue
            
        n_used = n_total
        status = "Tested"
        
        if test_type == "Shapiro–Wilk":
            if n_total < 3:
                results.append({'Group': cat, 'Statistic': None, 'p-value': None, 'N_total': n_total, 'N_used': n_used, 'Status': '(error) N < 3'})
                continue
            if n_total > 5000:
                data = data.sample(n=5000, random_state=42)
                n_used = 5000
                sampled_flag = True
        elif test_type == "D’Agostino–Pearson" and n_total < 8:
            results.append({'Group': cat, 'Statistic': None, 'p-value': None, 'N_total': n_total, 'N_used': n_used, 'Status': '(error) N < 8'})
            continue

        if test_type == "Shapiro–Wilk":
            stat, p = stats.shapiro(data)
        elif test_type == "D’Agostino–Pearson":
            stat, p = stats.normaltest(data)
        elif test_type == "Kolmogorov–Smirnov":
            z_data = (data - data.mean()) / data.std(ddof=0)
            stat, p = stats.kstest(z_data, 'norm')
        else: # Anderson-Darling
            result = stats.anderson(data, dist='norm', method='interpolate')
            stat, p = result.statistic, result.pvalue
            
        results.append({
            'Group': cat, 
            'Statistic': stat, 
            'p-value': p, 
            'N_total': n_total, 
            'N_used': n_used, 
            'Status': status
        })

    # Building the reproducible code for the student
    if test_type == "Shapiro–Wilk":
        code += "    if n_total < 3:\n        continue\n"
        code += "    if n_total > 5000:\n"
        code += "        data = data.sample(n=5000, random_state=42)\n"
        code += "    stat, p = stats.shapiro(data)\n"
    elif test_type == "D’Agostino–Pearson":
        code += "    if n_total < 8:\n        continue\n"
        code += "    stat, p = stats.normaltest(data)\n"
    elif test_type == "Kolmogorov–Smirnov":
        code += "    z_data = (data - data.mean()) / data.std(ddof=0)\n"
        code += "    stat, p = stats.kstest(z_data, 'norm')\n"
    else:
        code += "    res = stats.anderson(data, dist='norm', method='interpolate')\n"
        code += "    stat, p = res.statistic, res.pvalue\n"
        
    code += "    results.append({'Group': cat, 'Statistic': stat, 'p-value': p})\n\n"
    code += "results_df = pd.DataFrame(results)\nprint(results_df)"

    return pd.DataFrame(results), code, sampled_flag