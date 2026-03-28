import numpy as np
from scipy.stats import f
from scipy import stats
import pandas as pd

def run_normality_test_ovr(df, num_col, cat_col, target_cat, test_type):
    """
    Executes the selected normality test for a One-vs-Rest approach.
    Returns: (results_df, generated_code_string)
    """
    # 1. Crear los dos grupos: El elegido y el resto
    data_target = df[df[cat_col] == target_cat][num_col].dropna()
    data_rest = df[df[cat_col] != target_cat][num_col].dropna()
    
    groups_to_test = [
        (str(target_cat), data_target),
        ("Rest", data_rest)
    ]
    
    results = []

    # 2. Iterar sobre los dos grupos
    for group_name, data in groups_to_test:
        n_size = len(data)
        
        # Guardrails: minimum sample size requirements
        if n_size < 3 and test_type == "Shapiro–Wilk":
            results.append({'Group': group_name, 'Statistic': None, 'p-value': None, 'N': n_size, 'Status': '(error) N < 3'})
            continue
        if n_size < 8 and test_type == "D’Agostino–Pearson":
            results.append({'Group': group_name, 'Statistic': None, 'p-value': None, 'N': n_size, 'Status': '(error) N < 8'})
            continue
        if n_size == 0:
            results.append({'Group': group_name, 'Statistic': None, 'p-value': None, 'N': n_size, 'Status': '(error) Empty group'})
            continue

        stat, p, status = None, None, "test performed"
        
        if test_type == "Shapiro–Wilk":
            stat, p = stats.shapiro(data)
        elif test_type == "D’Agostino–Pearson":
            stat, p = stats.normaltest(data)
        elif test_type == "Kolmogorov–Smirnov":
            # Standardize before testing
            z_data = (data - data.mean()) / data.std(ddof=0)
            stat, p = stats.kstest(z_data, 'norm')
        else: # Anderson-Darling
            result = stats.anderson(data, dist='norm', method='interpolate')
            stat = result.statistic
            p = result.pvalue
            
        results.append({
            'Group': group_name, 
            'Statistic': stat, 
            'p-value': p, 
            'N': n_size, 
            'Status': status
        })

    results_df = pd.DataFrame(results)

    # 3. Generación de Código
    code = "import pandas as pd\n"
    code += "from scipy import stats\n\n"
    
    code += f"# Filter data for One-vs-Rest: '{target_cat}' vs 'Rest'\n"
    code += f"data_target = df[df['{cat_col}'] == '{target_cat}']['{num_col}'].dropna()\n"
    code += f"data_rest = df[df['{cat_col}'] != '{target_cat}']['{num_col}'].dropna()\n\n"
    
    code += f"groups_to_test = [('{target_cat}', data_target), ('Rest', data_rest)]\n"
    code += "results = []\n\n"
    
    code += "for group_name, data in groups_to_test:\n"
    
    if test_type == "Shapiro–Wilk":
        code += "    if len(data) >= 3:\n"
        code += "        stat, p = stats.shapiro(data)\n"
        code += "        results.append({'Group': group_name, 'Statistic': stat, 'p-value': p})\n"
    elif test_type == "D’Agostino–Pearson":
        code += "    if len(data) >= 8:\n"
        code += "        stat, p = stats.normaltest(data)\n"
        code += "        results.append({'Group': group_name, 'Statistic': stat, 'p-value': p})\n"
    elif test_type == "Kolmogorov–Smirnov":
        code += "    if len(data) > 0:\n"
        code += "        z_data = (data - data.mean()) / data.std(ddof=0)\n"
        code += "        stat, p = stats.kstest(z_data, 'norm')\n"
        code += "        results.append({'Group': group_name, 'Statistic': stat, 'p-value': p})\n"
    else:
        code += "    if len(data) > 0:\n"
        code += "        res = stats.anderson(data, dist='norm', method='interpolate')\n"
        code += "        results.append({'Group': group_name, 'Statistic': res.statistic, 'p-value': res.pvalue})\n"
        
    code += "\nresults_df = pd.DataFrame(results)\n"
    code += "print(results_df)\n"

    return results_df, code

def perform_ftest_ovr(df, num_col, cat_col, target_cat, alternative='two-sided', confidence=0.95):
    """
    Perform F-test to compare variances for a One-vs-Rest approach.
    Compares the 'target_cat' against all other categories combined.
    """
    group1_name = str(target_cat)
    group2_name = "Rest"
    
    # Filtrar datos: La categoría elegida vs el resto (!=)
    data1 = df[df[cat_col] == target_cat][num_col].dropna()
    data2 = df[df[cat_col] != target_cat][num_col].dropna()

    f_stat = np.var(data1, ddof=1) / np.var(data2, ddof=1)
    ddof1, ddof2 = len(data1) - 1, len(data2) - 1

    p_two = 2 * min(f.cdf(f_stat, ddof1, ddof2), 1 - f.cdf(f_stat, ddof1, ddof2))
    p_less = f.cdf(f_stat, ddof1, ddof2)
    p_greater = 1 - f.cdf(f_stat, ddof1, ddof2)

    p_values = {
        "two-sided": p_two,
        "less": p_less,
        "greater": p_greater
    }

    alpha = 1 - confidence
    lower_bound = f_stat / f.ppf(1 - alpha/2, ddof1, ddof2)
    upper_bound = f_stat / f.ppf(alpha/2, ddof1, ddof2)
    ci = (lower_bound, upper_bound)

    # Generación de código reflejando el filtro One-vs-Rest
    code = "import numpy as np\n"
    code += "from scipy.stats import f\n\n"
    
    code += f"# Filter data for One-vs-Rest: '{group1_name}' vs '{group2_name}'\n"
    code += f"data1 = df[df['{cat_col}'] == '{target_cat}']['{num_col}'].dropna()\n"
    code += f"data2 = df[df['{cat_col}'] != '{target_cat}']['{num_col}'].dropna()\n\n"
    
    code += "f_stat = np.var(data1, ddof=1) / np.var(data2, ddof=1)\n"
    code += "ddof1, ddof2 = len(data1)-1, len(data2)-1\n\n"
    
    if alternative == "two-sided":
        code += "p_value = 2 * min(f.cdf(f_stat, ddof1, ddof2), 1 - f.cdf(f_stat, ddof1, ddof2))\n"
    elif alternative == "less":
        code += "p_value = f.cdf(f_stat, ddof1, ddof2)\n"
    elif alternative == "greater":
        code += "p_value = 1 - f.cdf(f_stat, ddof1, ddof2)\n"

    code += "print(f'F-statistic: {f_stat:.4f}')\n"
    code += "print(f'p-value: {p_value:.4f}')\n"

    code += f"\n# Confidence interval for the ratio of variances\n"
    code += f"alpha = 1 - {confidence}\n"
    code += "lower_bound = f_stat / f.ppf(1 - alpha/2, ddof1, ddof2)\n"
    code += "upper_bound = f_stat / f.ppf(alpha/2, ddof1, ddof2)\n"
    code += "print(f'Confidence Interval for the ratio of variances: ({lower_bound:.4f}, {upper_bound:.4f})')\n"

    return f_stat, p_values[alternative], ci, code

def perform_levene_ovr(df, num_col, cat_col, target_cat, confidence=0.95):
    """
    Perform Levene's test for equal variances for a One-vs-Rest approach.
    """
    group1_name = str(target_cat)
    group2_name = "Rest"
    
    # Filtrar datos: La categoría elegida vs el resto (!=)
    data1 = df[df[cat_col] == target_cat][num_col].dropna()
    data2 = df[df[cat_col] != target_cat][num_col].dropna()

    stat, p_value = stats.levene(data1, data2, center='median')
    
    code = "import numpy as np\n"
    code += "from scipy import stats\n\n"
    
    code += f"# Filter data for One-vs-Rest: '{group1_name}' vs '{group2_name}'\n"
    code += f"data1 = df[df['{cat_col}'] == '{target_cat}']['{num_col}'].dropna()\n"
    code += f"data2 = df[df['{cat_col}'] != '{target_cat}']['{num_col}'].dropna()\n\n"
    
    code += "stat, p_value = stats.levene(data1, data2, center='median')\n"
    code += "print(f'Levene statistic: {stat:.4f}')\n"
    code += "print(f'p-value: {p_value:.4f}')\n"

    # Robust confidence interval for the ratio of variances using bootstrap
    boostrap_data = (np.array(data1), np.array(data2))
    
    ci = stats.bootstrap(
        boostrap_data, 
        lambda x, y, axis=-1: np.var(x, ddof=1, axis=axis) / np.var(y, ddof=1, axis=axis), 
        confidence_level=confidence, 
        n_resamples=2000, 
        method='percentile'
    ).confidence_interval
    ci_tuple = (ci.low, ci.high)
    
    code += f"\n# Bootstrap confidence interval for the ratio of variances\n"
    code += f"boostrap_data = (np.array(data1), np.array(data2))\n"
    code += f"ci = stats.bootstrap(\n"
    code += f"    boostrap_data, \n"
    code += f"    lambda x, y, axis=-1: np.var(x, ddof=1, axis=axis) / np.var(y, ddof=1, axis=axis), \n"
    code += f"    confidence_level={confidence}, \n"
    code += f"    n_resamples=2000, \n"
    code += f"    method='percentile'\n"
    code += f").confidence_interval\n"
    code += "print(f'Confidence Interval for the ratio of variances: ({ci.low:.4f}, {ci.high:.4f})')\n"

    return stat, p_value, ci_tuple, code

def perform_ttest_ovr(df, num_col, cat_col, target_cat, alternative='two-sided', confidence=0.95, equal_var=True):
    """Perform T-test using SciPy's built-in function, grouped by a categorical column."""
    
    group1 = str(target_cat)
    group2 = "Rest"
    
    x1 = df[df[cat_col] == target_cat][num_col].dropna()
    x2 = df[df[cat_col] != target_cat][num_col].dropna()
    
    # 2. Perform test
    res = stats.ttest_ind(x1, x2, equal_var=equal_var, alternative=alternative)
    
    t_stat = res.statistic
    p_val = res.pvalue
    
    ci_obj = res.confidence_interval(confidence_level=confidence)
    ci = (ci_obj.low, ci_obj.high)

    code = "from scipy import stats\n\n"
    code += f"#  '{group1}' vs '{group2}'\n"
    code += f"x1 = df[df['{cat_col}'] == '{group1}']['{num_col}'].dropna()\n"
    code += f"x2 = df[df['{cat_col}'] != '{group1}']['{num_col}'].dropna()\n\n"
    
    code += f"res = stats.ttest_ind(x1, x2, equal_var={equal_var}, alternative='{alternative}')\n\n"
    
    code += "print(f'T-statistic: {res.statistic:.4f}')\n"
    code += "print(f'p-value: {res.pvalue:.4f}')\n\n"
    
    code += f"# Confidence interval for the difference of means\n"
    code += f"ci = res.confidence_interval(confidence_level={confidence})\n"
    code += "print(f'Confidence Interval: ({ci.low:.4f}, {ci.high:.4f})')\n"

    return t_stat, p_val, ci, code


def perform_mannwhitney_ovr(df, num_col, cat_col, target_cat, alternative='two-sided', confidence=0.95):
    """Perform Mann-Whitney U test using SciPy's built-in function, grouped by a categorical column."""
    
    group1 = str(target_cat)
    group2 = "Rest"
    
    x1 = df[df[cat_col] == group1][num_col].dropna()
    x2 = df[df[cat_col] != group1][num_col].dropna()
    
    res = stats.mannwhitneyu(x1, x2, alternative=alternative)
    
    u_stat = res.statistic
    p_val = res.pvalue

    boostrap_data = (np.array(x1), np.array(x2))
    ci_obj = stats.bootstrap(
        boostrap_data, 
        lambda x, y, axis=-1: np.median(x, axis=axis) - np.median(y, axis=axis), 
        confidence_level=confidence, 
        n_resamples=2000, 
        method='percentile'
    ).confidence_interval
    
    # Convert to standard tuple for the UI
    ci = (ci_obj.low, ci_obj.high)

    code = "import numpy as np\n"
    code += "from scipy import stats\n\n"
    
    code += f"#  '{group1}' vs '{group2}'\n"
    code += f"x1 = df[df['{cat_col}'] == '{group1}']['{num_col}'].dropna()\n"
    code += f"x2 = df[df['{cat_col}'] != '{group1}']['{num_col}'].dropna()\n\n"
    
    code += f"res = stats.mannwhitneyu(x1, x2, alternative='{alternative}')\n\n"
    
    code += "print(f'U-statistic: {res.statistic:.4f}')\n"
    code += "print(f'p-value: {res.pvalue:.4f}')\n\n"
    
    code += f"# Bootstrap confidence interval for the difference of medians\n"
    code += f"boostrap_data = (np.array(x1), np.array(x2))\n"
    code += f"ci_obj = stats.bootstrap(\n"
    code += f"    boostrap_data, \n"
    code += f"    lambda x, y, axis=-1: np.median(x, axis=axis) - np.median(y, axis=axis), \n"
    code += f"    confidence_level={confidence}, \n"
    code += f"    n_resamples=2000, \n"
    code += f"    method='percentile'\n"
    code += f").confidence_interval\n"
    code += "print(f'Confidence Interval for the difference of medians: ({ci_obj.low:.4f}, {ci_obj.high:.4f})')\n"

    return u_stat, p_val, ci, code

def get_sample_difference_in_means_ovr(df, num_col, cat_col, target_cat):
    """Calculate the difference in means between two groups defined by a categorical column."""
    group1 = str(target_cat)

    mean1 = df[df[cat_col] == group1][num_col].dropna().mean()
    mean2 = df[df[cat_col] != group1][num_col].dropna().mean()

    code = f"mean1 = df[df['{cat_col}'] == '{group1}']['{num_col}'].dropna().mean()\n"
    code += f"mean2 = df[df['{cat_col}'] != '{group1}']['{num_col}'].dropna().mean()\n"
    code += f"difference = mean1 - mean2\n"

    return mean1 - mean2, code

def get_sample_difference_in_medians_ovr(df, num_col, cat_col, target_cat):
    """Calculate the difference in medians between two groups defined by a categorical column."""
    group1 = str(target_cat)

    median1 = df[df[cat_col] == group1][num_col].dropna().median()
    median2 = df[df[cat_col] != group1][num_col].dropna().median()
    
    code = f"median1 = df[df['{cat_col}'] == '{group1}']['{num_col}'].dropna().median()\n"
    code += f"median2 = df[df['{cat_col}'] != '{group1}']['{num_col}'].dropna().median()\n"
    code += f"difference = median1 - median2\n"

    return median1 - median2, code