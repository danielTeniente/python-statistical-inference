import numpy as np
from scipy.stats import f
from scipy import stats
import pandas as pd

def run_normality_test_ovr(df, num_col, cat_col, target_cat, test_type):
    """
    Executes the selected normality test for a One-vs-Rest approach.
    Returns: (results_df, generated_code_string, sampled_flag)
    """
    sampled_flag = False
    
    # OPTIMIZATION: Single boolean mask calculation (O(N) instead of O(2N))
    mask = df[cat_col] == target_cat
    data_target = df.loc[mask, num_col].dropna()
    data_rest = df.loc[~mask, num_col].dropna()
    
    groups_to_test = [
        (str(target_cat), data_target),
        ("Rest", data_rest)
    ]
    
    results = []

    for group_name, data in groups_to_test:
        n_total = len(data)
        
        if n_total == 0:
            results.append({'Group': group_name, 'Statistic': None, 'p-value': None, 'N_total': 0, 'N_used': 0, 'Status': '(error) Empty group'})
            continue

        n_used = n_total
        status = "Tested"

        # Rule 3: Guardrails and Undersampling for Shapiro-Wilk
        if test_type == "Shapiro–Wilk":
            if n_total < 3:
                results.append({'Group': group_name, 'Statistic': None, 'p-value': None, 'N_total': n_total, 'N_used': n_used, 'Status': '(error) N < 3'})
                continue
            if n_total > 5000:
                data = data.sample(n=5000, random_state=42)
                n_used = 5000
                sampled_flag = True
        elif test_type == "D’Agostino–Pearson" and n_total < 8:
            results.append({'Group': group_name, 'Statistic': None, 'p-value': None, 'N_total': n_total, 'N_used': n_used, 'Status': '(error) N < 8'})
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
            stat = result.statistic
            p = result.pvalue
            
        results.append({
            'Group': group_name, 
            'Statistic': stat, 
            'p-value': p, 
            'N_total': n_total, 
            'N_used': n_used, 
            'Status': status
        })

    results_df = pd.DataFrame(results)

    # Educational Code Generation
    code = "import pandas as pd\nfrom scipy import stats\n\n"
    code += "# Assuming 'df' is your loaded pandas DataFrame\n"
    code += f"# Efficient One-vs-Rest filtering: '{target_cat}' vs 'Rest'\n"
    code += f"mask = df['{cat_col}'] == '{target_cat}'\n"
    code += f"data_target = df.loc[mask, '{num_col}'].dropna()\n"
    code += f"data_rest = df.loc[~mask, '{num_col}'].dropna()\n\n"
    
    code += f"groups_to_test = [('{target_cat}', data_target), ('Rest', data_rest)]\n"
    code += "results = []\n\n"
    code += "for group_name, data in groups_to_test:\n"
    
    if test_type == "Shapiro–Wilk":
        code += "    if len(data) < 3: continue\n"
        code += "    if len(data) > 5000:\n"
        code += "        data = data.sample(n=5000, random_state=42) # Undersampling for Shapiro-Wilk constraints\n"
        code += "    stat, p = stats.shapiro(data)\n"
        code += "    results.append({'Group': group_name, 'Statistic': stat, 'p-value': p})\n"
    elif test_type == "D’Agostino–Pearson":
        code += "    if len(data) < 8: continue\n"
        code += "    stat, p = stats.normaltest(data)\n"
        code += "    results.append({'Group': group_name, 'Statistic': stat, 'p-value': p})\n"
    elif test_type == "Kolmogorov–Smirnov":
        code += "    if len(data) > 0:\n"
        code += "        z_data = (data - data.mean()) / data.std(ddof=0)\n"
        code += "        stat, p = stats.kstest(z_data, 'norm')\n"
        code += "        results.append({'Group': group_name, 'Statistic': stat, 'p-value': p})\n"
    else:
        code += "    if len(data) > 0:\n"
        code += "        res = stats.anderson(data, dist='norm', method='interpolate')\n"
        code += "        results.append({'Group': group_name, 'Statistic': res.statistic, 'p-value': res.pvalue})\n"
        
    code += "\nresults_df = pd.DataFrame(results)\nprint(results_df)\n"

    return results_df, code, sampled_flag


def perform_ftest_ovr(df, num_col, cat_col, target_cat, alternative='two-sided', confidence=0.95):
    """
    Perform F-test to compare variances for a One-vs-Rest approach.
    """
    group1_name = str(target_cat)
    group2_name = "Rest"
    
    mask = df[cat_col] == target_cat
    data1 = df.loc[mask, num_col].dropna()
    data2 = df.loc[~mask, num_col].dropna()

    f_stat = np.var(data1, ddof=1) / np.var(data2, ddof=1)
    ddof1, ddof2 = len(data1) - 1, len(data2) - 1

    p_two = 2 * min(f.cdf(f_stat, ddof1, ddof2), 1 - f.cdf(f_stat, ddof1, ddof2))
    p_less = f.cdf(f_stat, ddof1, ddof2)
    p_greater = 1 - f.cdf(f_stat, ddof1, ddof2)

    p_values = {"two-sided": p_two, "less": p_less, "greater": p_greater}

    alpha = 1 - confidence
    lower_bound = f_stat / f.ppf(1 - alpha/2, ddof1, ddof2)
    upper_bound = f_stat / f.ppf(alpha/2, ddof1, ddof2)
    ci = (lower_bound, upper_bound)

    code = "import numpy as np\nfrom scipy.stats import f\n\n"
    code += f"# Efficient filtering for '{group1_name}' vs '{group2_name}'\n"
    code += f"mask = df['{cat_col}'] == '{target_cat}'\n"
    code += f"data1 = df.loc[mask, '{num_col}'].dropna()\n"
    code += f"data2 = df.loc[~mask, '{num_col}'].dropna()\n\n"
    
    code += "f_stat = np.var(data1, ddof=1) / np.var(data2, ddof=1)\n"
    code += "ddof1, ddof2 = len(data1)-1, len(data2)-1\n\n"
    
    if alternative == "two-sided":
        code += "p_value = 2 * min(f.cdf(f_stat, ddof1, ddof2), 1 - f.cdf(f_stat, ddof1, ddof2))\n"
    elif alternative == "less":
        code += "p_value = f.cdf(f_stat, ddof1, ddof2)\n"
    elif alternative == "greater":
        code += "p_value = 1 - f.cdf(f_stat, ddof1, ddof2)\n"

    code += "print(f'F-statistic: {f_stat:.4f}, p-value: {p_value:.4f}')\n"
    code += f"\n# Confidence interval for the ratio of variances\n"
    code += f"alpha = 1 - {confidence}\n"
    code += "lower_bound = f_stat / f.ppf(1 - alpha/2, ddof1, ddof2)\n"
    code += "upper_bound = f_stat / f.ppf(alpha/2, ddof1, ddof2)\n"
    code += "print(f'Confidence Interval (Ratio of Variances): ({lower_bound:.4f}, {upper_bound:.4f})')\n"

    return f_stat, p_values[alternative], ci, code

def perform_levene_ovr(df, num_col, cat_col, target_cat, confidence=0.95):
    """
    Perform Levene's test for equal variances for a One-vs-Rest approach.
    Returns: (stat, p_value, ci_tuple, code, sampled_flag)
    """

    sampled_flag = False
    
    mask = df[cat_col] == target_cat
    data1 = df.loc[mask, num_col].dropna()
    data2 = df.loc[~mask, num_col].dropna()

    # The test itself handles large N well
    stat, p_value = stats.levene(data1, data2, center='median')
    
    # Rule 3: Bootstrap will crash the cloud with 250k rows. We must undersample proportionally.
    b_data1, b_data2 = data1, data2
    total_len = len(data1) + len(data2)
    
    if total_len > 5000:
        sampled_flag = True
        frac = 5000 / total_len
        b_data1 = data1.sample(frac=frac, random_state=42)
        b_data2 = data2.sample(frac=frac, random_state=42)

    boostrap_data = (np.array(b_data1), np.array(b_data2))
    
    ci = stats.bootstrap(
        boostrap_data, 
        lambda x, y, axis=-1: np.var(x, ddof=1, axis=axis) / np.var(y, ddof=1, axis=axis), 
        confidence_level=confidence, 
        n_resamples=2000, 
        method='percentile'
    ).confidence_interval
    ci_tuple = (ci.low, ci.high)

    code = "import numpy as np\nfrom scipy import stats\n\n"
    code += f"mask = df['{cat_col}'] == '{target_cat}'\n"
    code += f"data1 = df.loc[mask, '{num_col}'].dropna()\n"
    code += f"data2 = df.loc[~mask, '{num_col}'].dropna()\n\n"
    code += "stat, p_value = stats.levene(data1, data2, center='median')\n"
    code += "print(f'Levene statistic: {stat:.4f}, p-value: {p_value:.4f}')\n"
    
    if sampled_flag:
        code += "\n# Bootstrapping 250k rows causes memory overflow. Sampling to 5000 points for CI calculation.\n"
        code += "total_len = len(data1) + len(data2)\n"
        code += "frac = 5000 / total_len\n"
        code += "b_data1 = data1.sample(frac=frac, random_state=42)\n"
        code += "b_data2 = data2.sample(frac=frac, random_state=42)\n"
    else:
        code += "\nb_data1, b_data2 = data1, data2\n"

    code += f"boostrap_data = (np.array(b_data1), np.array(b_data2))\n"
    code += f"ci = stats.bootstrap(\n    boostrap_data,\n"
    code += f"    lambda x, y, axis=-1: np.var(x, ddof=1, axis=axis) / np.var(y, ddof=1, axis=axis),\n"
    code += f"    confidence_level={confidence}, n_resamples=2000, method='percentile'\n).confidence_interval\n"
    code += "print(f'Bootstrap CI (Ratio of Variances): ({ci.low:.4f}, {ci.high:.4f})')\n"

    return stat, p_value, ci_tuple, code, sampled_flag


def perform_ttest_ovr(df, num_col, cat_col, target_cat, alternative='two-sided', confidence=0.95, equal_var=True):
    """Perform T-test for One-vs-Rest."""
    mask = df[cat_col] == target_cat
    x1 = df.loc[mask, num_col].dropna()
    x2 = df.loc[~mask, num_col].dropna()
    
    res = stats.ttest_ind(x1, x2, equal_var=equal_var, alternative=alternative)
    ci_obj = res.confidence_interval(confidence_level=confidence)

    code = "from scipy import stats\n\n"
    code += f"mask = df['{cat_col}'] == '{target_cat}'\n"
    code += f"x1 = df.loc[mask, '{num_col}'].dropna()\n"
    code += f"x2 = df.loc[~mask, '{num_col}'].dropna()\n\n"
    code += f"res = stats.ttest_ind(x1, x2, equal_var={equal_var}, alternative='{alternative}')\n"
    code += "print(f'T-statistic: {res.statistic:.4f}, p-value: {res.pvalue:.4f}')\n"
    code += f"ci = res.confidence_interval(confidence_level={confidence})\n"
    code += "print(f'Confidence Interval: ({ci.low:.4f}, {ci.high:.4f})')\n"

    return res.statistic, res.pvalue, (ci_obj.low, ci_obj.high), code


def perform_mannwhitney_ovr(df, num_col, cat_col, target_cat, alternative='two-sided', confidence=0.95):
    """
    Perform Mann-Whitney U test. 
    Returns: (u_stat, p_val, ci, code, sampled_flag)
    """
    sampled_flag = False
    
    mask = df[cat_col] == target_cat
    x1 = df.loc[mask, num_col].dropna()
    x2 = df.loc[~mask, num_col].dropna()
    
    res = stats.mannwhitneyu(x1, x2, alternative=alternative)

    # Rule 3: Bootstrap protection for large N
    b_x1, b_x2 = x1, x2
    total_len = len(x1) + len(x2)
    
    if total_len > 5000:
        sampled_flag = True
        frac = 5000 / total_len
        b_x1 = x1.sample(frac=frac, random_state=42)
        b_x2 = x2.sample(frac=frac, random_state=42)

    boostrap_data = (np.array(b_x1), np.array(b_x2))
    ci_obj = stats.bootstrap(
        boostrap_data, 
        lambda x, y, axis=-1: np.median(x, axis=axis) - np.median(y, axis=axis), 
        confidence_level=confidence, 
        n_resamples=2000, 
        method='percentile'
    ).confidence_interval

    code = "import numpy as np\nfrom scipy import stats\n\n"
    code += f"mask = df['{cat_col}'] == '{target_cat}'\n"
    code += f"x1 = df.loc[mask, '{num_col}'].dropna()\n"
    code += f"x2 = df.loc[~mask, '{num_col}'].dropna()\n\n"
    code += f"res = stats.mannwhitneyu(x1, x2, alternative='{alternative}')\n"
    code += "print(f'U-statistic: {res.statistic:.4f}, p-value: {res.pvalue:.4f}')\n"
    
    if sampled_flag:
        code += "\n# Undersampling applied exclusively for Bootstrap calculations due to complexity O(B*N)\n"
        code += "total_len = len(x1) + len(x2)\n"
        code += "b_x1 = x1.sample(frac=5000/total_len, random_state=42)\n"
        code += "b_x2 = x2.sample(frac=5000/total_len, random_state=42)\n"
    else:
        code += "\nb_x1, b_x2 = x1, x2\n"

    code += f"boostrap_data = (np.array(b_x1), np.array(b_x2))\n"
    code += f"ci_obj = stats.bootstrap(\n    boostrap_data,\n"
    code += f"    lambda x, y, axis=-1: np.median(x, axis=axis) - np.median(y, axis=axis),\n"
    code += f"    confidence_level={confidence}, n_resamples=2000, method='percentile'\n).confidence_interval\n"
    code += "print(f'Bootstrap CI (Diff of Medians): ({ci_obj.low:.4f}, {ci_obj.high:.4f})')\n"

    return res.statistic, res.pvalue, (ci_obj.low, ci_obj.high), code, sampled_flag


def get_sample_difference_in_means_ovr(df, num_col, cat_col, target_cat):
    """Calculate the difference in means between two groups."""
    mask = df[cat_col] == target_cat
    mean1 = df.loc[mask, num_col].mean()
    mean2 = df.loc[~mask, num_col].mean()

    code = f"mask = df['{cat_col}'] == '{target_cat}'\n"
    code += f"mean1 = df.loc[mask, '{num_col}'].mean()\n"
    code += f"mean2 = df.loc[~mask, '{num_col}'].mean()\n"
    code += f"difference = mean1 - mean2\n"

    return mean1 - mean2, code

def get_sample_difference_in_medians_ovr(df, num_col, cat_col, target_cat):
    """Calculate the difference in medians between two groups."""
    mask = df[cat_col] == target_cat
    median1 = df.loc[mask, num_col].median()
    median2 = df.loc[~mask, num_col].median()
    
    code = f"mask = df['{cat_col}'] == '{target_cat}'\n"
    code += f"median1 = df.loc[mask, '{num_col}'].median()\n"
    code += f"median2 = df.loc[~mask, '{num_col}'].median()\n"
    code += f"difference = median1 - median2\n"

    return median1 - median2, code