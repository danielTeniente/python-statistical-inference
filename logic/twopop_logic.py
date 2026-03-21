import numpy as np
from scipy.stats import f
from scipy import stats
import matplotlib.pyplot as plt


def perform_ftest(df, num_col, cat_col, alternative='two-sided', confidence=0.95):
    """
    Perform F-test to compare variances of two independent samples 
    grouped by a categorical column.
    """
    categories = df[cat_col].dropna().unique()
    
    group1_name, group2_name = categories[0], categories[1]
    
    data1 = df[df[cat_col] == group1_name][num_col].dropna()
    data2 = df[df[cat_col] == group2_name][num_col].dropna()

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

    code = "import numpy as np\n"
    code += "from scipy.stats import f\n\n"
    
    code += f"# Filter data by categories '{group1_name}' and '{group2_name}'\n"
    code += f"data1 = df[df['{cat_col}'] == '{group1_name}']['{num_col}'].dropna()\n"
    code += f"data2 = df[df['{cat_col}'] == '{group2_name}']['{num_col}'].dropna()\n\n"
    
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

def perform_levene(df, num_col, cat_col, confidence=0.95):
    """Perform Levene's test for equal variances using a numeric and a categorical column."""
    
    categories = df[cat_col].dropna().unique()
    group1_name, group2_name = categories[0], categories[1]
    
    data1 = df[df[cat_col] == group1_name][num_col].dropna()
    data2 = df[df[cat_col] == group2_name][num_col].dropna()

    stat, p_value = stats.levene(data1, data2, center='median')
    
    code = "import numpy as np\n"
    code += "from scipy import stats\n\n"
    
    code += f"# Filter data by categories '{group1_name}' and '{group2_name}'\n"
    code += f"data1 = df[df['{cat_col}'] == '{group1_name}']['{num_col}'].dropna()\n"
    code += f"data2 = df[df['{cat_col}'] == '{group2_name}']['{num_col}'].dropna()\n\n"
    
    code += "stat, p_value = stats.levene(data1, data2, center='median')\n"
    code += "print(f'Levene statistic: {stat:.4f}')\n"
    code += "print(f'p-value: {p_value:.4f}')\n"

    # 5. Robust confidence interval for the ratio of variances using bootstrap
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

def plot_confidence_interval(low, high, estimated_value, title="Confidence Interval", 
    x_label="", y_label="", H0=0):
    """
    Generates a Forest Plot for a confidence interval (focusing on H0 = 1).
    Returns the figure and the equivalent Python code as a string.
    """
    # 1. Calculate the distances for the error bars
    left_dist = estimated_value - low
    right_dist = high - estimated_value

    # 2. Create the figure and axes
    fig, ax = plt.subplots(figsize=(8, 3))

    # 3. Draw the Null Hypothesis (H0) line and the error bar
    ax.axvline(x=H0, color='red', linestyle='--', linewidth=2, label='H₀ (Equal)')
    
    ax.errorbar(x=estimated_value, y=0,
                xerr=[[left_dist], [right_dist]], 
                fmt='o', 
                color='#1f77b4', 
                markersize=10, 
                capsize=8, 
                linewidth=2,
                label='Estimate (95% CI)')

    # 4. Aesthetic formatting of the axes
    ax.set_yticks([0])
    ax.set_yticklabels([y_label], fontsize=12)
    ax.set_ylim(-0.5, 0.5)
    ax.set_xlabel(x_label, fontsize=11)
    ax.set_title(title, fontsize=14, pad=15)
    
    ax.grid(axis='x', linestyle=':', alpha=0.7)
    ax.legend(loc='upper right')
    fig.tight_layout()

    # 5. Generate the equivalent Python code as a string
    code = "import matplotlib.pyplot as plt\n\n"
    code += f"estimated_value = {estimated_value}\n"
    code += f"ci_lower = {low}\n"
    code += f"ci_upper = {high}\n"
    code += "left_dist = estimated_value - ci_lower\n"
    code += "right_dist = ci_upper - estimated_value\n\n"
    code += "fig, ax = plt.subplots(figsize=(8, 3))\n"
    code += f"ax.axvline(x={H0}, color='red', linestyle='--', linewidth=2, label='H₀ (Equal)')\n"
    code += "ax.errorbar(x=estimated_value, y=0, xerr=[[left_dist], [right_dist]], fmt='o', color='#1f77b4', markersize=10, capsize=8, linewidth=2, label='Estimate (95% CI)')\n"
    code += "ax.set_yticks([0])\n"
    code += f"ax.set_yticklabels('{y_label}', fontsize=12)\n"
    code += "ax.set_ylim(-0.5, 0.5)\n"
    code += f"ax.set_xlabel('{x_label}', fontsize=11)\n"
    code += f"ax.set_title('{title}', fontsize=14, pad=15)\n"
    code += "ax.grid(axis='x', linestyle=':', alpha=0.7)\n"
    code += "ax.legend(loc='upper right')\n"
    code += "fig.tight_layout()\n"
    code += "plt.show()"

    # 6. Return both elements
    return fig, code

def perform_ttest(df, num_col, cat_col, alternative='two-sided', confidence=0.95, equal_var=True):
    """Perform T-test using SciPy's built-in function, grouped by a categorical column."""
    
    # 1. Extract categories and filter data
    categories = df[cat_col].dropna().unique()
    group1, group2 = categories[0], categories[1]
    
    x1 = df[df[cat_col] == group1][num_col].dropna()
    x2 = df[df[cat_col] == group2][num_col].dropna()
    
    # 2. Perform test
    res = stats.ttest_ind(x1, x2, equal_var=equal_var, alternative=alternative)
    
    t_stat = res.statistic
    p_val = res.pvalue
    
    ci_obj = res.confidence_interval(confidence_level=confidence)
    ci = (ci_obj.low, ci_obj.high)

    code = "from scipy import stats\n\n"
    code += f"# Filter data by categories '{group1}' and '{group2}'\n"
    code += f"x1 = df[df['{cat_col}'] == '{group1}']['{num_col}'].dropna()\n"
    code += f"x2 = df[df['{cat_col}'] == '{group2}']['{num_col}'].dropna()\n\n"
    
    code += f"res = stats.ttest_ind(x1, x2, equal_var={equal_var}, alternative='{alternative}')\n\n"
    
    code += "print(f'T-statistic: {res.statistic:.4f}')\n"
    code += "print(f'p-value: {res.pvalue:.4f}')\n\n"
    
    code += f"# Confidence interval for the difference of means\n"
    code += f"ci = res.confidence_interval(confidence_level={confidence})\n"
    code += "print(f'Confidence Interval: ({ci.low:.4f}, {ci.high:.4f})')\n"

    return t_stat, p_val, ci, code


def perform_mannwhitney(df, num_col, cat_col, alternative='two-sided', confidence=0.95):
    """Perform Mann-Whitney U test using SciPy's built-in function, grouped by a categorical column."""
    
    # 1. Extract categories and filter data
    categories = df[cat_col].dropna().unique()
        
    group1, group2 = categories[0], categories[1]
    
    x1 = df[df[cat_col] == group1][num_col].dropna()
    x2 = df[df[cat_col] == group2][num_col].dropna()
    
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
    
    code += f"# Filter data by categories '{group1}' and '{group2}'\n"
    code += f"x1 = df[df['{cat_col}'] == '{group1}']['{num_col}'].dropna()\n"
    code += f"x2 = df[df['{cat_col}'] == '{group2}']['{num_col}'].dropna()\n\n"
    
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

def get_sample_difference_in_means(df, num_col, cat_col):
    """Calculate the difference in means between two groups defined by a categorical column."""
    categories = df[cat_col].dropna().unique()
    group1, group2 = categories[0], categories[1]
    
    mean1 = df[df[cat_col] == group1][num_col].dropna().mean()
    mean2 = df[df[cat_col] == group2][num_col].dropna().mean()
    
    return mean1 - mean2

def get_sample_difference_in_medians(df, num_col, cat_col):
    """Calculate the difference in medians between two groups defined by a categorical column."""
    categories = df[cat_col].dropna().unique()
    group1, group2 = categories[0], categories[1]
    
    median1 = df[df[cat_col] == group1][num_col].dropna().median()
    median2 = df[df[cat_col] == group2][num_col].dropna().median()
    
    return median1 - median2