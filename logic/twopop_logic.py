import numpy as np
from scipy.stats import f
from scipy import stats
import matplotlib.pyplot as plt


def perform_ftest(df, col1, col2, alternative='two-sided', confidence=0.95):
    """Perform F-test to compare variances of two independent samples."""
    f_stat = np.var(df[col1], ddof=1) / np.var(df[col2], ddof=1)
    ddof1, ddof2 = len(df[col1])-1, len(df[col2])-1

    p_two = 2 * min(f.cdf(f_stat, ddof1, ddof2), 1 - f.cdf(f_stat, ddof1, ddof2))
    p_less = f.cdf(f_stat, ddof1, ddof2)
    p_greater = 1 - f.cdf(f_stat, ddof1, ddof2)

    p_values = {
        "two-sided": p_two,
        "less": p_less,
        "greater": p_greater
    }

    code = "from scipy.stats import f\n"
    code += f"f_stat = np.var(df['{col1}'], ddof=1) / np.var(df['{col2}'], ddof=1)\n"
    code += f"ddof1, ddof2 = len(df['{col1}'])-1, len(df['{col2}'])-1\n"
    if alternative == "two-sided":
        code += "p_value = 2 * min(f.cdf(f_stat, ddof1, ddof2), 1 - f.cdf(f_stat, ddof1, ddof2))\n"
    if alternative == "less":
        code += "p_value = f.cdf(f_stat, ddof1, ddof2)\n"
    if alternative == "greater":
        code += "p_value = 1 - f.cdf(f_stat, ddof1, ddof2)\n"

    code += "print(f'F-statistic: {f_stat:.4f}')\n"
    code += "print(f'p-value: {p_value:.4f}')\n"

    # Confidence interval
    alpha = 1 - confidence
    lower_bound = f_stat / f.ppf(1 - alpha/2, ddof1, ddof2)
    upper_bound = f_stat / f.ppf(alpha/2, ddof1, ddof2)
    ci = (lower_bound, upper_bound)
    code += f"\n# Confidence interval for the ratio of variances\n"
    code += f"alpha = 1 - {confidence}\n"
    code += f"lower_bound = f_stat / f.ppf(1 - alpha/2, ddof1, ddof2)\n"
    code += f"upper_bound = f_stat / f.ppf(alpha/2, ddof1, ddof2)\n"
    code += "print(f'Confidence Interval for the ratio of variances: ({lower_bound:.4f}, {upper_bound:.4f})')\n"

    return f_stat, p_values[alternative], ci, code

def perform_levene(df, col1, col2, confidence=0.95):
    "Perform Levene's test for equal variances."
    stat, p_value = stats.levene(df[col1], df[col2], center='median')
    code = "from scipy import stats\n"
    code += f"stat, p_value = stats.levene(df['{col1}'], df['{col2}'], center='median')\n"
    code += "print(f'Levene statistic: {stat:.4f}')\n"
    code += "print(f'p-value: {p_value:.4f}')\n"

    # robust confidence interval for the ratio of variances using bootstrap
    boostrap_data = (np.array(df[col1]), np.array(df[col2]))
    ci = stats.bootstrap(
        boostrap_data, 
        lambda x, y, axis=-1: np.var(x, ddof=1, axis=axis) / np.var(y, ddof=1, axis=axis), 
        confidence_level=confidence, 
        n_resamples=2000, 
        method='percentile'
    ).confidence_interval
    
    code += f"\n# Bootstrap confidence interval for the ratio of variances\n"
    code += f"import numpy as np\n"
    code += f"boostrap_data = (np.array(df['{col1}']), np.array(df['{col2}']))\n"
    code += f"ci = stats.bootstrap(boostrap_data, lambda x, y, axis=-1: np.var(x, ddof=1, axis=axis) / np.var(y, ddof=1, axis=axis), confidence_level={confidence}, n_resamples=2000, method='percentile').confidence_interval\n"
    code += "print(f'Confidence Interval for the ratio of variances: ({ci.low:.4f}, {ci.high:.4f})')\n"

    return stat, p_value, ci, code

def plot_confidence_interval(low, high, estimated_value, title="Confidence Interval", x_label="", y_label=""):
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
    ax.axvline(x=1.0, color='red', linestyle='--', linewidth=2, label='H₀ (Equal)')
    
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
    code += "ax.axvline(x=1.0, color='red', linestyle='--', linewidth=2, label='H₀ (Equal)')\n"
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

def perform_ttest(df, col1, col2, alternative='two-sided', confidence=0.95, equal_var=True):
    """Perform T-test using SciPy's built-in function."""
    
    x1, x2 = df[col1], df[col2]
    res = stats.ttest_ind(x1, x2, equal_var=equal_var, alternative=alternative)
    
    t_stat = res.statistic
    p_val = res.pvalue
    
    # Extraemos el intervalo de confianza (requiere SciPy 1.10.0 o superior)
    ci_obj = res.confidence_interval(confidence_level=confidence)
    ci = (ci_obj.low, ci_obj.high)

    # ----- Generación de Código -----
    code = "from scipy.stats import ttest_ind\n\n"
    code += f"x1, x2 = df['{col1}'], df['{col2}']\n"
    code += f"res = ttest_ind(x1, x2, equal_var={equal_var}, alternative='{alternative}')\n\n"
    
    code += "print(f'T-statistic: {res.statistic:.4f}')\n"
    code += "print(f'p-value: {res.pvalue:.4f}')\n\n"
    
    code += f"# Confidence interval for the difference of means\n"
    code += f"ci = res.confidence_interval(confidence_level={confidence})\n"
    code += "print(f'Confidence Interval: ({ci.low:.4f}, {ci.high:.4f})')\n"

    return t_stat, p_val, ci, code