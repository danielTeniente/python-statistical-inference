import numpy as np
from scipy.stats import f
from scipy import stats
import matplotlib.pyplot as plt

BOOTSTRAP_SAFETY_LIMIT = 5000

def apply_safety_sampling(data1, data2, var1_name="data1", var2_name="data2", limit=BOOTSTRAP_SAFETY_LIMIT):
    """
    Applies proportional stratified sampling if the total number of observations
    exceeds the safety limit. Returns the sampled series, a boolean flag, and 
    the reproducible python code string.
    """
    n1, n2 = len(data1), len(data2)
    total_len = n1 + n2
    is_sampled = False
    
    if total_len > limit:
        is_sampled = True
        frac = limit / total_len
        data1_boot = data1.sample(frac=frac)
        data2_boot = data2.sample(frac=frac)
        
        sampling_code = (
            f"# Safety undersampling for bootstrap CI (limit = {limit})\n"
            f"total_len = len({var1_name}) + len({var2_name})\n"
            f"frac = {limit} / total_len\n"
            f"{var1_name}_ci = {var1_name}.sample(frac=frac)\n"
            f"{var2_name}_ci = {var2_name}.sample(frac=frac)\n"
        )
    else:
        data1_boot = data1
        data2_boot = data2
        sampling_code = (
            f"{var1_name}_ci, {var2_name}_ci = {var1_name}, {var2_name}\n"
        )
        
    return data1_boot, data2_boot, is_sampled, sampling_code


# ===========================================================================
# STATISTICAL TESTS
# ===========================================================================

def perform_ftest(df, num_col, cat_col, alternative='two-sided', confidence=0.95):
    """
    Perform F-test to compare variances of two independent samples 
    grouped by a categorical column.
    """
    df_clean = df.dropna(subset=[cat_col])
    categories = df_clean[cat_col].unique()
    group1_name, group2_name = categories[0], categories[1]

    mask = df_clean[cat_col] == group1_name

    data1 = df_clean.loc[mask, num_col].dropna()
    data2 = df_clean.loc[~mask, num_col].dropna()

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

    # --- Code generation (reproducible snippet) ---
    code = "import numpy as np\n"
    code += "from scipy.stats import f\n\n"
    
    code += f"# Drop rows with missing values in '{cat_col}' and filter groups\n"
    code += f"df_clean = df.dropna(subset=['{cat_col}'])\n"
    code += f"mask = df_clean['{cat_col}'] == '{group1_name}'\n"
    code += f"data1 = df_clean.loc[mask, '{num_col}'].dropna()\n"
    code += f"data2 = df_clean.loc[~mask, '{num_col}'].dropna()\n\n"
    
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
    """
    Perform Levene's test for equal variances and compute a bootstrap confidence
    interval for the ratio of variances.
    """
    df_clean = df.dropna(subset=[cat_col])
    categories = df_clean[cat_col].unique()
    group1_name, group2_name = categories[0], categories[1]

    mask = df_clean[cat_col] == group1_name
    data1_full = df_clean.loc[mask, num_col].dropna()
    data2_full = df_clean.loc[~mask, num_col].dropna()

    # Levene's test
    stat, p_value = stats.levene(data1_full, data2_full, center='median')

    # ---------- Bootstrap CI with Refactored Safety Undersampling ----------
    data1_boot, data2_boot, is_sampled, sampling_code = apply_safety_sampling(
        data1_full, data2_full, var1_name="data1", var2_name="data2"
    )

    boot_data = (data1_boot, data2_boot)
    ci = stats.bootstrap(
        boot_data,
        lambda x, y, axis=-1: np.var(x, ddof=1, axis=axis) / np.var(y, ddof=1, axis=axis),
        confidence_level=confidence,
        n_resamples=2000,
        method='percentile'
    ).confidence_interval
    ci_tuple = (ci.low, ci.high)

    # --- Code generation (reproducible snippet) ---
    code = "import numpy as np\n"
    code += "from scipy import stats\n\n"
    
    code += "# Data preparation\n"
    code += f"df_clean = df.dropna(subset=['{cat_col}'])\n"
    code += f"mask = df_clean['{cat_col}'] == '{group1_name}'\n"
    code += f"data1 = df_clean.loc[mask, '{num_col}'].dropna()\n"
    code += f"data2 = df_clean.loc[~mask, '{num_col}'].dropna()\n\n"
    
    code += "stat, p_value = stats.levene(data1, data2, center='median')\n"
    code += "print(f'Levene statistic: {stat:.4f}')\n"
    code += "print(f'p-value: {p_value:.4f}')\n\n"

    code += "# Bootstrap CI for ratio of variances\n"
    code += sampling_code
    code += f"boot_data = (data1_ci, data2_ci)\n"
    code += "ci = stats.bootstrap(\n"
    code += "    boot_data,\n"
    code += "    lambda x, y, axis=-1: np.var(x, ddof=1, axis=axis) / np.var(y, ddof=1, axis=axis),\n"
    code += f"    confidence_level={confidence},\n"
    code += "    n_resamples=2000,\n"
    code += "    method='percentile'\n"
    code += ").confidence_interval\n"
    code += "print(f'Confidence Interval for the ratio of variances: ({ci.low:.4f}, {ci.high:.4f})')\n"

    return stat, p_value, ci_tuple, code, is_sampled


def perform_ttest(df, num_col, cat_col, selected_categories, alternative='two-sided', confidence=0.95, equal_var=True):
    """Perform T-test using SciPy's built-in function, respecting the selection order."""
    df_clean = df.dropna(subset=[cat_col])
    
    # Extraemos explícitamente el orden seleccionado por el usuario
    group1, group2 = selected_categories[0], selected_categories[1]

    # Filtramos usando las variables directamente
    mask1 = df_clean[cat_col] == group1
    mask2 = df_clean[cat_col] == group2
    
    x1 = df_clean.loc[mask1, num_col].dropna()
    x2 = df_clean.loc[mask2, num_col].dropna()

    res = stats.ttest_ind(x1, x2, equal_var=equal_var, alternative=alternative)
    t_stat = res.statistic
    p_val = res.pvalue
    ci_obj = res.confidence_interval(confidence_level=confidence)
    ci_tuple = (ci_obj.low, ci_obj.high)

    # Actualizamos el código que se muestra al usuario para reflejar el orden
    code = "from scipy import stats\n\n"
    code += f"# Prepare data\n"
    code += f"df_clean = df.dropna(subset=['{cat_col}'])\n"
    code += f"group1, group2 = '{group1}', '{group2}'\n"
    code += f"x1 = df_clean.loc[df_clean['{cat_col}'] == group1, '{num_col}'].dropna()\n"
    code += f"x2 = df_clean.loc[df_clean['{cat_col}'] == group2, '{num_col}'].dropna()\n\n"
    code += f"res = stats.ttest_ind(x1, x2, equal_var={equal_var}, alternative='{alternative}')\n"
    code += "print(f'T-statistic: {res.statistic:.4f}')\n"
    code += "print(f'p-value: {res.pvalue:.4f}')\n\n"
    code += f"ci = res.confidence_interval(confidence_level={confidence})\n"
    code += "print(f'Confidence Interval for difference of means: ({ci.low:.4f}, {ci.high:.4f})')\n"

    return t_stat, p_val, ci_tuple, code

def perform_mannwhitney(df, num_col, cat_col, alternative='two-sided', confidence=0.95):
    """Perform Mann-Whitney U test with bootstrap CI for difference of medians."""
    df_clean = df.dropna(subset=[cat_col])
    categories = df_clean[cat_col].unique()
    group1, group2 = categories[0], categories[1]

    mask = df_clean[cat_col] == group1
    x1_full = df_clean.loc[mask, num_col].dropna()
    x2_full = df_clean.loc[~mask, num_col].dropna()

    # Mann-Whitney test (fast, full data)
    res = stats.mannwhitneyu(x1_full, x2_full, alternative=alternative)
    u_stat = res.statistic
    p_val = res.pvalue

    # ---------- Bootstrap CI with Refactored Safety Undersampling ----------
    x1_boot, x2_boot, is_sampled, sampling_code = apply_safety_sampling(
        x1_full, x2_full, var1_name="x1", var2_name="x2"
    )

    boot_data = (x1_boot, x2_boot)
    ci_obj = stats.bootstrap(
        boot_data,
        lambda x, y, axis=-1: np.median(x, axis=axis) - np.median(y, axis=axis),
        confidence_level=confidence,
        n_resamples=2000,
        method='percentile'
    ).confidence_interval
    ci_tuple = (ci_obj.low, ci_obj.high)

    code = "import numpy as np\nfrom scipy import stats\n\n"
    code += "# Data preparation\n"
    code += f"df_clean = df.dropna(subset=['{cat_col}'])\n"
    code += f"mask = df_clean['{cat_col}'] == '{group1}'\n"
    code += f"x1 = df_clean.loc[mask, '{num_col}'].dropna()\n"
    code += f"x2 = df_clean.loc[~mask, '{num_col}'].dropna()\n\n"
    code += f"res = stats.mannwhitneyu(x1, x2, alternative='{alternative}')\n"
    code += "print(f'U-statistic: {res.statistic:.4f}')\n"
    code += "print(f'p-value: {res.pvalue:.4f}')\n\n"

    code += sampling_code
    code += "boot_data = (x1_ci, x2_ci)\n"
    code += "ci_obj = stats.bootstrap(\n"
    code += "    boot_data,\n"
    code += "    lambda x, y, axis=-1: np.median(x, axis=axis) - np.median(y, axis=axis),\n"
    code += f"    confidence_level={confidence},\n"
    code += "    n_resamples=2000,\n"
    code += "    method='percentile'\n"
    code += ").confidence_interval\n"
    code += "print(f'Confidence Interval for difference of medians: ({ci_obj.low:.4f}, {ci_obj.high:.4f})')\n"

    return u_stat, p_val, ci_tuple, code, is_sampled

def get_sample_difference_in_means(df, num_col, cat_col):
    """Calculate the difference in means between two groups."""
    df_clean = df.dropna(subset=[cat_col])
    categories = df_clean[cat_col].unique()
    group1, group2 = categories[0], categories[1]

    mask = df_clean[cat_col] == group1
    mean1 = df_clean.loc[mask, num_col].dropna().mean()
    mean2 = df_clean.loc[~mask, num_col].dropna().mean()

    code = f"df_clean = df.dropna(subset=['{cat_col}'])\n"
    code += f"mask = df_clean['{cat_col}'] == '{group1}'\n"
    code += f"mean1 = df_clean.loc[mask, '{num_col}'].dropna().mean()\n"
    code += f"mean2 = df_clean.loc[~mask, '{num_col}'].dropna().mean()\n"
    code += "diff_means = mean1 - mean2\n"
    code += "print(f'Difference in means: {diff_means:.4f}')\n"

    return mean1 - mean2, code

def get_sample_difference_in_medians(df, num_col, cat_col):
    """Calculate the difference in medians between two groups."""
    df_clean = df.dropna(subset=[cat_col])
    categories = df_clean[cat_col].unique()
    group1, group2 = categories[0], categories[1]

    mask = df_clean[cat_col] == group1
    median1 = df_clean.loc[mask, num_col].dropna().median()
    median2 = df_clean.loc[~mask, num_col].dropna().median()

    code = f"df_clean = df.dropna(subset=['{cat_col}'])\n"
    code += f"mask = df_clean['{cat_col}'] == '{group1}'\n"
    code += f"median1 = df_clean.loc[mask, '{num_col}'].dropna().median()\n"
    code += f"median2 = df_clean.loc[~mask, '{num_col}'].dropna().median()\n"
    code += "diff_medians = median1 - median2\n"
    code += "print(f'Difference in medians: {diff_medians:.4f}')\n"

    return median1 - median2, code

def get_sample_variance_ratio(df, num_col, cat_col):
    """Calculate the ratio of variances between two groups."""
    df_clean = df.dropna(subset=[cat_col])
    categories = df_clean[cat_col].unique()
    group1, group2 = categories[0], categories[1]

    mask = df_clean[cat_col] == group1
    # Usamos ddof=1 para obtener la varianza muestral (es el default en pandas, pero es buena práctica explicitarlo)
    var1 = df_clean.loc[mask, num_col].dropna().var(ddof=1)
    var2 = df_clean.loc[~mask, num_col].dropna().var(ddof=1)

    # Prevenir divisiones por cero en caso de que la varianza del grupo 2 sea 0
    if var2 == 0:
        ratio = float('inf') 
    else:
        ratio = var1 / var2

    code = f"df_clean = df.dropna(subset=['{cat_col}'])\n"
    code += f"mask = df_clean['{cat_col}'] == '{group1}'\n"
    code += f"var1 = df_clean.loc[mask, '{num_col}'].dropna().var(ddof=1)\n"
    code += f"var2 = df_clean.loc[~mask, '{num_col}'].dropna().var(ddof=1)\n"
    code += "variance_ratio = var1 / var2\n"
    code += "print(f'Variance ratio ({group1} / {group2}): {variance_ratio:.4f}')\n"

    return ratio, code

def plot_confidence_interval(low, high, estimated_value, title="Confidence Interval",
    x_label="", y_label="", H0=0):
    """
    Generates a Forest Plot for a confidence interval.
    """
    left_dist = estimated_value - low
    right_dist = high - estimated_value

    fig, ax = plt.subplots(figsize=(8, 3))

    ax.axvline(x=H0, color='red', linestyle='--', linewidth=2, label='H₀ (Equal)')
    
    ax.errorbar(x=estimated_value, y=0,
                xerr=[[left_dist], [right_dist]], 
                fmt='o', 
                color='#1f77b4', 
                markersize=10, 
                capsize=8, 
                linewidth=2,
                label='Estimate (CI)')

    ax.set_yticks([0])
    ax.set_yticklabels([y_label], fontsize=12)
    ax.set_ylim(-0.5, 0.5)
    ax.set_xlabel(x_label, fontsize=11)
    ax.set_title(title, fontsize=14, pad=15)
    
    ax.grid(axis='x', linestyle=':', alpha=0.7)
    ax.legend(loc='upper right')
    fig.tight_layout()

    # Code snippet
    code = "import matplotlib.pyplot as plt\n\n"
    code += f"estimated_value = {estimated_value}\n"
    code += f"ci_lower = {low}\n"
    code += f"ci_upper = {high}\n"
    code += "left_dist = estimated_value - ci_lower\n"
    code += "right_dist = ci_upper - estimated_value\n\n"
    code += "fig, ax = plt.subplots(figsize=(8, 3))\n"
    code += f"ax.axvline(x={H0}, color='red', linestyle='--', linewidth=2, label='H₀ (Equal)')\n"
    code += "ax.errorbar(x=estimated_value, y=0, xerr=[[left_dist], [right_dist]], fmt='o', color='#1f77b4', markersize=10, capsize=8, linewidth=2, label='Estimate (CI)')\n"
    code += "ax.set_yticks([0])\n"
    code += f"ax.set_yticklabels(['{y_label}'], fontsize=12)\n"
    code += "ax.set_ylim(-0.5, 0.5)\n"
    code += f"ax.set_xlabel('{x_label}', fontsize=11)\n"
    code += f"ax.set_title('{title}', fontsize=14, pad=15)\n"
    code += "ax.grid(axis='x', linestyle=':', alpha=0.7)\n"
    code += "ax.legend(loc='upper right')\n"
    code += "fig.tight_layout()\n"
    code += "plt.show()"

    return fig, code