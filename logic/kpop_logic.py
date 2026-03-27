import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pingouin as pg
import itertools
from scipy.stats import bootstrap
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scipy.stats import studentized_range

def perform_bartlett(df, num_col, cat_col):
    """Perform Bartlett's test for equal variances grouped by a categorical column."""
    categories = df[cat_col].dropna().unique()
    data_arrays = [df[df[cat_col] == cat][num_col].dropna() for cat in categories]
    stat, p_value = stats.bartlett(*data_arrays)

    code = "from scipy import stats\n\n"
    code += f"# Identify unique categories in '{cat_col}'\n"
    code += f"categories = df['{cat_col}'].dropna().unique()\n\n"
    code += f"# Create a list of arrays, one for each category\n"
    code += f"data_arrays = [df[df['{cat_col}'] == cat]['{num_col}'].dropna() for cat in categories]\n\n"
    code += f"# Perform Bartlett's test by unpacking the list\n"
    code += "stat, p_value = stats.bartlett(*data_arrays)\n\n"
    code += "print(f'Bartlett statistic: {stat:.4f}')\n"
    code += "print(f'p-value: {p_value:.4f}')\n"
    
    return stat, p_value, code

def perform_levene(df,num_col, cat_col):
    """Perform Levene's test for equal variances grouped by a categorical column."""
    categories = df[cat_col].dropna().unique()
    data_arrays = [df[df[cat_col] == cat][num_col].dropna() for cat in categories]
    stat, p_value = stats.levene(*data_arrays, center='median')
    
    code = "from scipy import stats\n\n"
    code += f"# Identify unique categories in '{cat_col}'\n"
    code += f"categories = df['{cat_col}'].dropna().unique()\n\n"
    code += f"# Create a list of arrays, one for each category\n"
    code += f"data_arrays = [df[df['{cat_col}'] == cat]['{num_col}'].dropna() for cat in categories]\n\n"
    code += f"# Perform Levene's test by unpacking the list\n"
    code += "stat, p_value = stats.levene(*data_arrays, center='median')\n\n"
    code += "print(f'Levene statistic: {stat:.4f}')\n"
    code += "print(f'p-value: {p_value:.4f}')\n"

    return stat, p_value, code

def perform_oneway_anova(df, num_col, cat_col):
    """Perform one-way ANOVA grouped by a categorical column."""
    categories = df[cat_col].dropna().unique()
    data_arrays = [df[df[cat_col] == cat][num_col].dropna() for cat in categories]
    stat, p_value = stats.f_oneway(*data_arrays)
    
    code = "from scipy import stats\n\n"
    code += f"# Identify unique categories in '{cat_col}'\n"
    code += f"categories = df['{cat_col}'].dropna().unique()\n\n"
    code += f"# Create a list of arrays, one for each category\n"
    code += f"data_arrays = [df[df['{cat_col}'] == cat]['{num_col}'].dropna() for cat in categories]\n\n"
    code += f"# Perform One-Way ANOVA by unpacking the list\n"
    code += "stat, p_value = stats.f_oneway(*data_arrays)\n\n"
    code += "print(f'ANOVA F-statistic: {stat:.4f}')\n"
    code += "print(f'p-value: {p_value:.4f}')\n"
    
    return stat, p_value, code

def perform_pairwise_tukeyhsd(df, num_col, cat_col, confidence=0.95):
    """
    Perform Tukey's HSD test for multiple comparisons grouped by a categorical column 
    and generates a Forest Plot.
    Assumes equal variances.
    Returns the tukey result object, the matplotlib figure, and the generated Python code.
    """
    alpha = 1 - confidence
    
    data = df[[num_col, cat_col]].dropna()
    tukey_result = pairwise_tukeyhsd(endog=data[num_col], groups=data[cat_col], alpha=alpha)
    tukey_df = pd.DataFrame(
        data=tukey_result._results_table.data[1:],
        columns=tukey_result._results_table.data[0]
    )
    
    comparisons = tukey_df["group1"].astype(str) + " - " + tukey_df["group2"].astype(str)
    mean_diff = tukey_df["meandiff"].astype(float)
    ci_low = tukey_df["lower"].astype(float)
    ci_high = tukey_df["upper"].astype(float)

    left_dist = mean_diff - ci_low
    right_dist = ci_high - mean_diff

    fig, ax = plt.subplots(figsize=(8, len(comparisons) * 0.8 + 1.5))
    
    # Null Hypothesis Line (H0 = 0 difference)
    ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='H₀ (No difference)')
    
    ax.errorbar(x=mean_diff, y=comparisons,
                xerr=[left_dist, right_dist], 
                fmt='o', 
                color='#1f77b4', 
                markersize=8, 
                capsize=5, 
                linewidth=2,
                label=f'Estimate ({int(confidence*100)}% CI)')

    ax.set_xlabel("Mean Difference", fontsize=11)
    ax.set_title("Tukey HSD Pairwise Comparisons", fontsize=14, pad=15)
    ax.grid(axis='x', linestyle=':', alpha=0.7)
    ax.legend(loc='best')
    fig.tight_layout()

    code = "import pandas as pd\n"
    code += "import matplotlib.pyplot as plt\n"
    code += "from statsmodels.stats.multicomp import pairwise_tukeyhsd\n\n"
    
    code += f"# Drop rows with missing values in either column\n"
    code += f"data = df[['{num_col}', '{cat_col}']].dropna()\n\n"
    
    code += f"# Perform Tukey HSD test\n"
    code += f"tukey_result = pairwise_tukeyhsd(endog=data['{num_col}'], groups=data['{cat_col}'], alpha={alpha:.2f})\n"
    code += "print(tukey_result)\n\n"
    
    code += "# Extract data for plotting\n"
    code += "tukey_df = pd.DataFrame(data=tukey_result._results_table.data[1:], columns=tukey_result._results_table.data[0])\n"
    code += "comparisons = tukey_df['group1'].astype(str) + ' - ' + tukey_df['group2'].astype(str)\n"
    code += "mean_diff = tukey_df['meandiff'].astype(float)\n"
    code += "ci_low = tukey_df['lower'].astype(float)\n"
    code += "ci_high = tukey_df['upper'].astype(float)\n\n"
    
    code += "left_dist = mean_diff - ci_low\n"
    code += "right_dist = ci_high - mean_diff\n\n"
    
    code += f"fig, ax = plt.subplots(figsize=(8, {len(comparisons) * 0.8 + 1.5:.1f}))\n"
    code += "ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='H₀ (No difference)')\n"
    code += f"ax.errorbar(x=mean_diff, y=comparisons, xerr=[left_dist, right_dist], fmt='o', color='#1f77b4', markersize=8, capsize=5, linewidth=2, label='Estimate ({int(confidence*100)}% CI)')\n"
    code += "ax.set_xlabel('Mean Difference', fontsize=11)\n"
    code += "ax.set_title('Tukey HSD Pairwise Comparisons', fontsize=14, pad=15)\n"
    code += "ax.grid(axis='x', linestyle=':', alpha=0.7)\n"
    code += "ax.legend(loc='best')\n"
    code += "fig.tight_layout()\n"
    code += "plt.show()\n"

    return tukey_result, fig, code

def perform_pairwise_gameshowell(df, num_col, cat_col, confidence=0.95):
    """
    Perform Games-Howell post-hoc test for multiple comparisons using Pingouin,
    grouped by a categorical column.
    Recalculates the Confidence Interval directly from standard error and df.
    """
    # 1. Clean data: drop rows where either the numeric or categorical value is missing
    data = df[[num_col, cat_col]].dropna()
    k = data[cat_col].nunique()
    
    # 2. Perform Games-Howell test
    gh_result = pg.pairwise_gameshowell(data=data, dv=num_col, between=cat_col)
    
    # 3. Calculate Confidence Intervals
    alpha = 1 - confidence
    q_critical = studentized_range.ppf(1 - alpha, k, gh_result['df'])
    margin_of_error = q_critical * (gh_result['se'] / np.sqrt(2))
    
    gh_result['lower'] = gh_result['diff'] - margin_of_error
    gh_result['upper'] = gh_result['diff'] + margin_of_error
    gh_result['is_significant'] = gh_result['pval'] < alpha

    comparisons = gh_result["A"].astype(str) + " - " + gh_result["B"].astype(str)
    mean_diff = gh_result["diff"].astype(float)
    ci_low = gh_result["lower"].astype(float)
    ci_high = gh_result["upper"].astype(float)

    left_dist = mean_diff - ci_low
    right_dist = ci_high - mean_diff

    fig, ax = plt.subplots(figsize=(8, len(comparisons) * 0.8 + 1.5))
    ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='H₀ (No difference)')
    
    ax.errorbar(x=mean_diff, y=comparisons, xerr=[left_dist, right_dist], 
                fmt='o', color='#ff7f0e', markersize=8, capsize=5, linewidth=2,
                label=f'Estimate ({int(confidence*100)}% CI)')

    ax.set_xlabel("Mean Difference", fontsize=11)
    ax.set_title("Games-Howell Pairwise Comparisons", fontsize=14, pad=15)
    ax.grid(axis='x', linestyle=':', alpha=0.7)
    ax.legend(loc='best')
    fig.tight_layout()

    code = "import pandas as pd\n"
    code += "import numpy as np\n"
    code += "import matplotlib.pyplot as plt\n"
    code += "import pingouin as pg\n"
    code += "from scipy.stats import studentized_range\n\n"
    
    code += f"# Drop rows with missing values\n"
    code += f"data = df[['{num_col}', '{cat_col}']].dropna()\n"
    code += f"k = data['{cat_col}'].nunique()\n\n"
    
    code += f"# Perform Games-Howell test\n"
    code += f"gh_result = pg.pairwise_gameshowell(data=data, dv='{num_col}', between='{cat_col}')\n\n"
    
    code += f"# Custom Confidence Interval calculation ({int(confidence*100)}%)\n"
    code += f"alpha = {1 - confidence:.3f}\n"
    code += "q_critical = studentized_range.ppf(1 - alpha, k, gh_result['df'])\n"
    code += "margin_of_error = q_critical * (gh_result['se'] / np.sqrt(2))\n\n"
    
    code += "gh_result['lower'] = gh_result['diff'] - margin_of_error\n"
    code += "gh_result['upper'] = gh_result['diff'] + margin_of_error\n"
    code += "gh_result['is_significant'] = gh_result['pval'] < alpha\n\n"
    
    code += "display_columns = ['A', 'B', 'diff', 'pval', 'lower', 'upper', 'is_significant']\n"
    code += "print(gh_result[display_columns].round(4))\n\n"
    
    code += "# Plotting logic\n"
    code += "comparisons = gh_result['A'].astype(str) + ' - ' + gh_result['B'].astype(str)\n"
    code += "left_dist = gh_result['diff'] - gh_result['lower']\n"
    code += "right_dist = gh_result['upper'] - gh_result['diff']\n\n"
    
    code += f"fig, ax = plt.subplots(figsize=(8, {len(comparisons) * 0.8 + 1.5:.1f}))\n"
    code += "ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='H₀ (No difference)')\n"
    code += f"ax.errorbar(x=gh_result['diff'], y=comparisons, xerr=[left_dist, right_dist], fmt='o', color='#ff7f0e', markersize=8, capsize=5, linewidth=2, label='Estimate ({int(confidence*100)}% CI)')\n"
    code += "ax.set_xlabel('Mean Difference', fontsize=11)\n"
    code += "ax.set_title('Games-Howell Pairwise Comparisons', fontsize=14, pad=15)\n"
    code += "ax.grid(axis='x', linestyle=':', alpha=0.7)\n"
    code += "ax.legend(loc='best')\n"
    code += "fig.tight_layout()\n"
    code += "plt.show()\n"

    display_cols = ['A', 'B', 'diff', 'se', 'df', 'pval', 'lower', 'upper', 'is_significant']
    
    return gh_result[display_cols], fig, code

def perform_krustall_wallis(df, num_col, cat_col):
    """Perform Kruskal-Wallis H-test for independent samples grouped by a categorical column."""

    categories = df[cat_col].dropna().unique()
    data_arrays = [df[df[cat_col] == cat][num_col].dropna() for cat in categories]
    stat, p_value = stats.kruskal(*data_arrays)
    
    code = "from scipy import stats\n\n"
    code += f"# Identify unique categories in '{cat_col}'\n"
    code += f"categories = df['{cat_col}'].dropna().unique()\n\n"
    code += f"# Create a list of arrays, one for each category\n"
    code += f"data_arrays = [df[df['{cat_col}'] == cat]['{num_col}'].dropna() for cat in categories]\n\n"
    code += f"# Perform Kruskal-Wallis H-test by unpacking the list\n"
    code += "stat, p_value = stats.kruskal(*data_arrays)\n\n"
    code += "print(f'Kruskal-Wallis statistic: {stat:.4f}')\n"
    code += "print(f'p-value: {p_value:.4f}')\n"
    
    return stat, p_value, code


def perform_bootstrap_pairwise_median(df, num_col, cat_col, confidence=0.95, n_resamples=2000):
    """
    Perform pairwise bootstrap confidence intervals for the difference of medians.
    Handles k populations based on a categorical column by calculating all pairwise combinations.
    Returns the results dataframe, the matplotlib figure, and the generated Python code.
    """
    results = []
    categories = df[cat_col].dropna().unique()

    for cat1, cat2 in itertools.combinations(categories, 2):
        arr1 = df[df[cat_col] == cat1][num_col].dropna().to_numpy()
        arr2 = df[df[cat_col] == cat2][num_col].dropna().to_numpy()
        
        # Protect against empty arrays if data is missing
        if len(arr1) == 0 or len(arr2) == 0:
            continue
            
        # Calculate dataset median difference        
        diff = np.median(arr1) - np.median(arr2)
        
        # Bootstrap
        ci_obj = bootstrap(
            (arr1, arr2), 
            lambda x, y, axis=-1: np.median(x, axis=axis) - np.median(y, axis=axis), 
            confidence_level=confidence, 
            n_resamples=n_resamples, 
            method='percentile'
        ).confidence_interval
        
        # Save results
        results.append({
            'A': cat1,
            'B': cat2,
            'diff': diff,
            'lower': ci_obj.low,
            'upper': ci_obj.high
        })
        
    # Convert to DataFrame
    res_df = pd.DataFrame(results)

    # 3. Prepare data for plotting
    comparisons = res_df['A'].astype(str) + " - " + res_df['B'].astype(str)
    median_diff = res_df['diff'].astype(float)
    ci_low = res_df['lower'].astype(float)
    ci_high = res_df['upper'].astype(float)

    left_dist = median_diff - ci_low
    right_dist = ci_high - median_diff

    # 4. Create Figure and Plot
    fig, ax = plt.subplots(figsize=(8, len(comparisons) * 0.8 + 1.5))
    
    # Null Hypothesis Line (H0 = 0 difference)
    ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='H₀ (No difference)')
    
    ax.errorbar(x=median_diff, y=comparisons,
                xerr=[left_dist, right_dist], 
                fmt='o', 
                color='#2ca02c',
                markersize=8, 
                capsize=5, 
                linewidth=2,
                label=f'Estimate ({int(confidence*100)}% CI)')

    ax.set_xlabel("Difference of Medians", fontsize=11)
    ax.set_title("Pairwise Bootstrap Comparisons (Medians)", fontsize=14, pad=15)
    ax.grid(axis='x', linestyle=':', alpha=0.7)
    ax.legend(loc='best')
    fig.tight_layout()

    # 5. Code Generation
    code = "import numpy as np\n"
    code += "import pandas as pd\n"
    code += "import matplotlib.pyplot as plt\n"
    code += "import itertools\n"
    code += "from scipy.stats import bootstrap\n\n"
    
    code += f"categories = df['{cat_col}'].dropna().unique()\n"
    code += "results = []\n\n"
    
    code += "# Calculate pairwise bootstrap intervals\n"
    code += "for cat1, cat2 in itertools.combinations(categories, 2):\n"
    code += f"    arr1 = df[df['{cat_col}'] == cat1]['{num_col}'].dropna().to_numpy()\n"
    code += f"    arr2 = df[df['{cat_col}'] == cat2]['{num_col}'].dropna().to_numpy()\n\n"
    code += "    if len(arr1) == 0 or len(arr2) == 0: continue\n\n"
    code += "    diff = np.median(arr1) - np.median(arr2)\n"
    code += "    ci_obj = bootstrap(\n"
    code += "        (arr1, arr2), \n"
    code += "        lambda x, y, axis=-1: np.median(x, axis=axis) - np.median(y, axis=axis), \n"
    code += f"        confidence_level={confidence}, \n"
    code += f"        n_resamples={n_resamples}, \n"
    code += "        method='percentile'\n"
    code += "    ).confidence_interval\n\n"
    code += "    results.append({'A': cat1, 'B': cat2, 'diff': diff, 'lower': ci_obj.low, 'upper': ci_obj.high})\n\n"
    
    code += "res_df = pd.DataFrame(results)\n"
    code += "print(res_df.round(4))\n\n"
    
    code += "# Plotting logic\n"
    code += "comparisons = res_df['A'].astype(str) + ' - ' + res_df['B'].astype(str)\n"
    code += "median_diff = res_df['diff'].astype(float)\n"
    code += "left_dist = median_diff - res_df['lower'].astype(float)\n"
    code += "right_dist = res_df['upper'].astype(float) - median_diff\n\n"
    
    code += f"fig, ax = plt.subplots(figsize=(8, {len(comparisons) * 0.8 + 1.5:.1f}))\n"
    code += "ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='H₀ (No difference)')\n"
    code += f"ax.errorbar(x=median_diff, y=comparisons, xerr=[left_dist, right_dist], fmt='o', color='#2ca02c', markersize=8, capsize=5, linewidth=2, label='Estimate ({int(confidence*100)}% CI)')\n"
    code += "ax.set_xlabel('Difference of Medians', fontsize=11)\n"
    code += "ax.set_title('Pairwise Bootstrap Comparisons (Medians)', fontsize=14, pad=15)\n"
    code += "ax.grid(axis='x', linestyle=':', alpha=0.7)\n"
    code += "ax.legend(loc='best')\n"
    code += "fig.tight_layout()\n"
    code += "plt.show()\n"

    return res_df, fig, code