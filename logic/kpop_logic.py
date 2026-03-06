import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pingouin as pg
import itertools
from scipy.stats import bootstrap
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scipy.stats import studentized_range

def perform_bartlett(df, cols):
    """Perform Bartlett's test for equal variances on multiple columns."""
    data_arrays = [df[col].dropna() for col in cols]
    stat, p_value = stats.bartlett(*data_arrays)
    
    args_str = ", ".join([f"df['{col}'].dropna()" for col in cols])
    code = "from scipy import stats\n"
    code += f"stat, p_value = stats.bartlett({args_str})\n"
    code += "print(f'Bartlett statistic: {stat:.4f}')\n"
    code += "print(f'p-value: {p_value:.4f}')\n"
    
    return stat, p_value, code

def perform_levene(df, cols):
    """Perform Levene's test for equal variances on multiple columns."""
    data_arrays = [df[col].dropna() for col in cols]
    stat, p_value = stats.levene(*data_arrays, center='median')
    
    args_str = ", ".join([f"df['{col}'].dropna()" for col in cols])
    code = "from scipy import stats\n"
    code += f"stat, p_value = stats.levene({args_str}, center='median')\n"
    code += "print(f'Levene statistic: {stat:.4f}')\n"
    code += "print(f'p-value: {p_value:.4f}')\n"
    
    return stat, p_value, code

def perform_oneway_anova(df, cols, equal_var=True):
    """Perform one-way ANOVA on multiple columns."""
    data_arrays = [df[col].dropna() for col in cols]
    stat, p_value = stats.f_oneway(*data_arrays, equal_var=equal_var)
    
    args_str = ", ".join([f"df['{col}'].dropna()" for col in cols])
    code = "from scipy import stats\n"
    code += f"stat, p_value = stats.f_oneway({args_str}, equal_var={equal_var})\n"
    code += "print(f'ANOVA F-statistic: {stat:.4f}')\n"
    code += "print(f'p-value: {p_value:.4f}')\n"
    
    return stat, p_value, code

def perform_pairwise_tukeyhsd(df, cols, confidence=0.95):
    """
    Perform Tukey's HSD test for multiple comparisons and generates a Forest Plot.
    Assumes equal variances.
    Returns the tukey result object, the matplotlib figure, and the generated Python code.
    """
    alpha = 1 - confidence
    data = df[cols].dropna().melt(var_name='group', value_name='value')
    tukey_result = pairwise_tukeyhsd(endog=data['value'], groups=data['group'], alpha=alpha)
    
    # 1. Extraer los datos de la tabla de resultados de Tukey
    tukey_df = pd.DataFrame(
        data=tukey_result._results_table.data[1:],
        columns=tukey_result._results_table.data[0]
    )
    comparisons = tukey_df["group1"] + " - " + tukey_df["group2"]
    mean_diff = tukey_df["meandiff"].astype(float)
    ci_low = tukey_df["lower"].astype(float)
    ci_high = tukey_df["upper"].astype(float)

    # 2. Calcular las distancias para las barras de error
    left_dist = mean_diff - ci_low
    right_dist = ci_high - mean_diff

    # 3. Crear la figura y el gráfico (altura dinámica según número de comparaciones)
    fig, ax = plt.subplots(figsize=(8, len(comparisons) * 0.8 + 1.5))
    
    # Línea de Hipótesis Nula (H0 = 0 diferencia)
    ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='H₀ (No difference)')
    
    ax.errorbar(x=mean_diff, y=comparisons,
                xerr=[left_dist, right_dist], 
                fmt='o', 
                color='#1f77b4', 
                markersize=8, 
                capsize=5, 
                linewidth=2,
                label=f'Estimate ({int(confidence*100)}% CI)')

    # 4. Formato estético
    ax.set_xlabel("Mean Difference", fontsize=11)
    ax.set_title("Tukey HSD Pairwise Comparisons", fontsize=14, pad=15)
    ax.grid(axis='x', linestyle=':', alpha=0.7)
    ax.legend(loc='best')
    fig.tight_layout()

    cols_str = ", ".join([f"'{col}'" for col in cols]) 
    
    code = "import pandas as pd\n"
    code += "import matplotlib.pyplot as plt\n"
    code += "from statsmodels.stats.multicomp import pairwise_tukeyhsd\n\n"
    code += f"data = df[[{cols_str}]].dropna().melt(var_name='group', value_name='value')\n"
    code += f"tukey_result = pairwise_tukeyhsd(endog=data['value'], groups=data['group'], alpha={alpha:.2f})\n"
    code += "print(tukey_result)\n\n"
    
    code += "# Extract data for plotting\n"
    code += "tukey_df = pd.DataFrame(data=tukey_result._results_table.data[1:], columns=tukey_result._results_table.data[0])\n"
    code += "comparisons = tukey_df['group1'] + ' - ' + tukey_df['group2']\n"
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

    # 6. Retornar el objeto de resultados, la figura y el código
    return tukey_result, fig, code


def perform_pairwise_gameshowell(df, cols, confidence=0.95):
    """
    Perform Games-Howell post-hoc test for multiple comparisons using Pingouin.
    Recalculates the Confidence Interval directly from standard error and df.
    """
    data = df[cols].dropna().melt(var_name='group', value_name='value')
    k = data['group'].nunique()
    
    gh_result = pg.pairwise_gameshowell(data=data, dv='value', between='group')
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

    cols_str = ", ".join([f"'{col}'" for col in cols]) 
    
    code = "import pandas as pd\n"
    code += "import numpy as np\n"
    code += "import matplotlib.pyplot as plt\n"
    code += "import pingouin as pg\n"
    code += "from scipy.stats import studentized_range\n\n"
    
    code += f"data = df[[{cols_str}]].dropna().melt(var_name='group', value_name='value')\n"
    code += "k = data['group'].nunique()\n"
    code += "gh_result = pg.pairwise_gameshowell(data=data, dv='value', between='group')\n\n"
    
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

def perform_krustall_wallis(df, cols):
    """Perform Kruskal-Wallis H-test for independent samples."""
    data_arrays = [df[col].dropna() for col in cols]
    stat, p_value = stats.kruskal(*data_arrays)
    
    args_str = ", ".join([f"df['{col}'].dropna()" for col in cols])
    code = "from scipy import stats\n"
    code += f"stat, p_value = stats.kruskal({args_str})\n"
    code += "print(f'Kruskal-Wallis statistic: {stat:.4f}')\n"
    code += "print(f'p-value: {p_value:.4f}')\n"
    
    return stat, p_value, code

def perform_bootstrap_pairwise_median(df, cols, confidence=0.95, n_resamples=2000):
    """
    Perform pairwise bootstrap confidence intervals for the difference of medians.
    Handles k populations by calculating all pairwise combinations.
    Returns the results dataframe, the matplotlib figure, and the generated Python code.
    """
    results = []
    
    for col1, col2 in itertools.combinations(cols, 2):
        valid_data = df[[col1, col2]].dropna()
        arr1 = valid_data[col1].to_numpy()
        arr2 = valid_data[col2].to_numpy()
        # dataset median difference        
        diff = np.median(arr1) - np.median(arr2)
        ci = bootstrap(
            (arr1, arr2), 
            lambda x, y, axis=-1: np.median(x, axis=axis) - np.median(y, axis=axis), 
            confidence_level=confidence, 
            n_resamples=n_resamples, 
            method='percentile'
        ).confidence_interval
        
        # Guardar resultados
        results.append({
            'A': col1,
            'B': col2,
            'diff': diff,
            'lower': ci.low,
            'upper': ci.high
        })
        
    # Convertir a DataFrame
    res_df = pd.DataFrame(results)

    # 2. Preparar datos para el gráfico
    comparisons = res_df['A'].astype(str) + " - " + res_df['B'].astype(str)
    median_diff = res_df['diff'].astype(float)
    ci_low = res_df['lower'].astype(float)
    ci_high = res_df['upper'].astype(float)

    left_dist = median_diff - ci_low
    right_dist = ci_high - median_diff

    # 3. Crear la figura y el gráfico
    fig, ax = plt.subplots(figsize=(8, len(comparisons) * 0.8 + 1.5))
    
    # Línea de Hipótesis Nula (H0 = 0 diferencia)
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

    cols_str = str(cols)
    
    code = "import numpy as np\n"
    code += "import pandas as pd\n"
    code += "import matplotlib.pyplot as plt\n"
    code += "import itertools\n"
    code += "from scipy.stats import bootstrap\n\n"
    
    code += f"cols = {cols_str}\n"
    code += "results = []\n\n"
    
    code += "# Calculate pairwise bootstrap intervals\n"
    code += "for col1, col2 in itertools.combinations(cols, 2):\n"
    code += "    valid_data = df[[col1, col2]].dropna()\n"
    code += "    arr1, arr2 = valid_data[col1].to_numpy(), valid_data[col2].to_numpy()\n"
    code += "    diff = np.median(arr1) - np.median(arr2)\n"
    code += "    ci = bootstrap(\n"
    code += "        (arr1, arr2), \n"
    code += "        lambda x, y, axis=-1: np.median(x, axis=axis) - np.median(y, axis=axis), \n"
    code += f"        confidence_level={confidence}, \n"
    code += f"        n_resamples={n_resamples}, \n"
    code += "        method='percentile'\n"
    code += "    ).confidence_interval\n"
    code += "    results.append({'A': col1, 'B': col2, 'diff': diff, 'lower': ci.low, 'upper': ci.high})\n\n"
    
    code += "res_df = pd.DataFrame(results)\n"
    code += "print(res_df)\n\n"
    
    code += "# Plotting\n"
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

    # 6. Retornar el dataframe de resultados, la figura y el código
    return res_df, fig, code
