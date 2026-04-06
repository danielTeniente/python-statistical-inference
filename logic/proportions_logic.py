import pandas as pd
import numpy as np
from scipy.stats import binomtest
from statsmodels.stats.proportion import proportions_ztest, proportion_confint, confint_proportions_2indep

################
# One proportion tests and confidence intervals
################

def perform_one_proportion_binomial_test(df, selected_column, p0=0.5, alternative='two-sided', success_term=None):
    """
    Perform a one-proportion binomial test handling both binary numeric (1/0) and categorical data.

    Parameters:
    - df: DataFrame containing the data.
    - selected_column: Column name for the binary outcome variable.
    - p0: Hypothesized proportion under the null hypothesis (default is 0.5).
    - alternative: Type of test ('two-sided', 'greater', 'less').
    - success_term: The specific value considered a "success" (e.g., 'Good'). 
                    If None, it assumes the column contains 1s and 0s.

    Returns:
    - statistic: The proportion of successes.
    - p_value: p-value of the test.
    - code: String containing the Python code to reproduce the test.
    """
    
    if success_term is not None:
        successes = (df[selected_column] == success_term).sum()
    else:
        successes = df[selected_column].sum()
        
    trials = len(df)
    
    # Ejecutar el test
    result = binomtest(successes, trials, p=p0, alternative=alternative)
    statistic = result.statistic
    p_value = result.pvalue

    # Generación dinámica del código en formato string
    code = "from scipy.stats import binomtest\n\n"
    
    if success_term is not None:
        # Nos aseguramos de poner comillas si el término es un string para el código exportado
        term_str = f"'{success_term}'" if isinstance(success_term, str) else str(success_term)
        code += f"successes = (df['{selected_column}'] == {term_str}).sum()\n"
    else:
        code += f"successes = df['{selected_column}'].sum()\n"
        
    code += f"trials = len(df)\n"
    code += f"result = binomtest(successes, trials, p={p0}, alternative='{alternative}')\n\n"
    code += "statistic = result.statistic\n"
    code += "p_value = result.pvalue\n"
    code += "print(f'Statistic: {statistic:.4f}')\n"
    code += "print(f'P-value: {p_value:.4f}')\n"
    
    return statistic, p_value, code

def perform_one_proportion_ztest(df, selected_column, p0=0.5, alternative='two-sided', success_term=None):
    """
    Perform a one-proportion z-test.

    Parameters:
    - df: DataFrame containing the data.
    - selected_column: Column name for the binary outcome variable.
    - p0: Hypothesized proportion under the null hypothesis (default is 0.5).
    - alternative: Type of test ('two-sided', 'greater', 'less').

    Returns:
    - z-statistic and p-value of the test.
    """
    if success_term is not None:
        successes = (df[selected_column] == success_term).sum()
    else:
        successes = df[selected_column].sum()
    trials = len(df)
    statistic, p_value = proportions_ztest(successes, trials, value=p0, alternative=alternative)

    code = f"from statsmodels.stats.proportion import proportions_ztest"
    if success_term is not None:
        term_str = f"'{success_term}'" if isinstance(success_term, str) else str(success_term)
        code += f"\nsuccesses = (df['{selected_column}'] == {term_str}).sum()"
    else:
        code += f"\nsuccesses = df['{selected_column}'].sum()"
    code += f"\ntrials = len(df)"
    code += f"\nstatistic, p_value = proportions_ztest(successes, trials, value=p0, alternative=alternative)"
    code += "\nprint(f'Z-Statistic: {statistic:.4f}')"
    code += "\nprint(f'P-value: {p_value:.4f}')"
    return statistic, p_value, code

def get_one_proportion_interval(df, selected_column, method='wilson', confidence=0.95, success_term=None):
    """
    Calculate the confidence interval for a proportion using a specified method.

    Parameters:
    - df: DataFrame containing the data.
    - selected_column: Column name for the binary outcome variable.
    - method: The method to use ('beta' for Clopper-Pearson, 'wilson', 'normal', etc.).
    - confidence: Confidence level for the interval (default is 0.95).
    - success_term: The value in the column representing a 'success'. If None, sums the column.

    Returns:
    - Tuple: (lower_bound, upper_bound).
    - String: Python code to reproduce the calculation.
    """
    
    # 1. Mapeo de nombres para que el 'print' del código generado se vea profesional
    method_names = {
        'beta': 'Clopper-Pearson',
        'wilson': 'Wilson',
        'normal': 'Wald (Normal)',
        'agresti_coull': 'Agresti-Coull',
        'jeffreys': 'Jeffreys'
    }
    method_display_name = method_names.get(method, method.capitalize())

    if success_term is not None:
        successes = (df[selected_column] == success_term).sum()
    else:        
        successes = df[selected_column].sum()

    trials = len(df)
    
    lower, upper = proportion_confint(successes, trials, alpha=1-confidence, method=method)

    code = "from statsmodels.stats.proportion import proportion_confint\n\n"
    
    if success_term is not None:
        term_str = f"'{success_term}'" if isinstance(success_term, str) else str(success_term)
        code += f"successes = (df['{selected_column}'] == {term_str}).sum()\n"
    else:
        code += f"successes = df['{selected_column}'].sum()\n"
        
    code += "trials = len(df)\n"
    
    alpha_val = 1 - confidence
    code += f"lower, upper = proportion_confint(successes, trials, alpha={alpha_val:.4f}, method='{method}')\n\n"
    code += f"print(f'{method_display_name} Confidence Interval: ({{lower:.4f}}, {{upper:.4f}})')\n"
    
    return (lower, upper), code

#################
# Two proportion tests and confidence intervals
#################

# Two-sample z-test for proportions
def perform_two_proportion_ztest(df, group_col, outcome_col, alternative='two-sided', success_term=None):
    """
    Perform a two-proportion z-test.

    Parameters:
    - df: DataFrame containing the data.
    - group_col: Column name for the grouping variable.
    - outcome_col: Column name for the binary outcome variable.
    - p0: Hypothesized difference in proportions under the null hypothesis (default is 0.0).
    - alternative: Type of test ('two-sided', 'greater', 'less').

    Returns:
    - z-statistic and p-value of the test.
    """
    groups = df[group_col].unique()
    if len(groups) != 2:
        raise ValueError("The grouping column must have exactly two unique values.")

    group1_name, group2_name = groups

    group1_data = df[df[group_col] == group1_name][outcome_col]
    group2_data = df[df[group_col] == group2_name][outcome_col]

    if success_term is not None:
        successes = np.array([(group1_data == success_term).sum(), 
                     (group2_data == success_term).sum()])
    else:        
        # implicitly assumes that the outcome column is binary with 1s representing successes and 0s representing failures
        successes = np.array([group1_data.sum(), group2_data.sum()])

 
    trials = [len(group1_data), len(group2_data)]
    
    statistic, p_value = proportions_ztest(successes, trials, alternative=alternative)

    code = "import numpy as np\n"
    code += f"from statsmodels.stats.proportion import proportions_ztest\n"
    code += f"group1_data = df[df['{group_col}'] == '{group1_name}']['{outcome_col}']\n"
    code += f"group2_data = df[df['{group_col}'] == '{group2_name}']['{outcome_col}']\n"
    if success_term is not None:
        term_str = f"'{success_term}'" if isinstance(success_term, str) else str(success_term)
        code += f"successes = np.array([(group1_data == {term_str}).sum(), (group2_data == {term_str}).sum()])\n" 
    else:
        code += f"successes = np.array([group1_data.sum(), group2_data.sum()])\n"
    code += f"trials = [len(group1_data), len(group2_data)]\n"
    code += f"statistic, p_value = proportions_ztest(successes, trials, alternative='{alternative}')\n"
    code += f"print(f'Z-Statistic: {statistic:.4f}')\n"
    code += f"print(f'P-value: {p_value:.4f}')\n"
    
    return statistic, p_value, code

def get_two_proportion_confint(df, group_col, outcome_col, 
    method='newcomb', confidence=0.95, success_term=None):
    # newcomb, wald
    """Calculate the confidence interval for the difference in proportions between two groups."""
    
    groups = df[group_col].unique()
    if len(groups) != 2:
        raise ValueError("The grouping column must have exactly two unique values.")

    group1_name, group2_name = groups

    group1_data = df[df[group_col] == group1_name][outcome_col]
    group2_data = df[df[group_col] == group2_name][outcome_col]

    if success_term is not None:
        successes = np.array([(group1_data == success_term).sum(), 
                     (group2_data == success_term).sum()])
    else:        
        # implicitly assumes that the outcome column is binary with 1s representing successes and 0s representing failures
        successes = np.array([group1_data.sum(), group2_data.sum()])

 
    trials = [len(group1_data), len(group2_data)]

    alpha = 1 - confidence
    lower, upper = confint_proportions_2indep(
        successes[0],
        trials[0],
        successes[1],
        trials[1],
        method=method,
        compare='diff',
        alpha=alpha
    )

    code = "import numpy as np\n"
    code += f"from statsmodels.stats.proportion import confint_proportions_2indep\n"
    code += f"group1_data = df[df['{group_col}'] == '{group1_name}']['{outcome_col}']\n"
    code += f"group2_data = df[df['{group_col}'] == '{group2_name}']['{outcome_col}']\n"
    if success_term is not None:
        term_str = f"'{success_term}'" if isinstance(success_term, str) else str(success_term)
        code += f"successes = np.array([(group1_data == {term_str}).sum(), (group2_data == {term_str}).sum()])\n"
    else:
        code += f"successes = np.array([group1_data.sum(), group2_data.sum()])\n"
    code += f"trials = [len(group1_data), len(group2_data)]\n"
    code += f"alpha = 1 - {confidence}\n"
    code += f"lower, upper = confint_proportions_2indep(successes[0], trials[0], successes[1], trials[1], method={method!r}, compare='diff', alpha=alpha)\n"
    code += f"print(f'{method.title()} Confidence Interval for Difference in Proportions: ({lower:.4f}, {upper:.4f})')\n"

    return (lower, upper), code
