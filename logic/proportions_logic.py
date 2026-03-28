from scipy.stats import binomtest, fisher_exact
from statsmodels.stats.proportion import proportions_ztest, proportion_confint

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
    
    # Flujo 1 y 2: Determinar cómo contar los éxitos
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

def perform_one_proportion_ztest(df, selected_column, p0=0.5, alternative='two-sided'):
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
    successes = df[selected_column].sum()
    trials = len(df)
    statistic, p_value = proportions_ztest(successes, trials, value=p0, alternative=alternative)

    code = f"from statsmodels.stats.proportion import proportions_ztest"
    code += f"\n\nsuccesses = df['{selected_column}'].sum()"
    code += f"\ntrials = len(df)"
    code += f"\nstatistic, p_value = proportions_ztest(successes, trials, value=p0, alternative=alternative)"
    code += "\nprint(f'Z-Statistic: {statistic:.4f}')"
    code += "\nprint(f'P-value: {p_value:.4f}')"
    return statistic, p_value, code

def get_clopper_person_interval(df, selected_column, confidence=0.95):
    """
    Calculate the Clopper-Pearson confidence interval for a proportion.

    Parameters:
    - df: DataFrame containing the data.
    - selected_column: Column name for the binary outcome variable.
    - confidence: Confidence level for the interval (default is 0.95).

    Returns:
    - Tuple containing the lower and upper bounds of the confidence interval.
    """
    successes = df[selected_column].sum()
    trials = len(df)
    lower, upper = proportion_confint(successes, trials, alpha=1-confidence, method='beta')

    code = f"from statsmodels.stats.proportion import proportion_confint"
    code += f"\n\nsuccesses = df['{selected_column}'].sum()"
    code += f"\ntrials = len(df)"
    code += f"\nlower, upper = proportion_confint(successes, trials, alpha=1-{confidence}, method='beta')"
    code += "\nprint(f'Clopper-Pearson Confidence Interval: ({lower:.4f}, {upper:.4f})')"
    return (lower, upper), code

def get_wilson_interval(df, selected_column, confidence=0.95):
    """
    Calculate the Wilson confidence interval for a proportion.

    Parameters:
    - df: DataFrame containing the data.
    - selected_column: Column name for the binary outcome variable.
    - confidence: Confidence level for the interval (default is 0.95).

    Returns:
    - Tuple containing the lower and upper bounds of the confidence interval.
    """
    successes = df[selected_column].sum()
    trials = len(df)
    lower, upper = proportion_confint(successes, trials, alpha=1-confidence, method='wilson')

    code = f"from statsmodels.stats.proportion import proportion_confint"
    code += f"\n\nsuccesses = df['{selected_column}'].sum()"
    code += f"\ntrials = len(df)"
    code += f"\nlower, upper = proportion_confint(successes, trials, alpha=1-{confidence}, method='wilson')"
    code += "\nprint(f'Wilson Confidence Interval: ({lower:.4f}, {upper:.4f})')"
    return (lower, upper), code

#################
# Two proportion tests and confidence intervals
#################

def perform_two_prop_fisher_exact_test(df, col1, col2):
    """
    Perform Fisher's exact test for two proportions.

    Parameters:
    - df: DataFrame containing the data.
    - col1: First binary column name.
    - col2: Second binary column name.

    Returns:
    - p-value of the test.
    """
    a 
    _, p_value = fisher_exact(contingency_table)

    code = f"from scipy.stats import fisher_exact"
    code += f"\n\ncontingency_table = pd.crosstab(df['{col1}'], df['{col2}'])"
    code += f"\n_, p_value = fisher_exact(contingency_table)"
    code += "\nprint(f'P-value: {p_value:.4f}')"
    return p_value, code
