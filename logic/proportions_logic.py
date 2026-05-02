import pandas as pd
import numpy as np
from scipy.stats import binomtest
from statsmodels.stats.proportion import proportions_ztest, proportion_confint, confint_proportions_2indep

################
# One proportion tests and confidence intervals
################

def perform_one_proportion_binomial_test(df, selected_column, p0=0.5, alternative='two-sided', success_term=None):
    if success_term is not None:
        successes = (df[selected_column] == success_term).sum()
        term_str = f"'{success_term}'" if isinstance(success_term, str) else str(success_term)
        code_successes = f"successes = (df['{selected_column}'] == {term_str}).sum()"
    else:
        successes = df[selected_column].sum()
        code_successes = f"successes = df['{selected_column}'].sum()"
        
    trials = len(df)
    result = binomtest(successes, trials, p=p0, alternative=alternative)
    
    code = f"""from scipy.stats import binomtest

            {code_successes}
            trials = len(df)
            result = binomtest(successes, trials, p={p0}, alternative='{alternative}')

            print(f'Statistic: {{result.statistic:.4f}}')
            print(f'P-value: {{result.pvalue:.4f}}')
            """
    return result.statistic, result.pvalue, code

def perform_one_proportion_ztest(df, selected_column, p0=0.5, alternative='two-sided', success_term=None):
    if success_term is not None:
        successes = (df[selected_column] == success_term).sum()
        term_str = f"'{success_term}'" if isinstance(success_term, str) else str(success_term)
        code_successes = f"successes = (df['{selected_column}'] == {term_str}).sum()"
    else:
        successes = df[selected_column].sum()
        code_successes = f"successes = df['{selected_column}'].sum()"
        
    trials = len(df)
    statistic, p_value = proportions_ztest(successes, trials, value=p0, alternative=alternative)

    code = f"""from statsmodels.stats.proportion import proportions_ztest

            {code_successes}
            trials = len(df)
            statistic, p_value = proportions_ztest(successes, trials, value={p0}, alternative='{alternative}')

            print(f'Z-Statistic: {{statistic:.4f}}')
            print(f'P-value: {{p_value:.4f}}')
            """
    return statistic, p_value, code

def get_one_proportion_interval(df, selected_column, method='wilson', confidence=0.95, success_term=None):
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
        term_str = f"'{success_term}'" if isinstance(success_term, str) else str(success_term)
        code_successes = f"successes = (df['{selected_column}'] == {term_str}).sum()"
    else:        
        successes = df[selected_column].sum()
        code_successes = f"successes = df['{selected_column}'].sum()"

    trials = len(df)
    alpha_val = 1 - confidence
    lower, upper = proportion_confint(successes, trials, alpha=alpha_val, method=method)

    code = f"""from statsmodels.stats.proportion import proportion_confint

            {code_successes}
            trials = len(df)
            lower, upper = proportion_confint(successes, trials, alpha={alpha_val:.4f}, method='{method}')

            print(f'{method_display_name} Confidence Interval: ({{lower:.4f}}, {{upper:.4f}})')
            """
    return (lower, upper), code

#################
# Two proportion tests and confidence intervals
#################

def perform_two_proportion_ztest(df, group_col, outcome_col, alternative='two-sided', success_term=None):
    groups = df[group_col].dropna().unique()
    if len(groups) != 2:
        raise ValueError("The grouping column must have exactly two unique values.")

    group1_name, group2_name = groups

    if success_term is not None:
        stats = (df[outcome_col] == success_term).groupby(df[group_col]).agg(['sum', 'count'])
        term_str = f"'{success_term}'" if isinstance(success_term, str) else str(success_term)
        code_stats = f"stats = (df['{outcome_col}'] == {term_str}).groupby(df['{group_col}']).agg(['sum', 'count'])"
    else:
        stats = df.groupby(group_col)[outcome_col].agg(['sum', 'count'])
        code_stats = f"stats = df.groupby('{group_col}')['{outcome_col}'].agg(['sum', 'count'])"

    # .loc asegura que el orden coincida con group1_name y group2_name (groupby ordena alfabéticamente por defecto)
    successes = stats.loc[[group1_name, group2_name], 'sum'].values
    trials = stats.loc[[group1_name, group2_name], 'count'].values
    
    statistic, p_value = proportions_ztest(successes, trials, alternative=alternative)

    code = f"""import numpy as np
    from statsmodels.stats.proportion import proportions_ztest

    {code_stats}

    successes = stats.loc[['{group1_name}', '{group2_name}'], 'sum'].values
    trials = stats.loc[['{group1_name}', '{group2_name}'], 'count'].values

    statistic, p_value = proportions_ztest(successes, trials, alternative='{alternative}')

    print(f'Z-Statistic: {{statistic:.4f}}')
    print(f'P-value: {{p_value:.4f}}')
    """
    return statistic, p_value, code

def get_two_proportion_confint(df, group_col, outcome_col, method='newcomb', confidence=0.95, success_term=None):
    groups = df[group_col].dropna().unique()
    if len(groups) != 2:
        raise ValueError("The grouping column must have exactly two unique values.")

    group1_name, group2_name = groups

    if success_term is not None:
        stats = (df[outcome_col] == success_term).groupby(df[group_col]).agg(['sum', 'count'])
        term_str = f"'{success_term}'" if isinstance(success_term, str) else str(success_term)
        code_stats = f"stats = (df['{outcome_col}'] == {term_str}).groupby(df['{group_col}']).agg(['sum', 'count'])"
    else:        
        stats = df.groupby(group_col)[outcome_col].agg(['sum', 'count'])
        code_stats = f"stats = df.groupby('{group_col}')['{outcome_col}'].agg(['sum', 'count'])"

    successes = stats.loc[[group1_name, group2_name], 'sum'].values
    trials = stats.loc[[group1_name, group2_name], 'count'].values

    alpha = 1 - confidence
    lower, upper = confint_proportions_2indep(
        successes[0], trials[0],
        successes[1], trials[1],
        method=method, compare='diff', alpha=alpha
    )

    code = f"""import numpy as np
    from statsmodels.stats.proportion import confint_proportions_2indep

    {code_stats}

    successes = stats.loc[['{group1_name}', '{group2_name}'], 'sum'].values
    trials = stats.loc[['{group1_name}', '{group2_name}'], 'count'].values

    lower, upper = confint_proportions_2indep(
        successes[0], trials[0], 
        successes[1], trials[1], 
        method='{method}', compare='diff', alpha={alpha:.4f}
    )

    print(f'{method.title()} Confidence Interval for Difference in Proportions: ({{lower:.4f}}, {{upper:.4f}})')
    """
    return (lower, upper), code