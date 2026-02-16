from scipy import stats

def run_normality_test(data, test_type):
    """
    Executes the selected normality test.
    Returns: (statistic, p_value, generated_code_string)
    """
    if test_type == "Shapiro–Wilk":
        stat, p = stats.shapiro(data)
        code = f"from scipy import stats\nstat, p = stats.shapiro(data)\nprint(f'Statistic: {{stat:.4f}}, p-value: {{p:.4f}}')"
    else:  # D’Agostino–Pearson
        stat, p = stats.normaltest(data)
        code = f"from scipy import stats\nstat, p = stats.normaltest(data)\nprint(f'Statistic: {{stat:.4f}}, p-value: {{p:.4f}}')"
    
    return stat, p, code