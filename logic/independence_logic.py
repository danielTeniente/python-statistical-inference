import pandas as pd
import numpy as np
from scipy.stats import fisher_exact, MonteCarloMethod, chi2_contingency

def get_contingency_table(df, var1_col, var2_col):
    """Generate a contingency table (crosstab) from the specified columns."""
    contingency_table = pd.crosstab(df[var1_col], df[var2_col])
    code = (
        "import pandas as pd\n\n"
        f"contingency_table = pd.crosstab(df['{var1_col}'], df['{var2_col}'])\n"
        "print('Contingency Table:')\n"
        "print(contingency_table)\n"
    )
    return contingency_table, code

def perform_fisher_exact_test(df, var1_col, var2_col, alternative='two-sided'):
    """
    Perform Fisher's Exact Test.
    Automatically uses Monte Carlo simulation for tables larger than 2x2.

    Parameters:
    - df: DataFrame containing the data.
    - var1_col: Column name representing the first categorical variable.
    - var2_col: Column name representing the second categorical variable.
    - alternative: Type of test ('two-sided', 'less', 'greater'). 
                   Note: 'less' and 'greater' are only valid for 2x2 tables.

    Returns:
    - statistic: The Odds Ratio (only for 2x2 tables) or None (for larger tables).
    - p_value: The calculated p-value.
    - code: String containing the Python code to reproduce the test.
    """
    contingency_table = pd.crosstab(df[var1_col], df[var2_col])
    
    statistic = None
    p_value = None
    
    if contingency_table.shape == (2, 2):
        statistic, p_value = fisher_exact(contingency_table, alternative=alternative)
        
        code = (
            "import pandas as pd\n"
            "from scipy.stats import fisher_exact\n\n"
            f"contingency_table = pd.crosstab(df['{var1_col}'], df['{var2_col}'])\n"
            f"statistic, p_value = fisher_exact(contingency_table, alternative='{alternative}')\n\n"
            "print('Contingency Table:')\n"
            "print(contingency_table)\n"
            "print(f'\\nOdds Ratio (Statistic): {statistic:.4f}')\n"
            "print(f'P-value: {p_value:.4f}')\n"
        )
        
    else:
        rng = np.random.default_rng()
        mc_method = MonteCarloMethod(rng=rng)
        
        result = fisher_exact(contingency_table, method=mc_method)
        p_value = result.pvalue
        
        code = (
            "import pandas as pd\n"
            "import numpy as np\n"
            "from scipy.stats import fisher_exact, MonteCarloMethod\n\n"
            f"contingency_table = pd.crosstab(df['{var1_col}'], df['{var2_col}'])\n"
            "rng = np.random.default_rng()\n"
            "method = MonteCarloMethod(rng=rng)\n"
            "result = fisher_exact(contingency_table, method=method) # Does not support alternative parameter\n\n"
            "print('Contingency Table:')\n"
            "print(contingency_table)\n"
            "print(f'\\nP-value (Monte Carlo): {result.pvalue:.4f}')\n"
        )

    return statistic, p_value, code

def perform_chi_square_test(df, var1_col, var2_col, correction=False):
    """
    Performs the Chi-square Test of Independence or Homogeneity (Proportions).
    
    Parameters:
    - df: DataFrame that contains all the data.
    - var1_col: Name of the first categorical variable.
    - var2_col: Name of the second categorical variable.
    - correction: Boolean. If True, applies Yates' continuity correction.
                  (Recommended for 2x2 tables).
                  
    Returns:
    - chi2_stat: The calculated Chi-square statistic.
    - p_value: The p-value.
    - code: String with the Python code to reproduce the test.
    """
    contingency_table = pd.crosstab(df[var1_col], df[var2_col])
    
    # Perform the Chi-square test
    chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table, correction=correction)
    
    code = (
        "import pandas as pd\n"
        "from scipy.stats import chi2_contingency\n\n"
        f"contingency_table = pd.crosstab(df['{var1_col}'], df['{var2_col}'])\n"
        f"chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table, correction={correction})\n\n"
        "print('Contingency Table:')\n"
        "print(contingency_table)\n"
        "print(f'\\nChi-square Statistic: {chi2_stat:.4f}')\n"
        "print(f'P-value: {p_value:.4f}')\n"
        "print(f'Degrees of Freedom (dof): {dof}')\n"
    )
    
    return chi2_stat, p_value, code