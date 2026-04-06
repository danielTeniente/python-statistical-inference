import pandas as pd
import numpy as np
from scipy.stats import fisher_exact, MonteCarloMethod

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
    Perform Fisher's Exact Test
    Automatically uses Monte Carlo simulation for tables larger than 2x2.

    Parameters:
    - df: DataFrame containing the data.
    - var1_col: Column name representing the first categorical variable.
    - var2_col: Column name representing the second categorical variable.
    - alternative: Type of test ('two-sided', 'less', 'greater'). 
                   Note: 'less' and 'greater' are only valid for 2x2 tables.

    Returns:
    - contingency_table: The generated crosstab (pandas DataFrame).
    - p_value: The calculated p-value.
    - statistic: The Odds Ratio (only for 2x2 tables) or None (for larger tables).
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
        metodo_mc = MonteCarloMethod(rng=rng)
        
        resultado = fisher_exact(contingency_table, method=metodo_mc)
        p_value = resultado.pvalue
        
        code = (
            "import pandas as pd\n"
            "import numpy as np\n"
            "from scipy.stats import fisher_exact, MonteCarloMethod\n\n"
            f"contingency_table = pd.crosstab(df['{var1_col}'], df['{var2_col}'])\n"
            "rng = np.random.default_rng()\n"
            "method = MonteCarloMethod(rng=rng)\n"
            "result = fisher_exact(contingency_table, method=method) #does not support alternative parameter\n\n"
            "print('Contingency Table:')\n"
            "print(contingency_table)\n"
            "print(f'\\nP-value (Monte Carlo): {result.pvalue:.4f}')\n"
        )

    return statistic, p_value, code