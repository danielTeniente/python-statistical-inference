import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
from statsmodels.stats.contingency_tables import Table2x2

# --- Internal Support Function (Backend) ---
def _get_base_chi2_stats(df, var1_col, var2_col):
    """Calculates the base contingency table and chi-square statistic without correction for association measures."""
    table = pd.crosstab(df[var1_col], df[var2_col])
    chi2_stat, p_value, _, _ = chi2_contingency(table, correction=False)
    n = table.values.sum()
    return table, chi2_stat, p_value, n

# --- 1. Cramér's V (For R x C tables) ---
def perform_cramers_v_test(df, var1_col, var2_col):
    """Performs the calculation for Cramér's V."""
    table, chi2_stat, p_value, n = _get_base_chi2_stats(df, var1_col, var2_col)
    k = min(table.shape)
    
    # Avoid division by zero if k=1
    cramers_v = np.sqrt(chi2_stat / (n * (k - 1))) if k > 1 else np.nan
    
    code = (
        "import pandas as pd\n"
        "import numpy as np\n"
        "from scipy.stats import chi2_contingency\n\n"
        f"table = pd.crosstab(df['{var1_col}'], df['{var2_col}'])\n"
        "chi2, p, _, _ = chi2_contingency(table, correction=False)\n"
        "n = table.values.sum()\n"
        "k = min(table.shape)\n"
        "cramers_v = np.sqrt(chi2 / (n * (k - 1)))\n\n"
        "print(f'Cramér\\'s V: {cramers_v:.4f}')\n"
        "print(f'P-value: {p:.4f}')\n"
    )
    
    return cramers_v, p_value, code

# --- 2. Pearson's Contingency Coefficient (For R x C tables) ---
def perform_pearsons_c_test(df, var1_col, var2_col):
    """Performs the calculation for Pearson's Contingency Coefficient (C)."""
    table, chi2_stat, p_value, n = _get_base_chi2_stats(df, var1_col, var2_col)
    
    pearson_c = np.sqrt(chi2_stat / (chi2_stat + n))
    
    code = (
        "import pandas as pd\n"
        "import numpy as np\n"
        "from scipy.stats import chi2_contingency\n\n"
        f"table = pd.crosstab(df['{var1_col}'], df['{var2_col}'])\n"
        "chi2, p, _, _ = chi2_contingency(table, correction=False)\n"
        "n = table.values.sum()\n"
        "pearson_c = np.sqrt(chi2 / (chi2 + n))\n\n"
        "print(f'Pearson\\'s C Coefficient: {pearson_c:.4f}')\n"
        "print(f'P-value: {p:.4f}')\n"
    )
    
    return pearson_c, p_value, code

# --- 3. Phi Coefficient (Only for 2x2 tables) ---
def perform_phi_coefficient_test(df, var1_col, var2_col):
    """Performs the calculation for the Phi Coefficient (Only 2x2)."""
    table, chi2_stat, p_value, n = _get_base_chi2_stats(df, var1_col, var2_col)
    
    if table.shape != (2, 2):
        raise ValueError("The Phi Coefficient is only valid for 2x2 contingency tables.")
        
    phi = np.sqrt(chi2_stat / n)
    
    code = (
        "import pandas as pd\n"
        "import numpy as np\n"
        "from scipy.stats import chi2_contingency\n\n"
        f"table = pd.crosstab(df['{var1_col}'], df['{var2_col}'])\n"
        "chi2, p, _, _ = chi2_contingency(table, correction=False)\n"
        "n = table.values.sum()\n"
        "phi = np.sqrt(chi2 / n)\n\n"
        "print(f'Phi Coefficient: {phi:.4f}')\n"
        "print(f'P-value: {p:.4f}')\n"
    )
    
    return phi, p_value, code

# --- 4. Odds Ratio (Only for 2x2 tables) ---
def perform_odds_ratio_test(df, var1_col, var2_col):
    """
    Calculates the Odds Ratio and its confidence intervals.
    Returns (odds_ratio, ci_low, ci_high, p_value, code).
    """
    # We use the support function only to obtain the associated p-value if needed
    table, _, p_value, _ = _get_base_chi2_stats(df, var1_col, var2_col)
    
    if table.shape != (2, 2):
        raise ValueError("The Odds Ratio is only valid for 2x2 contingency tables.")
        
    ct = Table2x2(table.values)
    odds_ratio = ct.oddsratio
    ci_low, ci_high = ct.oddsratio_confint()
    
    code = (
        "import pandas as pd\n"
        "from statsmodels.stats.contingency_tables import Table2x2\n"
        "from scipy.stats import chi2_contingency\n\n"
        f"table = pd.crosstab(df['{var1_col}'], df['{var2_col}'])\n"
        "ct = Table2x2(table.values)\n"
        "odds_ratio = ct.oddsratio\n"
        "ci_low, ci_high = ct.oddsratio_confint()\n\n"
        "# Reference P-value via Chi-square\n"
        "_, p, _, _ = chi2_contingency(table, correction=False)\n\n"
        "print(f'Odds Ratio: {odds_ratio:.4f}')\n"
        "print(f'Confidence Interval (95%): [{ci_low:.4f}, {ci_high:.4f}]')\n"
        "print(f'P-value: {p:.4f}')\n"
    )
    
    # Note: Returning a longer tuple here to accommodate the intervals
    return odds_ratio, ci_low, ci_high, p_value, code