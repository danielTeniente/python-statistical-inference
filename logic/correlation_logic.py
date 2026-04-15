import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr, kendalltau

# --- Internal Support Function (Backend) ---
def _bootstrap_ci(x, y, method='spearman', n_boot=5000, alpha=0.05):
    """Calculates bootstrap confidence intervals for correlation coefficients."""
    n = len(x)
    stats = []
    
    for _ in range(n_boot):
        idx = np.random.choice(range(n), n, replace=True)
        xb = x[idx]
        yb = y[idx]
        
        if method == 'spearman':
            r = spearmanr(xb, yb)[0]
        elif method == 'kendall':
            r = kendalltau(xb, yb)[0]
            
        stats.append(r)
        
    lower = np.percentile(stats, 100 * alpha / 2)
    upper = np.percentile(stats, 100 * (1 - alpha / 2))
    
    return lower, upper

# --- 1. Pearson Correlation ---
def perform_pearson_correlation(df, var1_col, var2_col):
    """Performs Pearson correlation test and calculates the 95% Confidence Interval."""
    # Drop NaNs to prevent scipy errors
    clean_df = df[[var1_col, var2_col]].dropna()
    x = clean_df[var1_col]
    y = clean_df[var2_col]
    
    result = pearsonr(x, y)
    corr_coeff = result.statistic
    p_value = result.pvalue
    ci = result.confidence_interval(confidence_level=0.95)
    ci_lower, ci_upper = ci.low, ci.high
    
    code = (
        "import pandas as pd\n"
        "from scipy.stats import pearsonr\n\n"
        f"clean_df = df[['{var1_col}', '{var2_col}']].dropna()\n"
        f"x = clean_df['{var1_col}']\n"
        f"y = clean_df['{var2_col}']\n\n"
        "result = pearsonr(x, y)\n"
        "corr_coeff = result.statistic\n"
        "p_value = result.pvalue\n"
        "ci = result.confidence_interval(confidence_level=0.95)\n\n"
        "print(f'Pearson Correlation: {corr_coeff:.4f}')\n"
        "print(f'P-value: {p_value:.4f}')\n"
        "print(f'95% CI: [{ci.low:.4f}, {ci.high:.4f}]')\n"
    )
    
    return corr_coeff, p_value, ci_lower, ci_upper, code

# --- 2. Spearman Rank Correlation ---
def perform_spearman_correlation(df, var1_col, var2_col, n_boot=5000):
    """Performs Spearman rank correlation and calculates a Bootstrap 95% Confidence Interval."""
    clean_df = df[[var1_col, var2_col]].dropna()
    x = clean_df[var1_col].values
    y = clean_df[var2_col].values
    
    corr_coeff, p_value = spearmanr(x, y)
    ci_lower, ci_upper = _bootstrap_ci(x, y, method='spearman', n_boot=n_boot)
    
    code = (
        "import pandas as pd\n"
        "import numpy as np\n"
        "from scipy.stats import spearmanr\n\n"
        f"clean_df = df[['{var1_col}', '{var2_col}']].dropna()\n"
        f"x = clean_df['{var1_col}'].values\n"
        f"y = clean_df['{var2_col}'].values\n\n"
        "corr_coeff, p_value = spearmanr(x, y)\n\n"
        "# Bootstrap 95% Confidence Interval\n"
        "n = len(x)\n"
        "stats = []\n"
        f"for _ in range({n_boot}):\n"
        "    idx = np.random.choice(range(n), n, replace=True)\n"
        "    r = spearmanr(x[idx], y[idx])[0]\n"
        "    stats.append(r)\n\n"
        "ci_lower = np.percentile(stats, 2.5)\n"
        "ci_upper = np.percentile(stats, 97.5)\n\n"
        "print(f'Spearman Correlation: {corr_coeff:.4f}')\n"
        "print(f'P-value: {p_value:.4f}')\n"
        "print(f'95% Bootstrap CI: [{ci_lower:.4f}, {ci_upper:.4f}]')\n"
    )
    
    return corr_coeff, p_value, ci_lower, ci_upper, code

# --- 3. Kendall Tau Correlation ---
def perform_kendall_correlation(df, var1_col, var2_col, n_boot=5000):
    """Performs Kendall Tau correlation test and calculates a Bootstrap 95% Confidence Interval."""
    clean_df = df[[var1_col, var2_col]].dropna()
    x = clean_df[var1_col].values
    y = clean_df[var2_col].values
    
    corr_coeff, p_value = kendalltau(x, y)
    ci_lower, ci_upper = _bootstrap_ci(x, y, method='kendall', n_boot=n_boot)
    
    code = (
        "import pandas as pd\n"
        "import numpy as np\n"
        "from scipy.stats import kendalltau\n\n"
        f"clean_df = df[['{var1_col}', '{var2_col}']].dropna()\n"
        f"x = clean_df['{var1_col}'].values\n"
        f"y = clean_df['{var2_col}'].values\n\n"
        "corr_coeff, p_value = kendalltau(x, y)\n\n"
        "# Bootstrap 95% Confidence Interval\n"
        "n = len(x)\n"
        "stats = []\n"
        f"for _ in range({n_boot}):\n"
        "    idx = np.random.choice(range(n), n, replace=True)\n"
        "    r = kendalltau(x[idx], y[idx])[0]\n"
        "    stats.append(r)\n\n"
        "ci_lower = np.percentile(stats, 2.5)\n"
        "ci_upper = np.percentile(stats, 97.5)\n\n"
        "print(f'Kendall Tau: {corr_coeff:.4f}')\n"
        "print(f'P-value: {p_value:.4f}')\n"
        "print(f'95% Bootstrap CI: [{ci_lower:.4f}, {ci_upper:.4f}]')\n"
    )
    
    return corr_coeff, p_value, ci_lower, ci_upper, code

def get_scatterplot(df, col_x, col_y, show_line=False):
    """
    Generates a scatterplot for two variables with an optional linear regression line.
    Returns the matplotlib figure and the equivalent Python code as a string.
    """
    # Drop NaNs to prevent errors in polyfit and plotting
    clean_df = df[[col_x, col_y]].dropna()
    x = clean_df[col_x]
    y = clean_df[col_y]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(x, y, alpha=0.7, edgecolors='w', linewidth=0.5)
    
    title_text = f'Scatterplot of {col_y} vs {col_x}'
    
    # Conditional logic for the regression line
    if show_line:
        # Calculate Pearson correlation for the dynamic title
        r_val, _ = pearsonr(x, y)
        title_text += f' (r = {r_val:.3f})'
        
        # Linear fit calculation
        coef = np.polyfit(x, y, 1)
        poly_fn = np.poly1d(coef)
        
        # Plot the regression line
        ax.plot(x, poly_fn(x), color='red', linewidth=2, label='Linear Fit')
        ax.legend()
    
    ax.set_title(title_text)
    ax.set_xlabel(col_x)
    ax.set_ylabel(col_y)
    
    # --- Reproducible Code Generation ---
    code = (
        "import matplotlib.pyplot as plt\n"
        "import numpy as np\n"
        "from scipy.stats import pearsonr\n\n"
        f"clean_df = df[['{col_x}', '{col_y}']].dropna()\n"
        f"x = clean_df['{col_x}']\n"
        f"y = clean_df['{col_y}']\n\n"
        "plt.figure(figsize=(10, 6))\n"
        "plt.scatter(x, y, alpha=0.7, edgecolors='w', linewidth=0.5)\n\n"
    )
    
    if show_line:
        code += (
            "r_val, _ = pearsonr(x, y)\n"
            "coef = np.polyfit(x, y, 1)\n"
            "poly_fn = np.poly1d(coef)\n"
            "plt.plot(x, poly_fn(x), color='red', linewidth=2, label='Linear Fit')\n"
            f"plt.title(f'Scatterplot of {col_y} vs {col_x} (r = {{r_val:.3f}})')\n"
            "plt.legend()\n"
        )
    else:
        code += f"plt.title('Scatterplot of {col_y} vs {col_x}')\n"
        
    code += (
        f"plt.xlabel('{col_x}')\n"
        f"plt.ylabel('{col_y}')\n"
        "plt.show()\n"
    )
    
    return fig, code

def get_correlation_heatmap(df, columns, method='pearson', shape='triangle'):
    """
    Generates a correlation heatmap for a list of columns.
    Supports different correlation methods ('pearson', 'spearman', 'kendall')
    and shapes ('triangle' or 'square'). Returns the figure and reproducible code.
    """
    # Filter the DataFrame to include only the selected columns
    subset_df = df[columns]
    
    # Calculate the correlation matrix
    corr_matrix = subset_df.corr(method=method)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Determine the mask based on the shape parameter
    if shape == 'triangle':
        # np.triu creates a mask for the upper triangle including the diagonal
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap="coolwarm", 
                    vmin=-1, vmax=1, fmt=".2f", ax=ax, linewidths=0.5)
    else: # square
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", 
                    vmin=-1, vmax=1, fmt=".2f", ax=ax, linewidths=0.5)
        
    ax.set_title(f"Correlation Matrix ({method.capitalize()})", pad=20)
    
    # --- Reproducible Code Generation ---
    # Convert the list of columns to a string representation for the code block
    cols_str = str(columns)
    
    code = (
        "import pandas as pd\n"
        "import numpy as np\n"
        "import matplotlib.pyplot as plt\n"
        "import seaborn as sns\n\n"
        f"columns = {cols_str}\n"
        "subset_df = df[columns]\n"
        f"corr_matrix = subset_df.corr(method='{method}')\n\n"
        "plt.figure(figsize=(10, 8))\n"
    )
    
    if shape == 'triangle':
        code += (
            "mask = np.triu(np.ones_like(corr_matrix, dtype=bool))\n"
            "sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', "
            "vmin=-1, vmax=1, fmt='.2f', linewidths=0.5)\n"
        )
    else:
        code += (
            "sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', "
            "vmin=-1, vmax=1, fmt='.2f', linewidths=0.5)\n"
        )
        
    code += (
        f"plt.title('Correlation Matrix ({method.capitalize()})', pad=20)\n"
        "plt.show()\n"
    )
    
    return fig, code