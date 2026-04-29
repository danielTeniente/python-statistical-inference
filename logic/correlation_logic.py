import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr, kendalltau

# --- Internal Support Function (Backend) ---
def _bootstrap_ci(x, y, method='spearman', n_boot=2000, alpha=0.05, max_bootstrap_n=5000):
    """Calculates bootstrap confidence intervals with a safety cap for large N."""
    n = len(x)
    
    # OPTIMIZATION 1: Cap the data size purely for the permutation test to avoid multi-hour calculations
    if n > max_bootstrap_n:
        idx_sample = np.random.choice(range(n), max_bootstrap_n, replace=False)
        x_boot_source = x[idx_sample]
        y_boot_source = y[idx_sample]
        n_boot_source = max_bootstrap_n
    else:
        x_boot_source = x
        y_boot_source = y
        n_boot_source = n

    stats = []
    
    for _ in range(n_boot):
        idx = np.random.choice(range(n_boot_source), n_boot_source, replace=True)
        xb = x_boot_source[idx]
        yb = y_boot_source[idx]
        
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
def perform_spearman_correlation(df, var1_col, var2_col, n_boot=2500):
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
def perform_kendall_correlation(df, var1_col, var2_col, n_boot=2500):
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
    """Generates scatterplot, sampling points for rendering performance while keeping math accurate."""
    clean_df = df[[col_x, col_y]].dropna()
    
    # Calculate the math (line) using FULL dataset
    x_full = clean_df[col_x]
    y_full = clean_df[col_y]
    
    # OPTIMIZATION 2: Cap scatter rendering to 3,000 points to prevent browser freezing
    MAX_SCATTER_POINTS = 3000
    if len(clean_df) > MAX_SCATTER_POINTS:
        plot_df = clean_df.sample(n=MAX_SCATTER_POINTS, random_state=42)
    else:
        plot_df = clean_df
        
    x_plot = plot_df[col_x]
    y_plot = plot_df[col_y]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(x_plot, y_plot, alpha=0.5, edgecolors='w', linewidth=0.5)
    
    title_text = f'Scatterplot of {col_y} vs {col_x}'
    if len(clean_df) > MAX_SCATTER_POINTS:
         title_text += f'\n(Rendered sample of {MAX_SCATTER_POINTS} points)'

    if show_line:
        r_val, _ = pearsonr(x_full, y_full)
        title_text += f'\nPearson r = {r_val:.3f}'
        
        coef = np.polyfit(x_full, y_full, 1)
        poly_fn = np.poly1d(coef)
        
        # Plot line across the full range of X
        ax.plot(x_full, poly_fn(x_full), color='red', linewidth=2, label='Linear Fit')
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
    subset_df = df[columns].dropna()
    
    # OPTIMIZATION 3: Protect Kendall Tau from freezing on matrix calculations
    if method == 'kendall' and len(subset_df) > 10000:
        subset_df = subset_df.sample(n=10000, random_state=42)
        
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