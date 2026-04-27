from scipy import stats
from scipy.stats import bootstrap
import numpy as np
from collections import namedtuple

ConfidenceInterval = namedtuple('ConfidenceInterval', ['low', 'high'])

def perform_ttest_with_ci(df,col, popmean=1.75, alternative='two-sided', confidence=0.95):
    """Perform one-sample t-test on the specified column of the DataFrame."""
    res =  stats.ttest_1samp(df[col], 
        popmean=popmean, alternative=alternative)
    ci = stats.t.interval(confidence, len(df[col])-1, 
        loc=df[col].mean(), scale=stats.sem(df[col]))
    code = "from scipy import stats \n"
    code += f"res = stats.ttest_1samp(df['{col}'], popmean={popmean}, alternative='{alternative}')\n"
    code += f"ci = stats.t.interval({confidence}, len(df['{col}'])-1, loc=df['{col}'].mean(), scale=stats.sem(df['{col}']))\n"
    code += "print(f't-statistic: {res.statistic:.4f}')\n"
    code += "print(f'p-value: {res.pvalue:.4f}')\n"
    code += "print(f'Confidence Interval: ({ci[0]:.4f}, {ci[1]:.4f})')\n"
    return res, ci, code

def perform_wilcoxon(df, col, popmean=1.75, alternative='two-sided'):
    """Perform Wilcoxon test and generate its educational code."""
    data = df[col].dropna()
    res = stats.wilcoxon(data - popmean, alternative=alternative)
    
    code = "from scipy import stats\n"
    code += f"data = df['{col}'].dropna()\n"
    code += f"res = stats.wilcoxon(data - {popmean}, alternative='{alternative}')\n"
    code += "print(f'Statistic: {res.statistic:.4f}')\n"
    code += "print(f'p-value: {res.pvalue:.4f}')\n"

    return res, code

def get_bootstrap_ci(df, col, confidence=0.95, threshold=5000):
    """Calculate Bootstrap CI with a safety threshold and generate its code."""
    data = df[col].dropna().values
    n = len(data)
    
    code = "\n# Bootstrap confidence interval for the median\n"
    code += "import numpy as np\n"
    code += "from scipy.stats import bootstrap\n"
    code += f"data = df['{col}'].dropna().values\n"
    
    if n > threshold:
        # Subsample for safety
        np.random.seed(42) # Fixed seed for reproducible educational output
        data_to_use = np.random.choice(data, size=threshold, replace=False)
        
        code += f"\n# Dataset exceeds {threshold} records. Taking a random subsample for performance.\n"
        code += "np.random.seed(42)\n"
        code += f"data_to_use = np.random.choice(data, size={threshold}, replace=False)\n"
    else:
        data_to_use = data
        code += "data_to_use = data\n"

    bootstrap_data = (data_to_use,)
    ci = bootstrap(
        bootstrap_data, np.median, 
        confidence_level=confidence, 
        n_resamples=2000, 
        method='percentile'
    ).confidence_interval
    
    code += "bootstrap_data = (data_to_use,)\n"
    code += f"ci = bootstrap(bootstrap_data, np.median, confidence_level={confidence}, n_resamples=2000, method='percentile').confidence_interval\n"
    code += "print(f'Bootstrap CI: ({ci.low:.4f}, {ci.high:.4f})')\n"
    
    return ConfidenceInterval(low=ci.low, high=ci.high), code

def get_exact_median_ci(df, col, confidence=0.95):
    """Calculate exact median CI using normal approximation and generate its code."""
    data = df[col].dropna().values
    n = len(data)
    
    data_sorted = np.sort(data)
    z_score = stats.norm.ppf(1 - (1 - confidence) / 2)
    lower_idx = max(0, int(np.floor((n / 2) - (z_score * np.sqrt(n) / 2))))
    upper_idx = min(n - 1, int(np.ceil((n / 2) + (z_score * np.sqrt(n) / 2))))
    
    ci = ConfidenceInterval(low=data_sorted[lower_idx], high=data_sorted[upper_idx])
    
    code = "\n# Exact confidence interval for the median using Order Statistics\n"
    code += "import numpy as np\n"
    code += "from scipy import stats\n"
    code += f"data = df['{col}'].dropna().values\n"
    code += "n = len(data)\n"
    code += "data_sorted = np.sort(data)\n"
    code += f"z_score = stats.norm.ppf(1 - (1 - {confidence}) / 2)\n"
    code += "lower_idx = max(0, int(np.floor((n / 2) - (z_score * np.sqrt(n) / 2))))\n"
    code += "upper_idx = min(n - 1, int(np.ceil((n / 2) + (z_score * np.sqrt(n) / 2))))\n"
    code += "ci_low, ci_high = data_sorted[lower_idx], data_sorted[upper_idx]\n"
    code += "print(f'Exact CI: ({ci_low:.4f}, {ci_high:.4f})')\n"
    
    return ci, code