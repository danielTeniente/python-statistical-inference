from scipy import stats
from scipy.stats import bootstrap
import numpy as np

def perform_ttest(df,col, popmean=1.75, alternative='two-sided', confidence=0.95):
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

def perform_wilcoxon(df, col, popmean=1.75, alternative='two-sided', confidence=0.95):
    """Perform Wilcoxon signed-rank test on the specified column of the DataFrame."""
    res = stats.wilcoxon(df[col] - popmean, alternative=alternative)
    code = "from scipy import stats \n"
    code += f"res = stats.wilcoxon(df['{col}'] - {popmean}, alternative='{alternative}')\n"
    code += "print(f'Statistic: {res.statistic:.4f}')\n"
    code += "print(f'p-value: {res.pvalue:.4f}')\n"

    # bootstrap confidence interval
    boostrap_data = (np.array(df[col]),)
    ci = bootstrap(boostrap_data, np.median, 
        confidence_level=confidence,
        n_resamples=2000, method='percentile'
        ).confidence_interval
    code += f"\n# Bootstrap confidence interval for the median\n"
    code += f"import numpy as np\n"
    code += f"from scipy.stats import bootstrap\n"
    code += f"boostrap_data = (np.array(df['{col}']),)\n"
    code += f"ci = bootstrap(boostrap_data, np.median, confidence_level={confidence}, n_resamples=2000, method='percentile').confidence_interval\n"
    code += "print(f'Confidence Interval: ({ci.low:.4f}, {ci.high:.4f})')\n"
    return res, ci, code