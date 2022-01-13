from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests

def test_all(info_df, fdr_correction=True, correction_alpha=0.05):
    ind_results = []
    for identifier in info_df.identifier.unique():
        test_df = info_df[info_df.identifier == identifier].copy()
        ind_results.append([identifier] + test_one_vs_both(test_df))
    results_df = pd.DataFrame(ind_results, columns=['identifier', 'delta_mean', 'p_value'])
    if fdr_correction:
        corr = multipletests(results_df['p_value'],
                             method='fdr_bh',
                             alpha=correction_alpha)
        results_df = results_df.assign(corr_pval=corr[1], reject_null=corr[0])
    return results_df


def test_one_vs_both(test_df):
    one_hit_samples = test_df[test_df.status == 'one'].positive_prob.values
    both_hit_samples = test_df[test_df.status == 'both'].positive_prob.values
    if one_hit_samples.shape[0] < 2 or both_hit_samples.shape[0] < 2:
        # if either one or both is 0, just set difference to 0
        delta_mean = 0
        p_value = 1.0
    else:
        delta_mean = np.mean(one_hit_samples) - np.mean(both_hit_samples)
        p_value = ttest_ind(one_hit_samples, both_hit_samples)[1]
    return [delta_mean, p_value]
