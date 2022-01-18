from pathlib import Path
import os
import sys
import glob

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, ttest_rel
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


def get_classifier_significance(identifiers,
                                preds_dir,
                                metric='aupr',
                                fdr_correction=True,
                                correction_alpha=0.05):
    """Determine which classifiers can distinguish between signal/shuffled."""

    class_df = []

    for identifier in identifiers:

        signal_results, shuffled_results = [], []
        signal_seeds, shuffled_seeds = [], []
        signal_folds, shuffled_folds = [], []

        signal_pattern = (
            '{}_expression_signal_classify_s*_metrics.tsv.gz'.format(identifier)
        )
        shuffled_pattern = (
            '{}_expression_shuffled_classify_s*_metrics.tsv.gz'.format(identifier)
        )

        try:
            signal_df = []
            for id_file in glob.glob(os.path.join(preds_dir, signal_pattern)):
                signal_df.append(pd.read_csv(id_file, sep='\t'))
            signal_df = pd.concat(signal_df)
            signal_df = (signal_df
              .loc[signal_df.data_type == 'test', :]
              .sort_values(by=['seed', 'fold'])
            )
            signal_results += signal_df[metric].values.tolist()
            signal_seeds += signal_df['seed'].values.tolist()
            signal_folds += signal_df['fold'].values.tolist()

            shuffled_df = []
            for id_file in glob.glob(os.path.join(preds_dir, shuffled_pattern)):
                shuffled_df.append(pd.read_csv(id_file, sep='\t'))
            shuffled_df = pd.concat(shuffled_df)
            shuffled_df = (shuffled_df
              .loc[shuffled_df.data_type == 'test', :]
              .sort_values(by=['seed', 'fold'])
            )
            shuffled_results += shuffled_df[metric].values.tolist()
            shuffled_seeds += shuffled_df['seed'].values.tolist()
            shuffled_folds += shuffled_df['fold'].values.tolist()
        except ValueError:
            print('No results file found for: ', identifier, file=sys.stderr)
            continue

        # make sure seeds and folds are in same order
        # this is necessary for paired t-test
        try:
            assert np.array_equal(signal_seeds, shuffled_seeds)
            assert np.array_equal(signal_folds, shuffled_folds)
        except AssertionError:
            print(identifier, file=sys.stderr)
            print(signal_seeds, shuffled_seeds, file=sys.stderr)
            print(signal_folds, shuffled_folds, file=sys.stderr)

        if np.array_equal(signal_results, shuffled_results):
            delta_mean = 0
            p_value = 1.0
        else:
            delta_mean = np.mean(signal_results) - np.mean(shuffled_results)
            p_value = ttest_rel(signal_results, shuffled_results)[1]
        class_df.append([identifier, delta_mean, p_value])

    class_df = pd.DataFrame(class_df, columns=['identifier', 'delta_mean', 'p_value'])

    if fdr_correction:
        corr = multipletests(class_df['p_value'],
                             method='fdr_bh',
                             alpha=correction_alpha)
        class_df = class_df.assign(corr_pval=corr[1], reject_null=corr[0])

    return class_df

