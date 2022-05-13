"""
Utilities for loading and preprocessing relevant data.
"""
import pandas as pd

def load_cnv_data(cnv_file, copy_samples, threshold='moderate'):
    """Load and threshold CNV data file from GDC."""

    # load raw copy info tsv file
    copy_thresh_df = (
        pd.read_csv(cnv_file, sep='\t', index_col=0)
          .drop(columns=['Locus ID', 'Cytoband'])
    )

    # drop everything after TCGA sample identifier, use sample as index
    copy_thresh_df.columns = copy_thresh_df.columns.str[0:15]

    # orient as samples x columns, harmonize samples, and fill NA values
    copy_samples = (
        copy_samples.intersection(set(copy_thresh_df.columns))
    )
    copy_thresh_df = (copy_thresh_df
        .T
        .loc[sorted(copy_samples)]
        .fillna(0)
        .astype(int)
    )

    # make sure there's no duplicate samples after we subset
    assert copy_thresh_df.index.duplicated().sum() == 0

    # thresholded copy number includes 5 values [-2, -1, 0, 1, 2], which
    # correspond to "deep loss", "moderate loss", "no change",
    # "moderate gain", and "deep gain", respectively.
    if threshold == 'moderate':
        # here we want to use "moderate" and "deep" loss/gain as 1
        copy_loss_df = (copy_thresh_df
            .replace(to_replace=[1, 2], value=0)
            .replace(to_replace=[-1, -2], value=1)
        )
        copy_gain_df = (copy_thresh_df
            .replace(to_replace=[1, 2], value=1)
            .replace(to_replace=[-1, -2], value=0)
        )
    elif threshold == 'deep':
        # here we want to use only "deep" loss/gain as 1
        copy_loss_df = (copy_thresh_df
            .replace(to_replace=[1, 2, -1], value=0)
            .replace(to_replace=[-2], value=1)
        )
        copy_gain_df = (copy_thresh_df
            .replace(to_replace=[2], value=1)
            .replace(to_replace=[1, -1, -2], value=0)
        )
    return copy_loss_df, copy_gain_df
