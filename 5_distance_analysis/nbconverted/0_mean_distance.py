#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pathlib import Path

import pandas as pd

import sys; sys.path.append('..')
import config as cfg

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[6]:


# whether to use expression or rppa data
# data_type = 'expression'
data_type = 'rppa'

# number of features to subset to, by mean absolute deviation
subset_mad_feats = 100


# ### Load expression data
# 
# We'll also subset to the top features by mean absolute deviation, if necessary.

# In[7]:


expression_data_file = (
    '/home/jake/research/mpmp/data/tcga_expression_matrix_processed.tsv.gz'
)
expression_sample_info = (
    '/home/jake/research/mpmp/data/sample_info/tcga_expression_sample_identifiers.tsv'
)

rppa_data_file = (
    '/home/jake/research/mpmp/data/tcga_rppa_matrix_processed.tsv'
)
rppa_sample_info = (
    '/home/jake/research/mpmp/data/sample_info/tcga_rppa_sample_identifiers.tsv'
)

if data_type == 'expression':
    data_df = pd.read_csv(expression_data_file, sep='\t', index_col=0)
    sample_info_df = pd.read_csv(expression_data_file, sep='\t', index_col=0)
elif data_type == 'rppa':
    data_df = pd.read_csv(rppa_data_file, sep='\t', index_col=0)
    sample_info_df = pd.read_csv(rppa_data_file, sep='\t', index_col=0)
    
print(data_df.shape)
data_df.iloc[:5, :5]


# In[10]:


if subset_mad_feats is not None:
    mad_ranking = (
        data_df.mad(axis=0)
               .sort_values(ascending=False)
    )
    top_feats = mad_ranking[:subset_mad_feats].index.astype(str).values
    print(top_feats[:5])
    data_df = data_df.reindex(top_feats, axis='columns')
    
print(data_df.shape)
data_df.head()


# ### Load sample-wise mutation data for Park et al. genes
# 
# This was computed and saved in `3_park_analysis.ipynb`.

# In[11]:


# load park et al. initial analysis results here
park_info_df = pd.read_csv(
    cfg.park_info_file, sep='\t'
)

print(park_info_df.shape)
park_info_df.head()

