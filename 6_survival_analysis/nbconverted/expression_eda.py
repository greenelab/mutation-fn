#!/usr/bin/env python
# coding: utf-8

# ## Explore one-hit vs. two-hit samples in expression space

# In[10]:


from pathlib import Path
import pickle as pkl

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import sys; sys.path.append('..')
import config as cfg
from data_utilities import load_cnv_data

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[11]:


# park et al. geneset info
park_loss_data = cfg.data_dir / 'park_loss_df.tsv'
park_gain_data = cfg.data_dir / 'park_gain_df.tsv'

# park et al. significant gene info
park_loss_sig_data = cfg.data_dir / 'park_loss_df_sig_only.tsv'
park_gain_sig_data = cfg.data_dir / 'park_gain_df_sig_only.tsv'

# park et al. gene/cancer type predictions
park_preds_dir = cfg.data_dir / 'park_genes_all_preds'

# mutation and copy number data
pancancer_pickle = Path('/home/jake/research/mpmp/data/pancancer_data.pkl')


# ### Load mutation info
# 
# For now, just use binary mutation status from the pancancer repo. In the future we could pull more granular info from MC3, but it would take some engineering of `1_get_mutation_counts` to do this for lots of genes.

# In[12]:


park_loss_df = pd.read_csv(park_loss_data, sep='\t', index_col=0)
park_loss_df.head()


# In[13]:


park_gain_df = pd.read_csv(park_gain_data, sep='\t', index_col=0)
park_gain_df.head()


# In[14]:


with open(pancancer_pickle, 'rb') as f:
    pancancer_data = pkl.load(f)


# In[15]:


# get (binary) mutation data
# 1 = observed non-silent mutation in this gene for this sample, 0 otherwise
mutation_df = pancancer_data[1]
print(mutation_df.shape)
mutation_df.iloc[:5, :5]


# ### Load copy number info
# 
# Get copy loss/gain info directly from GISTIC "thresholded" output. This should be the same as (or very similar to) what the Park et al. study uses.

# In[16]:


sample_freeze_df = pancancer_data[0]
copy_samples = set(sample_freeze_df.SAMPLE_BARCODE)
print(len(copy_samples))


# In[17]:


copy_loss_df, copy_gain_df = load_cnv_data(
    cfg.data_dir / 'pancan_GISTIC_threshold.tsv',
    copy_samples
)
print(copy_loss_df.shape)
copy_loss_df.iloc[:5, :5]


# In[18]:


print(copy_gain_df.shape)
copy_gain_df.iloc[:5, :5]

