#!/usr/bin/env python
# coding: utf-8

# ## Explore one-hit vs. two-hit samples in expression space

# In[1]:


from pathlib import Path
import pickle as pkl

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

import sys; sys.path.append('..')
import config as cfg
from data_utilities import load_cnv_data

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[2]:


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

# gene expression/rppa data files
data_type = 'gene expression'
subset_feats = 100
gene_expression_data_file = Path(
    '/home/jake/research/mpmp/data/tcga_expression_matrix_processed.tsv.gz'
)
rppa_data_file = Path(
    '/home/jake/research/mpmp/data/tcga_rppa_matrix_processed.tsv'
)


# ### Load mutation info
# 
# For now, just use binary mutation status from the pancancer repo. In the future we could pull more granular info from MC3, but it would take some engineering of `1_get_mutation_counts` to do this for lots of genes.

# In[3]:


park_loss_df = pd.read_csv(park_loss_data, sep='\t', index_col=0)
park_loss_df.head()


# In[4]:


park_gain_df = pd.read_csv(park_gain_data, sep='\t', index_col=0)
park_gain_df.head()


# In[5]:


with open(pancancer_pickle, 'rb') as f:
    pancancer_data = pkl.load(f)


# In[6]:


# get (binary) mutation data
# 1 = observed non-silent mutation in this gene for this sample, 0 otherwise
mutation_df = pancancer_data[1]
print(mutation_df.shape)
mutation_df.iloc[:5, :5]


# ### Load copy number info
# 
# Get copy loss/gain info directly from GISTIC "thresholded" output. This should be the same as (or very similar to) what the Park et al. study uses.

# In[7]:


sample_freeze_df = pancancer_data[0]
copy_samples = set(sample_freeze_df.SAMPLE_BARCODE)
print(len(copy_samples))


# In[8]:


copy_loss_df, copy_gain_df = load_cnv_data(
    cfg.data_dir / 'pancan_GISTIC_threshold.tsv',
    copy_samples
)
print(copy_loss_df.shape)
copy_loss_df.iloc[:5, :5]


# In[9]:


print(copy_gain_df.shape)
copy_gain_df.iloc[:5, :5]


# In[10]:


sample_freeze_df.head()


# ### Load expression data
# 
# We'll also subset to the top features by mean absolute deviation, if that option .

# In[11]:


if data_type == 'gene expression':
    exp_df = pd.read_csv(gene_expression_data_file, sep='\t', index_col=0)
elif data_type == 'rppa':
    exp_df = pd.read_csv(rppa_data_file, sep='\t', index_col=0)
    
print(exp_df.shape)
exp_df.iloc[:5, :5]


# In[12]:


# standardize features first
exp_df = pd.DataFrame(
    StandardScaler().fit_transform(exp_df),
    index=exp_df.index.copy(),
    columns=exp_df.columns.copy()
)
print(exp_df.shape)
exp_df.iloc[:5, :5]


# In[13]:


# subset to subset_feats features by mean absolute deviation
if subset_feats is not None:
    mad_ranking = (
        exp_df.mad(axis=0)
               .sort_values(ascending=False)
    )
    top_feats = mad_ranking[:subset_feats].index.astype(str).values
    print(top_feats[:5])
    exp_df = exp_df.reindex(top_feats, axis='columns')
    
print(exp_df.shape)
exp_df.iloc[:5, :5]


# ### Get sample info and groups for gene/cancer type

# In[14]:


def get_hits_for_gene_and_tissue(identifier, cancer_classification):
    """Given a gene and tissue, load the relevant mutation/CNV information,
    and divide the samples into groups to compare survival.
    """
    # get patient ids in given cancer type 
    gene, tissue = identifier.split('_')
    tissue_ids = (sample_freeze_df
        .query('DISEASE == @tissue')
        .SAMPLE_BARCODE
    )
    
    # get mutation and copy status
    mutation_status = mutation_df.loc[tissue_ids, gene]
    if cancer_classification == 'TSG':
        copy_status = copy_loss_df.loc[tissue_ids, gene]
    elif cancer_classification == 'Oncogene':
        copy_status = copy_gain_df.loc[tissue_ids, gene]
        
    # get hit groups from mutation/CNV data
    two_hit_samples = (mutation_status & copy_status).astype(int)
    one_hit_samples = (mutation_status ^ copy_status).astype(int)
        
    return pd.DataFrame(
        {'one_hit': one_hit_samples,
         'two_hit': two_hit_samples}
    )


# In[15]:


sample_mut_df = get_hits_for_gene_and_tissue('ATRX_LGG', 'TSG')

print(sample_mut_df.shape)
sample_mut_df.iloc[:5, :5]


# In[16]:


sample_mut_df.sum()

