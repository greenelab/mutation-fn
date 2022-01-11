#!/usr/bin/env python
# coding: utf-8

# ## Re-analysis of Park et al. findings using gene expression-based mutation signatures
# 
# TODO: document

# In[1]:


from pathlib import Path
import pickle as pkl

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import sys; sys.path.append('..')
import config as cfg

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[2]:


# park geneset info
park_loss_data = cfg.data_dir / 'park_loss_df.tsv'
park_gain_data = cfg.data_dir / 'park_gain_df.tsv'

# park gene/cancer type predictions
park_preds_dir = cfg.data_dir / 'park_genes_preds'

# mutation and copy number data
pancancer_pickle = Path('/home/jake/research/mpmp/data/pancancer_data.pkl')


# ### Load mutation/copy number info
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


mutation_df = pancancer_data[1]
print(mutation_df.shape)
mutation_df.iloc[:5, :5]


# In[7]:


copy_loss_df = pancancer_data[2]
print(copy_loss_df.shape)
copy_loss_df.iloc[:5, :5]


# In[8]:


copy_gain_df = pancancer_data[3]
print(copy_gain_df.shape)
copy_gain_df.iloc[:5, :5]


# ### Classify genes/cancer types into "classes"
# 
# In [Park et al. 2021](https://www.nature.com/articles/s41467-021-27242-3), they describe 4 "classes" of driver genes:
# 
# 1. Genes that function exclusively as one-hit drivers, no significant co-occurrence with CNAs (we aren't concerned with those here)
# 2. Genes that interact with CNA loss in at least one cancer type - "two-hit loss" drivers (i.e. classical tumor suppressors)
# 3. Genes that interact with CNA gain in at least one cancer type - "two-hit gain" drivers (for some examples/explanation of "two-hit" oncogenes, see [this paper](https://www.nature.com/articles/s41586-020-2175-2))
# 4. Genes that interact with both CNA loss and CNA gain across multiple cancer types - "two-hit loss and gain" drivers
# 
# Here, we label each of the genes from the Park et al. data with their "class", since we want to segment our analyses in this way too.

# In[9]:


# our datasets are already filtered for significance, so genes that appear
# in both loss/gain tables are class 4
# others are class 2/3 for loss/gain tables respectively

class_4_genes = (
    set(park_loss_df.Gene.unique()).intersection(
    set(park_gain_df.Gene.unique())
))
print(class_4_genes)


# In[10]:


def gene_to_class(g):
    return 'class 4' if g in class_4_genes else 'class 2'

loss_class = {g: gene_to_class(g) for g in park_loss_df.Gene.unique()}

park_loss_df['class'] = park_loss_df.Gene.map(loss_class)
park_loss_df.head()


# In[11]:


def gene_to_class(g):
    return 'class 4' if g in class_4_genes else 'class 3'

gain_class = {g: gene_to_class(g) for g in park_gain_df.Gene.unique()}

park_gain_df['class'] = park_gain_df.Gene.map(gain_class)
park_gain_df.head()


# ### Retrieve and format per-sample information
# 
# * Sample ID, gene/tissue (multi-index)
# * Gene classification
# * Mutation status for sample in gene
# * CNV status for sample in gene
# * Classifier probability

# In[30]:


from scipy.special import expit

def get_info_for_gene_and_tissue(identifier, class_df):
    gene, tissue = identifier.split('_')
    preds_file = park_preds_dir / 'expression_{}_raw_preds.tsv'.format(identifier)
    preds_df = pd.read_csv(preds_file, sep='\t', skiprows=1,
                           names=['sample_id', gene])
    
    # get predictions for identifier
    preds_df['identifier'] = identifier
    preds_df['positive_prob'] = expit(preds_df[gene])
    preds_df.drop(columns=[gene], inplace=True)
    
    # get mutation status for samples
    preds_df['mutation_status'] = mutation_df.loc[preds_df.index, gene]
    
    
    # get copy status for samples
    id_class = class_df.loc[identifier, 'classification']
    print(id_class)
    if id_class == 'TSG':
        copy_status = copy_loss_df.loc[preds_df.index, gene]
    elif id_class == 'Oncogene':
        copy_status = copy_gain_df.loc[preds_df.index, gene]
    preds_df['copy_status'] = copy_status
        
    def status_from_mut_info(row):
        if row['mutation_status'] == 1 and row['copy_status'] == 1:
            return 'both'
        elif row['mutation_status'] == 1 or row['copy_status'] == 1:
            return 'one'
        else:
            return 'none'
        
    preds_df['status'] = preds_df.apply(status_from_mut_info, axis=1)
    
    return preds_df

df = get_info_for_gene_and_tissue('TP53_STAD', park_loss_df)
print(df.mutation_status.isna().sum())
print(df.copy_status.isna().sum())
df.head(20)


# In[26]:


sns.set({'figure.figsize': (8, 6)})
sns.violinplot(df.positive_prob)


# In[44]:


order = ['none', 'one', 'both']
sns.set({'figure.figsize': (8, 6)})
sns.boxplot(data=df, x='status', y='positive_prob',
            order=order)

def get_counts(status):
    un = np.unique(status, return_counts=True)
    return {s: c for s, c in zip(*un)}

count_map = get_counts(df.status.values)
plt.xticks(np.arange(3),
           ['{} (n={})'.format(l, count_map[l]) for l in order])

