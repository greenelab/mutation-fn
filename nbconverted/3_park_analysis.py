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


mutation_df = pancancer_data[1]
print(mutation_df.shape)
mutation_df.iloc[:5, :5]


# ### Load copy number info
# 
# TODO: document

# In[7]:


# get copy loss/gain info directly from GISTIC output
# we use the preprocessing code here:
# https://github.com/greenelab/pancancer/blob/d1b3de7fa387d0a44d0a4468b0ac30918ed66886/scripts/initialize/process_copynumber.py#L21

copy_thresh_df = (
    pd.read_csv(Path('data', 'pancan_GISTIC_threshold.tsv'),
                sep='\t', index_col=0)
      .drop(columns=['Locus ID', 'Cytoband'])
)
copy_thresh_df.columns = copy_thresh_df.columns.str[0:15]

print(copy_thresh_df.shape)
copy_thresh_df.iloc[:5, :5]


# In[8]:


sample_freeze_df = pancancer_data[0]
copy_samples = list(
    set(sample_freeze_df.SAMPLE_BARCODE)
    .intersection(set(copy_thresh_df.columns))
)
print(len(copy_samples))


# In[9]:


# make sure we're not losing too many samples, a few is fine
print(set(sample_freeze_df.SAMPLE_BARCODE) - set(copy_thresh_df.columns))


# In[10]:


copy_thresh_df = (copy_thresh_df
    .T
    .loc[sorted(copy_samples)]
    .fillna(0)
    .astype(int)
)

print(copy_thresh_df.shape)
copy_thresh_df.iloc[:5, :5]


# In[11]:


# thresholded copy number includes 5 values [-2, -1, 0, 1, 2], which
# correspond to "deep loss", "moderate loss", "no change",
# "moderate gain", and "deep gain", respectively.
#
# here, we want to use "moderate" and "deep" loss/gain to define CNV
# loss/gain, as opposed to the more conservative approach of using
# "deep loss/gain" as in our classifiers

copy_loss_df = (copy_thresh_df
    .replace(to_replace=[1, 2], value=0)
    .replace(to_replace=[-1, -2], value=1)
)
print(copy_loss_df.shape)
copy_loss_df.iloc[:5, :5]


# In[12]:


copy_gain_df = (copy_thresh_df
    .replace(to_replace=[-1, -2], value=0)
    .replace(to_replace=[1, 2], value=1)
)
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

# In[13]:


# our datasets are already filtered for significance, so genes that appear
# in both loss/gain tables are class 4
# others are class 2/3 for loss/gain tables respectively

class_4_genes = (
    set(park_loss_df.Gene.unique()).intersection(
    set(park_gain_df.Gene.unique())
))
print(class_4_genes)


# In[14]:


def gene_to_class(g):
    return 'class 4' if g in class_4_genes else 'class 2'

loss_class = {g: gene_to_class(g) for g in park_loss_df.Gene.unique()}

park_loss_df['class'] = park_loss_df.Gene.map(loss_class)
park_loss_df.head()


# In[15]:


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

# In[22]:


from scipy.special import expit

def get_info_for_gene_and_tissue(identifier, classification):
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
    if classification == 'TSG':
        samples = preds_df.index.intersection(copy_loss_df.index)
        copy_status = copy_loss_df.loc[samples, gene]
    elif classification == 'Oncogene':
        samples = preds_df.index.intersection(copy_gain_df.index)
        copy_status = copy_gain_df.loc[samples, gene]
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


# In[23]:


plot_id = 'CDH1_BRCA'
df = get_info_for_gene_and_tissue(plot_id, 'TSG')
print(df.mutation_status.isna().sum())
print(df.copy_status.isna().sum())
df.head()


# In[24]:


sns.set({'figure.figsize': (8, 6)})
sns.violinplot(x=df.positive_prob)
plt.title('Distribution of positive probabilities for {}'.format(plot_id))


# In[25]:


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
plt.title(plot_id)


# ### Averages across each "class" of genes

# In[26]:


park_df = pd.concat((park_loss_df, park_gain_df))
print(park_df.shape)
park_df.head()


# In[27]:


park_info = []
for identifier in park_df.index:
    try:
        classification = park_df.loc[identifier, 'classification']
        info_df = get_info_for_gene_and_tissue(identifier, classification)
    except ValueError:
        classification = park_df.loc[identifier, 'classification'].values[0]
        info_df = get_info_for_gene_and_tissue(identifier, classification)
    except FileNotFoundError:
        continue
    park_info.append(info_df)
    
park_info_df = pd.concat(park_info)
print(park_info_df.shape)
park_info_df.head()


# In[28]:


def id_to_class(identifier):
    if type(park_df.loc[identifier, 'class']) == pd.Series:
        return park_df.loc[identifier, 'class'].values[0]
    else:
        return park_df.loc[identifier, 'class']

park_info_df['class'] = park_info_df['identifier'].apply(id_to_class)
park_info_df.head()


# In[29]:


park_info_df.groupby(by=['class']).count()


# In[30]:


order = ['none', 'one', 'both']
sns.set({'figure.figsize': (8, 6)})
sns.boxplot(data=park_info_df, x='status', y='positive_prob',
            order=order)
plt.title('Average over all genes/cancer types from Park et al.')

def get_counts(status):
    un = np.unique(status, return_counts=True)
    return {s: c for s, c in zip(*un)}

count_map = get_counts(park_info_df.status.values)
plt.xticks(np.arange(3),
           ['{} (n={})'.format(l, count_map[l]) for l in order])


# In[31]:


sns.set({'figure.figsize': (24, 6)})
fig, axarr = plt.subplots(1, 3)

for ix, class_label in enumerate(['class 2', 'class 3', 'class 4']):
    ax = axarr[ix]
    plot_df = park_info_df[park_info_df['class'] == class_label]
    sns.boxplot(data=plot_df, x='status', y='positive_prob',
                order=order, ax=ax)
    ax.set_title('Average over {} genes'.format(class_label))
    count_map = get_counts(plot_df.status.values)
    ax.set_xticks(np.arange(3), ['{} (n={})'.format(l, count_map[l]) for l in order])

