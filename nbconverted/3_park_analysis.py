#!/usr/bin/env python
# coding: utf-8

# ## Re-analysis of Park et al. findings using gene expression-based mutation signatures
# 
# In [Park et al. 2021](https://www.nature.com/articles/s41467-021-27242-3), the authors study interactions between point mutations and CNVs in TCGA data. Essentially, they do an overrepresentation analysis to identify genes/cancer types that have more samples with _both_ a point mutation and a CNV than expected by chance, implying that these genes tend to require "two hits" (the point mutation and the CNV in this case) to be activated (oncogenes) or inactivated (tumor suppressors).
# 
# Here, we want to take the genes/cancer types they identified, and analyze the functional effects in the context of our mutation prediction classifiers. Our hypothesis is that in the "two-hit" genes, samples with "two hits" (a point mutation and a CNV) will have a higher predicted mutation probability than samples with zero or one hit.

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
# Get copy loss/gain info directly from GISTIC "thresholded" output. This should be the same as (or very similar to) what the Park et al. study uses.

# In[7]:


# we use the data source and preprocessing code from the pancancer repo, here:
# https://github.com/greenelab/pancancer/blob/d1b3de7fa387d0a44d0a4468b0ac30918ed66886/scripts/initialize/process_copynumber.py#L21

copy_thresh_df = (
    pd.read_csv(cfg.data_dir / 'pancan_GISTIC_threshold.tsv',
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
print(sorted(set(sample_freeze_df.SAMPLE_BARCODE) - set(copy_thresh_df.columns)))


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
# In [the Park et al. paper](https://www.nature.com/articles/s41467-021-27242-3#Sec4), they describe 4 "classes" of driver genes:
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
print(sorted(class_4_genes))


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

# In[16]:


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


# In[17]:


plot_id = 'CDH1_BRCA'
df = get_info_for_gene_and_tissue(plot_id, 'TSG')
print(df.mutation_status.isna().sum())
print(df.copy_status.isna().sum())
df.head()


# In[18]:


sns.set({'figure.figsize': (8, 6)})
sns.violinplot(x=df.positive_prob)
plt.title('Distribution of positive probabilities for {}'.format(plot_id))


# In[19]:


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

# In[20]:


park_df = pd.concat((park_loss_df, park_gain_df))
print(park_df.shape)
park_df.head()


# In[21]:


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


# In[22]:


def id_to_class(identifier):
    if type(park_df.loc[identifier, 'class']) == pd.Series:
        return park_df.loc[identifier, 'class'].values[0]
    else:
        return park_df.loc[identifier, 'class']

park_info_df['class'] = park_info_df['identifier'].apply(id_to_class)
park_info_df.head()


# In[23]:


park_info_df.groupby(by=['class']).count()


# In[24]:


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


# In[25]:


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


# Looking at the box plots, we can see that in general, the samples with "both" a point mutation and a CNV in the gene of interest tend to score higher using our classifiers than samples with "one" of a point mutation or a CNV. This is despite the fact that our classifiers were trained using all of these samples ("one" or "both") as positive labels.
# 
# The next step is to break this down by gene - are there genes/cancer types where the functional effect of the "two hits" is clearer, or less clear? Are there genes where we see "two hits" at the genetic level but no functional effect/classifier difference, or are the sources of information basically redundant?

# ### Statistical testing for individual gene/cancer type combinations

# In[26]:


import utilities as ut


# In[27]:


info_compare_df = ut.test_all(park_info_df)
print(info_compare_df.shape)
print('reject null for:', info_compare_df.reject_null.sum(), '/', info_compare_df.shape[0])
info_compare_df.sort_values(by='corr_pval', ascending=True).head()


# In[28]:


# plot top three most significant
# and top three least significant
order = ['none', 'one', 'both']
def get_counts(status):
    un = np.unique(status, return_counts=True)
    return {s: c for s, c in zip(*un)}

sns.set({'figure.figsize': (24, 16)})
fig, axarr = plt.subplots(2, 3)

valid_df = info_compare_df[info_compare_df.p_value != 1.0]
plot_ids = (
    valid_df.sort_values(by='corr_pval', ascending=True).identifier.tolist()[:3] + 
    valid_df.sort_values(by='corr_pval', ascending=False).identifier.tolist()[:3]
)

for ix, identifier in enumerate(plot_ids):
    ax = axarr[ix // 3, ix % 3]
    plot_df = park_info_df[park_info_df.identifier == identifier]
    sns.boxplot(data=plot_df, x='status', y='positive_prob', order=order, ax=ax)
    count_map = get_counts(plot_df.status.values)
    ax.set_xticks(np.arange(3),
               ['{} (n={})'.format(l, count_map[l]) for l in order])
    ax.set_title(identifier)

