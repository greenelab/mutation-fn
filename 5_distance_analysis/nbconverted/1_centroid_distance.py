#!/usr/bin/env python
# coding: utf-8

# ## Calculate distance between means/medoids of mutation groupings
# 
# Our goal is to find an unsupervised way of calculating distance/similarity between our mutation groupings ("none"/"one"/"both") which isn't affected by sample size, to the degree that differentially expressed gene count was (see `4_de_analysis` notebooks).
# 
# Here, we'll try the extremely simple method of:
# 
# 1) taking the n-dimensional mean (centroid) or median (medoid) of each group  
# 2) calculating distance between the centroids and using this to define "expression similarity"
# 
# We'll try this for a few different feature selection/embedding methods, and for both gene expression and RPPA (protein expression) data.

# In[1]:


from pathlib import Path
import pickle as pkl
import itertools as it

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import sys; sys.path.append('..')
import config as cfg

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[2]:


# whether to use expression or rppa data
# data_type = 'expression'
data_type = 'rppa'

# how to calculate centroids, 'mean' or 'median'
centroid_method = 'mean'

# number of features to subset to, by mean absolute deviation
# TODO try this in PCA/UMAP space too
subset_mad_feats = 100


# ### Load expression data
# 
# We'll also subset to the top features by mean absolute deviation, if that option .

# In[3]:


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


# In[4]:


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


# ### Load Park et al. "hit" data
# 
# This was collated/formatted in `0_process_park.ipynb`

# In[5]:


with open(cfg.distance_gain_info, 'rb') as f:
    park_gain_info = pkl.load(f)
    
park_gain_info['TP53_BRCA'].head()


# In[6]:


with open(cfg.distance_loss_info, 'rb') as f:
    park_loss_info = pkl.load(f)
    
park_loss_info['TP53_BRCA'].head()


# ### Calculate distance between means/medians for given gene + cancer type

# In[7]:


from scipy.spatial.distance import pdist, squareform

def get_centroids_and_distance(identifier, info_df, centroid_method='mean'):
    
    groups = ['both', 'none', 'one']
    group_combinations = list(it.combinations(groups, 2))
    
    # get expression data for samples
    samples = info_df.index.intersection(data_df.index)
    info_df = info_df.reindex(samples)
    
    # if one group has no samples, we have to make sure to assign it 0 count
    class_counts = []
    hit_class_counts = info_df.groupby('num_hits').count().class_name
    for group in groups:
        if group in hit_class_counts.index:
            class_counts.append(hit_class_counts[group])
        else:
            class_counts.append(0)
    
    # group by number of hits, then calculate centroids
    centroids_df = (data_df
        .reindex(samples)
        .merge(info_df['num_hits'], left_index=True, right_index=True)
        .groupby('num_hits')
    )
    
    if centroid_method == 'mean':
        centroids_df = centroids_df.mean()
    elif centroid_method == 'median':
        centroids_df = centroids_df.median()
    else:
        raise NotImplementedError(
            'centroid method {} not implemented'.format(centroid_method)
        )
    
    # calculate distance between centroids
    # make sure this is in the same order for each identifier, and
    # handle NA distances here (if one group doesn't have any samples)
    dists = pdist(centroids_df.values, metric='euclidean')
    dist_combinations = list(it.combinations(hit_class_counts.index, 2))
    ordered_dists = []
    for cmb in group_combinations:
        if cmb not in dist_combinations:
            ordered_dists.append(np.nan)
        else:
            cmb_ix = dist_combinations.index(cmb)
            ordered_dists.append(dists[cmb_ix])
    
    return groups, group_combinations, class_counts, ordered_dists
    
get_centroids_and_distance('TP53_BRCA',
                           park_loss_info['TP53_BRCA'],
                           'median')


# ### Calculate centroid distance between "hits", per class
# 
# Class 1 = look at both loss and gain (should be one-hit in neither)  
# Class 2 = only look at loss (should be one-hit here)  
# Class 3 = only look at gain (should be one-hit here)  
# Class 4 = look at both loss and gain (should be one-hit in both)

# In[8]:


class_counts_df = {}
results_df = {}
counts_columns = None
results_columns = None

# get distances for copy loss, for class 1/2/4 genes
for identifier, loss_df in park_loss_info.items():
    
    if loss_df.head(1).class_name.values[0] == 'class 3':
        continue
        
    results = get_centroids_and_distance(identifier, loss_df, 'mean')
    
    if counts_columns is None:
        counts_columns = results[0]
    else:
        assert counts_columns == results[0]
        
    if results_columns is None:
        results_columns = ['{}/{}'.format(i, j) for i, j in results[1]]
            
    class_counts_df[identifier] = results[2]
    results_df[identifier] = results[3]
    
class_counts_loss_df = pd.DataFrame(
    class_counts_df.values(),
    index=class_counts_df.keys(),
    columns=counts_columns
)
    
results_loss_df = pd.DataFrame(
    results_df.values(),
    index=results_df.keys(),
    columns=results_columns
)
    
print(class_counts_loss_df.shape)
class_counts_loss_df.head()


# In[9]:


print(results_loss_df.shape)
results_loss_df.head()


# In[10]:


class_counts_df = {}
results_df = {}
counts_columns = None
results_columns = None

# get distances for copy gain, for class 1/3/4 genes
for identifier, gain_df in park_gain_info.items():
    
    if gain_df.head(1).class_name.values[0] == 'class 2':
        continue
        
    results = get_centroids_and_distance(identifier, gain_df, 'mean')
    
    if counts_columns is None:
        counts_columns = results[0]
    else:
        assert counts_columns == results[0]
        
    if results_columns is None:
        results_columns = ['{}/{}'.format(i, j) for i, j in results[1]]
            
    class_counts_df[identifier] = results[2]
    results_df[identifier] = results[3]
    
class_counts_gain_df = pd.DataFrame(
    class_counts_df.values(),
    index=class_counts_df.keys(),
    columns=counts_columns
)
    
results_gain_df = pd.DataFrame(
    results_df.values(),
    index=results_df.keys(),
    columns=results_columns
)
    
print(class_counts_gain_df.shape)
class_counts_gain_df.head()


# In[11]:


print(results_gain_df.shape)
results_gain_df.head()


# ### Plot centroid distance results
