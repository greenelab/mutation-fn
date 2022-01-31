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

import pandas as pd

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

# In[9]:


def get_centroids_and_distance(identifier, info_map, centroid_method='mean'):
    park_id_df = info_map[identifier]
    print(park_id_df.shape)
    print(park_id_df.head())
    print(park_id_df.index.duplicated().sum())
    # print(park_id_df[(park_id_df.copy_status > 0.0)].sort_values(by='sample_id').head(20))
    # samples = park_id_df.index.intersection(data_df.index)
    # park_id_df = park_id_df.reindex(samples)
    # expression_df = data_df.reindex(samples)
    # print(park_id_df.shape)
    # print(park_id_df.head())
    # print(expression_df.shape)
    # print(expression_df.iloc[:5, :5])
    hit_class_counts = park_id_df.groupby('num_hits').count().class_name
    return hit_class_counts
    # return (class_counts, centroids, distance)
    
print(get_centroids_and_distance('TP53_BRCA', park_loss_info))

