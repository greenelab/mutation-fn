#!/usr/bin/env python
# coding: utf-8

# ## Differential expression analysis control
# 
# As a way of quantifying how similar/different the expression profiles associated with different mutation patterns are, we want to count the number of differentially expressed genes between them. To make sure this makes sense, we first want to take some cancer subtypes we know are quite different, and compare the number of DE genes between them to the number of DE genes between random samples of the same size.
# 
# We expect to see that different subtypes have considerably more DE genes between them than random samples taken uniformly from the different subtypes.

# In[1]:


from pathlib import Path
import pickle as pkl

import pandas as pd

import sys; sys.path.append('..')
import config as cfg

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# ### Load datasets

# In[2]:


# load counts data
all_counts_df = pd.read_csv(cfg.processed_counts_file, sep='\t', index_col=0)
print(all_counts_df.shape)
all_counts_df.iloc[:5, :5]


# In[3]:


# load cancer types
sample_info_df = pd.read_csv(cfg.de_sample_info, sep='\t', index_col=0)
print(sample_info_df.shape)
sample_info_df.head()


# In[4]:


# load mutation status
pancancer_pickle = Path('/home/jake/research/mpmp/data/pancancer_data.pkl')
with open(pancancer_pickle, 'rb') as f:
    pancancer_data = pkl.load(f)
    
mutation_df = pancancer_data[1]
print(mutation_df.shape)
mutation_df.iloc[:5, :5]


# ### Differential expression between IDH1 mutant/wild-type samples in glioma
