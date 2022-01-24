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


# ### DE between IDH1 mutant/wild-type samples in low-grade glioma

# In[5]:


cfg.de_input_dir.mkdir(parents=True, exist_ok=True)
cfg.de_output_dir.mkdir(parents=True, exist_ok=True)

base_dir = str(cfg.de_base_dir)
output_dir = str(cfg.de_output_dir)


# In[6]:


# get LGG samples from counts data
lgg_samples = (
    sample_info_df[sample_info_df.cancer_type == 'LGG'].index
      .intersection(all_counts_df.index)
      .intersection(mutation_df.index)
)
lgg_counts_df = all_counts_df.loc[lgg_samples, :]
print(lgg_counts_df.shape)
lgg_counts_df.iloc[:5, :5]


# In[7]:


# save LGG samples to file, to be loaded by DESeq2
input_file = cfg.de_input_dir / 'lgg_counts.tsv'
input_str = str(input_file)

lgg_counts_df.to_csv(input_file, sep='\t')


# In[8]:


# get IDH1 mutation status
idh1_status_df = (mutation_df
    .loc[lgg_samples, ['IDH1']]
    .rename(columns={'IDH1': 'group'})
)
idh1_status_df.head()


# In[9]:


# save mutation status to file, to be loaded by DESeq2
input_metadata_file = cfg.de_input_dir / 'lgg_idh1_status.tsv'
input_metadata_str = str(input_metadata_file)

idh1_status_df.to_csv(input_metadata_file, sep='\t')


# In[10]:


get_ipython().run_line_magic('load_ext', 'rpy2.ipython')


# In[11]:


get_ipython().run_cell_magic('R', '-i base_dir -i input_metadata_str -i input_str -i output_dir', "\nsource(paste0(base_dir, '/de_analysis.R'))\n\nget_DE_stats_DESeq(input_metadata_str,\n                   input_str,\n                   'LGG_IDH1',\n                   output_dir)")


# ### DE between random samples in low-grade glioma
# 
# We do this to generate an empirical null distribution for our results in IDH1 mutants/wild-type samples.

# In[12]:


# number of random samples
n_samples = 5


# In[13]:


n_mutated = idh1_status_df.sum().values[0]
n_not_mutated = idh1_status_df.shape[0] - n_mutated
print(n_mutated, n_not_mutated)


# In[14]:


# we can use sklearn train_test_split to partition the data randomly
import numpy as np
from sklearn.model_selection import train_test_split

for sample_ix in range(n_samples):
    _, test_ixs = train_test_split(idh1_status_df.index,
                                   test_size=n_mutated,
                                   shuffle=True,
                                   random_state=sample_ix)
    labels_df = pd.DataFrame(
        np.zeros(idh1_status_df.shape[0]).astype(int),
        index=idh1_status_df.index.copy(),
        columns=['group']
    )
    labels_df.loc[test_ixs, 'group'] = 1
    
    save_file = cfg.de_input_dir / 'lgg_idh1_random_s{}.tsv'.format(sample_ix)
    print(str(save_file))
    labels_df.to_csv(save_file, sep='\t')


# In[15]:


input_metadata_dir = str(cfg.de_input_dir)


# In[ ]:


get_ipython().run_cell_magic('R', '-i base_dir -i input_str -i n_samples -i input_metadata_dir -i output_dir', "\nsource(paste0(base_dir, '/de_analysis.R'))\n\nfor (i in 0:(n_samples-1)) {\n    print(paste('Running: ', i))\n    input_metadata_str <- paste(\n        input_metadata_dir, '/lgg_idh1_random_s', i, '.tsv',\n        sep=''\n    )\n    get_DE_stats_DESeq(input_metadata_str,\n                       input_str,\n                       paste('LGG_IDH1_random_s', i, sep=''),\n                       output_dir)\n}")

