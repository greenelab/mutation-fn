#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pathlib import Path

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib_venn import venn2, venn2_unweighted

import sys; sys.path.append('..')
import config as cfg


# ### Load Vogelstein and Park gene info

# In[2]:


vogelstein_genes = '/'.join([
    cfg.vogelstein_base_url,
    cfg.vogelstein_commit,
    'data',
    'vogelstein_cancergenes.tsv'
])
print(vogelstein_genes)


# In[3]:


vogelstein_df = (
    pd.read_csv(vogelstein_genes, sep='\t')
      .rename(columns={'Gene Symbol'   : 'gene',
                       'Classification*': 'classification'})
)
vogelstein_df.head()


# In[4]:


park_loss_df = pd.read_csv(cfg.data_dir / 'park_loss_df.tsv', sep='\t')
park_loss_df.head()


# In[5]:


park_gain_df = pd.read_csv(cfg.data_dir / 'park_gain_df.tsv', sep='\t')
park_gain_df.head()


# ### Look at overlap between Park and Vogelstein genes

# In[6]:


sns.set_style('white')

park_loss_genes = set(park_loss_df.Gene.unique())
vogelstein_genes = set(vogelstein_df.gene.unique())

venn2_unweighted([park_loss_genes, vogelstein_genes], ['Park (loss)', 'Vogelstein'])
plt.title('Park/Vogelstein gene set overlap')


# In[8]:


print(len(park_loss_genes - vogelstein_genes))
print(park_loss_genes - vogelstein_genes)


# In[7]:


sns.set_style('white')

park_gain_genes = set(park_gain_df.Gene.unique())
vogelstein_genes = set(vogelstein_df.gene.unique())

venn2_unweighted([park_gain_genes, vogelstein_genes], ['Park (gain)', 'Vogelstein'])
plt.title('Park/Vogelstein gene set overlap')


# In[9]:


print(len(park_gain_genes - vogelstein_genes))
print(park_gain_genes - vogelstein_genes)

