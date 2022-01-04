#!/usr/bin/env python
# coding: utf-8

# ## Download multiple modalities of pan-cancer data from TCGA
# 
# The data is accessed directly from the [Genome Data Commons](https://gdc.cancer.gov/about-data/publications/pancanatlas).
# 
# NOTE: this download script uses the `md5sum` shell utility to verify file hashes. This script was developed and tested on a Linux machine, and `md5sum` commands may have to be changed to work on other platforms.

# In[1]:


import os
import pandas as pd
from urllib.request import urlretrieve


# First, we load a manifest file containing the GDC API ID and filename for each relevant file, as well as the md5 checksum to make sure the whole/uncorrupted file was downloaded.
# 
# The manifest included in this GitHub repo was downloaded from https://gdc.cancer.gov/node/971 on December 1, 2020.

# In[2]:


manifest_df = pd.read_csv(os.path.join('data', 'manifest.tsv'),
                          sep='\t', index_col=0)
manifest_df.head()


# ### Download gene expression data

# In[5]:


rnaseq_id, rnaseq_filename = manifest_df.loc['rna_seq'].id, manifest_df.loc['rna_seq'].filename
url = 'http://api.gdc.cancer.gov/data/{}'.format(rnaseq_id)
exp_filepath = os.path.join('data', rnaseq_filename)

if not os.path.exists(exp_filepath):
    urlretrieve(url, exp_filepath)
else:
    print('Downloaded data file already exists, skipping download')


# In[6]:


md5_sum = get_ipython().getoutput('md5sum $exp_filepath')
print(md5_sum[0])
assert md5_sum[0].split(' ')[0] == manifest_df.loc['rna_seq'].md5


# ### Download mutation data

# In[9]:


mutation_id, mutation_filename = manifest_df.loc['mutation'].id, manifest_df.loc['mutation'].filename
url = 'http://api.gdc.cancer.gov/data/{}'.format(mutation_id)
mutation_filepath = os.path.join('data', mutation_filename)

if not os.path.exists(mutation_filepath):
    urlretrieve(url, mutation_filepath)
else:
    print('Downloaded data file already exists, skipping download')


# In[10]:


md5_sum = get_ipython().getoutput('md5sum $exp_filepath')
print(md5_sum[0])
assert md5_sum[0].split(' ')[0] == manifest_df.loc['mutation'].md5

