#!/usr/bin/env python
# coding: utf-8

# ## Download multiple modalities of pan-cancer data from TCGA
# 
# The data is accessed directly from the [Genome Data Commons](https://gdc.cancer.gov/about-data/publications/pancanatlas).
# 
# NOTE: this download script uses the `md5sum` shell utility to verify file hashes. This script was developed and tested on a Linux machine, and `md5sum` commands may have to be changed to work on other platforms.

# In[1]:


import pandas as pd
from urllib.request import urlretrieve

import sys; sys.path.append('..')
import config as cfg


# First, we load a manifest file containing the GDC API ID and filename for each relevant file, as well as the md5 checksum to make sure the whole/uncorrupted file was downloaded.
# 
# The manifest included in this GitHub repo was downloaded from https://gdc.cancer.gov/node/971 on December 1, 2020.

# In[2]:


manifest_df = pd.read_csv(cfg.data_dir / 'manifest.tsv',
                          sep='\t', index_col=0)
manifest_df.head()


# ### Download gene expression data

# In[3]:


rnaseq_id, rnaseq_filename = manifest_df.loc['rna_seq'].id, manifest_df.loc['rna_seq'].filename
url = 'http://api.gdc.cancer.gov/data/{}'.format(rnaseq_id)
exp_filepath = cfg.data_dir / rnaseq_filename

if not exp_filepath.is_file():
    urlretrieve(url, exp_filepath)
else:
    print('Downloaded data file already exists, skipping download')


# In[4]:


md5_sum = get_ipython().getoutput('md5sum $exp_filepath')
print(md5_sum[0])
assert md5_sum[0].split(' ')[0] == manifest_df.loc['rna_seq'].md5


# ### Download mutation data

# In[5]:


mutation_id, mutation_filename = manifest_df.loc['mutation'].id, manifest_df.loc['mutation'].filename
url = 'http://api.gdc.cancer.gov/data/{}'.format(mutation_id)
mutation_filepath = cfg.data_dir / mutation_filename

if not mutation_filepath.is_file():
    urlretrieve(url, mutation_filepath)
else:
    print('Downloaded data file already exists, skipping download')


# In[6]:


md5_sum = get_ipython().getoutput('md5sum $mutation_filepath')
print(md5_sum[0])
assert md5_sum[0].split(' ')[0] == manifest_df.loc['mutation'].md5


# ### Download gene set from Park et al. paper
# 
# We want to download the set of genes analyzed in [Park et al. 2021](https://www.nature.com/articles/s41467-021-27242-3). We are particularly interested in the "Class 2/3/4" genes from Figure 1, which are inferred to be "two-hit" genes where non-synonymous mutations and CNVs tend to co-occur more often than would be expected by chance.

# In[7]:


import pandas as pd

park_df = pd.read_excel(
    'https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-021-27242-3/MediaObjects/41467_2021_27242_MOESM11_ESM.xlsx',
    sheet_name='SupplFig3', header=1
)
print(park_df.shape)
park_df.head()


# In[8]:


# the paper uses an FDR threshold of 0.1, so we do the same here
fdr_threshold = 0.1

park_loss_df = (park_df
  .loc[park_df.FDR < fdr_threshold, :]
  .iloc[:, :8]
  .set_index('Pair')
)
print(park_loss_df.shape)
print(park_loss_df.Gene.unique())
park_loss_df.head()


# In[9]:


park_gain_df = (park_df
  .loc[park_df['FDR.1'] < fdr_threshold, :]
  .iloc[:, 9:]
)
park_gain_df.columns = park_gain_df.columns.str.replace('.1', '')
park_gain_df.set_index('Pair', inplace=True)
print(park_gain_df.shape)
print(park_gain_df.Gene.unique())
park_gain_df.head()


# ### Download oncogene/tumor suppressor information for Park et al. genes

# In[10]:


# downloaded from https://www.sciencedirect.com/science/article/pii/S009286741830237X
# oncogene/TSG predictions for genes/cancer types using 20/20+ classifier
class_df = pd.read_excel(
    cfg.data_dir / '1-s2.0-S009286741830237X-mmc1.xlsx', 
    sheet_name='Table S1', index_col='KEY', header=2
)
class_df.rename(columns={'Tumor suppressor or oncogene prediction (by 20/20+)':
                         'classification'},
                inplace=True)

class_df.head()


# In[11]:


loss_class_df = (park_loss_df
    .merge(class_df.loc[:, ['classification']], left_index=True, right_index=True)
)

# format oncogene/TSG classification to work with vogelstein genes
loss_class_df['classification'] = (
    loss_class_df.classification.str.replace('possible ', '')
                                    .replace('tsg', 'TSG')
                                    .replace('oncogene', 'Oncogene')
)

print(loss_class_df.classification.unique())
loss_class_df.head()


# In[12]:


gain_class_df = (park_gain_df
    .merge(class_df.loc[:, ['classification']], left_index=True, right_index=True)
)

# format oncogene/TSG classification to work with vogelstein genes
gain_class_df['classification'] = (
    gain_class_df.classification.str.replace('possible ', '')
                                    .replace('tsg', 'TSG')
                                    .replace('oncogene', 'Oncogene')
)

print(gain_class_df.classification.unique())
gain_class_df.head()


# In[13]:


loss_class_df.to_csv(cfg.data_dir / 'park_loss_df.tsv', sep='\t')
gain_class_df.to_csv(cfg.data_dir / 'park_gain_df.tsv', sep='\t')

