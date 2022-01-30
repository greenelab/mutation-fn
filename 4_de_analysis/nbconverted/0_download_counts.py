#!/usr/bin/env python
# coding: utf-8

# ## Download pan-cancer RNA-seq read counts data from UCSC Xena Browser
# 
# The other RNA-seq data we downloaded in `0_data_download` contains pre-processed RPKM values. In order to do differential expression analysis, most methods recommend using count data, or something similar such as [RSEM expected counts](https://support.bioconductor.org/p/90672/#90678) (which is what we'll download here). This is because most DE methods do their own normalization to ensure that expression levels are comparable between samples, which is not necessarily true for quantities such as RPKM; [see here](https://hbctraining.github.io/DGE_workshop/lessons/02_DGE_count_normalization.html) for a more detailed explanation.
# 
# GDC does not seem to store RNA-seq read counts (that I'm aware of), so we'll download it from the UCSC Xena Browser instead. This data was generated as part of the Pan-Cancer Atlas project so it should apply to the same set of samples.

# In[1]:


import pandas as pd

import sys; sys.path.append('..')
import config as cfg

cfg.de_data_dir.mkdir(parents=True, exist_ok=True)
cfg.raw_de_data_dir.mkdir(parents=True, exist_ok=True)


# In[2]:


base_url = 'https://toil-xena-hub.s3.us-east-1.amazonaws.com/download/'
filename = 'tcga_gene_expected_count'

url = base_url + filename + '.gz'
output_filename = cfg.raw_de_data_dir / (filename + '.tsv.gz')

if not output_filename.is_file():
    print('Raw data file does not exist, downloading...')
    counts_df = pd.read_csv(url, sep='\t')
    counts_df.to_csv(output_filename, sep='\t')
else:
    print('Loading from existing raw data file')
    counts_df = pd.read_csv(output_filename, sep='\t', index_col=0)
    
counts_df.iloc[:5, :5]


# ## Process counts matrix

# In[3]:


print(counts_df.shape)

counts_df = (counts_df
    .set_index('sample')
    .dropna(axis='rows')
    .transpose()
    .sort_index(axis='rows')
    .sort_index(axis='columns')
)

counts_df.index.rename('sample_id', inplace=True)
counts_df.columns.name = None


# In[4]:


counts_df.iloc[:5, :5]


# In[5]:


# per the documentation for the Xena Browser, these are log-transformed
# expected counts - see: 
# https://toil-xena-hub.s3.us-east-1.amazonaws.com/download/tcga_gene_expected_count.json
#
# we want to un-log transform them here (2^x - 1), and round to the nearest integer,
# to prepare for DE analysis
print('After transform:', counts_df.min().min(), counts_df.max().max())
counts_df = ((2 ** counts_df) - 1).round(0).astype(int)
print('Before transform:', counts_df.min().min(), counts_df.max().max())


# In[6]:


counts_df.to_csv(cfg.processed_counts_file, sep='\t')

