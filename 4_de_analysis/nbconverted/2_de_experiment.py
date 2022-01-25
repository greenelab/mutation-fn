#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pathlib import Path
import pickle as pkl
import itertools as it

import pandas as pd

import sys; sys.path.append('..')
import config as cfg

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[2]:


# load gene expression counts data
all_counts_df = pd.read_csv(cfg.processed_counts_file, sep='\t', index_col=0)
print(all_counts_df.shape)
all_counts_df.iloc[:5, :5]


# In[3]:


# load park et al. initial analysis results here
park_info_df = pd.read_csv(
    cfg.park_info_file, sep='\t'
)

print(park_info_df.shape)
park_info_df.head()


# In[4]:


# for now, we'll just run these with a few hand-picked examples
# in the future we should run for all examples + do some kind of synthesis
class_1_id = 'NF2_KIRP'
class_2_id = 'PTEN_UCEC'
class_3_id = 'KRAS_COADREAD'
class_4_id = 'TP53_BRCA'

class_ids = [class_1_id, class_2_id, class_3_id, class_4_id]


# In[5]:


exp_output = str(cfg.de_input_dir / '{}_{}_{}_counts.tsv')
info_output = str(cfg.de_input_dir / '{}_{}_{}_info.tsv')

for class_id in class_ids:
    gene, cancer_type = class_id.split('_')
    for status_1, status_2 in it.combinations(
        ['none', 'one', 'both'], 2
    ):
        
        print(class_id, status_1, status_2)
        
        info_df = park_info_df[
            ((park_info_df.identifier == class_id) &
             (park_info_df.status.isin([status_1, status_2])))
        ].set_index('sample_id')
        
        samples = (
            info_df.index.intersection(all_counts_df.index)
        )
        exp_df = all_counts_df.loc[samples, :]
        info_df = info_df.loc[samples, :]
        
        exp_df.to_csv(
            exp_output.format(class_id, status_1, status_2),
            sep='\t'
        )
        info_df.to_csv(
            info_output.format(class_id, status_1, status_2),
            sep='\t'
        )

