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


# if True rerun DE analysis and overwrite existing results 
# if False look for existing results and don't rerun DE analysis
# (the latter makes the notebook run much faster)
RUN_DE_ANALYSIS = False


# In[3]:


cfg.de_input_dir.mkdir(parents=True, exist_ok=True)
cfg.de_output_dir.mkdir(parents=True, exist_ok=True)


# In[4]:


# load gene expression counts data
all_counts_df = pd.read_csv(cfg.processed_counts_file, sep='\t', index_col=0)
print(all_counts_df.shape)
all_counts_df.iloc[:5, :5]


# In[5]:


# load park et al. initial analysis results here
park_info_df = pd.read_csv(
    cfg.park_info_file, sep='\t'
)

print(park_info_df.shape)
park_info_df.head()


# In[6]:


# for now, we'll just run these with a few hand-picked examples
# in the future we should run for all examples + do some kind of synthesis
class_1_id = 'NF2_KIRP'
class_2_id = 'PTEN_UCEC'
class_3_id = 'KRAS_COADREAD'
class_4_id = 'TP53_BRCA'

class_ids = [class_1_id, class_2_id, class_3_id, class_4_id]


# In[7]:


exp_output = str(cfg.de_input_dir / '{}_{}_{}_counts.tsv')
info_output = str(cfg.de_input_dir / '{}_{}_{}_info.tsv')

for class_id in class_ids:
    gene, cancer_type = class_id.split('_')
    for status_1, status_2 in it.combinations(
        ['none', 'one', 'both'], 2
    ):
        
        print(class_id, status_1, status_2)
        
        # TODO figure out where the duplicates come from
        info_df = park_info_df[
            ((park_info_df.identifier == class_id) &
             (park_info_df.status.isin([status_1, status_2])))
        ].drop_duplicates()
        info_df.set_index('sample_id', inplace=True)
        info_df.rename(columns={'status': 'group'}, inplace=True)
        
        samples = (
            info_df.index.intersection(all_counts_df.index)
        )
        exp_df = all_counts_df.loc[samples, :]
        info_df = info_df.loc[samples, :]
        
        exp_df.to_csv(
            exp_output.format(class_id, status_1, status_2),
            sep='\t'
        )
        info_df.group.to_csv(
            info_output.format(class_id, status_1, status_2),
            sep='\t'
         )
    
info_df.head()


# In[8]:


base_dir = str(cfg.de_base_dir)
input_dir = str(cfg.de_input_dir)
output_dir = str(cfg.de_output_dir)


# In[9]:


get_ipython().run_line_magic('load_ext', 'rpy2.ipython')


# In[10]:


get_ipython().run_cell_magic('R', '-i RUN_DE_ANALYSIS -i base_dir -i input_dir -i output_dir ', "\nif (RUN_DE_ANALYSIS) {\n    source(paste0(base_dir, '/de_analysis.R'))\n\n    # this stuff probably shouldn't be hardcoded, but fine for now\n    identifiers <- c(\n        'NF2_KIRP',\n        'PTEN_UCEC',\n        'KRAS_COADREAD',\n        'TP53_BRCA'\n    )\n\n    status_combs <- list()\n    status_combs[[1]] <- c('none', 'one')\n    status_combs[[2]] <- c('none', 'both')\n    status_combs[[3]] <- c('one', 'both')\n\n    for (i in 1:length(identifiers)) {\n        identifier <- identifiers[i]\n        for (j in 1:length(status_combs)) {\n            status_1 <- status_combs[[j]][1]\n            status_2 <- status_combs[[j]][2]\n            counts_file <- paste(input_dir,\n                                 '/',\n                                 identifier,\n                                 '_',\n                                 status_1,\n                                 '_',\n                                 status_2,\n                                 '_counts.tsv',\n                                 sep='')\n            info_file <- paste(input_dir,\n                               '/',\n                               identifier,\n                               '_',\n                               status_1,\n                               '_',\n                               status_2,\n                               '_info.tsv',\n                               sep='')\n            print(counts_file)\n            print(info_file)\n            get_DE_stats_DESeq(info_file,\n                               counts_file,\n                               paste(identifier, '_', status_1, '_', status_2,\n                                     sep=''),\n                               output_dir)\n\n        }\n    }\n} else {\n    print('Skipping DE analysis, will use existing results files')\n}")

