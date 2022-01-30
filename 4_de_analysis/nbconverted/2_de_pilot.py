#!/usr/bin/env python
# coding: utf-8

# ## Differential expression analysis pilot study
# 
# As a way of quantifying how similar/different the expression profiles associated with different mutation patterns are, we want to count the number of differentially expressed genes between them. This is a pilot experiment using a small subset of the genes/cancer types from Park et al., to see how well the number of DE genes lines up with the expectations for one-hit vs. multi-hit drivers described in that paper.

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


# ### Load relevant data (gene expression & mutation status)

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


# ### Run differential expression for selected examples
# 
# For now, we'll just run these with a few hand-picked examples, one from each Park et al. "class". The "classes" are described in the `3_park_analysis.ipynb` notebook.

# In[6]:


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


# In[30]:


identifier = 'NF2_KIRP'
# adjusted p-value threshold
alpha = 0.05

none_one_de_results = pd.read_csv(
    cfg.de_output_dir / 'DE_stats_{}_none_one.txt'.format(identifier), sep='\t'
)
print(none_one_de_results.shape)
none_one_de_results.head()


# In[31]:


one_both_de_results = pd.read_csv(
    cfg.de_output_dir / 'DE_stats_{}_one_both.txt'.format(identifier), sep='\t'
)
print(one_both_de_results.shape)
one_both_de_results.head()


# In[32]:


none_both_de_results = pd.read_csv(
    cfg.de_output_dir / 'DE_stats_{}_none_both.txt'.format(identifier), sep='\t'
)
print(none_both_de_results.shape)
none_both_de_results.head()


# In[33]:


none_one_de_count = (
    (none_one_de_results.padj < alpha).sum()
)
one_both_de_count = (
    (one_both_de_results.padj < alpha).sum()
)
none_both_de_count = (
    (none_both_de_results.padj < alpha).sum()
)

print('DE genes for {} none vs. one:'.format(identifier), none_one_de_count)
print('DE genes for {} one vs. both:'.format(identifier), one_both_de_count)
print('DE genes for {} none vs. both:'.format(identifier), none_both_de_count)


# In[34]:


import matplotlib.pyplot as plt
import seaborn as sns

sns.set({'figure.figsize': (10, 8)})
fig, axarr = plt.subplots(2, 1)

sns.kdeplot(data=none_one_de_results.pvalue, label='none/one', ax=axarr[0])
sns.kdeplot(data=one_both_de_results.pvalue, label='one/both', ax=axarr[0])
sns.kdeplot(data=none_both_de_results.pvalue, label='none/both', ax=axarr[0])
axarr[0].set_title('Uncorrected p-value density distributions')
axarr[0].set_xlabel('Uncorrected p-value')
axarr[0].legend()

sns.kdeplot(data=none_one_de_results.padj, label='none/one', ax=axarr[1])
sns.kdeplot(data=one_both_de_results.padj, label='one/both', ax=axarr[1])
sns.kdeplot(data=none_both_de_results.padj, label='none/both', ax=axarr[1])
axarr[1].set_title('FDR corrected p-value density distributions')
axarr[1].set_xlabel('Corrected p-value')
axarr[1].legend()

plt.tight_layout()


# Results for all pilot examples are shown in [these Google slides](https://docs.google.com/presentation/d/1hjcxLztM2XbeNwIYL3fH0Xb9GsBCWFdMIH-9Q8jlXok/edit?usp=sharing). Within genes, we seem to see roughly the same patterns we saw with our mutation classifiers in `3_park_analysis`. However, it seems pretty clear from our examples that between-gene comparisons are going to be hard due to variability in sample size, as comparisons with more samples in the classes being compared are almost always going to have better power to detect DE genes.
# 
# Our goal in our next analysis, then, will be to find some sort of distance metric or projection that can compare groups of samples, while normalizing or averaging over varying sample counts.
