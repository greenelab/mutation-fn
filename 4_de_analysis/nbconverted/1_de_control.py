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


# In[2]:


# if True rerun DE analysis and overwrite existing results 
# if False look for existing results and don't rerun DE analysis
# (the latter makes the notebook run much faster)
RUN_DE_ANALYSIS = False


# ### Load datasets

# In[3]:


# load counts data
all_counts_df = pd.read_csv(cfg.processed_counts_file, sep='\t', index_col=0)
print(all_counts_df.shape)
all_counts_df.iloc[:5, :5]


# In[4]:


# load cancer types
sample_info_df = pd.read_csv(cfg.de_sample_info, sep='\t', index_col=0)
print(sample_info_df.shape)
sample_info_df.head()


# In[5]:


# load mutation status
pancancer_pickle = Path('/home/jake/research/mpmp/data/pancancer_data.pkl')
with open(pancancer_pickle, 'rb') as f:
    pancancer_data = pkl.load(f)
    
mutation_df = pancancer_data[1]
print(mutation_df.shape)
mutation_df.iloc[:5, :5]


# ### DE between IDH1 mutant/wild-type samples in low-grade glioma

# In[6]:


cfg.de_input_dir.mkdir(parents=True, exist_ok=True)
cfg.de_output_dir.mkdir(parents=True, exist_ok=True)

base_dir = str(cfg.de_base_dir)
output_dir = str(cfg.de_output_dir)


# In[7]:


# get LGG samples from counts data
lgg_samples = (
    sample_info_df[sample_info_df.cancer_type == 'LGG'].index
      .intersection(all_counts_df.index)
      .intersection(mutation_df.index)
)
lgg_counts_df = all_counts_df.loc[lgg_samples, :]
print(lgg_counts_df.shape)
lgg_counts_df.iloc[:5, :5]


# In[8]:


# save LGG samples to file, to be loaded by DESeq2
input_file = cfg.de_input_dir / 'lgg_counts.tsv'
input_str = str(input_file)

lgg_counts_df.to_csv(input_file, sep='\t')


# In[9]:


# get IDH1 mutation status
idh1_status_df = (mutation_df
    .loc[lgg_samples, ['IDH1']]
    .rename(columns={'IDH1': 'group'})
)
idh1_status_df.head()


# In[10]:


# save mutation status to file, to be loaded by DESeq2
input_metadata_file = cfg.de_input_dir / 'lgg_idh1_status.tsv'
input_metadata_str = str(input_metadata_file)

idh1_status_df.to_csv(input_metadata_file, sep='\t')


# In[11]:


get_ipython().run_line_magic('load_ext', 'rpy2.ipython')


# In[12]:


get_ipython().run_cell_magic('R', '-i RUN_DE_ANALYSIS -i base_dir -i input_metadata_str -i input_str -i output_dir ', "\nif (RUN_DE_ANALYSIS) {\n    source(paste0(base_dir, '/de_analysis.R'))\n\n    get_DE_stats_DESeq(input_metadata_str,\n                       input_str,\n                       'LGG_IDH1',\n                       output_dir)\n} else {\n    print('Skipping DE analysis, will use existing results files')\n}")


# ### DE between random samples in low-grade glioma
# 
# We do this to generate an empirical null distribution for our results in IDH1 mutants/wild-type samples.

# In[13]:


# number of random samples
n_samples = 5


# In[14]:


n_mutated = idh1_status_df.sum().values[0]
n_not_mutated = idh1_status_df.shape[0] - n_mutated
print(n_mutated, n_not_mutated)


# In[15]:


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


# In[16]:


input_metadata_dir = str(cfg.de_input_dir)


# In[17]:


get_ipython().run_cell_magic('R', '-i RUN_DE_ANALYSIS -i base_dir -i input_str -i n_samples -i input_metadata_dir -i output_dir', "\nif (RUN_DE_ANALYSIS) {\n    source(paste0(base_dir, '/de_analysis.R'))\n\n    for (i in 0:(n_samples-1)) {\n        print(paste('Running: ', i))\n        input_metadata_str <- paste(\n            input_metadata_dir, '/lgg_idh1_random_s', i, '.tsv',\n            sep=''\n        )\n        get_DE_stats_DESeq(input_metadata_str,\n                           input_str,\n                           paste('LGG_IDH1_random_s', i, sep=''),\n                           output_dir)\n    }\n} else {\n    print('Skipping DE analysis, will use existing results files')\n}")


# ### Compare IDH1 mutation DE results to randomly sampled results

# In[18]:


idh1_de_results_df = pd.read_csv(
    cfg.de_output_dir / 'DE_stats_LGG_IDH1.txt',
    sep='\t'
)

print(idh1_de_results_df.shape)
idh1_de_results_df.head()


# In[19]:


random_de_results = []
for i in range(n_samples):
    
    random_de_results.append(
        pd.read_csv(
            cfg.de_output_dir / 'DE_stats_LGG_IDH1_random_s{}.txt'.format(i),
            sep='\t'
        )
    )
    
print(random_de_results[0].shape)
random_de_results[0].head()


# In[20]:


# adjusted p-value threshold
alpha = 0.05

idh1_de_count = (
    (idh1_de_results_df.padj < alpha).sum()
)

random_de_count = [
    (random_de_results[ix].padj < alpha).sum() for ix in range(n_samples)
]

print('DE genes for IDH1 WT vs. mutant:', idh1_de_count)
print('DE genes for random size-matched samples:', random_de_count)


# In[30]:


import matplotlib.pyplot as plt
import seaborn as sns

sns.set({'figure.figsize': (8, 6)})

sns.kdeplot(data=idh1_de_results_df.pvalue, label='true')
for ix in range(n_samples):
    if ix == 0:
        sns.kdeplot(data=random_de_results[ix].pvalue,
                    color='red', label='random')
    else:
        sns.kdeplot(data=random_de_results[ix].pvalue, color='red')
plt.title('Uncorrected p-value density distributions')
plt.xlabel('uncorrected p-value')
plt.legend()


# In[31]:


import matplotlib.pyplot as plt
import seaborn as sns

sns.set({'figure.figsize': (8, 6)})

sns.kdeplot(data=idh1_de_results_df.padj, label='true')
for ix in range(n_samples):
    if ix == 0:
        sns.kdeplot(data=random_de_results[ix].padj,
                    color='red', label='random')
    else:
        sns.kdeplot(data=random_de_results[ix].padj, color='red')
plt.title('FDR corrected p-value density distributions')
plt.xlabel('Corrected p-value')
plt.legend()

