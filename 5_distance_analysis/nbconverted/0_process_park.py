#!/usr/bin/env python
# coding: utf-8

# ## Load and process Park et al. data
# 
# For each sample, we want to compute:
# 
# * (non-silent) binary mutation status in the gene of interest
# * binary copy gain/loss status in the gene of interest
# * what "class" the gene of interest is in (more detail on what this means below)
# 
# We'll save this to a file since the preprocessing takes a few minutes, so we can load it quickly in downstream analysis scripts.

# In[1]:


from pathlib import Path
import pickle as pkl

import pandas as pd

import sys; sys.path.append('..')
import config as cfg

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[2]:


# park et al. geneset info
park_loss_data = cfg.data_dir / 'park_loss_df.tsv'
park_gain_data = cfg.data_dir / 'park_gain_df.tsv'

# park et al. significant gene info
park_loss_sig_data = cfg.data_dir / 'park_loss_df_sig_only.tsv'
park_gain_sig_data = cfg.data_dir / 'park_gain_df_sig_only.tsv'

# mutation and copy number data
pancancer_pickle = Path('/home/jake/research/mpmp/data/pancancer_data.pkl')


# ### Load data from Park et al. supp. info

# In[3]:


park_loss_df = pd.read_csv(park_loss_data, sep='\t', index_col=0)
park_loss_df.head()


# In[4]:


park_gain_df = pd.read_csv(park_gain_data, sep='\t', index_col=0)
park_gain_df.head()


# ### Load mutation and CNV info

# In[5]:


with open(pancancer_pickle, 'rb') as f:
    pancancer_data = pkl.load(f)


# In[6]:


# get (binary) mutation data
# 1 = observed non-silent mutation in this gene for this sample, 0 otherwise
mutation_df = pancancer_data[1]
print(mutation_df.shape)
mutation_df.iloc[:5, :5]


# In[7]:


# we use the data source and preprocessing code from the pancancer repo, here:
# https://github.com/greenelab/pancancer/blob/d1b3de7fa387d0a44d0a4468b0ac30918ed66886/scripts/initialize/process_copynumber.py#L21

copy_thresh_df = (
    pd.read_csv(cfg.data_dir / 'pancan_GISTIC_threshold.tsv',
                sep='\t', index_col=0)
      .drop(columns=['Locus ID', 'Cytoband'])
)
copy_thresh_df.columns = copy_thresh_df.columns.str[0:15]

# thresholded copy number includes 5 values [-2, -1, 0, 1, 2], which
# correspond to "deep loss", "moderate loss", "no change",
# "moderate gain", and "deep gain", respectively.
print(copy_thresh_df.shape)
copy_thresh_df.iloc[:5, :5]


# In[8]:


sample_freeze_df = pancancer_data[0]
copy_samples = list(
    set(sample_freeze_df.SAMPLE_BARCODE)
    .intersection(set(copy_thresh_df.columns))
)
print(len(copy_samples))


# In[9]:


# make sure we're not losing too many samples, a few is fine
print(sorted(set(sample_freeze_df.SAMPLE_BARCODE) - set(copy_thresh_df.columns)))


# In[10]:


copy_thresh_df = (copy_thresh_df
    .T
    .loc[sorted(copy_samples)]
    .fillna(0)
    .astype(int)
)

print(copy_thresh_df.shape)
copy_thresh_df.iloc[:5, :5]


# In[11]:


# here, we want to use "moderate" and "deep" loss/gain to define CNV
# loss/gain (to match Park et al.)
#
# note that this is different to the more conservative approach of using
# "deep loss/gain" only as in our classifiers

copy_loss_df = (copy_thresh_df
    .replace(to_replace=[1, 2], value=0)
    .replace(to_replace=[-1, -2], value=1)
)
print(copy_loss_df.shape)
copy_loss_df.iloc[:5, :5]


# In[12]:


copy_gain_df = (copy_thresh_df
    .replace(to_replace=[-1, -2], value=0)
    .replace(to_replace=[1, 2], value=1)
)
print(copy_gain_df.shape)
copy_gain_df.iloc[:5, :5]


# ### Classify genes/cancer types into "classes"
# 
# In [the Park et al. paper](https://www.nature.com/articles/s41467-021-27242-3#Sec4), they describe 4 "classes" of driver genes:
# 
# 1. Genes that function exclusively as one-hit drivers, no significant co-occurrence with CNAs
# 2. Genes that interact with CNA loss in at least one cancer type - "two-hit loss" drivers (i.e. classical tumor suppressors)
# 3. Genes that interact with CNA gain in at least one cancer type - "two-hit gain" drivers (for some examples/explanation of "two-hit" oncogenes, see [this paper](https://www.nature.com/articles/s41586-020-2175-2))
# 4. Genes that interact with both CNA loss and CNA gain across multiple cancer types - "two-hit loss and gain" drivers
# 
# Here, we label each of the genes from the Park et al. data with their "class", since we want to segment our analyses in this way too.

# In[13]:


park_loss_sig_df = pd.read_csv(park_loss_sig_data, sep='\t', index_col=0)
park_gain_sig_df = pd.read_csv(park_gain_sig_data, sep='\t', index_col=0)

class_4_ids = (
    set(park_loss_sig_df.index.unique()).intersection(
    set(park_gain_sig_df.index.unique())
))

class_2_ids = set(park_loss_sig_df.index.unique()) - class_4_ids
class_3_ids = set(park_gain_sig_df.index.unique()) - class_4_ids

class_1_ids = (
    set(park_loss_df.index.unique()) - (
        class_4_ids.union(class_2_ids, class_3_ids)
    )
)
print(len(park_loss_df.index.unique()))
print('class 1:', len(class_1_ids))
print('class 2:', len(class_2_ids))
print('class 3:', len(class_3_ids))
print('class 4:', len(class_4_ids))
print(sorted(class_4_ids))


# In[14]:


def id_to_class(i):
    if i in class_2_ids:
        return 'class 2'
    elif i in class_3_ids:
        return 'class 3'
    elif i in class_4_ids:
        return 'class 4'
    else:
        return 'class 1'

loss_class = {i: id_to_class(i) for i in park_loss_df.index.unique()}

park_loss_df['class'] = park_loss_df.index.map(loss_class)
print(park_loss_df['class'].unique())
park_loss_df.head()


# In[15]:


gain_class = {i: id_to_class(i) for i in park_gain_df.index.unique()}

park_gain_df['class'] = park_gain_df.index.map(gain_class)
print(park_gain_df['class'].unique())
park_gain_df.head()


# In[16]:


sample_freeze_df.head()


# ### Retrieve and format per-sample information
# 
# We want to generate a dataframe with the following information:
# 
# * Sample ID, gene/tissue
# * Mutation status (binary) for sample in gene
# * CNV status (binary) for sample in gene, gain/loss for oncogene/TSG respectively
# * Park et al. gene "class" (class 1/2/3/4 as defined above)
# * Sample "number of hits" (none/one/both)

# In[17]:


def get_info_for_gene_and_tissue(identifier, all_info_df, copy_change):
    """Given a gene and tissue, load the relevant mutation information.
    
    'status' is what we will segment our plots by: 'none' == neither a point
    mutation or CNV observed for the given sample, 'one' == either a point
    mutation or CNV but not both, 'both' == both point mutation and CNV
    """
    info_df = {}
    gene, tissue = identifier.split('_')
    if tissue == 'COADREAD':
        tissue_samples = (
            sample_freeze_df[sample_freeze_df.DISEASE.isin(['COAD', 'READ'])]
              .SAMPLE_BARCODE
        )
    else:
        tissue_samples = (
            sample_freeze_df[sample_freeze_df.DISEASE == tissue]
              .SAMPLE_BARCODE
        )
    # TODO: not sure why these don't match
    tissue_samples = (
        mutation_df.index.intersection(tissue_samples)
                         .intersection(copy_loss_df.index)
                         .intersection(copy_gain_df.index)
    )
    class_name = (all_info_df
        .loc[all_info_df.index == identifier, ['class']]
    ).values[0]
    info_df['class_name'] = class_name
    
    # get mutation status for samples
    info_df['mutation_status'] = mutation_df.loc[tissue_samples, gene].values
    
    # get copy status for samples
    if copy_change == 'gain':
        info_df['cnv_status'] = copy_loss_df.loc[tissue_samples, gene].values
    elif copy_change == 'loss':
        info_df['cnv_status'] = copy_gain_df.loc[tissue_samples, gene].values
        
    info_df = pd.DataFrame(info_df, index=tissue_samples)
        
    def hits_from_mut_info(row):
        if row['mutation_status'] == 1 and row['cnv_status'] == 1:
            return 'both'
        elif row['mutation_status'] == 1 or row['cnv_status'] == 1:
            return 'one'
        else:
            return 'none'
        
    info_df['num_hits'] = info_df.apply(hits_from_mut_info, axis=1)
    
    return info_df

get_info_for_gene_and_tissue('TP53_BRCA', park_loss_df, 'loss')


# ### Format and pickle all per-sample info
# 
# We'll end up pickling a dict that maps each identifier (gene/cancer type combination) to a dataframe, assigning a "num_hits" class to each sample for that gene.
# 
# We'll create two of these, one for copy gains and one for copy losses, to be used downstream in our distance/similarity analyses.

# In[18]:


cfg.distance_data_dir.mkdir(exist_ok=True)

park_gain_num_hits = {}
for identifier in park_gain_df.index:
    park_gain_num_hits[identifier] = get_info_for_gene_and_tissue(identifier,
                                                                  park_gain_df,
                                                                  'gain')
    
park_gain_num_hits['TP53_BRCA'].head()


# In[19]:


with open(cfg.distance_gain_info, 'wb') as f:
    pkl.dump(park_gain_num_hits, f)


# In[20]:


park_loss_num_hits = {}
for identifier in park_loss_df.index:
    park_loss_num_hits[identifier] = get_info_for_gene_and_tissue(identifier,
                                                                  park_loss_df,
                                                                  'loss')
    
park_loss_num_hits['TP53_BRCA'].head()


# In[21]:


with open(cfg.distance_loss_info, 'wb') as f:
    pkl.dump(park_loss_num_hits, f)

