#!/usr/bin/env python
# coding: utf-8

# ## Survival analysis for Park et al. classes
# 
# For now, we'll just consider samples with a point mutation and CNV as "two-hit" samples, and samples with either but not both as "one-hit" samples. We could make this more granular in the future (i.e. use multiple point mutations as evidence of "two-hitness", or deep CNVs as "two-hit" samples as well).

# In[1]:


from pathlib import Path
import pickle as pkl

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sksurv.compare import compare_survival
from sksurv.nonparametric import kaplan_meier_estimator
from sksurv.util import Surv

import sys; sys.path.append('..')
import config as cfg


# In[2]:


MPMP_LOCATION = Path('/home/jake/research/mpmp')

# park et al. geneset info
park_loss_data = cfg.data_dir / 'park_loss_df.tsv'
park_gain_data = cfg.data_dir / 'park_gain_df.tsv'

# park et al. significant gene info
park_loss_sig_data = cfg.data_dir / 'park_loss_df_sig_only.tsv'
park_gain_sig_data = cfg.data_dir / 'park_gain_df_sig_only.tsv'


# ### Load information for Park et al. genes

# In[3]:


park_loss_sig_df = pd.read_csv(park_loss_sig_data, sep='\t', index_col=0)
park_loss_sig_df.head()


# In[4]:


park_gain_sig_df = pd.read_csv(park_gain_sig_data, sep='\t', index_col=0)
park_gain_sig_df.head()


# ### Get clinical endpoint info
# 
# Code for preprocessing clinical info is based on: https://github.com/greenelab/mpmp/blob/master/mpmp/utilities/data_utilities.py#L510
# 
# TODO: move this to a function/utilities file

# In[5]:


# use TCGA clinical data downloaded in mpmp repo
clinical_filename = (
    MPMP_LOCATION / 'data' / 'raw' / 'TCGA-CDR-SupplementalTableS1.xlsx'
)


# In[6]:


clinical_df = pd.read_excel(
    clinical_filename,
    sheet_name='TCGA-CDR',
    index_col='bcr_patient_barcode',
    engine='openpyxl'
)

clinical_df.index.rename('patient_id', inplace=True)

# drop numeric index column
clinical_df.drop(labels=['Unnamed: 0'], axis=1, inplace=True)

# we want to use age as a covariate
clinical_df.rename(columns={'age_at_initial_pathologic_diagnosis': 'age'},
                   inplace=True)

print(clinical_df.shape)
clinical_df.iloc[:5, :5]


# In[7]:


# we want to use overall survival as the target variable except for
# certain cancer types where progression-free intervals are typically
# used (since very few deaths are observed)
# this is recommended in https://doi.org/10.1016/j.cell.2018.02.052
pfi_cancer_types = [
    'BRCA', 'DLBC', 'LGG', 'PCPG', 'PRAD',
    'READ', 'TGCT', 'THCA', 'THYM'
]

clinical_df['time_in_days'] = clinical_df['OS.time']
clinical_df['status'] = clinical_df['OS'].astype('bool')

pfi_samples = clinical_df.type.isin(pfi_cancer_types)
clinical_df.loc[pfi_samples, 'time_in_days'] = clinical_df[pfi_samples]['PFI.time']
clinical_df.loc[pfi_samples, 'status'] = clinical_df[pfi_samples]['PFI'].astype('bool')

# clean up columns and drop samples with NA survival times
na_survival_times = (clinical_df['time_in_days'].isna())
cols_to_keep = ['status', 'time_in_days', 'age', 'type']
clinical_df = clinical_df.loc[~na_survival_times, cols_to_keep].copy()

# mean impute missing age values
clinical_df.age.fillna(clinical_df.age.mean(), inplace=True)

print(clinical_df.shape)
clinical_df.head()


# ### Get mutated samples info
# 
# TODO: move this to a function/utilities file

# In[8]:


# mutation and copy number data
pancancer_pickle = MPMP_LOCATION / 'data' / 'pancancer_data.pkl'

with open(pancancer_pickle, 'rb') as f:
    pancancer_data = pkl.load(f)


# In[9]:


# get (binary) mutation data
# 1 = observed non-silent mutation in this gene for this sample, 0 otherwise
mutation_df = pancancer_data[1]
print(mutation_df.shape)
mutation_df.iloc[:5, :5]


# In[10]:


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


# In[11]:


sample_freeze_df = pancancer_data[0]
copy_samples = list(
    set(sample_freeze_df.SAMPLE_BARCODE)
    .intersection(set(copy_thresh_df.columns))
)
print(len(copy_samples))


# In[12]:


# make sure we're not losing too many samples, a few is fine
print(sorted(set(sample_freeze_df.SAMPLE_BARCODE) - set(copy_thresh_df.columns)))


# In[13]:


copy_thresh_df = (copy_thresh_df
    .T
    .loc[sorted(copy_samples)]
    .fillna(0)
    .astype(int)
)

print(copy_thresh_df.shape)
copy_thresh_df.iloc[:5, :5]


# In[14]:


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


# In[15]:


copy_gain_df = (copy_thresh_df
    .replace(to_replace=[-1, -2], value=0)
    .replace(to_replace=[1, 2], value=1)
)
print(copy_gain_df.shape)
copy_gain_df.iloc[:5, :5]


# ### Get sample info and groups for gene/cancer type

# In[16]:


def get_groups_for_gene_and_tissue(identifier,
                                   cancer_classification,
                                   hits_classification):
    """Given a gene and tissue, load the relevant mutation/CNV information,
    and divide the samples into groups to compare survival.
    
    TODO document cancer_classification and hits_classification
    """
    # get patient ids (first 12 of TCGA identifier) with mutation info
    mut_patient_ids = mutation_df.index.str[:12]
    
    # get patient ids in given cancer type 
    gene, tissue = identifier.split('_')
    tissue_ids = (clinical_df
        .query('type == @tissue')
        .index
        # mutation_df and the CNV dfs have the same index, so we
        # only have to check this one rather than all of them
        .intersection(mut_patient_ids)
    )
    id_clinical_df = clinical_df.loc[tissue_ids, :].copy()
    
    # get mutation and copy status
    mutation_status = mutation_df.loc[
        mutation_df.index.str[:12].isin(mut_patient_ids), gene
    ]
    if cancer_classification == 'TSG':
        copy_status = copy_loss_df.loc[
            copy_loss_df.index.str[:12].isin(mut_patient_ids), gene
        ]
    elif cancer_classification == 'Oncogene':
        copy_status = copy_gain_df.loc[
            copy_gain_df.index.str[:12].isin(mut_patient_ids), gene
        ]
        
    mutation_status.index = mutation_status.index.str[:12]
    copy_status.index = copy_status.index.str[:12]
        
    # get groups from mutation/CNV data
    if hits_classification == 'one':
        id_clinical_df['is_mutated'] = (mutation_status | copy_status).astype(int)
        id_clinical_df['is_mutated_alt'] = (mutation_status & copy_status).astype(int)
    elif hits_classification == 'two':
        id_clinical_df['is_mutated'] = (mutation_status & copy_status).astype(int)
        id_clinical_df['is_mutated_alt'] = (mutation_status | copy_status).astype(int)
        
    return id_clinical_df


# In[17]:


# test for a selected example
identifier = 'RB1_BLCA'
cancer_classification = 'TSG'
hits_classification = 'two'

id_clinical_df = get_groups_for_gene_and_tissue(
    identifier, cancer_classification, hits_classification)

print(id_clinical_df.shape)
print(id_clinical_df.is_mutated.sum(), id_clinical_df.is_mutated_alt.sum())
print(id_clinical_df.isna().sum()) 
id_clinical_df.head()


# In[18]:


# plot groups
sns.set({'figure.figsize': (12, 6)})
sns.set_style('whitegrid')

fig, axarr = plt.subplots(1, 2)

def plot_id(identifier, id_clinical_df):
    
    gene, tissue = identifier.split('_')

    for ix, mut_col in enumerate(['is_mutated', 'is_mutated_alt']):

        ax = axarr[ix]

        n_mutant = id_clinical_df[mut_col].sum()
        n_wildtype = id_clinical_df.shape[0] - n_mutant

        for is_mutated in id_clinical_df[mut_col].unique():
            mask_mutated = (id_clinical_df[mut_col] == is_mutated)
            time_treatment, survival_prob_treatment = kaplan_meier_estimator(
                id_clinical_df['status'][mask_mutated],
                id_clinical_df['time_in_days'][mask_mutated]
            )

            def get_label(gene, is_mutated, n_mutant, n_wildtype):
                if is_mutated:
                    return '{} mutant (n={})'.format(gene, n_mutant)
                else:
                    return '{} wild-type (n={})'.format(gene, n_wildtype)

            # TODO: confidence intervals
            ax.step(time_treatment, survival_prob_treatment, where="post",
                    label=get_label(gene, is_mutated, n_mutant, n_wildtype))

        ax.legend()

        def get_n_hits(hits_class, ix):
            hits_desc = (['one', 'two'] if hits_class == 'one' else ['two', 'one'])
            return hits_desc[ix]

        ax.set_title('Survival for {}-hit classification'.format(
            get_n_hits(hits_classification, ix)))

        # hypothesis testing using log-rank test
        y = Surv.from_dataframe('status', 'time_in_days', id_clinical_df)
        chisq, p_val = compare_survival(y, id_clinical_df[mut_col].values)
        print(mut_col, 'chisq = {:.4f}'.format(chisq), 'p = {:.4e}'.format(p_val))

    plt.suptitle('Comparing mutation classes for {}'.format(identifier))
    
plot_id(identifier, id_clinical_df)


# ### Run for all "class 2" genes
# 
# We'll just start here for now, these are all tumor suppressors that Park et al. have annotated as "two-hit loss" drivers (i.e. classical tumor suppressors).
# 
# We want to see how many of them (if any) distinguish between survival groups more effectively for two-hit samples as opposed to one-hit samples (TODO: be a bit clearer with what we're doing here).

# In[19]:


class_2_ids = park_loss_sig_df.index.unique()
print(len(class_2_ids))
print(class_2_ids[:10])


# We want to build a dataframe with:
# * identifier
# * \# WT samples
# * \# mutant samples
# * survival p-value
# * \# WT samples (alt classification)
# * \# mutant samples (alt classification)
# * survival p-value (alt classification)

# In[20]:


class_2_surv_df = []
columns = [
    'identifier',
    'n_mutant',
    'n_wildtype',
    'chisq',
    'p_val',
    'n_mutant_alt',
    'n_wildtype_alt',
    'chisq_alt',
    'p_val_alt',
]

for identifier in class_2_ids:
    
    results = [identifier]
    
    id_clinical_df = get_groups_for_gene_and_tissue(
        identifier, 'TSG', 'two')
    
    for ix, mut_col in enumerate(['is_mutated', 'is_mutated_alt']):
    
        n_mutant = id_clinical_df[mut_col].sum()
        n_wildtype = id_clinical_df.shape[0] - n_mutant
        
        if n_mutant == 0 or n_wildtype == 0:
            print(identifier, mut_col, n_mutant, n_wildtype, file=sys.stderr)
            continue
    
        # hypothesis testing using log-rank test
        try:
            y = Surv.from_dataframe('status', 'time_in_days', id_clinical_df)
        except ValueError:
            # this happens for COADREAD, TODO fix it later
            print(identifier, file=sys.stderr)
            continue
        chisq, p_val = compare_survival(y, id_clinical_df[mut_col].values)
        results += [n_mutant, n_wildtype, chisq, p_val]
        
    if len(results) == len(columns):
        class_2_surv_df.append(results)
        
class_2_surv_df = pd.DataFrame(
    class_2_surv_df,
    columns=columns
)
class_2_surv_df.set_index('identifier')

class_2_surv_df['ts_diff'] = class_2_surv_df.chisq - class_2_surv_df.chisq_alt
class_2_surv_df['p_val_diff'] = class_2_surv_df.p_val - class_2_surv_df.p_val_alt

print(class_2_surv_df.shape)
class_2_surv_df.sort_values(by='ts_diff', ascending=False).head(20)


# In[21]:


sns.set({'figure.figsize': (10, 8)})
fig, axarr = plt.subplots(2, 1)

sns.kdeplot(data=class_2_surv_df.ts_diff, ax=axarr[0])
axarr[0].axvline(0, linestyle='--')
axarr[0].set_title('Test statistic diff, class 2 genes')
axarr[0].set_xlabel('TS(true labeling) - TS(alt labeling)')

metric = 'ts'
num_examples = 10

top_df = (class_2_surv_df
    .sort_values(by='{}_diff'.format(metric), ascending=False)
    .head(num_examples)
)
bottom_df = (class_2_surv_df
    .sort_values(by='{}_diff'.format(metric), ascending=False)
    .tail(num_examples)
)
plot_df = pd.concat((top_df, bottom_df)).reset_index()
sns.barplot(data=plot_df, x=plot_df.index, y='{}_diff'.format(metric),
            dodge=False, ax=axarr[1])
axarr[1].set_xticks([])
axarr[1].set_title('Top 10 diffs, class 2 genes')
axarr[1].set_xlabel('TS(true labeling) - TS(alt labeling)')

def show_values_on_bars(ax):
    for i in range(plot_df.shape[0]):
        _x = i
        _y = plot_df.loc[i, '{}_diff'.format(metric)]
        val = plot_df.loc[i, 'identifier']
        if _y > 0:
            ax.text(_x, _y + 10, val, ha="center", rotation=90) 
        else:
            ax.text(_x, _y - 100, val, ha="center", rotation=90)
            
axarr[1].set_ylim(-220, 120)
show_values_on_bars(axarr[1])

plt.tight_layout()


# In[25]:


sns.set({'figure.figsize': (12, 6)})
sns.set_style('whitegrid')

fig, axarr = plt.subplots(1, 2)

identifier = 'TP53_HNSC'

id_clinical_df = get_groups_for_gene_and_tissue(
    identifier, 'TSG', 'two')

plot_id(identifier, id_clinical_df)

