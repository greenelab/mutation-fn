#!/usr/bin/env python
# coding: utf-8

# ## Survival analysis proof-of-concept
# 
# Much of the preprocessing code is based on: https://github.com/greenelab/mpmp/blob/master/mpmp/utilities/data_utilities.py#L510

# In[1]:


from pathlib import Path

import pandas as pd


# ### Get clinical endpoint info

# In[2]:


# use TCGA clinical data downloaded in mpmp repo
mpmp_location = Path('/home/jake/research/mpmp')
clinical_filename = (
    mpmp_location / 'data' / 'raw' / 'TCGA-CDR-SupplementalTableS1.xlsx'
)


# In[3]:


clinical_df = pd.read_excel(
    clinical_filename,
    sheet_name='TCGA-CDR',
    index_col='bcr_patient_barcode',
    engine='openpyxl'
)

clinical_df.index.rename('sample_id', inplace=True)

# drop numeric index column
clinical_df.drop(labels=['Unnamed: 0'], axis=1, inplace=True)

# we want to use age as a covariate
clinical_df.rename(columns={'age_at_initial_pathologic_diagnosis': 'age'},
                   inplace=True)

print(clinical_df.shape)
clinical_df.iloc[:5, :5]


# In[4]:


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

