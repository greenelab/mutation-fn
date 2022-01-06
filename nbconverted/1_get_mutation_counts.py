#!/usr/bin/env python
# coding: utf-8

# ## Process MAF file from MC3 to get sample-specific information
# 
# We want to know if samples with mutations in tumor suppressors have mono-allelic or biallelic knockouts.

# In[1]:


import os
import gzip
from pathlib import Path

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[2]:


mc3_filename = Path('./data', 'mc3.v0.2.8.PUBLIC.maf.gz')

mutated_samples_dir = Path('./data', 'mutated_samples')
mutated_samples_dir.mkdir(exist_ok=True)

# gene to get/save mutation info for
gene = 'ARID1A'


# In[3]:


mc3 = gzip.open(mc3_filename, "rb")
maf_header = mc3.readline().decode('UTF-8').strip().split()
maf_ixs = {name: ix for ix, name in enumerate(maf_header)}

print(pd.Series(maf_header).head(10))


# In[4]:


mutated_samples_file = mutated_samples_dir / '{}_mutated_samples.tsv'.format(gene)

if mutated_samples_file.is_file():
    print('file already exists, loading from file')
    mutants_df = pd.read_csv(mutated_samples_file, sep='\t', index_col=0)
else:
    print('generating mutated samples from MC3 maf file')
    mutants = []
    for line in mc3:
        record = line.decode('UTF-8').strip().split("\t")
        hugo_symbol = record[maf_ixs['Hugo_Symbol']] # gene name
        tcga_id_raw = record[maf_ixs['Tumor_Sample_Barcode']] # tumor barcode
        tcga_id_raw_normal = record[maf_ixs['Matched_Norm_Sample_Barcode']] # normal barcode
        is_tumor = tcga_id_raw.split("-")[3].startswith("01")
        tss_code = tcga_id_raw.split("-")[1]

        if hugo_symbol == gene:
            mutants.append(record)
    mutants_df = pd.DataFrame(mutants, columns=maf_header)
    mutants_df.to_csv(mutated_samples_file, sep='\t')


# In[5]:


print(mutants_df.shape)
mutants_df.iloc[:5, :20]

