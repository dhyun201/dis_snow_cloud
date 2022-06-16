# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd

# Path
ipath = 'D:/PhD/0_SNPP.VIIRS_SC/File_list/'

# Variables
yymm_dim = ['202001', '202002', '202003']
date_ami = []  ;  date_viirs = []
  
# Loop YYYYMM
for k in range(len(yymm_dim)):
  f = open(ipath+'vnp10_filelist_'+yymm_dim[k]+'_Adding_AHI_Date.txt', mode='r')
  lines = f.readlines()
  
  # Loop Lines
  for j in range(len(lines)):
    tmp = lines[j].split('  ')      
    date_ami.append(tmp[0])    
    date_viirs.append(tmp[1][:-1])

## Dataframe 
df = pd.DataFrame(date_ami, columns=['date_ami'])
df['date_viirs'] = date_viirs
    
## Distribute Random [Train & Test]
train_df = df.sample(frac=0.7, replace=False, random_state=2021)
test_df  = df.drop(train_df.index)

## Save
opath = 'D:/PhD/0_VIIRS.SC_SI_Dataset/'
train_df.to_csv(opath+'date_sc_filelist_train.test_TRAIN_dataset.txt', sep='\t', index=False)
test_df.to_csv(opath+'date_sc_filelist_train.test_TEST_dataset.txt', sep='\t', index=False)

print('complete')