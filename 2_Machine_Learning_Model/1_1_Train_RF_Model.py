# -*- coding: utf-8 -*-
"""
Created on Wed May 12 15:41:05 2021

@author: donghyunjin
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler


#------------------------------------------------------------------------------
# Setting path
ipath = 'D:/PhD/0_Train.Test_Dataset/Train.Dataset/'
opath = 'D:/PhD/00_VERSION_3/3__RF_Make.Model/Train_Model_Design/v3_N.estimator.100_Max.Depth.30_Except.CH01.02.03.04.NDVI/'


#------------------------------------------------------------------------------
# Calc. the column on information files
file_nm = 'VIIRS.SC.Product_SC.CLD.Flag_NMSC.SC.Product_SC.Low.CLD.CLR_used_TRAIN.dataset_v1_mod10.bin'
data_info_size = os.path.getsize(ipath+file_nm)
data_info_lines = int(data_info_size / 24 / 4)

# Read VIIRS Snow cover, Snow-free land, Cloud Binary File
dataset = np.fromfile(ipath+file_nm,dtype=np.float32).reshape(data_info_lines, 24)

# dataframe count
df_col_nm = ['gk2a_ch05_ref', 'gk2a_ch06_ref', \
             'gk2a_ch07_bt',  'gk2a_ch14_bt', \
             'gk2a_ch15_bt', \
             'ndsi', 'ndwi', \
             'btd_14_07', 'gk2a_ch06_ano', 'gk2a_sza', 'gk2a_lat', 'gk2a_vza', \
             'gk2a_cld', 'gk2a_sc', 'viirs_data']
df_ct = len(df_col_nm)

# Data Frame
ix = np.arange(1, data_info_lines+1, 1)
df_t = pd.DataFrame(columns = ['gk2a_ch05_ref', 'gk2a_ch06_ref', \
                               'gk2a_ch07_bt',  'gk2a_ch14_bt', \
                               'gk2a_ch15_bt', \
                               'ndsi', 'ndwi', \
                               'btd_14_07', 'gk2a_ch06_ano', 'gk2a_sza', 'gk2a_lat', 'gk2a_vza',\
                               'gk2a_cld', 'gk2a_sc', 'viirs_data'])
    
df_t['gk2a_ch05_ref']    = pd.Series(index = ix, data=dataset[:, 4], dtype='float32')
df_t['gk2a_ch06_ref']    = pd.Series(index = ix, data=dataset[:, 5], dtype='float32')
df_t['gk2a_ch07_bt']     = pd.Series(index = ix, data=dataset[:, 6], dtype='float32')
df_t['gk2a_ch14_bt']     = pd.Series(index = ix, data=dataset[:, 7], dtype='float32')
df_t['gk2a_ch15_bt']     = pd.Series(index = ix, data=dataset[:, 8], dtype='float32')
df_t['gk2a_cld']         = pd.Series(index = ix, data=dataset[:,13], dtype='float32')
df_t['gk2a_sc']          = pd.Series(index = ix, data=dataset[:,14], dtype='float32')
df_t['viirs_data']       = pd.Series(index = ix, data=dataset[:,15], dtype='float32')
df_t['ndsi']             = pd.Series(index = ix, data=dataset[:,18], dtype='float32')
df_t['ndwi']             = pd.Series(index = ix, data=dataset[:,20], dtype='float32')
df_t['btd_14_07']        = pd.Series(index = ix, data=dataset[:,21], dtype='float32')
df_t['gk2a_ch06_ano']    = pd.Series(index = ix, data=dataset[:,23], dtype='float32')
df_t['gk2a_sza']         = pd.Series(index = ix, data=dataset[:, 9], dtype='float32')
df_t['gk2a_lat']         = pd.Series(index = ix, data=dataset[:,11], dtype='float32')
df_t['gk2a_vza']         = pd.Series(index = ix, data=dataset[:,10], dtype='float32')
#-----------------------------------------------------------------------------
# Filtering NaN
df_t = df_t.dropna(axis=0)

# Check the GK-2A Snow Cover - Cloud Flag
gk2a_sc_cld_ind = (df_t['gk2a_sc'] == 3.0) # GK-2A Snow Cover - Low. Cloud Flag
df_t_lcld_tmp = df_t[gk2a_sc_cld_ind]

viirs_sc_ind = (df_t_lcld_tmp['viirs_data'] == 1.0) | (df_t_lcld_tmp['viirs_data'] == 3.0) 
df_t_lcld = df_t_lcld_tmp[viirs_sc_ind]

# viirs_data Flag 값을 (1; Snow, 3; Cloud)에서 (1;Snow, 0;cloud)로 변경
viirs_cld_ind = (df_t_lcld['viirs_data'] == 3.0)
df_t_lcld['viirs_data'][viirs_cld_ind] = 0.0

# dataframe 변수 삭제
df_t_lcld       = df_t_lcld.drop(columns=['gk2a_cld'])
df_t_lcld       = df_t_lcld.drop(columns=['gk2a_sc'])
df_t_lcld_final = df_t_lcld.drop(columns=['viirs_data'])
print('[Filtering Dataframe] Complete')

del df_t, gk2a_sc_cld_ind, viirs_sc_ind, df_t_lcld_tmp

#------------------------------------------------------------------------------
# Fix variables\n",
df_t_lcld_final.loc[df_t_lcld_final.gk2a_ch05_ref < 0.01, 'gk2a_ch05_ref'] = 0.01  ;  df_t_lcld_final.loc[df_t_lcld_final.gk2a_ch05_ref > 1.0, 'gk2a_ch05_ref'] = 1.0
df_t_lcld_final.loc[df_t_lcld_final.gk2a_ch06_ref < 0.01, 'gk2a_ch06_ref'] = 0.01  ;  df_t_lcld_final.loc[df_t_lcld_final.gk2a_ch06_ref > 1.0, 'gk2a_ch06_ref'] = 1.0

df_t_lcld_final.loc[df_t_lcld_final.gk2a_ch07_bt < 240.0, 'gk2a_ch07_bt'] = 240.0  ;  df_t_lcld_final.loc[df_t_lcld_final.gk2a_ch07_bt > 380.0, 'gk2a_ch07_bt'] = 380.0
df_t_lcld_final.loc[df_t_lcld_final.gk2a_ch14_bt < 210.0, 'gk2a_ch14_bt'] = 210.0  ;  df_t_lcld_final.loc[df_t_lcld_final.gk2a_ch14_bt > 320.0, 'gk2a_ch14_bt'] = 320.0
df_t_lcld_final.loc[df_t_lcld_final.gk2a_ch15_bt < 210.0, 'gk2a_ch15_bt'] = 210.0  ;  df_t_lcld_final.loc[df_t_lcld_final.gk2a_ch15_bt > 310.0, 'gk2a_ch15_bt'] = 310.0

df_t_lcld_final.loc[df_t_lcld_final.ndwi < -0.5, 'ndwi'] = -0.5                    ;  df_t_lcld_final.loc[df_t_lcld_final.ndwi > 1.0, 'ndwi'] = 1.0
df_t_lcld_final.loc[df_t_lcld_final.ndsi < -1.0, 'ndsi'] = -1.0                    ;  df_t_lcld_final.loc[df_t_lcld_final.ndsi > 1.0, 'ndsi'] = 1.0

df_t_lcld_final.loc[df_t_lcld_final.btd_14_07 < -80.0, 'btd_14_07'] = -80.0        ;  df_t_lcld_final.loc[df_t_lcld_final.btd_14_07 > 0.0, 'btd_14_07'] = 0.0
df_t_lcld_final.loc[df_t_lcld_final.gk2a_ch06_ano < -2.0, 'gk2a_ch06_ano'] = -2.0  ;  df_t_lcld_final.loc[df_t_lcld_final.gk2a_ch06_ano > 2.0, 'gk2a_ch06_ano'] = 2.0
df_t_lcld_final.loc[df_t_lcld_final.gk2a_sza < 0.0, 'gk2a_sza'] = 0.0              ;  df_t_lcld_final.loc[df_t_lcld_final.gk2a_sza > 80.0, 'gk2a_sza'] = 80.0
df_t_lcld_final.loc[df_t_lcld_final.gk2a_lat < 20.0, 'gk2a_lat'] = 20.0            ;  df_t_lcld_final.loc[df_t_lcld_final.gk2a_lat > 80.0, 'gk2a_lat'] = 80.0
df_t_lcld_final.loc[df_t_lcld_final.gk2a_vza < 30.0, 'gk2a_vza'] = 30.0            ;  df_t_lcld_final.loc[df_t_lcld_final.gk2a_vza > 90.0, 'gk2a_vza'] = 90.0
print('[Min./Max. value of variables] Check complete')
#------------------------------------------------------------------------------

# Distribute Train & Validation
X_train, X_val, Y_train, Y_val = train_test_split(df_t_lcld_final, df_t_lcld['viirs_data'], test_size=0.4, random_state=0)

del df_t_lcld_final, df_t_lcld

# Normzalization or Standard
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_val_scaled = scaler.transform(X_val)
print('[Distribution Train & Validation, Normalization] Complete')

del X_train, X_val

#==============================================================================
n_estimators_dim = [50]  # v1
n_estimators_dim_output_nm = '100'
max_depth_dim = [25]  # v2
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
code_name_dim = n_estimators_dim
 
val_score = []  ;  train_score = []

#------------------------------------------------------------------------------
# n_estimators dim loop start
for k in range(len(code_name_dim)):
  rfc = RandomForestClassifier(n_estimators=n_estimators_dim[k], max_depth = max_depth_dim[0], random_state=0)
  rfc.fit(X_train_scaled, Y_train)
  val_score_tmp   = rfc.score(X_val_scaled, Y_val)       ;  val_score.append(val_score_tmp)
  train_score_tmp = rfc.score(X_train_scaled, Y_train)   ;  train_score.append(train_score_tmp)
  
  # Save the model depending on condidtion  
  joblib.dump(rfc, opath+'RF.model_v3_N.estimators.'+str(n_estimators_dim[k])+'_max.depth.'+str(max_depth_dim[0])+'_anaconda.pkl')
  
  print(k, ' / ', len(code_name_dim)-1)
  
#==============================================================================  
# Save the test/train score in csv file using data frame
# Setting the Dataframe
ix_out = np.arange(1, len(code_name_dim)+1, 1)
df_out = pd.DataFrame(columns = ['train_accuracy', 'validation_accuracy'])
df_out['train_accuracy']      = pd.Series(index = ix_out, data=train_score, dtype='float32')
df_out['validation_accuracy'] = pd.Series(index = ix_out, data=val_score, dtype='float32')

# Save the dataframe data in csv file
df_out.to_csv(opath+'validation.train_accuracy_RF_v3_N.estimators.'+n_estimators_dim_output_nm+'_max.depth.'+str(max_depth_dim[0])+'_anaconda.csv')

print("complete")