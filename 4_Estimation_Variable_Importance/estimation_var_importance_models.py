# -*- coding: utf-8 -*-
"""
Created on Wed May 12 15:41:05 2021

@author: donghyunjin
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os, math

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
import csv


# Setting path
ipath = 'D:/PhD/0_Train.Test_Dataset/Train.Dataset/'
opath = 'D:/PhD/00_VERSION_2/3__RF_Importance_value_for_Input_Parameter/'


# Calc. the column on information files
file_nm = 'VIIRS.SC.Product_SC.CLD.Flag_NMSC.SC.Product_SC.Hihg.Low.CLD.CLR_used_TRAIN.dataset_v1_mod10.bin'
data_info_size = os.path.getsize(ipath+file_nm)
data_info_lines = int(data_info_size / 24 / 4)

# Read VIIRS Snow cover, Snow-free land, Cloud Binary File
dataset = np.fromfile(ipath+file_nm,dtype=np.float32).reshape(data_info_lines, 24)


# dataframe count
df_col_nm = ['gk2a_ch01_ref', 'gk2a_ch02_ref', \
                              'gk2a_ch03_ref', 'gk2a_ch04_ref', \
                              'gk2a_ch05_ref', 'gk2a_ch06_ref', \
                              'gk2a_ch07_bt',  'gk2a_ch14_bt', \
                              'gk2a_ch15_bt', \
                              'ndsi', 'ndvi', 'ndwi', \
                              'btd_14_07', 'gk2a_ch06_ano', \
                              'gk2a_cld', 'gk2a_sc', 'viirs_data', 'sza']
df_ct = len(df_col_nm)

# Data Frame
ix = np.arange(1, data_info_lines+1, 1)
df_t = pd.DataFrame(columns = ['gk2a_ch01_ref', 'gk2a_ch02_ref', \
                               'gk2a_ch03_ref', 'gk2a_ch04_ref', \
                               'gk2a_ch05_ref', 'gk2a_ch06_ref', \
                               'gk2a_ch07_bt',  'gk2a_ch14_bt', \
                               'gk2a_ch15_bt', \
                               'ndsi', 'ndvi', 'ndwi', \
                               'btd_14_07', 'gk2a_ch06_ano', \
                               'gk2a_cld', 'gk2a_sc', 'viirs_data', 'sza'])
df_t['gk2a_ch01_ref']    = pd.Series(index = ix, data=dataset[:, 0], dtype='float32')
df_t['gk2a_ch02_ref']    = pd.Series(index = ix, data=dataset[:, 1], dtype='float32')
df_t['gk2a_ch03_ref']    = pd.Series(index = ix, data=dataset[:, 2], dtype='float32')
df_t['gk2a_ch04_ref']    = pd.Series(index = ix, data=dataset[:, 3], dtype='float32')
df_t['gk2a_ch05_ref']    = pd.Series(index = ix, data=dataset[:, 4], dtype='float32')
df_t['gk2a_ch06_ref']    = pd.Series(index = ix, data=dataset[:, 5], dtype='float32')
df_t['gk2a_ch07_bt']     = pd.Series(index = ix, data=dataset[:, 6], dtype='float32')
df_t['gk2a_ch14_bt']     = pd.Series(index = ix, data=dataset[:, 7], dtype='float32')
df_t['gk2a_ch15_bt']     = pd.Series(index = ix, data=dataset[:, 8], dtype='float32')
df_t['gk2a_cld']         = pd.Series(index = ix, data=dataset[:,13], dtype='float32')
df_t['gk2a_sc']          = pd.Series(index = ix, data=dataset[:,14], dtype='float32')
df_t['viirs_data']       = pd.Series(index = ix, data=dataset[:,15], dtype='float32')
df_t['ndsi']             = pd.Series(index = ix, data=dataset[:,18], dtype='float32')
df_t['ndvi']             = pd.Series(index = ix, data=dataset[:,19], dtype='float32')
df_t['ndwi']             = pd.Series(index = ix, data=dataset[:,20], dtype='float32')
df_t['btd_14_07']        = pd.Series(index = ix, data=dataset[:,21], dtype='float32')
df_t['gk2a_ch06_ano']    = pd.Series(index = ix, data=dataset[:,23], dtype='float32')
df_t['sza']              = pd.Series(index = ix, data=dataset[:, 9], dtype='float32')


# Filtering NaN
df_t = df_t.dropna(axis=0)

# Check the GK-2A Snow Cover - Cloud Flag
gk2a_sc_cld_ind = (df_t['gk2a_sc'] == 3.0) # GK-2A Snow Cover - Low. Cloud Flag
df_t_lcld_tmp = df_t[gk2a_sc_cld_ind]

viirs_sc_ind = (df_t_lcld_tmp['viirs_data'] == 1.0) | (df_t_lcld_tmp['viirs_data'] == 3.0) 
df_t_lcld = df_t_lcld_tmp[viirs_sc_ind]

del gk2a_sc_cld_ind, viirs_sc_ind, df_t_lcld_tmp

# dataframe 변수 삭제
df_t_lcld = df_t_lcld.drop(columns=['gk2a_cld'])  
df_t_lcld = df_t_lcld.drop(columns=['gk2a_sc'])
df_t_lcld = df_t_lcld.drop(columns=['sza'])
df_t_lcld_final = df_t_lcld.drop(columns=['viirs_data'])

# 변수 삭제
del df_t

# Distribute Train & Test
X_train, X_test, Y_train, Y_test = train_test_split(df_t_lcld_final, df_t_lcld['viirs_data'], \
                                                   test_size=0.4, random_state=0)

del df_t_lcld_final, df_t_lcld
    

# Normzalization or Standard
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

#del X_train, X_test

print("[ Ready to perform Random Forest Model ]")

#-----------------------------------------------------------------------------
n_estimators_dim = [300]
n_estimators_dim_nm = ['300']


# n_estimators dim loop start
for k in range(len(n_estimators_dim)):

  print("[ RF Model ] N-estimators : "+n_estimators_dim_nm[k])    

  rfc = RandomForestClassifier(n_estimators=n_estimators_dim[k], random_state=0)
  #rfc.fit(X_train, Y_train)
  rfc.fit(X_train, Y_train)

  #-----------------------------------------------------------------------------
  # Importance of variables for random forest
  #-----------------------------------------------------------------------------
  var_import = rfc.feature_importances_  # 변수별 중요도 값
  n_features = rfc.feature_importances_.shape[0]
  feature = X_train.columns.values       # 변수 이름

  # Imaging importance of variables 1
  df = pd.DataFrame({'feature': feature, 'importance': rfc.feature_importances_})
  df = df.sort_values('importance', ascending=True)
  x = df.feature
  y = df.importance
  ypos = np.arange(len(x))

# Write CSV file
  import csv
  df.to_csv(opath+"GK-2A.Snow.Product_Cloud__RF_importance_variable_n.estimators_"+n_estimators_dim_nm[k]+".csv")


  # plt.figure(figsize=(10,7))
  # plt.barh(x,y)
  # plt.yticks(ypos, x)
  # plt.xlabel('Importacne')
  # plt.ylabel('Variables')
  # #plt.xlim(0,0.2) 
  # plt.ylim(-1,len(x))
  #plt.show()

  # Save the importance of var. image
  #plt.savefig(opath+'image_GK-2A.Snow.Product_Cloud__RF_importance_variable_n.estimators_'+n_estimators_dim_nm[k]+'_test3.png', dpi=300, bbox_inches='tight')
  #plt.close()


  # #-----------------------------------------------------------------------------
  # # Imaging importance of variables 2
  # # Sorted var_import and feature (value and variable name)
  # var_import_tmp = np.reshape(var_import,(20,1))
  # feature_tmp = np.reshape(feature,(20,1))

  # var_import_dim = np.append(var_import_tmp, feature_tmp, axis=1)  # composit 'var_import_tmp' and 'feature_tmp'
  # feature_val = np.array(sorted(var_import_dim, key=lambda var_import_dim: var_import_dim[0]))  # Sorted composit dim.

  # # Imaging the importance of variables 
  # feature_tmp = feature_val[:,0].astype(np.float32)
  # plt.barh(np.arange(n_features), feature_tmp, align='center')
  # plt.yticks(np.arange(n_features), feature_val[:,1])
  # plt.xlabel("Random Forest Feature Importance")
  # plt.ylabel("Feature")
  # plt.ylim(-1, n_features)

  # # Save the importance of var. image
  # opath = 'D:/PhD/0_RF.Method/importance_variables/'
  # plt.savefig(opath+'image_RF_importance_variable_n.estimators_'+n_estimators_dim_nm[k]+'_test3.png', dpi=300, bbox_inches='tight')
  # plt.close()

  print(k, ' / ', len(n_estimators_dim)-1)






