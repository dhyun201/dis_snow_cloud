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
from sklearn.metrics import accuracy_score
import warnings; warnings.filterwarnings(action='ignore') # 경고 메시지 무시

from pylab import *


# -----------------------------------------------------------------------------
# Setting path
ipath_filelist = 'D:/PhD/0_Train.Test_Dataset/Test.Dataset_Filelist/'
ipath          = 'D:/PhD/0_Train.Test_Dataset/Test.Dataset_82.Scene_v3/'
ipath_model    = 'D:/PhD/00_VERSION_3/3__RF_Make.Model/Train_Model_Design/v3_N.estimator.50_Max.Depth.30_Except.CH01.02.03.04.NDVI/'
opath          = 'D:/PhD/00_VERSION_3/4__RF_Output_Composite_NMSC.Snow/1_Target_gk2a_cld_tmp/'

# Seizing Input dataset files list
f = open(ipath_filelist+"date_sc_filelist_train.test_TEST.82.Scene_dataset.txt", "r")
lines = f.readlines()

# -----------------------------------------------------------------------------
# # Setting path (all scene)
# ipath_filelist = 'D:/PhD/0_Train.Test_Dataset/'
# ipath_tmp      = 'D:/PhD/0_Train.Test_Dataset/Test.Dataset_All.Scene_v3/'
# ipath_model    = 'D:/PhD/00_VERSION_3/3__RF_Make.Model/Train_Model_Design/v3_N.estimator.50_Max.Depth.30_Except.CH01.02.03.04.NDVI/'
# opath          = 'D:/PhD/00_VERSION_3/4__RF_Output_Composite_NMSC.Snow/1_Target_gk2a_cld_all.scene/'

# # Seizing Input dataset files list
# f = open(ipath_filelist+"date_snow_filelist_train.test_TEST_dataset.txt", "r")
# lines = f.readlines()

#------------------------------------------------------------------------------
# Setting Input parameter Min. /Max.
gk2a_ch01_ref_min =    0.087875
gk2a_ch02_ref_min =    0.068522
gk2a_ch03_ref_min =    0.036655
gk2a_ch04_ref_min =    0.022195
gk2a_ch05_ref_min =    0.010000
gk2a_ch06_ref_min =    0.011170
gk2a_ch07_bt_min  =  242.479996
gk2a_ch14_bt_min  =  210.000000
gk2a_ch15_bt_min  =  210.000000
ndsi_min          =   -0.554348
ndvi_min          =   -0.554424
ndwi_min          =   -0.339889
btd_14_07_min     =  -80.000000
gk2a_ch06_ano_min =   -1.934124
gk2a_sza_min      =   31.055363
gk2a_lat_min      =   21.899480
gk2a_vza_min      =   31.905593

gk2a_ch01_ref_max =    1.000000
gk2a_ch02_ref_max =    1.000000
gk2a_ch03_ref_max =    1.000000
gk2a_ch04_ref_max =    1.000000
gk2a_ch05_ref_max =    1.000000
gk2a_ch06_ref_max =    1.000000
gk2a_ch07_bt_max  =  378.509979
gk2a_ch14_bt_max  =  310.849976
gk2a_ch15_bt_max  =  304.260010
ndsi_max          =    0.923196
ndvi_max          =    0.753739
ndwi_max          =    0.925926
btd_14_07_max     =   -0.410004
gk2a_ch06_ano_max =    1.875341
gk2a_sza_max      =   79.999969
gk2a_lat_max      =   78.844940
gk2a_vza_max      =   90.0



# -----------------------------------------------------------------------------
# Scene Loop Start
# -----------------------------------------------------------------------------
for j in range(len(lines)):
    
  # 82 Scene !!!!!!!!!!!!!!!!!!!!!!!!!!
  tmp = lines[j].split('  ')
  date_ami = tmp[0]  ;  date_viirs = tmp[1]



  # # Target for All scene  !!!!!!!!!!!!!!!!!!!!!
  # tmp = lines[j].split('\t')
  # date_ami = tmp[0]  ;  date_viirs = tmp[1][:-1]
  
  # ipath = ipath_tmp+date_ami[:6]+"/"

  # if j < 395:
  #   print(date_ami)
  #   continue

# -----------------------------------------------------------------------------
# File Exist Check
  f_exist = os.path.exists(ipath+"test.dataset_gk2a_sza_"+date_ami+".bin")
  if not (f_exist):
    print("[No File. AMI/VIIRS] : "+date_ami+" / "+date_viirs)
    continue
# -----------------------------------------------------------------------------
  sur_flag = np.fromfile(ipath+"test.dataset_out.flag_info_"+date_ami+".bin", dtype=np.int8).reshape(5500,5500)
# -----------------------------------------------------------------------------
  t_out_flag = np.fromfile(ipath+"test.dataset_out.flag_info_"+date_ami+".bin", dtype=np.int8)
  
# Seize the input data for data-frame
  ix = np.arange(1, len(t_out_flag)+1, 1)
  df_t = pd.DataFrame(columns = ['gk2a_ch05_ref', 'gk2a_ch06_ref', \
                                 'gk2a_ch07_bt',  'gk2a_ch14_bt', \
                                 'gk2a_ch15_bt', \
                                 'ndsi', 'ndwi', \
                                 'btd_14_07', 'gk2a_ch06_ano', 'gk2a_sza', 'gk2a_lat', 'gk2a_vza',  \
                                 'ind_x', 'ind_y', 'flag_info', \
                                 'viirs_data'])
 
  total_predict = pd.DataFrame(columns = ['gk2a_ch05_ref', 'gk2a_ch06_ref', \
                                 'gk2a_ch07_bt',  'gk2a_ch14_bt', \
                                 'gk2a_ch15_bt', \
                                 'ndsi', 'ndwi', \
                                 'btd_14_07', 'gk2a_ch06_ano', 'gk2a_sza', 'gk2a_lat', 'gk2a_vza',  \
                                 'ind_x', 'ind_y', 'flag_info', \
                                 'viirs_data'])  

# Seize the Input data for data-frame
#------------------------------------------------------------------------------
  df_t['flag_info']     = pd.Series(index=ix, data=t_out_flag, dtype='int8')  ;  del t_out_flag

  t_viirs_sc = np.fromfile(ipath+"test.dataset_viirs.snow_"+date_ami+".bin", dtype=np.int8)
  df_t['viirs_data']    = pd.Series(index=ix, data=t_viirs_sc, dtype='int8')  ;  del t_viirs_sc

  t_ind_x    = np.fromfile(ipath+"test.dataset_index_x_"+date_ami+".bin", dtype=np.int16)
#------------------------------------------------------------------------------    
  if sum(t_ind_x) == 0:
    print(date_ami)  
    continue
#------------------------------------------------------------------------------  
  df_t['ind_x']         = pd.Series(index=ix, data=t_ind_x,    dtype='int16') ;  del t_ind_x
  
  t_ind_y    = np.fromfile(ipath+"test.dataset_index_y_"+date_ami+".bin", dtype=np.int16)
  df_t['ind_y']         = pd.Series(index=ix, data=t_ind_y,    dtype='int16') ;  del t_ind_y  



#------------------------------------------------------------------------------  
 
  t_ch05_ref = np.fromfile(ipath+"test.dataset_gk2a_ch05_ref_"+date_ami+".bin", dtype=np.float32)
  df_t['gk2a_ch05_ref'] = pd.Series(index=ix, data=t_ch05_ref, dtype='f')     ;  del t_ch05_ref
  
  t_ch06_ref = np.fromfile(ipath+"test.dataset_gk2a_ch06_ref_"+date_ami+".bin", dtype=np.float32)
  df_t['gk2a_ch06_ref'] = pd.Series(index=ix, data=t_ch06_ref, dtype='f')     ;  del t_ch06_ref  

  t_ch07_bt  = np.fromfile(ipath+"test.dataset_gk2a_ch07_bt_"+date_ami+".bin", dtype=np.float32)
  df_t['gk2a_ch07_bt']  = pd.Series(index=ix, data=t_ch07_bt,  dtype='f')     ;  del t_ch07_bt

  t_ch14_bt  = np.fromfile(ipath+"test.dataset_gk2a_ch14_bt_"+date_ami+".bin", dtype=np.float32)
  df_t['gk2a_ch14_bt']  = pd.Series(index=ix, data=t_ch14_bt,  dtype='f')     ;  del t_ch14_bt
  
  t_ch15_bt  = np.fromfile(ipath+"test.dataset_gk2a_ch15_bt_"+date_ami+".bin", dtype=np.float32)
  df_t['gk2a_ch15_bt']  = pd.Series(index=ix, data=t_ch15_bt,  dtype='f')     ;  del t_ch15_bt  

#------------------------------------------------------------------------------
  t_ch06_ano = np.fromfile(ipath+"test.dataset_gk2a_ch06_ano_"+date_ami+".bin", dtype=np.float32)  
  df_t['gk2a_ch06_ano'] = pd.Series(index=ix, data=t_ch06_ano, dtype='f')     ;  del t_ch06_ano
  
  t_ndsi     = np.fromfile(ipath+"test.dataset_gk2a_ndsi_"+date_ami+".bin", dtype=np.float32)
  df_t['ndsi']          = pd.Series(index=ix, data=t_ndsi,     dtype='f')     ;  del t_ndsi
  
  t_ndwi     = np.fromfile(ipath+"test.dataset_gk2a_ndwi_"+date_ami+".bin", dtype=np.float32)  
  df_t['ndwi']          = pd.Series(index=ix, data=t_ndwi,     dtype='f')     ;  del t_ndwi
  
  t_btd_14_7 = np.fromfile(ipath+"test.dataset_gk2a_btd_11.2_3.8_"+date_ami+".bin", dtype=np.float32)   
  df_t['btd_14_07']     = pd.Series(index=ix, data=t_btd_14_7, dtype='f')     ;  del t_btd_14_7    

  t_sza = np.fromfile(ipath+"test.dataset_gk2a_sza_"+date_ami+".bin", dtype=np.float32)   
  df_t['gk2a_sza']     = pd.Series(index=ix, data=t_sza, dtype='f')     ;  del t_sza

  t_vza = np.fromfile(ipath+"test.dataset_gk2a_vza_"+date_ami+".bin", dtype=np.float32)   
  df_t['gk2a_vza']     = pd.Series(index=ix, data=t_vza, dtype='f')     ;  del t_vza
  
  t_lat = np.fromfile(ipath+"test.dataset_gk2a_lat_"+date_ami+".bin", dtype=np.float32)   
  df_t['gk2a_lat']     = pd.Series(index=ix, data=t_lat, dtype='f')     ;  del t_lat
# -----------------------------------------------------------------------------
# Extract Low confidence cloudy to discriminate snow/cloud pixel
  df_t_lcld_ind   = (df_t['flag_info'] == 6)              ;  df_t_lcld_tmp = df_t[df_t_lcld_ind]
  del df_t_lcld_ind, df_t

# -----------------------------------------------------------------------------
# Fix variables
  df_t_lcld_tmp.loc[df_t_lcld_tmp.gk2a_ch05_ref < 0.01, 'gk2a_ch05_ref'] = 0.01  ;  df_t_lcld_tmp.loc[df_t_lcld_tmp.gk2a_ch05_ref > 1.0, 'gk2a_ch05_ref'] = 1.0
  df_t_lcld_tmp.loc[df_t_lcld_tmp.gk2a_ch06_ref < 0.01, 'gk2a_ch06_ref'] = 0.01  ;  df_t_lcld_tmp.loc[df_t_lcld_tmp.gk2a_ch06_ref > 1.0, 'gk2a_ch06_ref'] = 1.0
  
  df_t_lcld_tmp.loc[df_t_lcld_tmp.gk2a_ch07_bt < 240.0, 'gk2a_ch07_bt'] = 240.0  ;  df_t_lcld_tmp.loc[df_t_lcld_tmp.gk2a_ch07_bt > 380.0, 'gk2a_ch07_bt'] = 380.0
  df_t_lcld_tmp.loc[df_t_lcld_tmp.gk2a_ch14_bt < 210.0, 'gk2a_ch14_bt'] = 210.0  ;  df_t_lcld_tmp.loc[df_t_lcld_tmp.gk2a_ch14_bt > 320.0, 'gk2a_ch14_bt'] = 320.0
  df_t_lcld_tmp.loc[df_t_lcld_tmp.gk2a_ch15_bt < 210.0, 'gk2a_ch15_bt'] = 210.0  ;  df_t_lcld_tmp.loc[df_t_lcld_tmp.gk2a_ch15_bt > 310.0, 'gk2a_ch15_bt'] = 310.0
  
  df_t_lcld_tmp.loc[df_t_lcld_tmp.gk2a_ch06_ano < -2.0, 'gk2a_ch06_ano'] = -2.0  ;  df_t_lcld_tmp.loc[df_t_lcld_tmp.gk2a_ch06_ano > 2.0, 'gk2a_ch06_ano'] = 2.0
  df_t_lcld_tmp.loc[df_t_lcld_tmp.ndwi < -0.5, 'ndwi'] = -0.5                    ;  df_t_lcld_tmp.loc[df_t_lcld_tmp.ndwi > 1.0, 'ndwi'] = 1.0  
  df_t_lcld_tmp.loc[df_t_lcld_tmp.ndsi < -1.0, 'ndsi'] = -1.0                    ;  df_t_lcld_tmp.loc[df_t_lcld_tmp.ndsi > 1.0, 'ndsi'] = 1.0
  df_t_lcld_tmp.loc[df_t_lcld_tmp.btd_14_07 < -80.0, 'btd_14_07'] = -80.0        ;  df_t_lcld_tmp.loc[df_t_lcld_tmp.btd_14_07 > 0.0, 'btd_14_07'] = 0.0  
  
  df_t_lcld_tmp.loc[df_t_lcld_tmp.gk2a_sza < 0.0, 'gk2a_sza'] = 0.0              ;  df_t_lcld_tmp.loc[df_t_lcld_tmp.gk2a_sza > 80.0, 'gk2a_sza'] = 80.0  
  df_t_lcld_tmp.loc[df_t_lcld_tmp.gk2a_lat < 20.0, 'gk2a_lat'] = 20.0            ;  df_t_lcld_tmp.loc[df_t_lcld_tmp.gk2a_lat > 80.0, 'gk2a_lat'] = 80.0
  df_t_lcld_tmp.loc[df_t_lcld_tmp.gk2a_vza < 30.0, 'gk2a_vza'] = 30.0            ;  df_t_lcld_tmp.loc[df_t_lcld_tmp.gk2a_vza > 90.0, 'gk2a_vza'] = 90.0
  
  print('[Min./Max. value of varialbes] Check complete')
# -----------------------------------------------------------------------------
# Check data frame 2
  df_t_lcld = df_t_lcld_tmp.drop(columns=['flag_info'])    
  df_t_lcld = df_t_lcld.drop(columns=['ind_x'])      
  df_t_lcld = df_t_lcld.drop(columns=['ind_y'])     
  df_t_lcld = df_t_lcld.drop(columns=['viirs_data'])   
  
# -----------------------------------------------------------------------------
# Setting Hyper-parameter for Random Forest 
  n_estimators_val = 50  
  max_depth_val    = 30
  
# -----------------------------------------------------------------------------
  out_df_t = df_t_lcld_tmp
   
#------------------------------------------------------------------------------
# Normzalization or Standard
  out_df_t_scaled = pd.DataFrame(columns = ['gk2a_ch05_ref', 'gk2a_ch06_ref', \
                                'gk2a_ch07_bt',  'gk2a_ch14_bt', \
                                'gk2a_ch15_bt', \
                                'ndsi', 'ndwi', \
                                'btd_14_07', 'gk2a_ch06_ano', 'gk2a_sza', 'gk2a_lat', 'gk2a_vza'])
  
  out_df_t_scaled['gk2a_ch05_ref'] = ( out_df_t['gk2a_ch05_ref'] - gk2a_ch05_ref_min ) / ( gk2a_ch05_ref_max - gk2a_ch05_ref_min )
  out_df_t_scaled['gk2a_ch06_ref'] = ( out_df_t['gk2a_ch06_ref'] - gk2a_ch06_ref_min ) / ( gk2a_ch06_ref_max - gk2a_ch06_ref_min )
  
  out_df_t_scaled['gk2a_ch07_bt']  = ( out_df_t['gk2a_ch07_bt'] - gk2a_ch07_bt_min ) / ( gk2a_ch07_bt_max - gk2a_ch07_bt_min )
  out_df_t_scaled['gk2a_ch14_bt']  = ( out_df_t['gk2a_ch14_bt'] - gk2a_ch14_bt_min ) / ( gk2a_ch14_bt_max - gk2a_ch14_bt_min )
  out_df_t_scaled['gk2a_ch15_bt']  = ( out_df_t['gk2a_ch15_bt'] - gk2a_ch15_bt_min ) / ( gk2a_ch15_bt_max - gk2a_ch15_bt_min )
  
  out_df_t_scaled['ndsi']          = ( out_df_t['ndsi'] - ndsi_min ) / ( ndsi_max - ndsi_min )
  out_df_t_scaled['ndwi']          = ( out_df_t['ndwi'] - ndwi_min ) / ( ndwi_max - ndwi_min )
  
  out_df_t_scaled['btd_14_07']     = ( out_df_t['btd_14_07'] - btd_14_07_min ) / ( btd_14_07_max - btd_14_07_min )
  out_df_t_scaled['gk2a_ch06_ano'] = ( out_df_t['gk2a_ch06_ano'] - gk2a_ch06_ano_min ) / ( gk2a_ch06_ano_max - gk2a_ch06_ano_min )
  out_df_t_scaled['gk2a_sza']      = ( out_df_t['gk2a_sza'] - gk2a_sza_min ) / ( gk2a_sza_max - gk2a_sza_min )
  out_df_t_scaled['gk2a_lat']      = ( out_df_t['gk2a_lat'] - gk2a_lat_min ) / ( gk2a_lat_max - gk2a_lat_min )
  out_df_t_scaled['gk2a_vza']      = ( out_df_t['gk2a_vza'] - gk2a_vza_min ) / ( gk2a_vza_max - gk2a_vza_min )  
  
  df_t_scaled = np.array(out_df_t_scaled)
  del out_df_t_scaled, df_t_lcld

# # -----------------------------------------------------------------------------      
# # Call the model
#   loaded_model = joblib.load(ipath_model+"RF.model_N.estimators.100_max.depth.35_involve.sza.pkl")
#   print("[Random Forest Model Read] Complete")

# -----------------------------------------------------------------------------      
# Call the model
  loaded_model = joblib.load(ipath_model+"RF.model_v3_N.estimators.50_max.depth.30_anaconda.pkl")
  print("[Random Forest Model Read] Complete")

# -----------------------------------------------------------------------------  
# Discrimination
  predicted = loaded_model.predict(df_t_scaled)
  
# -----------------------------------------------------------------------------  
#   Append "total_predict : ind_x, ind_y", "total_out_df_t : predict"
  out_df_t['output'] = predicted[:].astype('f')
  total_predict = total_predict.append(out_df_t)
  print("[Scene Test Complete] : "+date_ami)
# -----------------------------------------------------------------------------
# Calc. Accuracy using VIIRS Snow Cover Data
  df_t_lcld_ind_1 = (total_predict['viirs_data'] == 1) | (total_predict['viirs_data'] == 3) 
  inp_out = total_predict[df_t_lcld_ind_1].astype('f')
  accuracy  = accuracy_score(inp_out['viirs_data'], inp_out['output'])
  print('Accuracy: {score:.3f}'.format(score=accuracy))
# -----------------------------------------------------------------------------
# Make output file
  output = np.zeros([5500,5500], dtype=np.float32)
  total_predict['ind_y'] = total_predict['ind_y'].astype('int16')
  total_predict['ind_x'] = total_predict['ind_x'].astype('int16')
  if (len(total_predict['output']) != 0):
    output[total_predict['ind_y'], total_predict['ind_x']] = total_predict['output']
# -----------------------------------------------------------------------------
# Seize the surface information 
  output_ind = (output == 1.) & (sur_flag == 6.) ;  sur_flag[output_ind] = 7.  # Snow
  output_ind = (output == 0.) & (sur_flag == 6.) ;  sur_flag[output_ind] = 8.  # Cloud
  del output, output_ind  
  
# -----------------------------------------------------------------------------  
# Save the file
  f = open(opath+"output_RF.model_v3_"+date_ami+".bin", "wb")
  sur_flag.tofile(f)  ;  f.close()



