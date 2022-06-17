# -*- coding: utf-8 -*-
"""
Created on Fri Aug  6 14:52:20 2021

@author: donghyunjin
"""

import numpy as np
import glob
import os


# Path 
# All Scene
ipath_gk2a = "D:/PhD/0_GK2A.Snow/data_nmsc_all.scene/"
ipath_dl   = "D:/PhD/00_VERSION_3/7__DL_Output_Composite_NMSC.Snow/1_Target_gk2a_cld_all.scene/"
ipath_ndsi = "D:/PhD/0_Train.Test_Dataset/Test.Dataset_All.Scene_v3/"
opath      = "D:/PhD/00_VERSION_3/7__DL_Output_Composite_NMSC.Snow/2_Composite_gk2a_sc_all.scene/"


# Search the Files
files_dl = glob.glob(ipath_dl+"output_DNN.model_v3_*.bin")
files_dl.sort()


# Start Loop for Scene
for k in range(len(files_dl)):
  
  # Seize date & name (82 Scene)
  fn_dl = os.path.split(files_dl[k])[1]
  date_nm = fn_dl.split('_')[3][:-4]

#------------------------------------------------------------------------------
# File Exist Check
  # 82 Scene
  #f_exist = os.path.exists(ipath_gk2a+"gk2a_ami_le2_scsi_fd020ge_"+date_nm+".bin")
  
  # All Scene
  f_exist = os.path.exists(ipath_gk2a+date_nm[0:6]+"/gk2a_ami_le2_scsi_fd020ge_"+date_nm+".bin")
  if not (f_exist):
    print("[No File. AMI/VIIRS] : "+date_nm)
    continue

# All Scene
  gk2a_sc   = np.fromfile(ipath_gk2a+date_nm[0:6]+"/gk2a_ami_le2_scsi_fd020ge_"+date_nm+".bin", dtype=np.int8).reshape(5500,5500) 
  dnn_sc    = np.fromfile(ipath_dl+"output_DNN.model_v3_"+date_nm+".bin", dtype=np.int8).reshape(5500,5500)
  gk2a_ndsi = np.fromfile(ipath_ndsi+date_nm[0:6]+"/test.dataset_gk2a_ndsi_"+date_nm+".bin", dtype=np.float32).reshape(5500,5500)
  gk2a_btd  = np.fromfile(ipath_ndsi+date_nm[0:6]+"/test.dataset_gk2a_btd_11.2_3.8_"+date_nm+".bin", dtype=np.float32).reshape(5500,5500)    


#------------------------------------------------------------------------------  
  # Seize the output variable 1
  output_sc = gk2a_sc  # GK-2A SCSI Data
  
  output_sc = np.where(output_sc == 4, 6, output_sc)  # Sea-ice
  output_sc = np.where(output_sc == 5, 7, output_sc)  # Ice-free water

  gk2a_sc_dnn_sc_ind = (dnn_sc == 7)  & (gk2a_sc != 1)   ; output_sc[gk2a_sc_dnn_sc_ind] = 4  # Snow (DNN)
  gk2a_sc_dnn_cld_ind = (dnn_sc == 8) & (gk2a_sc != 1)   ; output_sc[gk2a_sc_dnn_cld_ind] = 5 # Cloud(DNN)

#------------------------------------------------------------------------------      
  # Snow re-check 1
  gk2a_ndsi_no_snow_ind = (output_sc == 4) & (gk2a_ndsi < 0.0)
  output_sc[gk2a_ndsi_no_snow_ind] = 5
  
  # Snow re-check 2
  gk2a_ndsi_no_snow_ind = (output_sc == 4) & (gk2a_btd < -15.0)
  output_sc[gk2a_ndsi_no_snow_ind] = 5
  
  # Write the final output_1
  with open(opath+"gk2a.sc_dnn.sc_"+date_nm+"_snow.recheck_ndsi_btd.bin", "wb") as f:
    f.write(output_sc)

  
print("Complete!")
