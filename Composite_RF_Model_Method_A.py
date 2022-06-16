# -*- coding: utf-8 -*-
"""
Created on Fri Aug  6 14:52:20 2021

@author: donghyunjin
"""

import numpy as np
import glob
import os

#------------------------------------------------------------------------------
# Path (82 Scene)
ipath_gk2a = "D:/PhD/0_GK2A.Snow/data_nmsc/"
ipath_rf   = "D:/PhD/00_VERSION_3/4__RF_Output_Composite_NMSC.Snow/1_Target_gk2a_cld_tmp/"
ipath_ndsi = "D:/PhD/0_Train.Test_Dataset/Test.Dataset_82.Scene_v3/"
opath      = "D:/PhD/00_VERSION_3/4__RF_Output_Composite_NMSC.Snow/2_Composit_gk2a_sc_tmp/"

# Search the Files
files_rf = glob.glob(ipath_rf+"output_RF.model_v3_*.bin")
files_rf.sort()


# Start Loop for Scene
for k in range(len(files_rf)):
  
  # Seize date & name
  fn_rf = os.path.split(files_rf[k])[1]
  date_nm = fn_rf.split('_')[3][:-4]

#------------------------------------------------------------------------------  
# File Exist Check
  f_exist = os.path.exists(ipath_gk2a+"gk2a_ami_le2_scsi_fd020ge_"+date_nm+".bin")
  if not (f_exist):
    print("[No File. AMI/VIIRS] : "+date_nm)
    continue

#------------------------------------------------------------------------------  
  # Read the Gk-2A Snow / ML.AI. based Snow
  gk2a_sc = np.fromfile(ipath_gk2a+"gk2a_ami_le2_scsi_fd020ge_"+date_nm+".bin", dtype=np.int8).reshape(5500,5500)
  rf_sc   = np.fromfile(files_rf[k], dtype=np.int8).reshape(5500,5500)

  gk2a_ndsi = np.fromfile(ipath_ndsi+"test.dataset_gk2a_ndsi_"+date_nm+".bin", dtype=np.float32).reshape(5500,5500)
  gk2a_btd  = np.fromfile(ipath_ndsi+"test.dataset_gk2a_btd_11.2_3.8_"+date_nm+".bin", dtype=np.float32).reshape(5500,5500)

#------------------------------------------------------------------------------    
  # Seize the output variable
  output_sc = gk2a_sc
  
  output_sc = np.where(output_sc == 4, 6, output_sc)
  output_sc = np.where(output_sc == 5, 7, output_sc)

  gk2a_sc_rf_sc_ind  = (rf_sc == 7) & (gk2a_sc != 1)  ; output_sc[gk2a_sc_rf_sc_ind] = 4  # Re-snow
  gk2a_sc_rf_cld_ind = (rf_sc == 8) & (gk2a_sc != 1)  ; output_sc[gk2a_sc_rf_cld_ind] = 5  # Re-cloud
 
#------------------------------------------------------------------------------      
  # Snow re-check 1
  gk2a_ndsi_no_snow_ind = (output_sc == 4) & (gk2a_ndsi < 0.0)
  output_sc[gk2a_ndsi_no_snow_ind] = 5
  
  # Snow re-check 2
  gk2a_ndsi_no_snow_ind = (output_sc == 4) & (gk2a_btd < -15.0)
  output_sc[gk2a_ndsi_no_snow_ind] = 5

#------------------------------------------------------------------------------        
  # Write the final output
  f = open(opath+"output_GK2A.RF_SC.CLD_"+date_nm+"_snow.recheck_ndsi_btd.bin", "wb")
  output_sc.tofile(f)  ;  f.close()
  
  # with open(opath+"gk2a.sc_rf.sc_"+date_nm+".bin", "wb") as f:
  #   f.write(output_sc)
  
  print(date_nm)
  
print("Complete")  
