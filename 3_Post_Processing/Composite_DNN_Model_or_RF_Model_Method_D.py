# -*- coding: utf-8 -*-
"""
Created on Fri Aug  6 14:52:20 2021

@author: donghyunjin
"""

import numpy as np
import glob
import os


# Path 
# 82 Scene
ipath_gk2a = "D:/PhD/0_GK2A.Snow/data_nmsc/"
ipath_dl   = "D:/PhD/00_VERSION_3/7__DL_Output_Composite_NMSC.Snow/2_Composite_gk2a_sc_all.scene/"
ipath_rf   = "D:/PhD/00_VERSION_3/4__RF_Output_Composite_NMSC.Snow/2_Composit_gk2a_sc_all.scene/snow_recheck_ndsi_btd/"
opath      = "D:/PhD/00_VERSION_3/10__Composit_RF_or_DL.Output/All.Scene/"

# Search the Files
files_dl = glob.glob(ipath_dl+"gk2a.sc_dnn.sc_*_snow.recheck_ndsi_btd.bin")
files_dl.sort()


# Start Loop for Scene
for k in range(len(files_dl)):
  
  # Seize date & name (82 Scene)
  fn_dl = os.path.split(files_dl[k])[1]
  date_nm = fn_dl.split('_')[2]
  
#------------------------------------------------------------------------------
# File Exist Check
  # All Scene
  f_exist = os.path.exists(ipath_rf+"output_GK2A.RF_SC.CLD_"+date_nm+"_snow.recheck_ndsi_btd.bin")
  if not (f_exist):
    print("[No File. AMI/VIIRS] : "+date_nm)
    continue

# All Scene
  rf_sc     = np.fromfile(ipath_rf+"output_GK2A.RF_SC.CLD_"+date_nm+"_snow.recheck_ndsi_btd.bin", dtype=np.int8).reshape(5500,5500)  
  dnn_sc    = np.fromfile(ipath_dl+"gk2a.sc_dnn.sc_"+date_nm+"_snow.recheck_ndsi_btd.bin", dtype=np.int8).reshape(5500,5500)

#------------------------------------------------------------------------------  
  # Seize the output variable 1
  output_sc = rf_sc
  # RF SC - 0: Night / 1: Snow / 2: Snow-free land / 3: Cloud / 4: Snow (RF) / 5: Cloud (RF) / 6: Sea-ice / 7: Ice-free water
  # DL SC - 0: Night / 1: Snow / 2: Snow-free land / 3: Cloud / 4: Snow (DL) / 5: Cloud (DL) / 6: Sea-ice / 7: Ice-free water
  
  #output_sc = np.where(output_sc == 4, 5, output_sc)
  
  dl_sc_ind = (dnn_sc == 4)
  output_sc[dl_sc_ind] = 4
  
  
  # Write the final output_1
  with open(opath+"gk2a.sc_rf.sc_or_dnn.sc_"+date_nm+".bin", "wb") as f:
    f.write(output_sc)  

  
print("Complete!")
