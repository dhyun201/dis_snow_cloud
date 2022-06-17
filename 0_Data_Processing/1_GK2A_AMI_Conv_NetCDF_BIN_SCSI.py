from netCDF4 import Dataset
import netCDF4 as nc
import numpy as np
import glob, os
import math
import time
import datetime
from datetime import timedelta


## Setting path
ipath_l1b = '/ecoface2/DHJIN/DATA_GK2A.SCSI/SCSI/202002/'
opath_l1b = '/data/DHJIN/DATA_GK2A.AMI/L2_SCSI/'


fn = glob.glob(ipath_l1b+'gk2a_ami_le2_scsi_fd020ge_20200201*00.nc')
fn.sort()

for kk in range(len(fn)):

  tmp = os.path.split(fn[kk])
  tmp_1 = tmp[1].split('_')
  date = tmp_1[5][:-3]

  print(date)

  yy = date[0:4]
  mm = date[4:6]
  dd = date[6:8]
  hh = date[8:10]
  mn = date[10:12]
  gk2a_date = yy+mm+dd+hh+mn
  print(gk2a_date)


  if not os.path.isdir(opath_l1b):
    os.system("mkdir -p "+opath_l1b)
    print("Make OUTPUT directory")

 ## Check the files
  oistat_scsi  = os.path.exists(opath_l1b+'gk2a_ami_le2_scsi_fd020ge_'+gk2a_date+'.bin')

  if (oistat_scsi):
    print('Exist OUTPUT Files : '+gk2a_date)
    continue

 
  ## Check the files
  istat_scsi  = os.path.exists(ipath_l1b+'gk2a_ami_le2_scsi_fd020ge_'+gk2a_date+'.nc')

  if not (istat_scsi):
    print('No Files : '+gk2a_date)
    continue


## Setting Filename
  fn_scsi = glob.glob(ipath_l1b+'gk2a_ami_le2_scsi_fd020ge_'+gk2a_date+'.nc')

  fn_scsi.sort()

  ## Setting public constant variables
  gx_2km  =  5500 ; gy_2km  =  5500


  print(' [ Setting Complete : Variables ] ')
  print(' [ Start Code ] ')

  for k in range(0,1):

    start_time = time.time()

    #date_name = name[25:-3]
    date_name = gk2a_date

    print('------------------------------------')
    print(date_name, k, len(fn_scsi))  

  ## GK-2A AMI Snow Cover
    D_scsi = Dataset(fn_scsi[k], mode='r')  
    key_group = list(D_scsi.variables.keys())
    gk2a_sc_2km = D_scsi.variables[key_group[0]][:]

    print(' [ Reading Complete : Snow Cover ] ')
 
##---------------------------------------------------------

    ## Writing Reflectance Binary File
    binfile = open(opath_l1b+'gk2a_ami_le2_scsi_fd020ge_'+date_name+'.bin', 'wb')
    binfile.write(gk2a_sc_2km)
    binfile.close()
  
    print(' [ Writing Complete : Snow Cover Files ] ') 
 
    print("--- %s seconds ---" % (time.time() - start_time))
 
