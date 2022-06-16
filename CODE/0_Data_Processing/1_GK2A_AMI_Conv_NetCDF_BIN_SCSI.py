from netCDF4 import Dataset
import netCDF4 as nc
import numpy as np
import glob, os
import math
#from pylab import *
import time
import datetime
from datetime import timedelta



## Setting Time
#start = datetime.datetime(2020, 1, 1, 0, 0)
#end   = datetime.datetime(2020, 3, 31, 23, 50).strftime('%Y%m%d%H%M')
#delta = timedelta(minutes = 10)
#delta = timedelta(hours = 1)

#dt = start
#date_list = [dt.strftime('%Y%m%d%H%M')]

#while 1:
#  dt = dt + delta
#  date_list.append(dt.strftime('%Y%m%d%H%M'))
#  if (dt.strftime('%Y%m%d%H%M')==end): break

ipath_l1b = '/ecoface2/DHJIN/DATA_GK2A.SCSI/SCSI/202003/'
opath_l1b = '/data/DHJIN/DATA_GK2A.AMI/L2_SCSI/'


fn = glob.glob(ipath_l1b+'gk2a_ami_le2_scsi_fd020ge_20200305*00.nc')
fn.sort()

for kk in range(len(fn)):

  tmp = os.path.split(fn[kk])
  tmp_1 = tmp[1].split('_')
  date = tmp_1[5][:-3]

  print(date)

#ipath_filelist = '/ecoface/DHJIN/'
#yymm = '202001'
#yymm = '201912'

#f_txt = open(ipath_filelist+'viirs_sc.si.ist_filelist_gk2atime_'+yymm+'_07.txt', 'r')
#f_txt = open(ipath_filelist+'viirs_sc.si.ist_filelist_gk2atime_'+yymm+'.txt', 'r')
#lines = f_txt.readlines()

#for date in lines:


  yy = date[0:4]
  mm = date[4:6]
  dd = date[6:8]
  hh = date[8:10]
  mn = date[10:12]
  gk2a_date = yy+mm+dd+hh+mn
  print(gk2a_date)

  #opath_l1b = '/ecoface2/DHJIN/DATA_GK2A.SCSI/SCSI_bin/202003/'

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
 
'''
## L1B --> Reflectane
  rad_vis = ch01_data_dn * nml_gain_vis(1) + nml_offset_vis(1)
  ref_vis = rad_vis * nml_r_to_a_vis(1)

## L1B --> BT
  rad_ir  = ch07_data_dn * nml_gain_ir(1) + nml_offset_ir(1)
  c1 = 2 * planck_c * light_speed*2
  c2 = (planck_c * light_speed) / boltz_c
  planck_c = 6.62607004E-34
  light_speed = 2.99792458E8
  boltz_c = 1.38064852E-23
  rad_tmp = c2 * ( nml_wavenum_ir(1) * 100. ) / &
            #-----------------------------------#
            ( log( (c1 * (nml_wavenum_ir(1) * 100.)**3) / (rad_ir  * 1.E-5) + 1.) ) 
  bt_ir   =  nml_coef_c0(1) + nml_coef_c1(1) * rad_tmp + nml_coef_c2(1) * rad_tmp**2
'''
