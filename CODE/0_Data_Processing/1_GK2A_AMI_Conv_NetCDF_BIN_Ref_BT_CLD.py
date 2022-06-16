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
start = datetime.datetime(2020, 2, 18, 3, 0)
end   = datetime.datetime(2019, 2, 18, 4, 0).strftime('%Y%m%d%H%M')
#delta = timedelta(minutes = 10)
delta = timedelta(hours = 1)

dt = start
date_list = [dt.strftime('%Y%m%d%H%M')]

while 1:
  dt = dt + delta
  date_list.append(dt.strftime('%Y%m%d%H%M'))
  if (dt.strftime('%Y%m%d%H%M')==end): break

for date in date_list:

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

  #ipath_cld = '/data/GK2A/cld/'+yy+mm+'/'+dd+'/'+hh+'/'
  #ipath_l1b = '/data/GK2A/L1B/'+yy+mm+'/'+dd+'/'+hh+'/'
  ipath_cld = '/data/DHJIN/DATA_GK2A.AMI/202002180300/'
  ipath_l1b = ipath_cld
  opath_cld = '/data/DHJIN/DATA_GK2A.AMI/CLD/1_BIN.File/'+yy+mm+'/'+dd+'/'+hh+'/'
  opath_l1b = '/data/DHJIN/DATA_GK2A.AMI/L1B/1_BIN.File/'+yy+mm+'/'+dd+'/'+hh+'/'

  if not os.path.isdir(opath_cld):
    os.system("mkdir -p "+opath_cld)
    print("Make OUTPUT directory")

  if not os.path.isdir(opath_l1b):
    os.system("mkdir -p "+opath_l1b)
    print("Make OUTPUT directory")

 ## Check the files
  oistat_cld   = os.path.exists(opath_cld+'gk2a_ami_le2_cld_fd020ge_'+gk2a_date+'.bin')
  oistat_ch01  = os.path.exists(opath_l1b+'gk2a_ami_le1b_vi004_fd020ge_'+gk2a_date+'.bin')
  oistat_ch02  = os.path.exists(opath_l1b+'gk2a_ami_le1b_vi005_fd020ge_'+gk2a_date+'.bin')
  oistat_ch03  = os.path.exists(opath_l1b+'gk2a_ami_le1b_vi006_fd020ge_'+gk2a_date+'.bin')
  oistat_ch04  = os.path.exists(opath_l1b+'gk2a_ami_le1b_vi008_fd020ge_'+gk2a_date+'.bin')
  oistat_ch05  = os.path.exists(opath_l1b+'gk2a_ami_le1b_nr013_fd020ge_'+gk2a_date+'.bin')
  oistat_ch06  = os.path.exists(opath_l1b+'gk2a_ami_le1b_nr016_fd020ge_'+gk2a_date+'.bin')
  oistat_ch07  = os.path.exists(opath_l1b+'gk2a_ami_le1b_sw038_fd020ge_'+gk2a_date+'.bin')
  oistat_ch14  = os.path.exists(opath_l1b+'gk2a_ami_le1b_ir112_fd020ge_'+gk2a_date+'.bin')
  oistat_ch15  = os.path.exists(opath_l1b+'gk2a_ami_le1b_ir123_fd020ge_'+gk2a_date+'.bin')

  if (oistat_cld and oistat_ch01 and oistat_ch02 and oistat_ch03 and oistat_ch04 and \
          oistat_ch05 and oistat_ch06 and oistat_ch07 and oistat_ch14 and oistat_ch15):
    print('Exist OUTPUT Files : '+gk2a_date)
    continue

 
  ## Check the files
  istat_cld   = os.path.exists(ipath_cld+'gk2a_ami_le2_cld_fd020ge_'+gk2a_date+'.nc')
  istat_ch01  = os.path.exists(ipath_l1b+'gk2a_ami_le1b_vi004_fd010ge_'+gk2a_date+'.nc')
  istat_ch02  = os.path.exists(ipath_l1b+'gk2a_ami_le1b_vi005_fd010ge_'+gk2a_date+'.nc')
  istat_ch03  = os.path.exists(ipath_l1b+'gk2a_ami_le1b_vi006_fd005ge_'+gk2a_date+'.nc')
  istat_ch04  = os.path.exists(ipath_l1b+'gk2a_ami_le1b_vi008_fd010ge_'+gk2a_date+'.nc')
  istat_ch05  = os.path.exists(ipath_l1b+'gk2a_ami_le1b_nr013_fd020ge_'+gk2a_date+'.nc')
  istat_ch06  = os.path.exists(ipath_l1b+'gk2a_ami_le1b_nr016_fd020ge_'+gk2a_date+'.nc')
  istat_ch07  = os.path.exists(ipath_l1b+'gk2a_ami_le1b_sw038_fd020ge_'+gk2a_date+'.nc')
  istat_ch14  = os.path.exists(ipath_l1b+'gk2a_ami_le1b_ir112_fd020ge_'+gk2a_date+'.nc')
  istat_ch15  = os.path.exists(ipath_l1b+'gk2a_ami_le1b_ir123_fd020ge_'+gk2a_date+'.nc')

  if not (istat_cld and istat_ch01 and istat_ch02 and istat_ch03 and istat_ch04 and \
          istat_ch05 and istat_ch06 and istat_ch07 and istat_ch14 and istat_ch15):
    print('No Files : '+gk2a_date)
    continue


## Setting Filename
  fn_cld = glob.glob(ipath_cld+'gk2a_ami_le2_cld_fd020ge_'+gk2a_date+'.nc')
  fn_ch01 = glob.glob(ipath_l1b+'gk2a_ami_le1b_vi004_fd010ge_'+gk2a_date+'.nc')
  fn_ch02 = glob.glob(ipath_l1b+'gk2a_ami_le1b_vi005_fd010ge_'+gk2a_date+'.nc')
  fn_ch03 = glob.glob(ipath_l1b+'gk2a_ami_le1b_vi006_fd005ge_'+gk2a_date+'.nc')
  fn_ch04 = glob.glob(ipath_l1b+'gk2a_ami_le1b_vi008_fd010ge_'+gk2a_date+'.nc')
  fn_ch05 = glob.glob(ipath_l1b+'gk2a_ami_le1b_nr013_fd020ge_'+gk2a_date+'.nc')
  fn_ch06 = glob.glob(ipath_l1b+'gk2a_ami_le1b_nr016_fd020ge_'+gk2a_date+'.nc')
  fn_ch07 = glob.glob(ipath_l1b+'gk2a_ami_le1b_sw038_fd020ge_'+gk2a_date+'.nc')
  fn_ch14 = glob.glob(ipath_l1b+'gk2a_ami_le1b_ir112_fd020ge_'+gk2a_date+'.nc')
  fn_ch15 = glob.glob(ipath_l1b+'gk2a_ami_le1b_ir123_fd020ge_'+gk2a_date+'.nc')

  fn_cld.sort()
  fn_ch01.sort()
  fn_ch02.sort()
  fn_ch03.sort()
  fn_ch04.sort()
  fn_ch05.sort()
  fn_ch06.sort()
  fn_ch07.sort()
  fn_ch14.sort()
  fn_ch15.sort()


  ## Setting public constant variables
  gx_2km  =  5500 ; gy_2km  =  5500
  gx_1km  = 11000 ; gy_1km  = 11000
  gx_500m = 22000 ; gy_500m = 22000


  ## Setting Visible constant variables
  nml_gain_vis1   =  0.363545805215835   ; nml_gain_vis2   =  0.343625485897064
  nml_gain_vis3   =  0.154856294393539   ; nml_gain_vis4   =  0.0457241721451282
  nml_gain_vis5   =  0.0346878096461296  ; nml_gain_vis6   =  0.0498007982969284

  nml_offset_vis1 = -7.27090454101562    ; nml_offset_vis2 = -6.87249755859375
  nml_offset_vis3 = -6.19424438476562    ; nml_offset_vis4 = -3.65792846679687
  nml_offset_vis5 = -1.38751220703125    ; nml_offset_vis6 = -0.996017456054687

  nml_r_to_a_vis1 =  0.0015582450        ; nml_r_to_a_vis2 =  0.0016595767
  nml_r_to_a_vis3 =  0.0019244840        ; nml_r_to_a_vis4 =  0.0032723873
  nml_r_to_a_vis5 =  0.0087081313        ; nml_r_to_a_vis6 =  0.0129512876


  ## Setting IR constant variables
  nml_wavenum_ir07  = 2612.677373521110
  nml_wavenum_ir14  =  891.713057301260
  nml_wavenum_ir15  =  810.609007871230

  nml_gain_ir07    = -0.00108296517282724000 
  nml_gain_ir14    = -0.02167448587715620000 
  nml_gain_ir15    = -0.02337997220456600000 

  nml_offset_ir07  =  17.69998741149900000000 
  nml_offset_ir14  = 176.71343994140600000000 
  nml_offset_ir15  = 190.64962768554600000000 

  nml_coef_c0_07    = -0.447843939824124 
  nml_coef_c0_14    = -0.249111718496148 
  nml_coef_c0_15    = -0.458113885722738 

  nml_coef_c1_07    =  1.000655680903890 
  nml_coef_c1_14    =  1.001211668737560 
  nml_coef_c1_15    =  1.002455209755350 

  nml_coef_c2_07    =  -6.338240899124480E-08
  nml_coef_c2_14    =  -1.131679640116650E-06
  nml_coef_c2_15    =  -2.530643147204760E-06

  planck_c    = 6.62607004E-34
  light_speed = 2.99792458E8
  boltz_c     = 1.38064852E-23

  c1 = 2 * planck_c * light_speed ** 2
  c2 = (planck_c * light_speed) / boltz_c

  print(' [ Setting Complete : Variables ] ')
  print(' [ Start Code ] ')

  for k in range(len(fn_cld)):

    start_time = time.time()

    name = os.path.basename(fn_cld[k])
    output_name = name[0:-3]
    date_name = name[25:-3]

    print('------------------------------------')
    print(date_name, k, len(fn_cld))  

    ## Cloud Mask
    D = Dataset(fn_cld[k], mode='r')
    key_group = list(D.variables.keys())
    cld_data = D.variables[key_group[0]][:]
    print(' [ Reading Complete : CLD ] ')
 
    binfile = open(opath_cld+output_name+'.bin', 'wb')
    binfile.write(cld_data)
    binfile.close()

    ## GK-2A AMI Band 1
    D_ch01 = Dataset(fn_ch01[k], mode='r')  
    key_group = list(D_ch01.variables.keys())
    ch01_data_dn = D_ch01.variables[key_group[0]][:]
    ch01_data_dn = ch01_data_dn.astype('float32')
    ch01_data_dn[ch01_data_dn == 32768.] = np.NaN

    ## GK-2A AMI Band 2
    D_ch02 = Dataset(fn_ch02[k], mode='r')  
    key_group = list(D_ch02.variables.keys())
    ch02_data_dn = D_ch02.variables[key_group[0]][:]
    ch02_data_dn = ch02_data_dn.astype('float32')
    ch02_data_dn[ch02_data_dn == 32768.] = np.NaN

    ## GK-2A AMI Band 3
    D_ch03 = Dataset(fn_ch03[k], mode='r')
    key_group = list(D_ch03.variables.keys())
    ch03_data_dn = D_ch03.variables[key_group[0]][:]
    ch03_data_dn = ch03_data_dn.astype('float32')
    ch03_data_dn[ch03_data_dn == 32768.] = np.NaN

    ## GK-2A AMI Band 4
    D_ch04 = Dataset(fn_ch04[k], mode='r')
    key_group = list(D_ch04.variables.keys())
    ch04_data_dn = D_ch04.variables[key_group[0]][:]
    ch04_data_dn = ch04_data_dn.astype('float32')
    ch04_data_dn[ch04_data_dn == 32768.] = np.NaN

    ## GK-2A AMI Band 5
    D_ch05 = Dataset(fn_ch05[k], mode='r')
    key_group = list(D_ch05.variables.keys())
    ch05_data_dn = D_ch05.variables[key_group[0]][:]
    ch05_data_2km = ch05_data_dn.astype('float32')
    ch05_data_2km[ch05_data_2km == 32768.] = np.NaN

    ## GK-2A AMI Band 6
    D_ch06 = Dataset(fn_ch06[k], mode='r')
    key_group = list(D_ch06.variables.keys())
    ch06_data_dn = D_ch06.variables[key_group[0]][:]
    ch06_data_2km = ch06_data_dn.astype('float32')
    ch06_data_2km[ch06_data_2km == 32768.] = np.NaN
    print(' [ Reading Complete : Visible Channel ] ')

    ## GK-2A AMI Band 7
    D_ch07 = Dataset(fn_ch07[k], mode='r')
    key_group = list(D_ch07.variables.keys())
    ch07_data_dn = D_ch07.variables[key_group[0]][:]
    ch07_data_2km = ch07_data_dn.astype('float32')
    ch07_data_2km[ch07_data_2km == 32768.] = np.NaN

    ## GK-2A AMI Band 14
    D_ch14 = Dataset(fn_ch14[k], mode='r')
    key_group = list(D_ch14.variables.keys())
    ch14_data_dn = D_ch14.variables[key_group[0]][:]
    ch14_data_2km = ch14_data_dn.astype('float32')
    ch14_data_2km[ch14_data_2km == 32768.] = np.NaN

    ## GK-2A AMI Band 15
    D_ch15 = Dataset(fn_ch15[k], mode='r')
    key_group = list(D_ch15.variables.keys())
    ch15_data_dn = D_ch15.variables[key_group[0]][:]
    ch15_data_2km = ch15_data_dn.astype('float32')
    ch15_data_2km[ch15_data_2km == 32768.] = np.NaN
    print(' [ Reading Complete : IR Channel ] ')
 
##---------------------------------------------------------

    ch01_data_2km = np.zeros((5500,5500), dtype='uint16')
    ch02_data_2km = np.zeros((5500,5500), dtype='uint16')
    ch03_data_2km = np.zeros((5500,5500), dtype='uint16')
    ch04_data_2km = np.zeros((5500,5500), dtype='uint16')
 
##---------------------------------------------------------
    gy_2km_group  =  550  ;  gx_2km_group  = 550
    gy_1km_group  = 1100  ;  gx_1km_group  = 1100
    gy_500m_group = 2200  ;  gx_500m_group = 2200


    for jj in range(0,10):
      for ii in range(0,10):
 
        px_500m = ii*2200 ;  py_500m = jj*2200  
        px_1km  = ii*1100 ;  py_1km  = jj*1100
        px_2km  = ii*550  ;  py_2km  = jj*550

        ## Resamlping 1km to 2km
        tmp = ch01_data_dn[py_1km:py_1km+1100,px_1km:px_1km+1100].reshape(gy_2km_group, gy_1km_group//gy_2km_group,\
          gx_2km_group, gx_1km_group//gx_2km_group)
        tmp = tmp.astype('uint16')
        tmp_1 = np.nanmean(tmp, axis=-1)
        ch01_data_2km[py_2km:py_2km+550,px_2km:px_2km+550] = np.nanmean(tmp_1,1)

        tmp = ch02_data_dn[py_1km:py_1km+1100,px_1km:px_1km+1100].reshape(gy_2km_group, gy_1km_group//gy_2km_group,\
          gx_2km_group, gx_1km_group//gx_2km_group)
        tmp = tmp.astype('uint16')
        tmp_1 = np.nanmean(tmp, axis=-1)
        ch02_data_2km[py_2km:py_2km+550,px_2km:px_2km+550] = np.nanmean(tmp_1,1)

        tmp = ch04_data_dn[py_1km:py_1km+1100,px_1km:px_1km+1100].reshape(gy_2km_group, gy_1km_group//gy_2km_group,\
          gx_2km_group, gx_1km_group//gx_2km_group)
        tmp = tmp.astype('uint16')
        tmp_1 = np.nanmean(tmp, axis=-1)
        ch04_data_2km[py_2km:py_2km+550,px_2km:px_2km+550] = np.nanmean(tmp_1,1)

        ## Resamlping 500m to 2km
        tmp = ch03_data_dn[py_500m:py_500m+2200,px_500m:px_500m+2200].reshape(gy_2km_group, gy_500m_group//gy_2km_group,\
          gx_2km_group, gx_500m_group//gx_2km_group)
        tmp = tmp.astype('uint16')
        tmp_1 = np.nanmean(tmp, axis=-1)
        ch03_data_2km[py_2km:py_2km+550,px_2km:px_2km+550] = np.nanmean(tmp_1,1)

    print( ' [ Resampling Complete : Visible Channels ] ' )

    ch01_data_2km = ch01_data_2km.astype('float32')
    ch02_data_2km = ch02_data_2km.astype('float32')
    ch03_data_2km = ch03_data_2km.astype('float32')
    ch04_data_2km = ch04_data_2km.astype('float32')

    gk2a_ch01_ref_2km = np.zeros((5500,5500), dtype='float32')
    gk2a_ch02_ref_2km = np.zeros((5500,5500), dtype='float32')
    gk2a_ch03_ref_2km = np.zeros((5500,5500), dtype='float32')
    gk2a_ch04_ref_2km = np.zeros((5500,5500), dtype='float32')
    gk2a_ch05_ref_2km = np.zeros((5500,5500), dtype='float32')
    gk2a_ch06_ref_2km = np.zeros((5500,5500), dtype='float32')

    ## Calculating Radiometric Calibration (Visible Channels)
    rad_vis_ch01 = ch01_data_2km * nml_gain_vis1 + nml_offset_vis1
    gk2a_ch01_ref_2km = rad_vis_ch01 * nml_r_to_a_vis1 * 10000.
    gk2a_ch01_ref_2km = gk2a_ch01_ref_2km.astype('int16')

    rad_vis_ch02 = ch02_data_2km * nml_gain_vis2 + nml_offset_vis2
    gk2a_ch02_ref_2km = rad_vis_ch02 * nml_r_to_a_vis2 * 10000.
    gk2a_ch02_ref_2km = gk2a_ch02_ref_2km.astype('int16')

    rad_vis_ch03 = ch03_data_2km * nml_gain_vis3 + nml_offset_vis3
    gk2a_ch03_ref_2km = rad_vis_ch03 * nml_r_to_a_vis3 * 10000.
    gk2a_ch03_ref_2km = gk2a_ch03_ref_2km.astype('int16')

    rad_vis_ch04 = ch04_data_2km * nml_gain_vis4 + nml_offset_vis4
    gk2a_ch04_ref_2km = rad_vis_ch04 * nml_r_to_a_vis4 * 10000.
    gk2a_ch04_ref_2km = gk2a_ch04_ref_2km.astype('int16')

    rad_vis_ch05 = ch05_data_2km * nml_gain_vis5 + nml_offset_vis5
    gk2a_ch05_ref_2km = rad_vis_ch05 * nml_r_to_a_vis5 * 10000.
    gk2a_ch05_ref_2km = gk2a_ch05_ref_2km.astype('int16')

    rad_vis_ch06 = ch06_data_2km * nml_gain_vis6 + nml_offset_vis6
    gk2a_ch06_ref_2km = rad_vis_ch06 * nml_r_to_a_vis6 * 10000.
    gk2a_ch06_ref_2km = gk2a_ch06_ref_2km.astype('int16')

    print(' [ Calibration Complete : Visible Files ] ') 

    ## Writing Reflectance Binary File
    binfile = open(opath_l1b+'gk2a_ami_le1b_vi004_fd020ge_'+date_name+'.bin', 'wb')
    binfile.write(gk2a_ch01_ref_2km)
    binfile.close()
  
    binfile = open(opath_l1b+'gk2a_ami_le1b_vi005_fd020ge_'+date_name+'.bin', 'wb')
    binfile.write(gk2a_ch02_ref_2km)
    binfile.close()
  
    binfile = open(opath_l1b+'gk2a_ami_le1b_vi006_fd020ge_'+date_name+'.bin', 'wb')
    binfile.write(gk2a_ch03_ref_2km)
    binfile.close()
  
    binfile = open(opath_l1b+'gk2a_ami_le1b_vi008_fd020ge_'+date_name+'.bin', 'wb')
    binfile.write(gk2a_ch04_ref_2km)
    binfile.close()
  
    binfile = open(opath_l1b+'gk2a_ami_le1b_nr013_fd020ge_'+date_name+'.bin', 'wb')
    binfile.write(gk2a_ch05_ref_2km)
    binfile.close()
  
    binfile = open(opath_l1b+'gk2a_ami_le1b_nr016_fd020ge_'+date_name+'.bin', 'wb')
    binfile.write(gk2a_ch06_ref_2km)
    binfile.close()
  
##-------------------------------------------------------------
    rad_ir_ch07 = np.zeros((5500,5500), dtype='float32')
    rad_tmp_ch07 = np.zeros((5500,5500), dtype='float32')
    gk2a_ch07_bt_2km = np.zeros((5500,5500), dtype='float32')

    rad_ir_ch14 = np.zeros((5500,5500), dtype='float32')
    rad_tmp_ch14 = np.zeros((5500,5500), dtype='float32')
    gk2a_ch14_bt_2km = np.zeros((5500,5500), dtype='float32')

    rad_ir_ch15 = np.zeros((5500,5500), dtype='float32')
    rad_tmp_ch15 = np.zeros((5500,5500), dtype='float32')
    gk2a_ch15_bt_2km = np.zeros((5500,5500), dtype='float32')


    ## Calculating Radiometric Calibration (IR Channels)
    ## ch 07
    rad_ir_ch07 = ch07_data_2km * nml_gain_ir07 + nml_offset_ir07
    rad_tmp_ch07 = c2 * ( nml_wavenum_ir07 * 100. ) / \
               ( np.log( (c1 * (nml_wavenum_ir07 * 100.)**3) / (rad_ir_ch07 * 1.E-5) + 1.) )
    gk2a_ch07_bt_2km = \
               nml_coef_c0_07 + nml_coef_c1_07 * rad_tmp_ch07 + nml_coef_c2_07 * rad_tmp_ch07**2
    #gk2a_ch07_bt_2km = ( gk2a_ch07_bt_2km - 100 ) * 100.


    ## ch 14
    rad_ir_ch14 = ch14_data_2km * nml_gain_ir14 + nml_offset_ir14
    rad_tmp_ch14 = c2 * ( nml_wavenum_ir14 * 100. ) / \
               ( np.log( (c1 * (nml_wavenum_ir14 * 100.)**3) / (rad_ir_ch14 * 1.E-5) + 1.) )
    gk2a_ch14_bt_2km = \
               nml_coef_c0_14 + nml_coef_c1_14 * rad_tmp_ch14 + nml_coef_c2_14 * rad_tmp_ch14**2
    #gk2a_ch14_bt_2km = ( gk2a_ch14_bt_2km - 100 ) * 100.


    ## ch 15
    rad_ir_ch15 = ch15_data_2km * nml_gain_ir15 + nml_offset_ir15
    rad_tmp_ch15 = c2 * ( nml_wavenum_ir15 * 100. ) / \
               ( np.log( (c1 * (nml_wavenum_ir15 * 100.)**3) / (rad_ir_ch15 * 1.E-5) + 1.) )
    gk2a_ch15_bt_2km = \
               nml_coef_c0_15 + nml_coef_c1_15 * rad_tmp_ch15 + nml_coef_c2_15 * rad_tmp_ch15**2
    #gk2a_ch15_bt_2km = ( gk2a_ch15_bt_2km - 100 ) * 100.


    #gk2a_ch07_bt_2km = gk2a_ch07_bt_2km.astype(int16)
    #gk2a_ch14_bt_2km = gk2a_ch14_bt_2km.astype(int16)
    #gk2a_ch15_bt_2km = gk2a_ch15_bt_2km.astype(int16)

    print(' [ Calibration Complete : IR Files ] ') 

    ## Writing Reflectance File
    binfile = open(opath_l1b+'gk2a_ami_le1b_sw038_fd020ge_'+date_name+'.bin', 'wb')
    binfile.write(gk2a_ch07_bt_2km)
    binfile.close()
  
    binfile = open(opath_l1b+'gk2a_ami_le1b_ir112_fd020ge_'+date_name+'.bin', 'wb')
    binfile.write(gk2a_ch14_bt_2km)
    binfile.close()

    binfile = open(opath_l1b+'gk2a_ami_le1b_ir123_fd020ge_'+date_name+'.bin', 'wb')
    binfile.write(gk2a_ch15_bt_2km)
    binfile.close()

    print(' [ Writing Complete : BT Files ] ') 
 
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
