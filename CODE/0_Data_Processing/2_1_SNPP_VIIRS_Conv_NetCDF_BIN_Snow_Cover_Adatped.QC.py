# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 14:57:45 2020

@author: jindonghyun
"""

from netCDF4 import Dataset
import netCDF4 as nc
import numpy as np
import glob, os


ipath = '/ecoface/DHJIN/DATA_VIIRS.SC/0_NC.File/202001/'
opath = '/ecoface/DHJIN/DATA_VIIRS.SC/1_BIN.File/202001/'

if not os.path.isdir(opath):
    os.makedirs(opath)
    print(" Make OUTPUT directory ")
    
fn = glob.glob(ipath+'VNP10*.nc')
fn.sort()

f = open(opath+'vnp10_filelist_202001.txt', mode = 'wt')

for k in range(len(fn)):
  print(k, fn[k])

  name = os.path.basename(fn[k])
  output_name = name[0:-4]

  print('-------------------------------------')
  print(output_name, k)

  D = Dataset(fn[k], mode='r')
  key_group = list(D.groups.keys())

  print(key_group)
  
## Bring the Geolocation Information
  key_geo = list(D.dimensions.keys())
  D_vy = D.dimensions[key_geo[0]]
  vy = D_vy.size
  D_vx = D.dimensions[key_geo[1]]
  vx = D_vx.size

## Write output_name, vx, vy
  f.write(output_name+'   '+str(vx)+'   '+str(vy)+'\n')

## Geolocation Data, IST Data Groupd Loop 
  for i in range(len(key_group)):
    print(i, key_group[i])

    if key_group[i] == 'GeolocationData':
      print('NetCDF Group : ', key_group[i])

      D2 = D.groups[key_group[i]]
      key = list(D2.variables.keys())[:]

## Data in the groups Loop
      for j in range(len(key)):
        print(' Data : ', key[j])

        if key[j] == 'latitude':
            lat_tmp = D2.variables[key[j]][:]
            lat = np.flip((np.flip(lat_tmp, 0)), 1)
            lat.filled().tofile(opath+output_name+'_'+key[j]+'.bin')
        elif key[j] == 'longitude':
            lon_tmp = D2.variables[key[j]][:]
            lon = np.flip((np.flip(lon_tmp, 0)), 1)
            lon.filled().tofile(opath+output_name+'_'+key[j]+'.bin')

    elif key_group[i] == 'SnowData':
      print('NetCDF Group : ', key_group[i])

      D2 = D.groups[key_group[i]]
      key = list(D2.variables.keys())[:]
      
## Data in the groups Loop
      for j in range(len(key)):
        print(' Data : ', key[j])

        if key[j] == 'Basic_QA':
            basic_qa_tmp = D2.variables[key[j]][:]
            data_basic_qa = np.flip((np.flip(basic_qa_tmp,0)),1)

        elif key[j] == 'NDSI_Snow_Cover':
            ndsi_sc_tmp = D2.variables[key[j]][:]
            data_ndsi_sc = np.flip((np.flip(ndsi_sc_tmp,0)),1)

        if j == 3: ## [ j==3 : NDSI_Snow_Cover ]
## Adapted QA to NDSI_Snow_Cover
          out_data = data_basic_qa
          out_data = np.where(data_ndsi_sc <= 100, data_ndsi_sc, out_data)
          out_data = np.where(data_basic_qa == 1, 150, out_data)
          out_data = np.where(data_basic_qa == 2, 150, out_data)
          out_data = np.where(data_basic_qa == 3, 150, out_data)

## Write with Binary file
          binfile = open(opath+output_name+'_QA.NDSI.SC.bin', 'wb')
          binfile.write(out_data)
          binfile.close()

  D.close()
f.close()
