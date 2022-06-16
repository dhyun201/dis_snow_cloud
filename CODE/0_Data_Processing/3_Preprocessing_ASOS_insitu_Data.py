# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 14:08:15 2021

@author: donghyunjin
"""
#------------------------------------------------------------------------------
# ASOS 관측지점 정보 csv 파일 읽기
def read2information(filename):
  import pandas as pd
  import numpy as np
    
# CSV 파일을 데이터프레임으로 읽기 - 공공기관은 engine='python'으로 해야함 -> 인코드 : euc-kr임 (일반적으로 인코드 : utf-8)
  df_kor_tmp = pd.read_csv(filename, engine='python')

# 데이터프레임 
  df_kor_1 = df_kor_tmp.drop(columns=['시작일'])
  df_kor_2 = df_kor_1.drop(columns=['종료일'])
  df_kor_3 = df_kor_2.drop(columns=['지점명'])                                  
  df_kor_4 = df_kor_3.drop(columns=['지점주소'])  
  df_kor_5 = df_kor_4.drop(columns=['관리관서'])  
  df_kor_6 = df_kor_5.drop(columns=['기압계(관측장비지상높이(m))'])    
  df_kor_7 = df_kor_6.drop(columns=['기온계(관측장비지상높이(m))'])    
  df_kor_8 = df_kor_7.drop(columns=['풍속계(관측장비지상높이(m))'])    
  df_kor   = df_kor_8.drop(columns=['강우계(관측장비지상높이(m))'])    
  
  del df_kor_tmp, df_kor_1, df_kor_2, df_kor_3, df_kor_4, df_kor_5, df_kor_6, df_kor_7, df_kor_8
  
# 데이터프레임 class name 변경  
  df_kor.rename(columns={'지점':'Number'}, inplace=True)
  df_kor.rename(columns={'위도':'Latitude'}, inplace=True)  
  df_kor.rename(columns={'경도':'Longitude'}, inplace=True)
  df_kor.rename(columns={'노장해발고도(m)':'DEM'}, inplace=True)

# 데이터타입 'object'를 'float64'로 변경
  df_fi = df_kor.astype({'Latitude':'float'})
  #df_fi = df_kor.astpye({'Latitude':'float64'})
  #df_kor_final = pd.to_numeric(df_kor, errors='coerce')  # If ‘coerce’, then invalid parsing will be set as NaN
  #df_kor_final = pd.to_numeric(df_kor, errors='ignore')  # If ‘ignore’, then invalid parsing will return the input

  return df_fi

#------------------------------------------------------------------------------
# 2020년 01월 01일 ~ 03월 31일 ASOS 기반 적설자료 읽기
def read2snow4insitu(filename):
  import pandas as pd
  import numpy as np

# CSV 파일을 데이터프레임으로 읽기 - 공공기관은 engine='python'으로 해야함 -> 인코드 : euc-kr임 (일반적으로 인코드 : utf-8)
  fn = "D:/PhD/0_Data_in-situ_ASOS/OBS_ASOS_TIM_20211129140456.csv"
  df_tmp = pd.read_csv(fn, engine='python')

# 데이터프레임 
  df_1 = df_tmp.drop(columns=['지점명'])
  df   = df_1.drop(columns=['3시간신적설(cm)'])
  
# 데이터프레임 class name 변경  
  df.rename(columns={'지점':'Number'}, inplace=True)
  df.rename(columns={'일시':'Date'}, inplace=True)  
  df.rename(columns={'적설(cm)':'Snow_depth'}, inplace=True)  
  
# 문자열(object)을 datetime64 타입으로 변경  
  # df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d %H:%M:%S', errors='raise')  

  # df['Birth_date']       = df['Date'].dt.date         # YYYY-MM-DD(문자)
  # df['Birth_year']       = df['Date'].dt.year         # 연(4자리숫자)
  # df['Birth_month']      = df['Date'].dt.month        # 월(숫자)
  # df['Birth_month_name'] = df['Date'].dt.month_name() # 월(문자)

  # df['Birth_day']        = df['Date'].dt.day          # 일(숫자)
  # df['Birth_time']       = df['Date'].dt.time         # HH:MM:SS(문자)
  # df['Birth_hour']       = df['Date'].dt.hour         # 시(숫자)
  # df['Birth_minute']     = df['Date'].dt.minute       # 분(숫자)
  # df['Birth_second']     = df['Date'].dt.second       # 초(숫자)
  
  return df

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
import numpy as np
import os

# Main code
ipath = "D:/PhD/0_Data_in-situ_ASOS/"
ipath_gk2a_latlon = "F:/Instrument_AMI/GK2A.AMI_DATA/latlon/"
opath_txt = ipath+"insitu_snow_infor/"
if not (os.path.exists(opath_txt)): 
  os.makedirs(opath_txt)

#------------------------------------------------------------------------------
# GK-2A 위도, 경도 자료 읽기
lat = np.fromfile(ipath_gk2a_latlon+"Lat_2km.bin", dtype='f').reshape(5500,5500)
lon = np.fromfile(ipath_gk2a_latlon+"Lon_2km.bin", dtype='f').reshape(5500,5500)

#------------------------------------------------------------------------------
# csv 파일 이름 설정
fn_info = ipath+"META_information_in-situ_point_20211129140727.csv"
fn_sc = ipath+"OBS_ASOS_TIM_20211129140456.csv"

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# csv 파일(ASOS 전체 관측지점 정보) 읽기
df_tmp = read2information(fn_info)
df_asos_info = df_tmp.dropna(axis=0)  ;  del df_tmp

array_asos_info_no  = np.array(df_asos_info['Number'])
array_asos_info_lat = np.array(df_asos_info['Latitude'])
array_asos_info_lon = np.array(df_asos_info['Longitude'])
array_asos_info_dem = np.array(df_asos_info['DEM'])

df_asos_info.info()

#------------------------------------------------------------------------------
# ASOS 전체 관측소 정보(Point No., Latitude, Longitude, DEM) Text 파일로 저장
f = open(opath_txt+"META_information_in-situ_point.txt", "a")
for k in range(len(array_asos_info_no)):
  row_str = str(array_asos_info_no[k])+"  "+str(array_asos_info_lat[k])+"  "\
            +str(array_asos_info_lon[k])+"  "+str(array_asos_info_dem[k])+"\n"  
  f.write(row_str)            
f.close()

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# Snow in-situ 자료 읽기
df_sc = read2snow4insitu(fn_sc)

#------------------------------------------------------------------------------
# 적설이 있는 지역의 관측소 No. 파악
#insitu_no = (df_sc['Number'])
insitu_no = df_sc['Number'].unique()

#------------------------------------------------------------------------------
# 적설이 있는 관측소 정보 Txt 파일로 저장
f = open(opath_txt+"snow_META_information_in-situ_point.txt", "a")
#f.write("No.  Latitude  Longitude DEM  \n")
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
for k in range(len(insitu_no)):

  ind = (array_asos_info_no == insitu_no[k])
  row_str = str(array_asos_info_no[ind][0])+"  "+str(array_asos_info_lat[ind][0])+"  "\
            +str(array_asos_info_lon[ind][0])+"  "+str(array_asos_info_dem[ind][0])+"\n"
  print(row_str)
  f.write(row_str)
  print(k, " / ", len(insitu_no))   
  
f.close()  
  