#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# For discriminating snow and cloud, Deep learning code

# Version 0.0 : 2021-11-10 = Deep learning 모델 층 구조 확립 목적

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
import os

#------------------------------------------------------------------------------
# Setting path
ipath = "D:/PhD/0_Train.Test_Dataset/Train.Dataset/"

opath_txt = "D:/PhD/00_VERSION_3/6__DL_Make.Model/Train_Model.Design/2_Validation.Accuracy__Step6__Stacked.Accuray_EPOCHs/"
if not os.path.isdir(opath_txt):
       os.makedirs(opath_txt)
       print("make OUTPUT_txt directory")

opath_model = opath_txt

import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
#------------------------------------------------------------------------------
start = time.time()

# Calc. the column on information files
file_nm = 'VIIRS.SC.Product_SC.CLD.Flag_NMSC.SC.Product_SC.Low.CLD.CLR_used_TRAIN.dataset_v1_mod10.bin'
data_info_size = os.path.getsize(ipath+file_nm)
data_info_lines = int(data_info_size / 24 / 4)

# Read VIIRS Snow cover, Snow-free land, Cloud Binary File
dataset = np.fromfile(ipath+file_nm,dtype=np.float32).reshape(data_info_lines, 24)
#------------------------------------------------------------------------------
# dataframe count
df_col_nm = ['gk2a_ch05_ref', 'gk2a_ch06_ref', 'gk2a_ch07_bt',  'gk2a_ch14_bt', 'gk2a_ch15_bt', \
             'ndsi', 'ndwi', \
             'btd_14_07', 'gk2a_ch06_ano', 'gk2a_sza', 'gk2a_lat', 'gk2a_vza',  \
             'gk2a_cld', 'gk2a_sc', 'viirs_data']
df_ct = len(df_col_nm)

# Data Frame
ix = np.arange(1, data_info_lines+1, 1)
df_t = pd.DataFrame(columns = ['gk2a_ch05_ref', 'gk2a_ch06_ref', \
                               'gk2a_ch07_bt',  'gk2a_ch14_bt', 'gk2a_ch15_bt', \
                               'ndsi', 'ndwi', \
                               'btd_14_07', 'gk2a_ch06_ano', 'gk2a_sza', 'gk2a_lat', 'gk2a_vza',\
                               'gk2a_cld', 'gk2a_sc', 'viirs_data'])

df_t['gk2a_ch05_ref']    = pd.Series(index = ix, data=dataset[:, 4], dtype='float32')
df_t['gk2a_ch06_ref']    = pd.Series(index = ix, data=dataset[:, 5], dtype='float32')
df_t['gk2a_ch07_bt']     = pd.Series(index = ix, data=dataset[:, 6], dtype='float32')
df_t['gk2a_ch14_bt']     = pd.Series(index = ix, data=dataset[:, 7], dtype='float32')
df_t['gk2a_ch15_bt']     = pd.Series(index = ix, data=dataset[:, 8], dtype='float32')
df_t['gk2a_cld']         = pd.Series(index = ix, data=dataset[:,13], dtype='float32')
df_t['gk2a_sc']          = pd.Series(index = ix, data=dataset[:,14], dtype='float32')
df_t['viirs_data']       = pd.Series(index = ix, data=dataset[:,15], dtype='float32')
df_t['ndsi']             = pd.Series(index = ix, data=dataset[:,18], dtype='float32')
df_t['ndwi']             = pd.Series(index = ix, data=dataset[:,20], dtype='float32')
df_t['btd_14_07']        = pd.Series(index = ix, data=dataset[:,21], dtype='float32')
df_t['gk2a_ch06_ano']    = pd.Series(index = ix, data=dataset[:,23], dtype='float32')

df_t['gk2a_sza']         = pd.Series(index = ix, data=dataset[:, 9], dtype='float32')
df_t['gk2a_lat']         = pd.Series(index = ix, data=dataset[:,11], dtype='float32')
df_t['gk2a_vza']         = pd.Series(index = ix, data=dataset[:,10], dtype='float32')
#------------------------------------------------------------------------------
# Filtering NaN
df_t = df_t.dropna(axis=0)

# Check the GK-2A Snow Cover - Cloud Flag
gk2a_sc_cld_ind = (df_t['gk2a_sc'] == 3.0) # GK-2A Snow Cover - Low. Cloud Flag
df_t_lcld_tmp = df_t[gk2a_sc_cld_ind]

viirs_sc_ind = (df_t_lcld_tmp['viirs_data'] == 1.0) | (df_t_lcld_tmp['viirs_data'] == 3.0) 
df_t_lcld = df_t_lcld_tmp[viirs_sc_ind]

# viirs_data Flag 값을 (1; Snow, 3; Cloud)에서 (1;Snow, 0;cloud)로 변경
viirs_cld_ind = (df_t_lcld['viirs_data'] == 3.0)
df_t_lcld['viirs_data'][viirs_cld_ind] = 0.0

# dataframe 변수 삭제
df_t_lcld       = df_t_lcld.drop(columns=['gk2a_cld'])  
df_t_lcld       = df_t_lcld.drop(columns=['gk2a_sc'])
df_t_lcld_final = df_t_lcld.drop(columns=['viirs_data'])
print('[Filtering Dataframe] Complete')

del df_t, gk2a_sc_cld_ind, viirs_sc_ind, df_t_lcld_tmp

#------------------------------------------------------------------------------
# Fix variables
df_t_lcld_final.loc[df_t_lcld_final.gk2a_ch05_ref < 0.01, 'gk2a_ch05_ref'] = 0.01  ;  df_t_lcld_final.loc[df_t_lcld_final.gk2a_ch05_ref > 1.0, 'gk2a_ch05_ref'] = 1.0
df_t_lcld_final.loc[df_t_lcld_final.gk2a_ch06_ref < 0.01, 'gk2a_ch06_ref'] = 0.01  ;  df_t_lcld_final.loc[df_t_lcld_final.gk2a_ch06_ref > 1.0, 'gk2a_ch06_ref'] = 1.0

df_t_lcld_final.loc[df_t_lcld_final.gk2a_ch07_bt < 240.0, 'gk2a_ch07_bt'] = 240.0  ;  df_t_lcld_final.loc[df_t_lcld_final.gk2a_ch07_bt > 380.0, 'gk2a_ch07_bt'] = 380.0
df_t_lcld_final.loc[df_t_lcld_final.gk2a_ch14_bt < 210.0, 'gk2a_ch14_bt'] = 210.0  ;  df_t_lcld_final.loc[df_t_lcld_final.gk2a_ch14_bt > 320.0, 'gk2a_ch14_bt'] = 320.0
df_t_lcld_final.loc[df_t_lcld_final.gk2a_ch15_bt < 210.0, 'gk2a_ch15_bt'] = 210.0  ;  df_t_lcld_final.loc[df_t_lcld_final.gk2a_ch15_bt > 310.0, 'gk2a_ch15_bt'] = 310.0

df_t_lcld_final.loc[df_t_lcld_final.ndwi < -0.5, 'ndwi'] = -0.5                    ;  df_t_lcld_final.loc[df_t_lcld_final.ndwi > 1.0, 'ndwi'] = 1.0
df_t_lcld_final.loc[df_t_lcld_final.ndsi < -1.0, 'ndsi'] = -1.0                    ;  df_t_lcld_final.loc[df_t_lcld_final.ndsi > 1.0, 'ndsi'] = 1.0

df_t_lcld_final.loc[df_t_lcld_final.btd_14_07 < -80.0, 'btd_14_07'] = -80.0        ;  df_t_lcld_final.loc[df_t_lcld_final.btd_14_07 > 0.0, 'btd_14_07'] = 0.0
df_t_lcld_final.loc[df_t_lcld_final.gk2a_ch06_ano < -2.0, 'gk2a_ch06_ano'] = -2.0  ;  df_t_lcld_final.loc[df_t_lcld_final.gk2a_ch06_ano > 2.0, 'gk2a_ch06_ano'] = 2.0

df_t_lcld_final.loc[df_t_lcld_final.gk2a_sza < 0.0, 'gk2a_sza'] = 0.0              ;  df_t_lcld_final.loc[df_t_lcld_final.gk2a_sza > 80.0, 'gk2a_sza'] = 80.0
df_t_lcld_final.loc[df_t_lcld_final.gk2a_lat < 20.0, 'gk2a_lat'] = 20.0            ;  df_t_lcld_final.loc[df_t_lcld_final.gk2a_lat > 80.0, 'gk2a_lat'] = 80.0
df_t_lcld_final.loc[df_t_lcld_final.gk2a_vza < 30.0, 'gk2a_vza'] = 30.0            ;  df_t_lcld_final.loc[df_t_lcld_final.gk2a_vza > 90.0, 'gk2a_vza'] = 90.0
print('[Min./Max. value of variables] Check complete')
#------------------------------------------------------------------------------
# Distribute Train & Validation
X_train, X_val, Y_train, Y_val = train_test_split(df_t_lcld_final, df_t_lcld['viirs_data'], test_size=0.4, random_state=0) 
    
del df_t_lcld_final, df_t_lcld

# Normzalization or Standard
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_val_scaled = scaler.transform(X_val)
print('[Distribution Train & Validation, Normalization] Complete')
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Input, Dropout
from keras.callbacks import EarlyStopping
from keras.models import load_model
from keras.utils.vis_utils import plot_model
import matplotlib.pyplot as plt

#------------------------------------------------------------------------------
# 모델 층 구조 및 하이퍼파라미터 후보들 셋팅
input_parameter_ct = 12
#------------------------------------------------------------------------------
# 고정 Parameter
# Model 구조
node_dim = [500]
layer_dim = ['4']
l1_dim = [0]  # regularizier L1
l2_dim = [0]  # regularizier L2
keep_prob_dim = [1.0]  # Dropout
dropout_dim = [0.0]

# Model fit 시, 조건
stop_patience_dim = [20]
learning_rate_dim = [0.0001]  # learning rate
learning_rate_dim_nm = ['0.0001']

batch_size_dim = [128]

#------------------------------------------------------------------------------
# Test  Parameter
epoch_dim = [400]  ;  epoch_dim_str = ['400']

ct = 2
ct_dim = np.arange(ct)

#------------------------------------------------------------------------------

# 모델 구성
for k in range(len(epoch_dim)):  
  f = open(opath_txt+"v3_DL.method_Loss.Acc_Epochs."+str(epoch_dim[k])+"_"+str(ct)+".txt", "w")
  for j in range(len(ct_dim)):  
    for i in range(1):  
      for t in range(1):  

        print("---------------------------------------------------------------")
        name = "Layer:"+str(layer_dim[0])+"  Node:"+str(node_dim[0])+ \
               " Dropout:"+str(dropout_dim[0])+" Regularization L1:"+str(l1_dim[0])+" L2:"+str(l2_dim[0])+\
               " Early.Stopping.Patience:"+str(stop_patience_dim[0])+" Learning.Rate:"+str(learning_rate_dim[0])+\
               " Epoch:"+str(epoch_dim[k])+ " Batch_size:"+str(batch_size_dim[0])

#------------------------------------------------------------------------------
# 모델 디자인 정의
        model = Sequential([Input(shape=(input_parameter_ct,))])

        regularizer_l1 = tf.keras.regularizers.l1(l1_dim[0])
        regularizer_l2 = tf.keras.regularizers.l2(l2_dim[0])

        model.add(Dense(units=node_dim[0], activation='relu', kernel_regularizer=regularizer_l1, activity_regularizer=regularizer_l2))
        if keep_prob_dim[0] != 1.0:
          model.add(Dropout(1-keep_prob_dim[0], seed=2)) 
        model.add(Dense(units=node_dim[0], activation='relu', kernel_regularizer=regularizer_l1, activity_regularizer=regularizer_l2))
        if keep_prob_dim[0] != 1.0:
          model.add(Dropout(1-keep_prob_dim[0], seed=2))         
        model.add(Dense(units=node_dim[0], activation='relu', kernel_regularizer=regularizer_l1, activity_regularizer=regularizer_l2))                      
        if keep_prob_dim[0] != 1.0:
          model.add(Dropout(1-keep_prob_dim[0], seed=2))         
        model.add(Dense(units=node_dim[0], activation='relu', kernel_regularizer=regularizer_l1, activity_regularizer=regularizer_l2))
        if keep_prob_dim[0] != 1.0:
          model.add(Dropout(1-keep_prob_dim[0], seed=2))         

        model.add(Dense(1, activation='sigmoid'))
        print('[DNN Model] model layer & node design complete')  

#------------------------------------------------------------------------------
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate_dim[0]), loss='binary_crossentropy', metrics=['accuracy'])
#------------------------------------------------------------------------------
        early_stopping = EarlyStopping(patience = stop_patience_dim[0])
        hist = model.fit(X_train_scaled, Y_train, epochs=epoch_dim[k], verbose=1, batch_size=batch_size_dim[0], validation_data=(X_val_scaled, Y_val), callbacks=[early_stopping])  ## verbose : 훈련과정 보이기                         
        print('[DNN Model] model fitting complete')
#------------------------------------------------------------------------------
# 모델 평가하기
        loss_and_metrics = model.evaluate(X_val_scaled, Y_val)
        print('')
        print('loss : '+str(loss_and_metrics[0]))
        print('accuracy : '+str(loss_and_metrics[1]))

# Text
        loss_val = "{:.4f}".format(loss_and_metrics[0])
        acc_val = "{:.4f}".format(loss_and_metrics[1])
        f.write(name+"  loss:"+str(loss_val)+"  accuracy:"+str(acc_val)+"\n")

  f.close()

print("All complete")
