import numpy as np
import pandas as pd
import torch.cuda

# from psm_main import *
# test_psm(nrows=87841)
# from swat_main import *
# test_swat(nrows=449919)

# swat test 449919 train 495000
# 87841 test
# df = pd.read_csv('data/psm/test.csv')
# shape = df.shape
# pass
# a = torch.cuda.is_available()


# from smd_main import *
# test_smd_for_all_entity(nrows=None)

# train_smd(nrows=1000)
# import shutil
#
# shutil.make_archive('D:\\压缩结果', 'zip', 'D:/learn/Q-LSTM/trained_model')

# import matplotlib.pyplot as plt
# # smap_msl數據集
# import numpy as np
# a_1_train = np.load('data/smap_msl/data/train/P-1.npy')
# a_1_test = np.load('data/smap_msl/data/test/P-1.npy')
# plt.plot(a_1_test[:, 0])
# plt.show()

import os

# df = pd.read_csv('data/smap_msl/labeled_anomalies.csv')
# smaps_pos = df['spacecraft'] == 'SMAP'
# smaps = df[smaps_pos]
# dirs = [i.split('.')[0] for i in smaps['chan_id'].values]
# train_set = np.load('data/smap_msl/data/train/' + dirs[0] + '.npy')
# test_set = np.load('data/smap_msl/data/test/' + dirs[0] + '.npy')
#
# col_num = train_set.shape[1]
# a = train_set[:, 0]
# from sklearn.preprocessing import MinMaxScaler
# a = a.reshape(-1, 1)
# a = MinMaxScaler().fit_transform(a)
# df = pd.read_csv('data/smap_msl/labeled_anomalies.csv')
# channel_info = df[df['chan_id'] == 'S-1']
# # 初始化数组，设置异常位置值为1
# labels = np.zeros(channel_info['num_values'])
# anomaly_loc = channel_info['anomaly_sequences']
# a = anomaly_loc.values[0]
# b = str(a[0])
# c = eval(b)
# a = [[1, 2], [2, 3], [3, 4], [4, 5]]
# b = str(a)
# c = eval(b)
# from smap_main import train_smap, test_smap_for_all_channel

# train_smap()
# test_smap_for_all_channel()

from msl_main import train_msl, test_msl_for_all_channel

# train_msl()
test_msl_for_all_channel()


pass
