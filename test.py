import time

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

# from msl_main import train_msl, test_msl_for_all_channel
#
# # train_msl()
# test_msl_for_all_channel()


# df1 = pd.read_csv('data/wadi/WADI_attackdata.csv', usecols=['Date', 'Time'])
# df1 = pd.read_csv('data/wadi/WADI_14days.csv', skiprows=4, index_col=0)
# a = np.where(np.isnan(df1))
pass
# df1['Timestamp'] = df1['Date'] + " " + df1['Time']
# df1["Timestamp"] = pd.to_datetime(df1["Timestamp"], format="%m/%d/%Y %I:%M:%S.000 %p")
# df1["unix"] = df1["Timestamp"].astype(np.int64)
# df = df.dropna(axis=1, how='all')
# df = df.drop(['Date', 'Time'], axis=1)
# col_names = df.columns
# res = []
# for i in col_names:
#     res.append(i)


# s = '\\WIN-25J4RO10SBF\LOG_DATA\SUTD_WADI\LOG_DATA\LEAK_DIFF_PRESSURE'
# 9/25/20176:00:00.000 PM
# 9/25/2017 6:00:00.000 PM
# arr = s.split("\\")
# for i in res:
#     print(i)
# 1506362400000000000
# 1506362401000000000
# 1507549816
# 1507549816000000000

# from datetime import datetime, timedelta, timezone
#
# res = [['9/10/17 19:25:00', '9/10/17 19:50:16'], ['10/10/17 10:24:10', '10/10/17 10:34:00'],
#        ['10/10/17 10:55:00', '10/10/17 11:24:00'], ['10/10/17 11:30:40', '10/10/17 11:44:50'],
#        ['10/10/17 13:39:30', '10/10/17 13:50:40'], ['10/10/17 14:48:17', '10/10/17 14:59:55'],
#        ['10/10/17 17:40:00', '10/10/17 17:49:40'], ['10/10/17 10:55:00', '10/10/17 10:56:27'],
#        ['11/10/17 11:17:54', '11/10/17 11:31:20'], ['11/10/17 11:36:31', '11/10/17 11:47:00'],
#        ['11/10/17 11:59:00', '11/10/17 12:05:00'], ['11/10/17 12:07:30', '11/10/17 12:10:52'],
#        ['11/10/17 12:16:00', '11/10/17 12:25:36'], ['11/10/17 15:26:30', '11/10/17 15:37:00']]
# labels = np.zeros(df1.shape[0])
# for i in range(len(res)):
#     tz_utc_8 = timezone(timedelta(hours=8))
#
#     start = datetime.strptime(res[i][0], "%d/%m/%y %H:%M:%S")
#     # 设置时区
#     start_timestamp = (int(time.mktime(start.timetuple())) + (8 * 60 * 60)) * (10 ** 9)
#
#     end = datetime.strptime(res[i][1], "%d/%m/%y %H:%M:%S")
#     end_timestamp = (int(time.mktime(end.timetuple())) + (8 * 60 * 60)) * (10 ** 9)
#
#     abnormal = df1[(df1['unix'] >= start_timestamp) & (df1['unix'] <= end_timestamp)]
#     abnormal_idx = abnormal.index
#     labels[abnormal_idx] = 1
#
# res = 0
# for i in labels:
#     if i == 1:
#         res += 1
# print(res)

# d = datetime.strptime(res[0][1], "%d/%m/%y %H:%M:%S")
# a = d.timetuple()
# b = time.mktime(a)
# c = int(b)
# d = c * (10 ** 9)
# from wadi_main import test_wadi, train_wadi
#
# # train_wadi(nrows=None)
# test_wadi(nrows=None)


# from smap_main import test_smap_for_all_channel
#
# test_smap_for_all_channel()

# from msl_main import test_msl_for_all_channel
#
# test_msl_for_all_channel()

# array = [[0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6], [5, 6, 7], [6, 7, 8], [7, 8, 9], [8, 9, 10], [9, 10, 11]]
# myarray = np.array(array)


from sklearn.preprocessing import MinMaxScaler

ip = '127.0.0.1'
ip = ip.replace(".", "_")
print(ip)
import os

folder = os.path.exists('web/trained_model/' + ip)
if not folder:
    os.makedirs('web/trained_model/' + ip)
# influx读取数据
import influxdb_client

bucket = "monitor"
org = "seu"
token = "gZTu3-P2pKcGQI-wBgHUT1nRIckb7N_drF-r9YKUdbszy1hTrN3BwIR5CdFHshzGcW81n_SbjfI5-RQsUz11zA=="
url = "http://101.35.159.221:8086"
client = influxdb_client.InfluxDBClient(url=url, token=token, org=org)
query_api = client.query_api()
flux = 'from(bucket: "monitor")|> range(start: -1h)|> filter(fn: (r) => r["_measurement"] == "cpu2" or r[' \
       '"_measurement"] == "disk" or r["_measurement"] == "memory" or r["_measurement"] == "net")|> filter(fn: (r) => ' \
       'r["address"] == "http://1.15.117.64:8081")|> drop(columns: ["result", "address", "_measurement", "_start", ' \
       '"_stop"])|> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value") '
# 3天数据量大概要半分钟
df_result = query_api.query_data_frame(flux)
df_result.to_csv('monitor.csv', index=False)
df = pd.read_csv('monitor.csv')
df = df.drop(['result', 'table'], axis=1)
df["_time"] = pd.to_datetime(df['_time'])
# df["_time"] = df["_time"].astype('int64')
df.index = df['_time']
df = df.resample('30S').mean()
df = df.interpolate(method='linear')
print(df.columns)

import torch
import pandas as pd
import matplotlib.pyplot as plt
from Q_LSTM import LSTM
from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler
from utils import sliding_windows, quantile_loss, get_device
from eval_methods import *

seq_length = 4
cols = df.columns
for idx in range(len(cols)):
    col = cols[idx]
    training_set = df[col].to_frame()
    training_set = training_set.iloc[:, 0:1].values
    sc = MinMaxScaler()
    training_data = sc.fit_transform(training_set)

    x, y = sliding_windows(training_data, seq_length)
    dataX = Variable(torch.Tensor(np.array(x)))
    dataY = Variable(torch.Tensor(np.array(y)))

    # 超参
    num_epochs = 2000
    learning_rate = 0.01
    input_size = 1
    hidden_size = 2
    num_layers = 1
    num_classes = 1

    lstm = LSTM(num_classes, input_size, hidden_size, seq_length, num_layers)
    # 将模型转移到指定设备上
    device = get_device()
    lstm = lstm.to(device)
    dataX = dataX.to(device)
    dataY = dataY.to(device)
    optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate)
    # Train the model
    for epoch in range(num_epochs):
        output_low, output_high = lstm(dataX)
        optimizer.zero_grad()

        # obtain the loss function
        loss_low = torch.sum(quantile_loss(0.01, dataY, output_low), dim=0)
        loss_high = torch.sum(quantile_loss(0.99, dataY, output_high), dim=0)
        loss = loss_low + loss_high
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))
    torch.save(lstm.state_dict(), 'web/trained_model/' + ip + '/model_{}'.format(col))

    # 可视化结果
    lstm.eval()
    train_predict_low, train_predict_high = lstm(dataX)

    data_predict_low = train_predict_low.data.cpu().numpy()
    data_predict_high = train_predict_high.data.cpu().numpy()
    dataY_plot = dataY.data.cpu().numpy()

    data_predict_low = sc.inverse_transform(data_predict_low)
    data_predict_high = sc.inverse_transform(data_predict_high)
    dataY_plot = sc.inverse_transform(dataY_plot)

    plt.plot(data_predict_high, color='blue', label='high quantile')
    plt.plot(dataY_plot, color='green', label='origin')
    plt.plot(data_predict_low, color='red', label='low quantile')
    plt.suptitle('Time-Series Prediction Train, column name: {}'.format(col))
    plt.legend()
    plt.show()

pass
