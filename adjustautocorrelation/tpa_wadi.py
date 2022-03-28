import time
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader

from SwatDataset import SwatDataset
from TPA import TPA
from eval_methods import *
from utils import sliding_windows, quantile_loss, get_device

# %matplotlib inline
torch.set_default_tensor_type(torch.DoubleTensor)

def train_wadi(seq_length: int = 60, nrows: int = 100):
    training_set = pd.read_csv('../data/wadi/WADI_14days.csv', skiprows=4, nrows=nrows, index_col=0)
    training_set = training_set.dropna(axis=1, how='all')
    training_set = training_set.drop(['Date', 'Time'], axis=1)
    # for i in list(training_set):
    #     training_set[i] = training_set[i].fillna(0.0, inplace=True)
    training_set = training_set.astype(float)
    sc = MinMaxScaler()
    training_data = sc.fit_transform(training_set)

    # 超参
    num_epochs = 200
    learning_rate = 0.01
    input_size = training_set.shape[1]
    hidden_size = 64
    num_layers = 1
    ar_len = 2

    model = TPA(seq_length, hidden_size, num_layers, ar_len, input_size)
    dataset = SwatDataset(training_data, seq_length)
    dataLoader = DataLoader(dataset=dataset, batch_size=30000, num_workers=0)
    # 将模型转移到指定设备上
    device = get_device()
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # Train the model
    for epoch in range(num_epochs):
        loss_sum = 0
        for i, (dataX, dataY) in enumerate(dataLoader):
            dataX = dataX.to(device)
            dataY = dataY.to(device)
            output_low, output_high = model(dataX)
            # output = model(dataX)
            optimizer.zero_grad()

            # obtain the loss function
            loss_low = torch.sum(quantile_loss(0.05, dataY, output_low), dim=0)
            loss_high = torch.sum(quantile_loss(0.95, dataY, output_high), dim=0)
            loss = loss_low + loss_high
            loss_sum += loss.item()
            loss.backward()
            optimizer.step()
        # if epoch % 100 == 0:
        print("Epoch: %d, loss: %1.5f" % (epoch, loss_sum))
    torch.save(model.state_dict(), 'model/wadi_tpa')

    # # 可视化结果
    # model.eval()
    # model.load_state_dict(torch.load('model/swat_tpa'))
    # dataLoader = DataLoader(dataset=dataset, batch_size=training_data.shape[0], num_workers=0)
    # for i, (dataX, dataY) in enumerate(dataLoader):
    #     # x, y = sliding_windows(training_data, seq_length)
    #     # dataX = Variable(torch.Tensor(np.array(x)))
    #     # dataY = Variable(torch.Tensor(np.array(y)))
    #     dataX = dataX.to(device)
    #     train_predict_low, train_predict_high = model(dataX)

    #     data_predict_low = train_predict_low.data.cpu().numpy()
    #     data_predict_high = train_predict_high.data.cpu().numpy()
    #     dataY_plot = dataY.data.cpu().numpy()

    #     data_predict_low = sc.inverse_transform(data_predict_low)
    #     data_predict_high = sc.inverse_transform(data_predict_high)
    #     dataY_plot = sc.inverse_transform(dataY_plot)
    #     for col in range(training_set.shape[1]):
    #         plt.plot(data_predict_high[:, col], color='blue', label='high quantile')
    #         plt.plot(dataY_plot[:, col], color='green', label='origin')
    #         plt.plot(data_predict_low[:, col], color='red', label='low quantile')
    #         plt.suptitle('Time-Series Prediction Train, column name: {}'.format(col))
    #         plt.legend()
    #         plt.show()


def get_labels(seq_length: int = 4, nrows: int = 1000):
    testing_set = pd.read_csv('../data/wadi/WADI_attackdata.csv', usecols=['Date', 'Time'], nrows=nrows)
    testing_set['Timestamp'] = testing_set['Date'] + " " + testing_set['Time']
    testing_set["Timestamp"] = pd.to_datetime(testing_set["Timestamp"], format="%m/%d/%Y %I:%M:%S.000 %p")
    testing_set["unix"] = testing_set["Timestamp"].astype(np.int64)
    abnormal_range = [['9/10/17 19:25:00', '9/10/17 19:50:16'], ['10/10/17 10:24:10', '10/10/17 10:34:00'],
                      ['10/10/17 10:55:00', '10/10/17 11:24:00'], ['10/10/17 11:30:40', '10/10/17 11:44:50'],
                      ['10/10/17 13:39:30', '10/10/17 13:50:40'], ['10/10/17 14:48:17', '10/10/17 14:59:55'],
                      ['10/10/17 17:40:00', '10/10/17 17:49:40'], ['10/10/17 10:55:00', '10/10/17 10:56:27'],
                      ['11/10/17 11:17:54', '11/10/17 11:31:20'], ['11/10/17 11:36:31', '11/10/17 11:47:00'],
                      ['11/10/17 11:59:00', '11/10/17 12:05:00'], ['11/10/17 12:07:30', '11/10/17 12:10:52'],
                      ['11/10/17 12:16:00', '11/10/17 12:25:36'], ['11/10/17 15:26:30', '11/10/17 15:37:00']]
    labels = np.zeros(testing_set.shape[0])
    for i in range(len(abnormal_range)):
        start = datetime.strptime(abnormal_range[i][0], "%d/%m/%y %H:%M:%S")
        # 手动设置时区为东八区
        start_timestamp = (int(time.mktime(start.timetuple())) + (8 * 60 * 60)) * (10 ** 9)

        end = datetime.strptime(abnormal_range[i][1], "%d/%m/%y %H:%M:%S")
        end_timestamp = (int(time.mktime(end.timetuple())) + (8 * 60 * 60)) * (10 ** 9)

        abnormal = testing_set[(testing_set['unix'] >= start_timestamp) & (testing_set['unix'] <= end_timestamp)]
        abnormal_idx = abnormal.index
        labels[abnormal_idx] = 1
    # 通过滑动窗口的y对应的label值
    _, sliding_labels = sliding_windows(labels, seq_length=seq_length)
    sliding_labels = sliding_labels.astype(np.int32)
    return sliding_labels

def cat(mlist):
  for i in range(len(mlist) - 1):
    if i == 0:
        c = np.concatenate((mlist[i], mlist[i + 1]), axis=0)
    else:
        c = np.concatenate((c, mlist[i + 1]), axis=0)
  return c

def test_wadi(seq_length: int = 4, nrows: int = 1000):
    l = get_labels(seq_length=seq_length + 1, nrows=nrows)
    testing_set = pd.read_csv('../data/wadi/WADI_attackdata.csv', nrows=nrows, index_col=0)
    testing_set = testing_set.dropna(axis=1, how='all')
    testing_set = testing_set.drop(['Date', 'Time'], axis=1)
    for i in list(testing_set):
        testing_set[i].fillna(0, inplace=True)
    testing_set = testing_set.astype(float)
    testing_set = testing_set.values
    sc = MinMaxScaler()
    testing_data = sc.fit_transform(testing_set)

    dataset = SwatDataset(testing_data, seq_length)
    dataLoader = DataLoader(dataset=dataset, batch_size=5000, num_workers=0)
    input_size = testing_set.shape[1]
    hidden_size = 168
    num_layers = 1
    ar_len = 2

    model = TPA(seq_length, hidden_size, num_layers, ar_len, input_size)
    if torch.cuda.is_available():
        model.load_state_dict(torch.load('model/wadi_tpa'))
    else:
      model.load_state_dict(torch.load('model/wadi_tpa', map_location=torch.device('cpu')))
    model.eval()
    device = get_device()
    model = model.to(device)
    total_x = []
    total_y = []
    total_low = []
    total_high = []
    for _, (dataX, dataY) in enumerate(dataLoader):
        dataX = dataX.to(device)
        dataY = dataY.to(device)
        test_predict_low, test_predict_high = model(dataX)

        data_predict_low = test_predict_low.data.cpu().numpy()
        data_predict_high = test_predict_high.data.cpu().numpy()
        dataY_plot = dataY.data.cpu().numpy()
        total_y.append(dataY_plot)
        total_low.append(data_predict_low)
        total_high.append(data_predict_high)
    total_y = np.array(total_y)
    total_low = np.array(total_low)
    total_high = np.array(total_high)
    dataY_plot = cat(total_y)
    data_predict_low = cat(total_low)
    data_predict_high = cat(total_high)
    for i in range(dataX.shape[2]):
        if 15 <= i <= 35:
            plt.plot(data_predict_high[:, i], color='blue', label='high quantile')
            plt.plot(dataY_plot[:, i], color='green', label='origin')
            plt.plot(data_predict_low[:, i], color='red', label='low quantile')
            # plt.suptitle('Time-Series Prediction Test, column name: {}'.format(i))
            plt.legend()
            # plt.savefig('saved_fig/swat/swat_pred{}'.format(idx))
            plt.show()

    abnormal = np.where((dataY_plot < data_predict_low) | (dataY_plot > data_predict_high), 1, 0)
    final_res = np.mean(abnormal, axis=1)  # 每个样本获取到的异常分数
    final_res = np.where(final_res > 0.5, 1, 0)
    same = 0
    print(len(l))
    print(len(final_res))
    for i in range(len(l)):
        if final_res[i] == l[i]:
            same += 1
    print('按每个维度投票之后的准确率值: ' + str(same / len(l)))
    t, th = bf_search(np.mean(abnormal, axis=1), l, start=0., end=0.9, step_num=int((0.9 - 0.) / 0.001),
                      display_freq=100)


if __name__ == '__main__':
  # pass
  #   train_wadi(seq_length=4, nrows=1000)
    test_wadi(seq_length=4, nrows=None)
