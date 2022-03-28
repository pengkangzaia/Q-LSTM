import time
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.utils.data as data_utils
from sklearn.preprocessing import MinMaxScaler
from torch.autograd import Variable

from Q_LSTM import LSTM
from eval_methods import *
from utils import sliding_windows, quantile_loss, get_device


def get_columns():
    df = pd.read_csv('data/wadi/WADI_14days.csv', skiprows=4, index_col=0)
    df = df.dropna(axis=1, how='all')
    df = df.drop(['Date', 'Time'], axis=1)
    col_names_res = []
    for i in df.columns:
        col_names_res.append(i.split('\\')[-1])
    return col_names_res


def train_wadi(start_col: int, end_col: int, seq_length: int = 4, nrows: int = 100):
    wadi_columns = get_columns()
    # for idx in range(len(wadi_columns) - 1, -1, -1):
    # for idx in range(len(wadi_columns)):
    for idx in range(start_col, end_col + 1, 1):
        col = wadi_columns[idx]
        training_set = pd.read_csv('data/wadi/WADI_14days.csv', skiprows=4, index_col=0, nrows=nrows)
        training_set = training_set.dropna(axis=1, how='all')
        training_set = training_set.drop(['Date', 'Time'], axis=1)
        training_set.columns = wadi_columns
        training_set = training_set[col]
        training_set.fillna(0, inplace=True)
        training_set = training_set.astype(float)
        training_set = training_set.values.reshape(-1, 1)
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
        torch.save(lstm.state_dict(), 'trained_model/wadi/saved_model{}'.format(idx))

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
        plt.suptitle('WADI Time-Series Prediction Train, column name: {}'.format(col))
        plt.legend()
        plt.show()


def get_labels(seq_length: int = 4, nrows: int = 1000):
    testing_set = pd.read_csv('data/wadi/WADI_attackdata.csv', usecols=['Date', 'Time'], nrows=nrows)
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


def test_wadi(seq_length: int = 4, nrows: int = 1000):
    abnormal_rates = []
    abnormals = []
    wadi_columns = get_columns()
    sliding_labels = get_labels(seq_length=seq_length, nrows=nrows)
    for idx in range(len(wadi_columns)):
        col = wadi_columns[idx]
        testing_set = pd.read_csv('data/wadi/WADI_attackdata.csv', index_col=0, nrows=nrows)
        testing_set = testing_set.dropna(axis=1, how='all')
        testing_set = testing_set.drop(['Date', 'Time'], axis=1)
        testing_set.columns = wadi_columns
        testing_set = testing_set[col]
        testing_set.fillna(0, inplace=True)
        testing_set = testing_set.astype(float)
        testing_set = testing_set.values.reshape(-1, 1)
        sc = MinMaxScaler()
        testing_data = sc.fit_transform(testing_set)

        x, y = sliding_windows(testing_data, seq_length)
        dataX = Variable(torch.Tensor(np.array(x)))
        dataY = Variable(torch.Tensor(np.array(y)))
        test_size = len(y)

        # 参数
        input_size = 1
        hidden_size = 2
        num_layers = 1
        num_classes = 1
        model = LSTM(num_classes, input_size, hidden_size, seq_length, num_layers)
        if torch.cuda.is_available():
            model.load_state_dict(torch.load('trained_model/wadi/saved_model{}'.format(idx)))
        else:
            model.load_state_dict(
                torch.load('trained_model/wadi/saved_model{}'.format(idx), map_location=torch.device('cpu')))
        model.eval()
        device = get_device()
        model = model.to(device)
        dataX = dataX.to(device)
        dataY = dataY.to(device)
        test_predict_low, test_predict_high = model(dataX)

        data_predict_low = test_predict_low.data.cpu().numpy()
        data_predict_high = test_predict_high.data.cpu().numpy()
        dataY_plot = dataY.data.cpu().numpy()
        plt.plot(data_predict_high, color='blue', label='high quantile')
        plt.plot(dataY_plot, color='green', label='origin')
        plt.plot(data_predict_low, color='red', label='low quantile')
        # plt.suptitle('WADI Time-Series Prediction Test, column name: {}'.format(col))
        # plt.suptitle('Q-LSTM Prediction, column name: {}'.format(col))
        plt.legend()
        plt.savefig('saved_fig/wadi/wadi_pred{}'.format(idx))
        plt.show()

        abnormal = np.where((dataY_plot < data_predict_low) | (dataY_plot > data_predict_high), 1, 0)
        abnormal = np.squeeze(abnormal, axis=1)
        same = 0
        for i in range(test_size):
            if abnormal[i] == sliding_labels[i]:
                same += 1
        abnormal_rates.append(same / test_size)
        abnormals.append(abnormal)
    print('预测正确率:' + str(abnormal_rates))
    avg_rate = np.array(abnormal_rates).mean()
    print('预测平均正确率: ' + str(avg_rate))

    final_res = np.mean(abnormals, axis=0)
    final_res = np.where(final_res > 0.5, 1, 0)
    same = 0
    for i in range(len(sliding_labels)):
        if final_res[i] == sliding_labels[i]:
            same += 1
    print('按每个维度投票之后的准确率值: ' + str(same / len(sliding_labels)))
    t, th = bf_search(np.mean(abnormals, axis=0), sliding_labels, start=0., end=0.9, step_num=int((0.9 - 0.) / 0.001),
                      display_freq=100)
