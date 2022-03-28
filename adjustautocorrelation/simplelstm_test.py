import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader

from SwatDataset import SwatDataset
from SimpleLSTM import LSTM
from eval_methods import *
from utils import sliding_windows, quantile_loss, get_device

torch.set_default_tensor_type(torch.DoubleTensor)
str_cols = 'feature_0,feature_1,feature_2,feature_3,feature_4,feature_5,feature_6,feature_7,feature_8,feature_9,' \
           'feature_10,feature_11,feature_12,feature_13,feature_14,feature_15,feature_16,feature_17,feature_18,' \
           'feature_19,feature_20,feature_21,feature_22,feature_23,feature_24'
psm_columns = str_cols.split(',')


def train_psm(seq_length: int = 4, nrows: int = 1000):
    training_set = pd.read_csv('../data/psm/train.csv', nrows=nrows)
    training_set.fillna(0, inplace=True)  # imputes missing values
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

    model = LSTM(input_size, input_size, hidden_size, seq_length, num_layers)
    # model = TPA(seq_length, hidden_size, num_layers, ar_len, input_size)
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
            output = model(dataX)
            optimizer.zero_grad()

            loss = torch.mean((dataY - output) ** 2)
            # obtain the loss function
            loss_sum += loss.item()
            loss.backward()
            optimizer.step()
        # if epoch % 100 == 0:
        print("Epoch: %d, loss: %1.5f" % (epoch, loss_sum))
    torch.save(model.state_dict(), 'model/psm_lstm')


def cat(mlist):
    if len(mlist) == 1:
        return np.squeeze(mlist)
    for i in range(len(mlist) - 1):
        if i == 0:
            c = np.concatenate((mlist[i], mlist[i + 1]), axis=0)
        else:
            c = np.concatenate((c, mlist[i + 1]), axis=0)
    return c

def get_labels(seq_length: int = 4, nrows: int = 1000):
    testing_set = pd.read_csv('../data/psm/test_label.csv', nrows=nrows)
    labels = testing_set['label'].values
    _, slided_labels = sliding_windows(labels, seq_length=seq_length)
    slided_labels = slided_labels.astype(np.int32)
    return slided_labels


def test_psm(seq_length: int = 4, nrows: int = 1000):
    l = get_labels(seq_length=seq_length + 1, nrows=nrows)
    testing_set = pd.read_csv('../data/psm/test.csv', sep=',', nrows=nrows)
    testing_set = testing_set.astype(float)
    testing_set = testing_set.values
    sc = MinMaxScaler()
    testing_data = sc.fit_transform(testing_set)

    dataset = SwatDataset(testing_data, seq_length)
    dataLoader = DataLoader(dataset=dataset, batch_size=5000, num_workers=0)
    input_size = testing_set.shape[1]
    hidden_size = 64
    num_layers = 1
    ar_len = 2


    model = LSTM(input_size, input_size, hidden_size, seq_length, num_layers)
    # model = TPA(seq_length, hidden_size, num_layers, ar_len, input_size)
    if torch.cuda.is_available():
        model.load_state_dict(torch.load('model/psm_lstm'))
    else:
      model.load_state_dict(torch.load('model/psm_lstm', map_location=torch.device('cpu')))
    model.eval()
    device = get_device()
    model = model.to(device)
    total_y = []
    total_out = []
    for _, (dataX, dataY) in enumerate(dataLoader):
        dataX = dataX.to(device)
        dataY = dataY.to(device)
        test_predict = model(dataX)

        data_predict = test_predict.data.cpu().numpy()
        dataY_plot = dataY.data.cpu().numpy()
        total_y.append(dataY_plot)
        total_out.append(data_predict)
    total_y = np.array(total_y)
    dataY_plot = cat(total_y)
    total_out = np.array(total_out)
    data_predict = cat(total_out)
    for i in range(dataX.shape[2]):
        plt.plot(data_predict[:, i], color='blue', label='predict')
        plt.plot(dataY_plot[:, i], color='green', label='origin')
        plt.suptitle('Time-Series Prediction Test, column name: {}'.format(i))
        plt.legend()
        # plt.savefig('saved_fig/swat/swat_pred{}'.format(idx))
        plt.show()

    abnormal = (dataY_plot - data_predict) ** 2
    final_res = np.mean(abnormal, axis=1)  # 每个样本获取到的异常分数
    t, th = bf_search(np.mean(abnormal, axis=1), l, start=0., end=0.05, step_num=int((0.9 - 0.) / 0.0001),
                      display_freq=100)


test_psm(4, None)
