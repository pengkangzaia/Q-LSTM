import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader

from SwatDataset import SwatDataset
from TPA import TPA
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
    torch.save(model.state_dict(), 'model/psm_tpa')


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

    model = TPA(seq_length, hidden_size, num_layers, ar_len, input_size)
    if torch.cuda.is_available():
        model.load_state_dict(torch.load('model/psm_tpa'))
    else:
        model.load_state_dict(torch.load('model/psm_tpa', map_location=torch.device('cpu')))
    model.eval()
    device = get_device()
    model = model.to(device)
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
        plt.plot(data_predict_high[:, i], color='blue', label='high quantile')
        plt.plot(dataY_plot[:, i], color='green', label='origin')
        plt.plot(data_predict_low[:, i], color='red', label='low quantile')
        plt.suptitle('Time-Series Prediction Test, column name: {}'.format(i))
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
    train_psm(4, None)
    test_psm(4, 100)
