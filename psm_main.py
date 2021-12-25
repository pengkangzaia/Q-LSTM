import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Q_LSTM import LSTM
from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler
from utils import sliding_windows, quantile_loss, get_device

str_cols = 'feature_0,feature_1,feature_2,feature_3,feature_4,feature_5,feature_6,feature_7,feature_8,feature_9,' \
           'feature_10,feature_11,feature_12,feature_13,feature_14,feature_15,feature_16,feature_17,feature_18,' \
           'feature_19,feature_20,feature_21,feature_22,feature_23,feature_24'
psm_columns = str_cols.split(',')


def train_psm(seq_length: int = 4, nrows: int = 1000):
    for idx in range(len(psm_columns)):
        col = psm_columns[idx]
        training_set = pd.read_csv('data/psm/train.csv', usecols=[col], nrows=nrows)
        training_set[col].fillna(0, inplace=True)  # imputes missing values
        training_set = training_set.astype(float)
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
            loss_low = torch.sum(quantile_loss(0.05, dataY, output_low), dim=0)
            loss_high = torch.sum(quantile_loss(0.95, dataY, output_high), dim=0)
            loss = loss_low + loss_high
            loss.backward()
            optimizer.step()
            if epoch % 100 == 0:
                print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))
        torch.save(lstm.state_dict(), 'trained_model/psm/saved_model{}'.format(idx))

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
        plt.suptitle('PSM Training Abnormal Detection for column: {}'.format(col))
        plt.legend()
        plt.show()


def get_labels(seq_length: int = 4, nrows: int = 1000):
    testing_set = pd.read_csv('data/psm/test_label.csv', nrows=nrows)
    labels = testing_set['label'].values
    _, slided_labels = sliding_windows(labels, seq_length=seq_length)
    slided_labels = slided_labels.astype(np.int32)
    return slided_labels


def test_psm(seq_length: int = 4, nrows: int = 1000):
    abnormal_rates = []
    abnormals = []
    slided_labels = get_labels(seq_length=seq_length, nrows=nrows)
    for idx in range(len(psm_columns)):
        col = psm_columns[idx]
        testing_set = pd.read_csv('data/psm/test.csv', usecols=[col], sep=',', nrows=nrows)
        testing_set = testing_set.iloc[:, 0:1]
        testing_set = testing_set.astype(float)
        testing_set = testing_set.values
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
            model.load_state_dict(torch.load('trained_model/psm/saved_model{}'.format(idx)))
        else:
            model.load_state_dict(torch.load('trained_model/psm/saved_model{}'.format(idx), map_location=torch.device('cpu')))
        # 将模型转移到指定设备上
        device = get_device()
        print(device)
        model = model.to(device)
        dataX = dataX.to(device)
        print(dataX.device)
        dataY = dataY.to(device)
        model.eval()
        test_predict_low, test_predict_high = model(dataX)

        data_predict_low = test_predict_low.data.cpu().numpy()
        data_predict_high = test_predict_high.data.cpu().numpy()
        dataY_plot = dataY.data.cpu().numpy()
        plt.plot(data_predict_high, color='blue', label='high quantile')
        plt.plot(dataY_plot, color='green', label='origin')
        plt.plot(data_predict_low, color='red', label='low quantile')
        plt.suptitle('PSM Test Abnormal Detection for column: {}'.format(col))
        plt.legend()
        plt.savefig('saved_fig/psm/psm_pred{}'.format(idx))
        plt.show()

        abnormal = np.where((dataY_plot < data_predict_low) | (dataY_plot > data_predict_high), 1, 0)
        abnormal = np.squeeze(abnormal, axis=1)
        same = 0
        for i in range(test_size):
            if abnormal[i] == slided_labels[i]:
                same += 1
        abnormal_rates.append(same / test_size)
        abnormals.append(abnormal)
    print('预测正确率:' + str(abnormal_rates))
    avg_rate = np.array(abnormal_rates).mean()
    print('预测平均正确率: ' + str(avg_rate))

    final_res = np.mean(abnormals, axis=0)
    final_res = np.where(final_res > 0.5, 1, 0)
    same = 0
    for i in range(len(slided_labels)):
        if final_res[i] == slided_labels[i]:
            same += 1
    print('按每个维度投票之后的准确率值: ' + str(same / len(slided_labels)))
