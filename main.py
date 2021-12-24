import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Q_LSTM import LSTM
from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler
from utils import sliding_windows, quantile_loss

str_cols = 'FIT101;LIT101;MV101;P101;P102;AIT201;AIT202;AIT203;FIT201;MV201;P201;P202;P203;P204;P205;P206;DPIT301' \
           ';FIT301;LIT301;MV301;MV302;MV303;MV304;P301;P302;AIT401;AIT402;FIT401;LIT401;P401;P402;P403;P404;UV401' \
           ';AIT501;AIT502;AIT503;AIT504;FIT501;FIT502;FIT503;FIT504;P501;P502;PIT501;PIT502;PIT503;FIT601;P601;P602' \
           ';P603'
swat_columns = str_cols.split(';')


def train_swat(seq_length: int = 4, nrows: int = 100):
    for idx in range(len(swat_columns)):
        col = swat_columns[idx]
        training_set = pd.read_csv('SWaT_Dataset_Normal_v1.csv', usecols=[col], nrows=nrows)
        training_set[col] = training_set[col].apply(lambda x: str(x).replace(",", "."))
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
        torch.save(lstm.state_dict(), 'trained_model/saved_model{}'.format(idx))

        # 可视化结果
        lstm.eval()
        train_predict_low, train_predict_high = lstm(dataX)

        data_predict_low = train_predict_low.data.numpy()
        data_predict_high = train_predict_high.data.numpy()
        dataY_plot = dataY.data.numpy()

        data_predict_low = sc.inverse_transform(data_predict_low)
        data_predict_high = sc.inverse_transform(data_predict_high)
        dataY_plot = sc.inverse_transform(dataY_plot)

        plt.plot(data_predict_high, color='blue', label='high quantile')
        plt.plot(dataY_plot, color='green', label='origin')
        plt.plot(data_predict_low, color='red', label='low quantile')
        plt.suptitle('Time-Series Prediction Train, column name: {}'.format(col))
        plt.legend()
        plt.show()


def get_labels(seq_length: int = 4, nrows: int = 1000):
    testing_set = pd.read_csv('SWaT_Dataset_Attack_v0.csv', usecols=['Normal/Attack'], sep=';', nrows=nrows)
    labels = [float(label != 'Normal') for label in testing_set["Normal/Attack"].values]
    _, l = sliding_windows(labels, seq_length=seq_length)
    l = l.astype(np.int32)
    return l


def test_swat(seq_length: int = 4, nrows: int = 1000):
    abnormal_rates = []
    abnormals = []
    l = get_labels(seq_length=seq_length, nrows=nrows)
    for idx in range(len(swat_columns)):
        col = swat_columns[idx]
        testing_set = pd.read_csv('SWaT_Dataset_Attack_v0.csv', usecols=[col, 'Normal/Attack'], sep=';', nrows=nrows)
        testing_set[col] = testing_set[col].apply(lambda x: str(x).replace(",", "."))
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
        model.load_state_dict(torch.load('trained_model/saved_model{}'.format(idx)))
        model.eval()
        test_predict_low, test_predict_high = model(dataX)

        data_predict_low = test_predict_low.data.numpy()
        data_predict_high = test_predict_high.data.numpy()
        dataY_plot = dataY.data.numpy()
        plt.plot(data_predict_high, color='blue', label='high quantile')
        plt.plot(dataY_plot, color='green', label='origin')
        plt.plot(data_predict_low, color='red', label='low quantile')
        plt.suptitle('Time-Series Prediction Test, column name: {}'.format(col))
        plt.legend()
        plt.show()

        abnormal = np.where((dataY_plot < data_predict_low) | (dataY_plot > data_predict_high), 1, 0)
        abnormal = np.squeeze(abnormal, axis=1)
        same = 0
        for i in range(test_size):
            if abnormal[i] == l[i]:
                same += 1
        abnormal_rates.append(same / test_size)
        abnormals.append(abnormal)
    print('预测正确率:' + str(abnormal_rates))
    avg_rate = np.array(abnormal_rates).mean()
    print('预测平均正确率: ' + str(avg_rate))

    final_res = np.mean(abnormals, axis=0)
    final_res = np.where(final_res > 0.5, 1, 0)
    same = 0
    for i in range(len(l)):
        if final_res[i] == l[i]:
            same += 1
    print('按每个维度投票之后的准确率值: ' + str(same / len(l)))
