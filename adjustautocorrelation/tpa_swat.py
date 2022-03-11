import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.autograd import Variable
from torch.utils.data import DataLoader

from TPA import TPA
from eval_methods import *
from utils import sliding_windows, quantile_loss, get_device
from SwatDataset import SwatDataset

# %matplotlib inline

torch.set_default_tensor_type(torch.DoubleTensor)


def train_swat(seq_length: int = 60, nrows: int = 100):
    training_set = pd.read_csv('../data/swat/SWaT_Dataset_Normal_v1.csv', nrows=nrows, index_col="Timestamp")
    training_set = training_set.drop(["Normal/Attack"], axis=1)
    for i in list(training_set):
        training_set[i] = training_set[i].apply(lambda x: str(x).replace(",", "."))
    training_set = training_set.astype(float)
    sc = MinMaxScaler()
    training_data = sc.fit_transform(training_set)

    # 超参
    num_epochs = 10
    learning_rate = 0.01
    input_size = training_set.shape[1]
    hidden_size = 64
    num_layers = 1
    ar_len = 24

    model = TPA(seq_length, hidden_size, num_layers, ar_len, input_size)
    dataset = SwatDataset(training_data, seq_length)
    dataLoader = DataLoader(dataset=dataset, batch_size=1000, num_workers=0)
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
            loss_low = torch.sum(quantile_loss(0.01, dataY, output_low), dim=0)
            loss_high = torch.sum(quantile_loss(0.99, dataY, output_high), dim=0)
            loss = loss_low + loss_high
            loss_sum += loss.item()
            loss.backward()
            optimizer.step()
        # if epoch % 100 == 0:
        print("Epoch: %d, loss: %1.5f" % (epoch, loss_sum))
    torch.save(model.state_dict(), 'model/swat_tpa')

    # 可视化结果
    model.eval()
    dataLoader = DataLoader(dataset=dataset, batch_size=100000, num_workers=0)
    for i, (dataX, dataY) in enumerate(dataLoader):
        # x, y = sliding_windows(training_data, seq_length)
        # dataX = Variable(torch.Tensor(np.array(x)))
        # dataY = Variable(torch.Tensor(np.array(y)))
        train_predict_low, train_predict_high = model(dataX)

        data_predict_low = train_predict_low.data.cpu().numpy()
        data_predict_high = train_predict_high.data.cpu().numpy()
        dataY_plot = dataY.data.cpu().numpy()

        data_predict_low = sc.inverse_transform(data_predict_low)
        data_predict_high = sc.inverse_transform(data_predict_high)
        dataY_plot = sc.inverse_transform(dataY_plot)
        for col in range(training_set.shape[1]):
            plt.plot(data_predict_high[:, col], color='blue', label='high quantile')
            plt.plot(dataY_plot[:, col], color='green', label='origin')
            plt.plot(data_predict_low[:, col], color='red', label='low quantile')
            plt.suptitle('Time-Series Prediction Train, column name: {}'.format(col))
            plt.legend()
            plt.show()


def get_labels(seq_length: int = 4, nrows: int = 1000):
    testing_set = pd.read_csv('../data/swat/SWaT_Dataset_Attack_v0.csv', usecols=['Normal/Attack'], sep=';',
                              nrows=nrows)
    labels = [float(label != 'Normal') for label in testing_set["Normal/Attack"].values]
    _, l = sliding_windows(labels, seq_length=seq_length)
    l = l.astype(np.int32)
    return l


def test_swat(seq_length: int = 4, nrows: int = 1000):
    l = get_labels(seq_length=seq_length + 1, nrows=nrows)
    testing_set = pd.read_csv('../data/swat/SWaT_Dataset_Attack_v0.csv', sep=';', nrows=nrows, index_col="Timestamp")
    testing_set = testing_set.drop(["Normal/Attack"], axis=1)
    for i in list(testing_set):
        testing_set[i] = testing_set[i].apply(lambda x: str(x).replace(",", "."))
    testing_set = testing_set.astype(float)
    testing_set = testing_set.values
    sc = MinMaxScaler()
    testing_data = sc.fit_transform(testing_set)

    dataset = SwatDataset(testing_data, seq_length)
    dataLoader = DataLoader(dataset=dataset, batch_size=10000, num_workers=0)
    input_size = testing_set.shape[1]
    hidden_size = 64
    num_layers = 1
    ar_len = 24

    model = TPA(seq_length, hidden_size, num_layers, ar_len, input_size)
    if torch.cuda.is_available():
        model.load_state_dict(torch.load('model/swat_tpa'))
    else:
        model.load_state_dict(torch.load('model/swat_tpa', map_location=torch.device('cpu')))
    model.eval()
    device = get_device()
    model = model.to(device)
    for _, (dataX, dataY) in enumerate(dataLoader):
        dataX = dataX.to(device)
        dataY = dataY.to(device)
        test_predict_low, test_predict_high = model(dataX)

        data_predict_low = test_predict_low.data.cpu().numpy()
        data_predict_high = test_predict_high.data.cpu().numpy()
        dataY_plot = dataY.data.cpu().numpy()
        a = dataX.shape[2]
        print(a)
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
        for i in range(len(l)):
            if final_res[i] == l[i]:
                same += 1
        print('按每个维度投票之后的准确率值: ' + str(same / len(l)))
        t, th = bf_search(np.mean(abnormal, axis=1), l, start=0., end=0.9, step_num=int((0.9 - 0.) / 0.001),
                          display_freq=100)


if __name__ == '__main__':
    train_swat(seq_length=60, nrows=2000)
    # test_swat(seq_length=60, nrows=1000)
