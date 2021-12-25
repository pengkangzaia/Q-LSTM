import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.autograd import Variable

from Q_LSTM import LSTM
from eval_methods import *
from utils import sliding_windows, quantile_loss, get_device

# 28 files in total
# entity_ids = ["1-1", "1-2", "1-3", "1-4", "1-5", "1-6", "1-7", "1-8",
#               "2-1", "2-2", "2-3", "2-4", "2-5", "2-6", "2-7", "2-8", "2-9",
#               "3-1", "3-2", "3-3", "3-4", "3-5", "3-6", "3-7", "3-8", "3-9", "3-10", "3-11"]
entity_ids = ["1-1", "1-2"]


def train_smd(seq_length: int = 4, nrows: int = 1000):
    for i in range(len(entity_ids)):
        df = pd.read_csv('data/smd/train/machine-' + entity_ids[i] + '.txt', header=None, nrows=nrows)
        col_num = df.shape[1]
        for j in range(col_num):
            training_set = pd.read_csv('data/smd/train/machine-' + entity_ids[i] + '.txt', header=None, nrows=nrows,
                                       usecols=[j])
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
                if epoch % 200 == 0:
                    print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))
            torch.save(lstm.state_dict(), 'trained_model/smd/saved_model_id{}_{}'.format(entity_ids[i], j))

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
            plt.suptitle('SMD Training Abnormal Detection for machine {} {}th column'.format(entity_ids[i], j))
            plt.legend()
            plt.show()


def get_labels(entity_id: str, seq_length: int = 4, nrows: int = 1000):
    testing_set = pd.read_csv('data/smd/test_label/machine-' + entity_id + '.txt', header=None, nrows=nrows)
    labels = testing_set.values
    _, slided_labels = sliding_windows(labels, seq_length=seq_length)
    slided_labels = slided_labels.astype(np.int32)
    return slided_labels


def test_smd_for_all_entity(seq_length: int = 4, nrows: int = 1000):
    preds, accs = [], []
    for entity_id in entity_ids:
        pred, acc = test_psm_for_entity(entity_id, seq_length, nrows)
        preds.append(pred)
        accs.append(acc)
    print("所有entity的最佳F-score平均值为：" + str(np.array(accs).mean()))


# 返回两个值
#   1. 根据上下边界计算的越界预测值
#   2. bf_search搜索出的最佳准确率
def test_psm_for_entity(entity_id: str, seq_length: int = 4, nrows: int = 1000):
    abnormal_rates = []
    abnormals = []
    df = pd.read_csv('data/smd/test/machine-' + entity_id + '.txt', header=None, nrows=nrows)
    col_num = df.shape[1]
    slided_labels = get_labels(entity_id=entity_id, seq_length=seq_length, nrows=nrows)
    for idx in range(col_num):
        testing_set = pd.read_csv('data/smd/test/machine-' + entity_id + '.txt', usecols=[idx], header=None,
                                  nrows=nrows)
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
            model.load_state_dict(torch.load('trained_model/smd/saved_model_id{}_{}'.format(entity_id, idx)))
        else:
            model.load_state_dict(
                torch.load('trained_model/smd/saved_model_id{}_{}'.format(entity_id, idx),
                           map_location=torch.device('cpu')))
        # 将模型转移到指定设备上
        device = get_device()
        model = model.to(device)
        dataX = dataX.to(device)
        dataY = dataY.to(device)
        model.eval()
        test_predict_low, test_predict_high = model(dataX)

        data_predict_low = test_predict_low.data.cpu().numpy()
        data_predict_high = test_predict_high.data.cpu().numpy()
        dataY_plot = dataY.data.cpu().numpy()
        plt.plot(data_predict_high, color='blue', label='high quantile')
        plt.plot(dataY_plot, color='green', label='origin')
        plt.plot(data_predict_low, color='red', label='low quantile')
        plt.suptitle('SMD Test Abnormal Detection machine {} {}th column'.format(entity_id, idx))
        plt.legend()
        plt.savefig('saved_fig/smd/smd_pred{}_{}'.format(entity_id, idx))
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
    t, th = bf_search(np.mean(abnormals, axis=0), slided_labels, start=0., end=0.9, step_num=int((0.9 - 0.) / 0.001),
                      display_freq=100)
    return np.mean(abnormals, axis=0), t[0]
