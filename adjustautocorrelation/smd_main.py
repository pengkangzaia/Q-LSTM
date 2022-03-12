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
# 28 files in total
entity_ids = ["1-1", "1-2", "1-3", "1-4", "1-5", "1-6", "1-7", "1-8",
              "2-1", "2-2", "2-3", "2-4", "2-5", "2-6", "2-7", "2-8", "2-9",
              "3-1", "3-2", "3-3", "3-4", "3-5", "3-6", "3-7", "3-8", "3-9", "3-10", "3-11"]


def cat(mlist):
    if len(mlist) == 1:
        return np.squeeze(mlist)
    for i in range(len(mlist) - 1):
        if i == 0:
            c = np.concatenate((mlist[i], mlist[i + 1]), axis=0)
        else:
            c = np.concatenate((c, mlist[i + 1]), axis=0)
    return c


def train_smd(seq_length: int = 4, nrows: int = 1000):
    for i in range(len(entity_ids)):
        entity_id = entity_ids[i]
        print(entity_ids[i])
        training_set = pd.read_csv('../data/smd/train/machine-' + entity_id + '.txt', header=None, nrows=nrows)
        training_set = training_set.astype(float)
        sc = MinMaxScaler()
        training_data = sc.fit_transform(training_set)

        # 超参
        num_epochs = 200
        learning_rate = 0.01
        input_size = training_set.shape[1]
        hidden_size = 2
        num_layers = 1
        ar_len = 2

        model = TPA(seq_length, hidden_size, num_layers, ar_len, input_size)
        dataset = SwatDataset(training_data, seq_length)
        dataLoader = DataLoader(dataset=dataset, batch_size=20000, num_workers=0)
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
            # print("Epoch: %d, loss: %1.5f" % (epoch, loss_sum))
        torch.save(model.state_dict(), 'model/smd/smd{}_tpa'.format(entity_id))


def get_labels(entity_id: str, seq_length: int = 4, nrows: int = 1000):
    testing_set = pd.read_csv('../data/smd/test_label/machine-' + entity_id + '.txt', header=None, nrows=nrows)
    labels = testing_set.values
    _, slided_labels = sliding_windows(labels, seq_length=seq_length)
    slided_labels = slided_labels.astype(np.int32)
    return slided_labels


def test_smd_for_all_entity(seq_length: int = 4, nrows: int = 1000):
    preds, accs = [], []
    for entity_id in entity_ids:
        pred, acc = test_smd_for_entity(entity_id, seq_length, nrows)
        preds.append(pred)
        accs.append(acc)
    print("所有entity的最佳F-score平均值为：" + str(np.array(accs).mean()))


# 返回两个值
#   1. 根据上下边界计算的越界预测值
#   2. bf_search搜索出的最佳准确率
def test_smd_for_entity(entity_id: str, seq_length: int = 4, nrows: int = 1000):
    slided_labels = get_labels(entity_id=entity_id, seq_length=seq_length + 1, nrows=nrows)
    slided_labels = slided_labels.squeeze()

    testing_set = pd.read_csv('../data/smd/test/machine-' + entity_id + '.txt', header=None, nrows=nrows)
    testing_set = testing_set.astype(float)
    testing_set = testing_set.values
    sc = MinMaxScaler()
    testing_data = sc.fit_transform(testing_set)

    # 超参
    input_size = testing_set.shape[1]
    hidden_size = 2
    num_layers = 1
    ar_len = 2

    model = TPA(seq_length, hidden_size, num_layers, ar_len, input_size)
    dataset = SwatDataset(testing_data, seq_length)
    dataLoader = DataLoader(dataset=dataset, batch_size=20000, num_workers=0)

    if torch.cuda.is_available():
        model.load_state_dict(torch.load('model/smd/smd{}_tpa'.format(entity_id)))
    else:
        model.load_state_dict(
            torch.load('model/smd/smd{}_tpa'.format(entity_id), map_location=torch.device('cpu')))
    # 将模型转移到指定设备上
    device = get_device()
    model = model.to(device)
    model.eval()
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
    t, th = bf_search(np.mean(abnormal, axis=1), slided_labels, start=0., end=0.9, step_num=int((0.9 - 0.) / 0.001),
                      display_freq=100)
    return np.mean(abnormal, axis=0), t[0]


if __name__ == '__main__':
    test_smd_for_all_entity(4, 100)
