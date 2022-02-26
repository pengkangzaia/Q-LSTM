import os

import influxdb_client
import matplotlib.pyplot as plt
import pandas as pd
from flask import Flask, request
from sklearn.preprocessing import MinMaxScaler

from Q_LSTM import *
from eval_methods import *
from utils import sliding_windows, quantile_loss

app = Flask(__name__)


@app.route("/pred", methods=["POST"])
def predict():
    data = request.get_json()
    # data数据需要标准化到0-1
    data = np.array(data)

    abnormals = []
    for idx in range(len(data[0])):
        seq_length = data.shape[0] - 1
        d = np.expand_dims(data[:, idx], 1)
        x = np.expand_dims(d[:-1], 0)
        y = np.expand_dims(np.expand_dims(d[-1], 0), 0)
        dataX = Variable(torch.Tensor(np.array(x)))
        dataY = Variable(torch.Tensor(np.array(y)))

        model = LSTM(num_classes=1, input_size=1, hidden_size=2, seq_length=seq_length, num_layers=1)
        model.load_state_dict(
            torch.load('../trained_model/psm/saved_model{}'.format(idx), map_location=torch.device('cpu')))
        model.eval()
        test_predict_low, test_predict_high = model(torch.FloatTensor(dataX))
        data_predict_low = test_predict_low.data.cpu().numpy()
        data_predict_high = test_predict_high.data.cpu().numpy()
        dataY_plot = np.array(torch.FloatTensor(dataY))
        abnormal = np.where((dataY_plot < data_predict_low) | (dataY_plot > data_predict_high), 1, 0)
        abnormal = np.squeeze(abnormal, axis=1)
        abnormals.append(abnormal)
    final_res = np.mean(abnormals, axis=0)
    # 从数据库中取阈值 t
    t = 0.5
    final_res = np.where(final_res > t, 1, 0)
    final_res = np.squeeze(final_res).tolist()
    return str(final_res)


@app.route("/train", methods=["GET"])
def train():
    ip = request.args.get("ip")
    ip = ip.replace(".", "_")
    folder = os.path.exists('trained_model/' + ip)
    if not folder:
        os.makedirs('trained_model/' + ip)
    # 1. 读取数据
    bucket = "monitor"
    org = "seu"
    token = "gZTu3-P2pKcGQI-wBgHUT1nRIckb7N_drF-r9YKUdbszy1hTrN3BwIR5CdFHshzGcW81n_SbjfI5-RQsUz11zA=="
    url = "http://101.35.159.221:8086"
    client = influxdb_client.InfluxDBClient(url=url, token=token, org=org)
    query_api = client.query_api()
    flux = 'from(bucket: "monitor")|> range(start: -60m)|> filter(fn: (r) => r["_measurement"] == "cpu2" or r[' \
           '"_measurement"] == "disk" or r["_measurement"] == "memory" or r["_measurement"] == "net")|> filter(fn: (r) => ' \
           'r["address"] == "http://1.15.117.64:8081")|> drop(columns: ["result", "address", "_measurement", "_start", ' \
           '"_stop"])|> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value") '
    # 3天数据量大概要半分钟
    df = query_api.query_data_frame(flux)
    df = df.drop(['result', 'table'], axis=1)
    df["_time"] = pd.to_datetime(df['_time'])
    df.index = df['_time']
    df = df.resample('30S').mean()
    df = df.interpolate(method='linear')
    # 训练模型
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
        torch.save(lstm.state_dict(), 'trained_model/' + ip + '/model_{}'.format(col))

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
    # 4. 可能还需要做的：实时读取进度
    return "1"




if __name__ == "__main__":
    app.run()
