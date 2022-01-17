import numpy as np
from flask import Flask, request

from Q_LSTM import *

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


if __name__ == "__main__":
    app.run()
