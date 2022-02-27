import numpy as np
import torch


# seq_length=n 长度为前n-1的历史值，n为当前值
def sliding_windows(data, seq_length):
    x, y = [], []
    for i in range(len(data) - seq_length + 1):
        _x = data[i:(i + seq_length - 1)]
        _y = data[i + seq_length - 1]
        x.append(_x)
        y.append(_y)
    return np.array(x), np.array(y)


# Loss function
def quantile_loss(q, y, f):
    e = (y - f)
    a = torch.max(q * e, (q - 1) * e)
    b = torch.mean(a, dim=-1)
    return b


def get_device():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return device
