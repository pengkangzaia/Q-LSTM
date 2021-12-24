import numpy as np
import torch


# def sliding_windows(data, labels, seq_length):
#     x = []
#     y = []
#     l = []
#
#     for i in range(len(data) - seq_length - 1):
#         _x = data[i:(i + seq_length)]
#         _y = data[i + seq_length]
#         x.append(_x)
#         y.append(_y)
#         l.append(labels[i + seq_length])
#
#     return np.array(x), np.array(y), np.array(l)


def sliding_windows(data, seq_length):
    x = []
    y = []

    for i in range(len(data) - seq_length - 1):
        _x = data[i:(i + seq_length)]
        _y = data[i + seq_length]
        x.append(_x)
        y.append(_y)

    return np.array(x), np.array(y)


# Loss function
def quantile_loss(q, y, f):
    e = (y - f)
    a = torch.max(q * e, (q - 1) * e)
    b = torch.mean(a, dim=-1)
    return b






