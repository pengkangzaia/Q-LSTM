import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


class TPA_Attention(nn.Module):
    def __init__(self, seq_length, hidden_size):
        super(TPA_Attention, self).__init__()
        self.n_filters = 32
        self.filter_size = 1
        self.conv = nn.Conv2d(1, self.n_filters, (seq_length, self.filter_size))
        self.Wa = nn.Parameter(torch.rand(self.n_filters, hidden_size))
        self.Whv = nn.Linear(self.n_filters + hidden_size, hidden_size)

    def forward(self, hs, ht):
        hs = hs.unsqueeze(1)
        H = self.conv(hs)[:, :, 0]  # B x n_filters x hidden_size
        H = H.transpose(1, 2)
        alpha = torch.sigmoid(torch.sum((H @ self.Wa) * ht.unsqueeze(-1), dim=-1))  # B x hidden_size
        V = torch.sum(H * alpha.unsqueeze(-1), dim=1)  # B x n_filters
        vh = torch.cat([V, ht], dim=1)
        return self.Whv(vh)


class TPA(nn.Module):
    def __init__(self, seq_length, hidden_size, n_layers, ar_len, input_size):
        super(TPA, self).__init__()
        self.input_proj = nn.Linear(input_size, hidden_size)
        self.lstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size,
                            num_layers=n_layers, batch_first=True)
        self.att = TPA_Attention(seq_length, hidden_size)
        self.out_proj = nn.Linear(hidden_size, input_size)
        self.ar = nn.Linear(ar_len, 1)

        self.ar_len = ar_len
        self.fc_low = nn.Linear(hidden_size, input_size)
        self.fc_high = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        # batch_size, seq_len, input_size = x.size()
        px = F.relu(self.input_proj(x))
        hs, (ht, _) = self.lstm(px)
        ht = ht[-1]
        final_h = self.att(hs, ht)
        out = self.out_proj(final_h) + self.ar(x[:, -self.ar_len:].transpose(1, 2))[:, :, 0]
        out_low = self.fc_low(out)
        out_high = self.fc_high(out)
        return out_low, out_high
