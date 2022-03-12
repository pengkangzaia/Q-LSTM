import torch
import torch.nn as nn
from torch.autograd import Variable
from utils import get_device


class LSTM(nn.Module):

    def __init__(self, num_classes, input_size, hidden_size, seq_length, num_layers):
        super(LSTM, self).__init__()

        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_length = seq_length

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)

        self.fc = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        ula, (h_out, _) = self.lstm(x)
        ula = ula.view(x.size(0), x.size(1), self.hidden_size)[:, -1]
        out = self.fc(ula)

        return out
