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

        self.fc_low = nn.Linear(hidden_size, num_classes)
        self.fc_high = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size))

        c_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size))
        device = get_device()
        x = x.to(device)
        h_0 = h_0.to(device)
        c_0 = c_0.to(device)

        # Propagate input through LSTM
        ula, (h_out, _) = self.lstm(x, (h_0, c_0))

        h_out = h_out.view(-1, self.hidden_size)

        out_low = self.fc_low(h_out)
        out_high = self.fc_high(h_out)

        return out_low, out_high


