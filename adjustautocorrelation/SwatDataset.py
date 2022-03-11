from torch.utils.data import Dataset


class SwatDataset(Dataset):
    def __init__(self, raw_X, seq_length):
        self.X = raw_X
        self.seq_length = seq_length

    def __len__(self):
        return self.X.shape[0] - self.seq_length

    def __getitem__(self, idx):
        return self.X[idx:idx + self.seq_length], self.X[idx + self.seq_length]
