import torch
from torch.utils.data import Dataset


class BenchmarkDataset(Dataset):
    def __init__(self, shot, dset_size, n_channels):
        self.shot = shot
        self.dset_size = dset_size
        self.n_channels = n_channels
        self.task_group_names = [f'proxy_{i}' for i in range(n_channels)]
        self.TASK_GROUP_NAMES = self.task_group_names

    def __len__(self):
        return self.dset_size
    
    def __getitem__(self, idx):
        X = torch.rand(self.n_channels, self.shot*2, 3, 224, 224)
        Y = torch.rand(self.n_channels, self.shot*2, 1, 224, 224)
        M = torch.ones_like(Y)
        t_idx = torch.arange(self.n_channels)
        g_idx = torch.LongTensor([0])[0]
        channel_mask = torch.ones(self.n_channels, self.n_channels, dtype=torch.bool)

        return X, Y, M, t_idx, g_idx, channel_mask
