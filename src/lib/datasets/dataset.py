import os
import numpy as np
import torch
from torch.utils.data import Dataset
from src.lib.datasets.data_loader import csv_loader

class EmbryoDataset(Dataset):
    def __init__(self, transform=None, root=None, split_list=None, train=True):
        self.transform = transform
        self.root = root
        self.train = train
        with open(split_list, 'r') as f:
            self.file_list = [line.rstrip() for line in f]
        with open(os.path.join(self.root, 'labels', 'born.txt'), 'r') as f:
            self.born_list = [line.rstrip() for line in f]
        with open(os.path.join(self.root, 'labels', 'abort.txt'), 'r') as f:
            self.abort_list = [line.rstrip() for line in f]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, i):
        input = csv_loader(os.path.join(self.root, 'input', self.file_list[i], 'criteria.csv'))
        if self.file_list[i] in self.born_list:
            label = np.array([1])
        elif self.file_list[i] in self.abort_list:
            label = np.array([0])
        else:
            raise ValueError('Unknown file name: {}'.format(self.file_list[i]))

        # if self.transform:
        #     input = self.transform(input)

        return torch.tensor(input), torch.tensor(label)
