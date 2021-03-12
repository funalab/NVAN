import os
import numpy as np
import random
import torch
from torch.utils.data import Dataset
from src.lib.datasets.data_loader import csv_loader, csv_loader_criteria_list

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
        self.criteria_list = csv_loader_criteria_list(os.path.join(self.root, 'input', self.file_list[0], 'criteria.csv'))

    def __len__(self):
        return len(self.file_list)

    def get_input(self, i):
        input = csv_loader(os.path.join(self.root, 'input', self.file_list[i], 'criteria.csv'))
        return self.normalization(self.transform(input))

    def get_label(self, i):
        if self.file_list[i] in self.born_list:
            label = np.array([1])
        elif self.file_list[i] in self.abort_list:
            label = np.array([0])
        else:
            raise ValueError('Unknown file name: {}'.format(self.file_list[i]))
        return label

    def normalization(self, vec):
        return np.array([(v - np.mean(v)) / np.std(v) for v in vec]).astype(np.float32)

    def transform(self, vec):
        return np.array(vec[int(random.uniform(0, 20)):]).astype(np.float32)

    def __getitem__(self, i):
        input, label = self.get_input(i), self.get_label(i)

        return torch.tensor(input), torch.tensor(label)
