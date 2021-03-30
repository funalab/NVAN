import os
import numpy as np
import random
import torch
from torch.utils.data import Dataset
from src.lib.datasets.data_loader import csv_loader, csv_loader_criteria_list, image_loader

class EmbryoDataset(Dataset):
    def __init__(self, root=None, split_list=None, train=True, delete_tp=20):
        self.root = root
        self.train = train
        with open(split_list, 'r') as f:
            self.file_list = [line.rstrip() for line in f]
        with open(os.path.join(self.root, 'labels', 'born.txt'), 'r') as f:
            self.born_list = [line.rstrip() for line in f]
        with open(os.path.join(self.root, 'labels', 'abort.txt'), 'r') as f:
            self.abort_list = [line.rstrip() for line in f]
        self.criteria_list = csv_loader_criteria_list(os.path.join(self.root, 'input', self.file_list[0], 'criteria.csv'))
        self.eps = 0.000001
        self.delete_tp = delete_tp

    def __len__(self):
        return len(self.file_list)

    def get_input(self, i):
        input = csv_loader(os.path.join(self.root, 'input', self.file_list[i], 'criteria.csv'))
        return input

    def get_label(self, i):
        if self.file_list[i] in self.born_list:
            label = np.array([1])
        elif self.file_list[i] in self.abort_list:
            label = np.array([0])
        else:
            raise ValueError('Unknown file name: {}'.format(self.file_list[i]))
        return label

    def normalization(self, vec):
        vec = vec.transpose(1, 0)
        return np.array([(v - np.mean(v)) / (np.std(v) + self.eps) for v in vec]).transpose(1, 0).astype(np.float32)

    def augmentation(self, vec):
        #s = int(random.uniform(0, self.delete_tp))
        e = int(random.uniform(0, self.delete_tp)) + 1
        vec_aug = np.array(vec[:-e]).astype(np.float32)
        return vec_aug

    def __getitem__(self, i):
        input, label = self.get_input(i), self.get_label(i)
        if self.train:
            input = self.augmentation(input)
        input = self.normalization(input)
        return torch.tensor(input), torch.tensor(label)


class EmbryoImageDataset(Dataset):
    def __init__(self, root=None, split_list=None, train=True, delete_tp=20):
        self.root = root
        self.train = train
        with open(split_list, 'r') as f:
            self.file_list = [line.rstrip() for line in f]
        with open(os.path.join(self.root, 'labels', 'born.txt'), 'r') as f:
            self.born_list = [line.rstrip() for line in f]
        with open(os.path.join(self.root, 'labels', 'abort.txt'), 'r') as f:
            self.abort_list = [line.rstrip() for line in f]
        self.criteria_list = csv_loader_criteria_list(os.path.join(self.root, 'input', self.file_list[0], 'criteria.csv'))
        self.eps = 0.000001
        self.delete_tp = delete_tp

    def __len__(self):
        return len(self.file_list)

    def get_image(self, i):
        images = image_loader(os.path.join(self.root, 'image', self.file_list[i]))
        return images

    def get_label(self, i):
        if self.file_list[i] in self.born_list:
            label = np.array([1])
        elif self.file_list[i] in self.abort_list:
            label = np.array([0])
        else:
            raise ValueError('Unknown file name: {}'.format(self.file_list[i]))
        return label

    def normalization(self, images):
        for t in len(images):
            images[t] = (images[t] - np.mean(images[t])) / np.std(images)
        return images.astype(np.float32)

    def __getitem__(self, i):
        images, label = self.get_image(i), self.get_label(i)
        images = self.normalization(images)
        return torch.tensor(images), torch.tensor(label)
