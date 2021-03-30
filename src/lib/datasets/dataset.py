import os
import numpy as np
import random
import skimage.io as io
from skimage import transform
from glob import glob

import torch
from torch.utils.data import Dataset
from src.lib.datasets.data_loader import csv_loader, csv_loader_criteria_list

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
    def __init__(self, root=None, split_list=None, train=True, delete_tp=20, ip_size=[16,48,48]):
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
        self.ip_size = ip_size

    def __len__(self):
        return len(self.file_list)

    def get_image(self, i):
        image_path = np.sort(glob(os.path.join(self.root, 'images_preprocess', self.file_list[i], '*.tif')))
        if len(image_path) == 0:
            image_path = np.sort(glob(os.path.join(self.root, 'images', self.file_list[i], '*.tif')))
            images = []
            for p in range(len(image_path)):
                img = io.imread(image_path[p])
                img = transform.resize(img, self.ip_size, order=1, preserve_range=True)
                img = (img - np.mean(img)) / np.std(img)
                images.append(img)
            os.makedirs(os.path.join(self.root, 'images_preprocess', self.file_list[i]), exist_ok=True)
            filename = os.path.join(self.root, 'images_preprocess', self.file_list[i], 'images.tif')
            io.imsave(filename, np.array(images).astype(np.float32))
        else:
            e = int(random.uniform(0, self.delete_tp)) + 1
            images = io.imread(image_path[0])[:-e]
            
        return np.array(images).astype(np.float32)

    def get_label(self, i):
        if self.file_list[i] in self.born_list:
            label = np.array([1])
        elif self.file_list[i] in self.abort_list:
            label = np.array([0])
        else:
            raise ValueError('Unknown file name: {}'.format(self.file_list[i]))
        return label

    def __getitem__(self, i):
        images, label = self.get_image(i), self.get_label(i)
        return torch.tensor(images), torch.tensor(label)
