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
    def __init__(self, root=None, split_list=None,basename='input', train=True, delete_tp=20, delete_variable=None):
        self.root = root
        self.basename = basename
        self.train = train
        with open(split_list, 'r') as f:
            self.file_list = [line.rstrip() for line in f]
        with open(os.path.join(self.root, 'labels', 'born.txt'), 'r') as f:
            self.born_list = [line.rstrip() for line in f]
        with open(os.path.join(self.root, 'labels', 'abort.txt'), 'r') as f:
            self.abort_list = [line.rstrip() for line in f]
        self.criteria_list = csv_loader_criteria_list(os.path.join(self.root, basename, self.file_list[0], 'criteria.csv'))
        self.eps = 0.000001
        self.delete_tp = delete_tp
        if delete_variable != None and delete_variable != []:
            self.delete_variable = np.flip(np.sort(eval(delete_variable)))
            for d in self.delete_variable:
                self.criteria_list = np.delete(self.criteria_list, obj=d, axis=0)
        else:
            self.delete_variable = None

    def __len__(self):
        return len(self.file_list)

    def get_input(self, i):
        input = csv_loader(os.path.join(self.root, self.basename, self.file_list[i], 'criteria.csv'))
        if self.delete_variable is not None:
            for d in self.delete_variable:
                input = np.delete(input, obj=d, axis=1)
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
        e = torch.randint(0, self.delete_tp, (1,)).numpy()[0] + 1
        vec_aug = np.array(vec[:-e]).astype(np.float32)
        return vec_aug

    def __getitem__(self, i):
        input, label = self.get_input(i), self.get_label(i)
        if self.train:
            input = self.augmentation(input)
        input = self.normalization(input)
        return torch.tensor(input), torch.tensor(label)


class EmbryoImageDataset(Dataset):
    def __init__(self, root=None, split_list=None, basename='images', train=True, delete_tp=50, ip_size=[16,48,48], model='Conv2DLSTM'):
        self.root = root
        self.basename = basename
        self.train = train
        with open(split_list, 'r') as f:
            self.file_list = [line.rstrip() for line in f]
        with open(os.path.join(self.root, 'labels', 'born.txt'), 'r') as f:
            self.born_list = [line.rstrip() for line in f]
        with open(os.path.join(self.root, 'labels', 'abort.txt'), 'r') as f:
            self.abort_list = [line.rstrip() for line in f]
        self.criteria_list = None
        self.eps = 0.000001
        self.delete_tp = delete_tp
        self.ip_size = ip_size
        self.model = model
            
    def __len__(self):
        return len(self.file_list)

    def get_image(self, i):
        image_path = np.sort(glob(os.path.join(self.root, '{}_preprocess'.format(self.basename), self.file_list[i], '*.tif')))
        if len(image_path) == 0:
            image_path = np.sort(glob(os.path.join(self.root, self.basename, self.file_list[i], '*.tif')))
            images = []
            for p in range(len(image_path)):
                img = io.imread(image_path[p])
                img = transform.resize(img, self.ip_size, order=1, preserve_range=True)
                img = (img - np.mean(img)) / np.std(img)
                images.append(img)
            os.makedirs(os.path.join(self.root, '{}_preprocess'.format(self.basename), self.file_list[i]), exist_ok=True)
            filename = os.path.join(self.root, '{}_preprocess'.format(self.basename), self.file_list[i], 'images.tif')
            io.imsave(filename, np.array(images).astype(np.float32))
        else:
            if self.model == 'Conv5DLSTM':
                if self.train:
                    images = io.imread(image_path[0])[:-self.e]
                else:
                    images = io.imread(image_path[0])
            else:
                if self.train:
                    e = torch.randint(0, self.delete_tp, (1,)).numpy()[0] + 1
                    images = io.imread(image_path[0])[:-e]
                else:
                    images = io.imread(image_path[0])

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
        if self.model in ['Conv2DLSTM', 'Conv3DLSTM']:
            images, label = self.get_image(i), self.get_label(i)
            return torch.tensor(images), torch.tensor(label)
        elif self.model == 'Conv5DLSTM':
            if self.train:
                self.e = torch.randint(0, self.delete_tp, (1,)).numpy()[0] + 1
            self.basename = 'images_BF'
            images_2d = torch.tensor(self.get_image(i)).unsqueeze(1)
            self.basename = 'images'
            images_3d = torch.tensor(self.get_image(i))
            images = torch.cat((images_2d, images_3d), dim=1)
            label = self.get_label(i)
            return images, torch.tensor(label)
