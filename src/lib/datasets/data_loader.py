import sys
import os
import csv
import numpy as np

from torch.nn.utils.rnn import pad_sequence

end_time = 360

def csv_loader(path):
    with open(path, 'r') as f:
        reader = csv.reader(f)
        vec = [row[1:] for row in reader]
        vec.pop(0)
        vec = np.array(vec).astype(np.float32)
    return vec[:end_time]

def csv_loader_criteria_list(path):
    with open(path, 'r') as f:
        reader = csv.reader(f)
        vec = [row[1:] for row in reader]
    return np.array(vec[0])

def label_loader(file_list):
    label = []
    for fl in file_list:
        if fl[:fl.find('/')] == 'born':
            label.append(0)
        elif fl[:fl.find('/')] == 'abort':
            label.append(1)
        else:
            sys.exit()

def pad_collate(batch):
    (xx, yy) = zip(*batch)
    x_lens = [len(x) for x in xx]
    y_lens = [len(y) for y in yy]
    xx_pad = pad_sequence(xx, batch_first=True, padding_value=0)
    yy_pad = pad_sequence(yy, batch_first=True, padding_value=0)

    return xx_pad, yy_pad
