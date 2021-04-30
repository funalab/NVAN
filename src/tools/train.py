import os
import sys
import argparse
import configparser
import multiprocessing
from datetime import datetime
import pytz
import math
import matplotlib.pylab as plt
#plt.use('Agg')
sys.path.append(os.getcwd())
import random
import numpy as np
from glob import glob

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

from src.lib.utils.utils import get_model, get_dataset
from src.lib.utils.cmd_args import create_dataset_parser, create_classifier_parser, create_runtime_parser
from src.lib.trainer.trainer import Trainer
from src.lib.datasets.data_loader import pad_collate
from src.lib.datasets.sampler import BalancedBatchSampler


def main(argv=None):

    ''' ConfigParser '''
    # parsing arguments from command-line or config-file
    if argv is None:
        argv = sys.argv

    conf_parser = argparse.ArgumentParser(
        description=__doc__,  # printed with -h/--help
        formatter_class=argparse.RawDescriptionHelpFormatter,
        add_help=False
    )
    conf_parser.add_argument("-c", "--conf_file", help="Specify config file", metavar="FILE_PATH")
    args, remaining_argv = conf_parser.parse_known_args()

    dataset_conf_dict = {}
    classifier_conf_dict = {}
    runtime_conf_dict = {}

    if args.conf_file is not None:
        config = configparser.ConfigParser()
        config.read([args.conf_file])
        dataset_conf_dict = dict(config.items("Dataset"))
        classifier_conf_dict = dict(config.items("Model"))
        runtime_conf_dict = dict(config.items("Runtime"))

    ''' Parameters '''
    # Dataset options
    dataset_parser, dataset_args, remaining_argv = \
        create_dataset_parser(remaining_argv, **dataset_conf_dict)
    # Classifier options
    classifier_parser, classifier_args, remaining_argv = \
        create_classifier_parser(remaining_argv, **classifier_conf_dict)
    # Runtime options
    runtime_parser, runtime_args, remaining_argv = \
        create_runtime_parser(remaining_argv, **runtime_conf_dict)

    # merge options
    parser = argparse.ArgumentParser(
        description='PyTorch implementation of embryo classification',
        parents=[conf_parser, dataset_parser, classifier_parser, runtime_parser])
    args = parser.parse_args()

    # Seed
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(args.seed)

    ''' Classifier '''
    # Initialize the classifier to train
    classifier = get_model(args)
    if args.init_classifier is not None:
        print('Load classifier from', args.init_classifier)
        classifier = torch.load(args.init_classifier)

    # Prepare device
    if args.device == 'cuda:0':
        classifier = classifier.to(args.device)
    else:
        classifier = classifier.to(args.device)

    ''' Dataset '''
    # Load dataset
    train_dataset, validation_dataset = get_dataset(args)
    print('-- train_dataset.size = {}\n-- validation_dataset.size = {}'.format(
        train_dataset.__len__(), validation_dataset.__len__()))

    train_sampler = BalancedBatchSampler(train_dataset, n_classes=args.num_classes, n_samples=args.batchsize)

    ''' Iterator '''
    # Set up iterators
    train_iterator = DataLoader(
        dataset=train_dataset,
        #batch_size=int(args.batchsize),
        batch_sampler=train_sampler,
        collate_fn=pad_collate
    )
    validation_iterator = DataLoader(
        dataset=validation_dataset,
        batch_size=int(args.val_batchsize),
        shuffle=False,
        collate_fn=pad_collate
    )

    ''' Optimizer '''
    # Initialize an optimizer
    if args.optimizer == 'SGD':
        optimizer = optim.SGD(
            params=classifier.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay
            )
    elif args.optimizer == 'Adadelta':
        optimizer = optim.Adadelta(
            params=classifier.parameters(),
            lr=args.lr,
            rho=args.momentum,
            weight_decay=args.weight_decay
            )
    elif args.optimizer == 'Adagrad':
        optimizer = optim.Adagrad(
            params=classifier.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
            )
    elif args.optimizer == 'Adam':
        optimizer = optim.Adam(
            params=classifier.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
            )
    elif args.optimizer == 'AdamW':
        optimizer = optim.AdamW(
            params=classifier.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
            )
    elif args.optimizer == 'SparseAdam':
        optimizer = optim.SparseAdam(
            params=classifier.parameters(),
            lr=args.lr
            )
    elif args.optimizer == 'Adamax':
        optimizer = optim.Adamax(
            params=classifier.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
            )
    elif args.optimizer == 'ASGD':
        optimizer = optim.ASGD(
            params=classifier.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
            )
    elif args.optimizer == 'RMSprop':
        optimizer = optim.RMSprop(
            params=classifier.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay
            )
    else:
        raise ValueError('Unknown optimizer name: {}'.format(args.optimizer))

    # Make Directory
    current_datetime = datetime.now(pytz.timezone('Asia/Tokyo')).strftime('%Y%m%d_%H%M%S')
    save_dir = args.save_dir + '_' + str(current_datetime)
    os.makedirs(save_dir, exist_ok=True)

    ''' Graph Visualization '''
    if eval(args.graph):
        print('Making the graph of model.', end='')
        dummy_x, _ = train_dataset.__getitem__(0)
        input_names = ['input']
        print('.', end='')
        output_names = ['output']
        print('.', end='')
        torch.onnx.export(classifier, dummy_x.unsqueeze(0), os.path.join(save_dir, 'graph.onnx'), verbose=True, input_names=input_names, output_names=output_names)
        print('Success!')

    # Training
    trainer_args = {
        'optimizer' : optimizer,
        'epoch' : args.epoch,
        'save_dir' : save_dir,
        'eval_metrics' : args.eval_metrics,
        'device' : args.device
    }

    trainer = Trainer(**trainer_args)
    trainer.train(
        model=classifier,
        train_iterator=train_iterator,
        validation_iterator=validation_iterator
    )

    if args.delete_variable is not None:
        aw_list = np.load(os.path.join(save_dir, 'aw_best_val.npz'), allow_pickle=True)['arr_0']
        aw = np.ones((args.input_dim, 487))
        for p in range(len(aw_list)):
            aw *= aw_list[p][0, :,:487].cpu().numpy()
        aw = aw ** (1/len(aw_list))
        aw = np.sum(aw, axis=1)
        index = np.argmin(aw)
        new_delete_variable = []
        cnt = 0
        for i in range(11):
            if i in eval(args.delete_variable):
                new_delete_variable.append(i)
            else:
                if cnt == index:
                    new_delete_variable.append(i)
                cnt += 1
        print(new_delete_variable)

        set_num = int(args.split_list_train[len('datasets/split_list/mccv/set'):args.split_list_train.rfind('/')])
        filename = os.path.join('confs', 'models', 'mccv_sv', 'NVAN', 'train_set{0}_sv{1:02d}.cfg'.format(set_num, len(new_delete_variable)))
        with open(filename, 'w') as f:
            f.write('[Dataset]\n')
            f.write('root_path = datasets\n')
            f.write('split_list_train = datasets/split_list/mccv/set{}/train.txt\n'.format(set_num))
            f.write('split_list_validation = datasets/split_list/mccv/set{}/validation.txt\n'.format(set_num))
            f.write('basename =input\n\n')

            f.write('[Model]\n')
            f.write('model = NVAN\n')
            f.write('# init_classifier =\n')
            f.write('input_dim = {}\n'.format(11 - len(new_delete_variable)))
            f.write('num_classes = 2\n')
            f.write('num_layers = 2\n')
            f.write('hidden_dim = 128\n')
            f.write('dropout = 0.5\n')
            f.write('lossfun = nn.BCEWithLogitsLoss()\n')
            f.write('eval_metrics = f1\n\n')

            f.write('[Runtime]\n')
            f.write('save_dir = results/train_NVAN_set{0:02d}_sv{1:02d}\n'.format(set_num, len(new_delete_variable)))
            f.write('batchsize = 2\n')
            f.write('val_batchsize = 1\n')
            f.write('epoch = {}\n'.format(args.epoch))
            f.write('optimizer = Adadelta\n')
            f.write('lr = 1.0\n')
            f.write('momentum = 0.95\n')
            f.write('weight_decay = 0.001\n')
            f.write('delete_tp = 50\n')
            f.write('# cuda:0 or cpu\n')
            f.write('device = {}\n'.format(args.device))
            f.write('seed = 0\n')
            f.write('phase = train\n')
            f.write('graph = False\n')
            f.write('# 0: number, 1: volume_mean, 2: volume_sd, 3: surface_mean, 4: surface_sd,\n')
            f.write('# 5: aspect_mean, 6: aspect_sd, 7: solidity_mean, 8: solidity_sd, 9: centroid_mean, 10: centroid_sd\n')
            f.write('delete_variable = {}   # input_dim == 11 - len(delete_variable)\n'.format(new_delete_variable))

if __name__ == '__main__':
    main()
