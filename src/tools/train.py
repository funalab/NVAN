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

        # print('Making the graph of model.', end='')
        # from torchviz import make_dot
        # print('.', end='')
        # dummy_x, _ = train_dataset.__getitem__(0)
        # dummy_y = classifier(dummy_x.unsqueeze(0))
        # print('.', end='')
        # dot = make_dot(dummy_y.mean(), params=dict(classifier.named_parameters()))
        # dot.format = 'png'
        # print('.', end='')
        # dot.render(os.path.join(save_dir, 'graph'))
        # print('Success!')

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

if __name__ == '__main__':
    main()
