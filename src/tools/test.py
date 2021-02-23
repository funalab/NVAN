import os
import sys
import json
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

from src.lib.utils.utils import get_model, get_test_dataset
from src.lib.utils.cmd_args import create_dataset_parser, create_classifier_parser, create_runtime_parser
from src.lib.trainer.trainer import Tester
from src.lib.datasets.data_loader import pad_collate


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

    ''' Classifier '''
    # Initialize the classifier to train
    classifier = get_model(args)
    if args.init_classifier is not None:
        print('Load classifier from', args.init_classifier)
        classifier = torch.load(args.init_classifier)
        classifier.phase = args.phase


    # Prepare device
    if args.device is 'cuda:0':
        classifier = classifier.to(args.device)
    else:
        classifier = classifier.to(args.device)

    ''' Dataset '''
    # Load dataset
    test_dataset = get_test_dataset(args)
    print('-- test_dataset.size = {}'.format(
        test_dataset.__len__()))

    ''' Iterator '''
    # Set up iterators
    test_iterator = DataLoader(
        dataset=test_dataset,
        batch_size=int(args.val_batchsize),
        shuffle=False,
        collate_fn=pad_collate
    )

    # Make Directory
    current_datetime = datetime.now(pytz.timezone('Asia/Tokyo')).strftime('%Y%m%d_%H%M%S')
    save_dir = args.save_dir + '_' + str(current_datetime)
    os.makedirs(save_dir, exist_ok=True)

    tester_args = {
        'save_dir' : save_dir,
        'file_list' : test_dataset.file_list
        }
    tester = Tester(**tester_args)
    result, _ = tester.test(classifier, test_iterator, phase='test')
    with open(os.path.join(save_dir, 'log'), 'w') as f:
        json.dump(result, f, indent=4)

if __name__ == '__main__':
    main()
