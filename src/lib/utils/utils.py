import os
import torch.nn as nn
from src.lib.models.net import LSTMClassifier, LSTMAttentionClassifier, LSTMMultiAttentionClassifier,BiDirectionalLSTMClassifier, MuVAN
from src.lib.datasets.dataset import EmbryoDataset
from src.lib.datasets.transforms import normalize


def get_model(args):
    model = eval(args.model)(
            input_dim=args.input_dim,
            embed_dim=args.embed_dim,
            num_classes=args.num_classes,
            num_layers=args.num_layers,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            lossfun=eval(args.lossfun),
            sharping_factor=args.sharping_factor,
            phase=args.phase
        )
    return model

def get_dataset(args):
    train_dataset = EmbryoDataset(
        transform=normalize(),
        root=args.root_path,
        split_list=args.split_list_train,
        train=True
    )
    validation_dataset = EmbryoDataset(
        transform=normalize(),
        root=args.root_path,
        split_list=args.split_list_validation,
        train=False
    )
    return train_dataset, validation_dataset


def get_test_dataset(args):
    test_dataset = EmbryoDataset(
        transform=normalize(),
        root=args.root_path,
        split_list=args.split_list_test,
        train=False
    )
    return test_dataset

def print_args(dataset_args, model_args, updater_args, runtime_args):
    """ Export config file
    Args:
        dataset_args    : Argument Namespace object for loading dataset
        model_args      : Argument Namespace object for Generator and Discriminator
        updater_args    : Argument Namespace object for Updater
        runtime_args    : Argument Namespace object for runtime parameters
    """
    dataset_dict = {k: v for k, v in vars(dataset_args).items() if v is not None}
    model_dict = {k: v for k, v in vars(model_args).items() if v is not None}
    updater_dict = {k: v for k, v in vars(updater_args).items() if v is not None}
    runtime_dict = {k: v for k, v in vars(runtime_args).items() if v is not None}
    print('============================')
    print('[Dataset]')
    for k, v in dataset_dict.items():
        print('%s = %s' % (k, v))
    print('\n[Model]')
    for k, v in model_dict.items():
        print('%s = %s' % (k, v))
    print('\n[Updater]')
    for k, v in updater_dict.items():
        print('%s = %s' % (k, v))
    print('\n[Runtime]')
    for k, v in runtime_dict.items():
        print('%s = %s' % (k, v))
    print('============================\n')


def export_to_config(save_dir, dataset_args, model_args, updater_args, runtime_args):
    """ Export config file
    Args:
        save_dir (str)      : /path/to/save_dir
        dataset_args (dict) : Dataset arguments
        model_args (dict)   : Model arguments
        updater_args (dict) : Updater arguments
        runtime_args (dict) : Runtime arguments
    """
    dataset_dict = {k: v for k, v in vars(dataset_args).items() if v is not None}
    model_dict = {k: v for k, v in vars(model_args).items() if v is not None}
    updater_dict = {k: v for k, v in vars(updater_args).items() if v is not None}
    runtime_dict = {k: v for k, v in vars(runtime_args).items() if v is not None}
    with open(os.path.join(save_dir, 'parameters.cfg'), 'w') as txt_file:
        txt_file.write('[Dataset]\n')
        for k, v in dataset_dict.items():
            txt_file.write('%s = %s\n' % (k, v))
        txt_file.write('\n[Model]\n')
        for k, v in model_dict.items():
            txt_file.write('%s = %s\n' % (k, v))
        txt_file.write('\n[Updater]\n')
        for k, v in updater_dict.items():
            txt_file.write('%s = %s\n' % (k, v))
        txt_file.write('\n[Runtime]\n')
        for k, v in runtime_dict.items():
            txt_file.write('%s = %s\n' % (k, v))
        txt_file.write('\n[MN]\n')
