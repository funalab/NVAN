import os
import torch.nn as nn
# from src.lib.models import LSTM, LSTMAttention, LSTMMultiAttention, MuVAN, iMuVAN
from src.lib.models.LSTM import LSTM
from src.lib.models.LSTMAttention import LSTMAttention
from src.lib.models.LSTMMultiAttention import LSTMMultiAttention
from src.lib.models.MuVAN import MuVAN
from src.lib.models.NVAN import NVAN
from src.lib.models.Transformer import Transformer
from src.lib.models.Conv2DLSTM import Conv2DLSTM
from src.lib.models.Conv3DLSTM import Conv3DLSTM
from src.lib.models.Conv5DLSTM import Conv5DLSTM
from src.lib.datasets.dataset import EmbryoDataset, EmbryoImageDataset


def get_model(args):
    model_list = ['LSTM', 'LSTMAttention', 'LSTMMultiAttention', 'MuVAN', 'NVAN']
    if args.model in model_list:
        model = eval(args.model)(
                input_dim=args.input_dim,
                num_classes=args.num_classes,
                num_layers=args.num_layers,
                hidden_dim=args.hidden_dim,
                dropout=args.dropout,
                lossfun=eval(args.lossfun),
                phase=args.phase
            )
    elif args.model == 'Transformer':
        model = eval(args.model)(
                input_dim=args.input_dim,
                num_classes=args.num_classes,
                num_layers=args.num_layers,
                hidden_dim=args.hidden_dim,
                num_head=args.num_head,
                dropout=args.dropout,
                lossfun=eval(args.lossfun),
                phase=args.phase
        )
    elif args.model == 'Conv2DLSTM' or 'Conv3DLSTM' or 'Conv5DLSTM':
        model = eval(args.model)(
                input_dim=args.input_dim,
                num_classes=args.num_classes,
                num_layers=args.num_layers,
                hidden_dim=args.hidden_dim,
                dropout=args.dropout,
                ip_size=eval(args.ip_size),
                lossfun=eval(args.lossfun),
                phase=args.phase
        )
    else:
        raise ValueError('Unknown model name: {}'.format(args.model))

    return model

def get_dataset(args):
    model_list = ['LSTM', 'LSTMAttention', 'LSTMMultiAttention', 'MuVAN', 'NVAN', 'Transformer']
    if args.model in model_list:
        train_dataset = EmbryoDataset(
            root=args.root_path,
            split_list=args.split_list_train,
            basename=args.basename,
            train=True,
            delete_tp=args.delete_tp,
            delete_variable=args.delete_variable
        )
        validation_dataset = EmbryoDataset(
            root=args.root_path,
            split_list=args.split_list_validation,
            basename=args.basename,
            train=False,
            delete_tp=None,
            delete_variable=args.delete_variable
        )
    elif args.model == 'Conv2DLSTM' or 'Conv3DLSTM' or 'Conv5DLSTM':
        train_dataset = EmbryoImageDataset(
            root=args.root_path,
            split_list=args.split_list_train,
            basename=args.basename,
            train=True,
            delete_tp=args.delete_tp,
            ip_size=eval(args.ip_size),
            model=args.model
        )
        validation_dataset = EmbryoImageDataset(
            root=args.root_path,
            split_list=args.split_list_validation,
            basename=args.basename,
            train=False,
            delete_tp=0,
            ip_size=eval(args.ip_size),
            model=args.model
        )
    else:
        raise ValueError('Unknown model name: {}'.format(args.model))

    return train_dataset, validation_dataset


def get_test_dataset(args):
    model_list = ['LSTM', 'LSTMAttention', 'LSTMMultiAttention', 'MuVAN', 'NVAN', 'Transformer']
    if args.model in model_list:
        test_dataset = EmbryoDataset(
            root=args.root_path,
            split_list=args.split_list_test,
            basename=args.basename,
            train=False,
            delete_tp=None,
            delete_variable=args.delete_variable
        )
    elif args.model == 'Conv2DLSTM' or 'Conv3DLSTM' or 'Conv5DLSTM':
        test_dataset = EmbryoImageDataset(
            root=args.root_path,
            split_list=args.split_list_test,
            basename=args.basename,
            train=False,
            delete_tp=0,
            ip_size=eval(args.ip_size),
            model=args.model
        )
    else:
        raise ValueError('Unknown model name: {}'.format(args.model))
    
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
