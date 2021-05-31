import argparse
from distutils.util import strtobool


def create_dataset_parser(remaining_argv, **conf_dict):
    input_formats = ['csv']
    # Dataset options
    parser = argparse.ArgumentParser(description='Public Dataset Parameters', add_help=False)
    parser.set_defaults(**conf_dict)
    parser.add_argument('--root_path', type=str,
                        help='/path/to/dataset')
    parser.add_argument('--split_list_train', type=str,
                        help='/path/to/train_list.txt (list of {train,validation} files)')
    parser.add_argument('--split_list_validation', type=str,
                        help='/path/to/validation_list.txt (list of {train,validation} files)')
    parser.add_argument('--input_format', choices=input_formats,
                        help='Input format {"csv"}')
    parser.add_argument('--basename', type=str,
                        help='basename of input directory')
    args, remaining_argv = parser.parse_known_args(remaining_argv)

    return parser, args, remaining_argv


def create_classifier_parser(remaining_argv, **conf_dict):
    activation_types = ['relu', 'leaky_relu', 'swish']
    normalization_types = ['BatchNormalization', 'LayerNormalization', 'InstanceNormalization', 'GroupNormalization']
    # generator class options
    parser = argparse.ArgumentParser(description='Model Parameters', add_help=False)
    parser.set_defaults(**conf_dict)

    parser.add_argument('--init_classifier',
                        help='Initialize the Classifier file')
    parser.add_argument('--input_dim', type=int,
                        help='')
    parser.add_argument('--padding_idx', type=int,
                        help='')
    parser.add_argument('--num_classes', type=int,
                        help='')
    parser.add_argument('--num_layers', type=int,
                        help='')
    parser.add_argument('--hidden_dim', type=int,
                        help='')
    parser.add_argument('--base_ch', type=int,
                        help='')
    parser.add_argument('--num_head', type=int,
                        help='')
    parser.add_argument('--dropout', type=float,
                        help='')
    parser.add_argument('--lossfun',
                        help='')
    parser.add_argument('--eval_metrics',
                        help='')
    args, remaining_argv = parser.parse_known_args(remaining_argv)

    return parser, args, remaining_argv


def create_runtime_parser(remaining_argv, **conf_dict):
    optimizer_list = ['SGD', 'MomentumSGD', 'Adam']
    # Model runtime options (for adversarial network model)
    parser = argparse.ArgumentParser(description='Runtime Parameters', add_help=False)
    parser.set_defaults(**conf_dict)
    parser.add_argument('--save_dir', type=str,
                        help='Root directory which trained files are saved')
    parser.add_argument('--batchsize', '-B', type=int,
                        help='Learning minibatch size, default=32')
    parser.add_argument('--val_batchsize', '-b', type=int,
                        help='Validation minibatch size')
    parser.add_argument('--epoch', '-E', type=int,
                        help='Number of epochs to train, default=10')
    parser.add_argument('--optimizer', choices=optimizer_list,
                        help='Optimizer name for generator {"MomentumSGD", "SGD", "Adam"}')
    parser.add_argument('--lr', type=float,
                        help='Initial learning rate ("alpha" in case of Adam)')
    parser.add_argument('--momentum', type=float,
                        help='Momentum (used in MomentumSGD)')
    parser.add_argument('--weight_decay', type=float,
                        help='Weight decay for optimizer scheduling')
    parser.add_argument('--delete_tp', type=int,
                        help='Number of time point deleted')
    parser.add_argument('--device',
                        help='GPU ID (negative value indicates CPU')
    parser.add_argument('--phase',
                        help='Specify mode (train, test)')
    parser.add_argument('--delete_variable',
                        help='Specify list of index of variable')
    args, remaining_argv = parser.parse_known_args(remaining_argv)

    return parser, args, remaining_argv
