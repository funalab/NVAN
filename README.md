# NVAN (Normalized Multi-view Attention Network)

This is the code for [hoge](fuga).
This project is carried out in cooperation with [Funahashi Lab. at Keio University](https://fun.bio.keio.ac.jp/) and two labs: [Kobayashi Lab. at the University of Tokyo](http://research.crmind.net/) and Yamagata Lab. at Kindai University.


## Overview

Normalized Multi-view Attention Network (NVAN) performs the task of classification using multivariate time-series data as input.
The multivariate time-series data in this project was obtained from images segmented by [QCANet](https://github.com/funalab/QCANet).


## Performance


## Requirements

- [Python 3.6](https://www.python.org/downloads/)
- [PyTorch 1.5+](https://chainer.org/)
- [NumPy](http://www.numpy.org)
- [scikit-image](http://scikit-image.org/)
- [Matplotlib](https://matplotlib.org/)



## QuickStart

1. Download this repository by `git clone`.

   ```sh
   % git clone https://gitlab.com/funalab/embryo_classification.git
   ```

2. Install requirements.

   - CPU version
     ```sh
     % cd embryo_classification/
     % python -m venv venv
     % source ./venv/bin/activate
     % pip install -r requirements.txt
     ```

   - GPU version (requires Anaconda)
     ```sh
     % cd embryo_classification/
     % conda create --name nvan --file conda-spec-file.txt
     % conda activate nvan
     ```

3. Inference on example test dataset.

   To run NVAN, follow the commands below.
   If you want to use a GPU, specify `cuda:[GPU ID]` (`cuda:0` if GPU ID is `0`) in device variable of `confs/models/test_best.cfg`.
   ```sh
   % ./scripts/run_test.sh
   ```

   After running the command, the results will be generated in the `results/test_NVAN` directory.


## How to train and run NVAN

1. Train NVAN with example dataset

   Run the following command to train NVAN on the datasets/input_example dataset, which is time-series segmentation images acquired by QCANet into multivariate time series data.
   The training results will be generated in the results/train_NVAN_example directory.
   ```sh
   % ./scripts/run_train_example.sh
   ```

   The training conditions are described in `confs/models/train_example.cfg`.
   The details of the parameters are as follows.


    ```
    [Dataset]
    root_path                                   : Specify root directory path for training data.
    split_list_train                            : Specify the path of the file in which the input file name used for training is enumerated.
    split_list_validation                       : Specify the path of the file in which the input file name used for validation is enumerated.
    basename                                    : Specify the name of the directory where multivariate time series data is stored.

    [Model]
    model                                       : Specify model name {``NVAN''}.
    init_classifier                             : Initialize the classifier from given file.
    input_dim                                   : Specify dimensions of input multivariate.
    num_classes                                 : Specify number of label class. 
    num_layers                                  : Specify number of layers in LSTM.
    hidden_dim                                  : Specify number of hidden size in LSTM.
    base_ch                                     : Specify number of channels in convolution layer.
    dropout                                     : Specify the dropout rate for fully connected layers.
    lossfun                                     : Specify loss function.
    eval_metrics                                : Specify the metrics to be used to determine the convergence of learning.

    [Runtime]
    save_dir                                    : Specify output files directory where classification and model file will be stored.
    batchsize                                   : Specify minibatch size in training.
    val_batchsize                               : Specify minibatch size in validation.
    epoch                                       : Specify the number of sweeps over the dataset to train.
    optimizer                                   : Specify optimizer {``SGD'', ``Adadelta'', ``Adagrad'', ``Adam'', ``AdamW'', ``SparseAdam'', ``Adamax'', ``ASGD'', ``RMSprop''}.
    lr                                          : Specify initial learning rate for optimizer.
    momentum                                    : Specify momentum for optimizer.
    weight_decay                                : Specify weight decay (L2 norm) for optimizer.
    delete_tp                                   : Specify the number of time points to randomly delete the length of the input time series data. (parameter for augmentation in training.)
    device                                      : Specify `cpu' or `cuda:[GPU ID]' (`cuda:0` if GPU ID is `0`)
    seed                                        : Specify the seed of the random number.
    phase                                       : Specify the phase {``train'' or ``test''}
    graph                                       : Specify `True` to generate a computational graph.
    ```

2. Run inference in trained NVAN
   The trained NVAN is `results/train_NVAN_example/best_model.npz`, which was generated in the previous step.
   Specify this file path as `init_classifier` in `confs/models/test_example.cfg`. After that, you can run the following command to infer the learned NVAN.
   The results of the inference will be generated in the `results/test_NVAN_example` directory.
   ``sh
   % ./scripts/run_test_example.sh
   ``


## Acknowledgement

The microscopic images used to generate multivariate time-series data included in this repository is provided by Yamagata Lab., Kindai University.
The development of this algorithm was funded by JSPS KAKENHI Grant Numbers XXX to [Akira Funahashi](https://github.com/funasoul).