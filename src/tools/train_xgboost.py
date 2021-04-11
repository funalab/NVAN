#!/usr/bin/env python

import os
import json
import pytz
from copy import copy
from glob import glob
from datetime import datetime
import argparse
import xgboost as xgb
import pandas as pd
import sklearn as sk
import matplotlib.pyplot as plt

from statistics import mean
from statistics import stdev


def getPartialMatrix( vPd, vRank, colNum = 10 , pc = 0):
    vRankSr = vRank.iloc[ :, pc ].abs().sort_values(ascending = False).index
    new_vPd = vPd[ vRankSr[ 0:colNum]  ]
    return( new_vPd )


def main():
    ap = argparse.ArgumentParser(description='python train_xgboost.py')
    ap.add_argument('--root', '-r', nargs='?', default='/home/tokuoka/git/embryo_classification', help='Specify root path')
    ap.add_argument('--save_dir', nargs='?', default='results/train_xgboost', help='Specify output files directory for create figures')
    ap.add_argument('--set', type=int, default=1, help='Specify index of set in MCCV')
    ap.add_argument('--input', nargs='?', default='RAW', help='Specify input type [RAW, PCA]')
    args = ap.parse_args()
    
    allData = pd.read_csv(os.path.join(args.root, 'datasets', 'tree', 'learningInput.csv'))
    allData = allData[ allData.learningInput != 'no_pups']
    vRank = pd.read_csv(os.path.join(args.root, 'datasets', 'tree', 'PCARotation.csv'), index_col=0)

    split_list_train = os.path.join(args.root, 'datasets', 'split_list', 'mccv', 'set{}'.format(args.set), 'train.txt')
    split_list_val = os.path.join(args.root, 'datasets', 'split_list', 'mccv', 'set{}'.format(args.set), 'validation.txt')
    split_list_test = os.path.join(args.root, 'datasets', 'split_list', 'mccv', 'set{}'.format(args.set), 'test.txt')
    with open(split_list_train, 'r') as f:
        file_list_train = [line.rstrip() for line in f]
    with open(split_list_val, 'r') as f:
        file_list_val = [line.rstrip() for line in f]
    with open(split_list_test, 'r') as f:
        file_list_test = [line.rstrip() for line in f]

    trainData, valData, testData = copy(allData), copy(allData), copy(allData)
    for i in file_list_val + file_list_test:
        trainData = trainData[trainData.name != i]
    for i in file_list_train + file_list_test:
        valData = valData[valData.name != i]
    for i in file_list_train + file_list_val:
        testData = testData[testData.name != i]

    x_train = trainData.iloc[:,:(trainData.shape[1]-1)]
    y_train = trainData['learningInput'].str.replace('born', '0').str.replace('abort', '1').astype(int)
    x_val = valData.iloc[:,:(valData.shape[1]-1)]
    y_val = valData['learningInput'].str.replace('born', '0').str.replace('abort', '1').astype(int)
    x_test = testData.iloc[:,:(testData.shape[1]-1)]
    y_test = testData['learningInput'].str.replace('born', '0').str.replace('abort', '1').astype(int)

    # Make Directory
    current_datetime = datetime.now(pytz.timezone('Asia/Tokyo')).strftime('%Y%m%d_%H%M%S')
    save_dir = args.save_dir + '_set{}_'.format(args.set) + str(current_datetime)
    os.makedirs(save_dir, exist_ok=True)

    numRank = [100,200,300,400,500,600,700,800,900,1000]
    numDepth = [1,2,3,4,5]
    best_rank = 0
    best_depth = 0
    best_f1_val = 0
    log_iteration = {}
    for r in numRank:
        for d in numDepth:
            partialx_train = getPartialMatrix(x_train, vRank, r, 0)
            partialx_val = getPartialMatrix(x_val, vRank, r, 0)

            eval_set = [(partialx_train, y_train),(partialx_val, y_val)]
            eval_metric = ["logloss"]

            clf = xgb.XGBClassifier(n_estimators=10000, learning_rate=0.001, max_depth=d, gamma=0, subsample=1.0)
            clf.fit(partialx_train, y_train, eval_metric=eval_metric, eval_set=eval_set, early_stopping_rounds=100, verbose=False)
    
            clf_best = xgb.XGBClassifier(n_estimators=clf.get_booster().best_iteration, learning_rate=0.001, max_depth=d, gamma=0, subsample=1.0)
            clf_best.fit(partialx_train, y_train, eval_metric=eval_metric , eval_set=eval_set, verbose=False)

            y_train_pred = clf_best.predict(partialx_train)
            y_val_pred = clf_best.predict(partialx_val)

            acc_train = sk.metrics.accuracy_score(y_train , y_train_pred)
            acc_val = sk.metrics.accuracy_score(y_val , y_val_pred)
            pre_val = sk.metrics.precision_score(y_val , y_val_pred, pos_label=1)
            rec_val = sk.metrics.recall_score(y_val , y_val_pred, pos_label=1)
            f1_val = sk.metrics.f1_score(y_val , y_val_pred, pos_label=1)

            log = {'rank': r, 'depth': d, 'accuracy_train': acc_train, 'accuracy_validation': acc_val,
                   'precision_validation': pre_val, 'recall_validation': rec_val, 'f1_validation': f1_val}
            with open(os.path.join(save_dir, 'log'), 'a') as f:
                json.dump(log, f, indent=4)
            if best_f1_val <= f1_val:
                print('Updated best model.')
                best_f1_val = f1_val
                best_rank = r
                best_depth = d
                clf_best_best = clf_best

    log = {'best f1 validation': best_f1_val, 'best rank': best_rank, 'best depth': best_depth, 'best iteration': clf_best_best.get_booster().best_iteration}
    with open(os.path.join(save_dir, 'best_result'), 'a') as f:
        json.dump(log, f, indent=4)
    print(log)
    
    partialx_test = getPartialMatrix(x_test, vRank, best_rank, 0)
    clf_best_best.predict(partialx_test)
    y_test_pred = clf_best_best.predict(partialx_test)
    acc_test = sk.metrics.accuracy_score(y_test, y_test_pred)
    pre_test = sk.metrics.precision_score(y_test, y_test_pred, pos_label=1)
    rec_test = sk.metrics.recall_score(y_test, y_test_pred, pos_label=1)
    f1_test = sk.metrics.f1_score(y_test, y_test_pred, pos_label=1)

    TP, TN, FP, FN = 0, 0, 0, 0
    for i in range(len(y_test)):
        if y_test.iloc[i] == y_test_pred[i]:
            if y_test.iloc[i] == 1:
                TP += 1
            elif y_test.iloc[i] == 0:
                TN += 1
        else:
            if y_test.iloc[i] == 1:
                FN += 1
            elif y_test.iloc[i] == 0:
                FP += 1

    log = {'accuracy': acc_test,
           'precision': pre_test,
           'recall': rec_test,
           'f1': f1_test,
           'TP': TP,
           'TN': TN,
           'FP': FP,
           'FN': FN,
           'AUROC': 0.0,
           'AUPR': 0.0
    }
    with open(os.path.join(save_dir, 'test_result'), 'a') as f:
        json.dump(log, f, indent=4)
    print('y_test: {}'.format(list(y_test)))
    print('y_pred: {}'.format(list(y_test_pred)))
    print(log)

if __name__ == '__main__':
    main()
