import os
import json
import time
import numpy as np
from sklearn import metrics

import torch
import torch.nn as nn


class Trainer(object):

    def __init__(self, **kwargs):
        self.optimizer = kwargs['optimizer']
        self.epoch = kwargs['epoch']
        self.save_dir = kwargs['save_dir']
        self.eval_metrics = kwargs['eval_metrics']
        self._best_accuracy = 0.0
        self.results = {}


    def train(self, model, train_iterator, validation_iterator):
        tester_args = {
            'save_dir' : self.save_dir
            }
        validator = Tester(**tester_args)
        start = time.time()

        for epoch in range(1, self.epoch + 1):
            print('[epoch {}]'.format(epoch))
            # turn on network training mode
            model.train()

            loss_train = self._train_step(model, train_iterator)
            eval_results, loss_train_2 = validator.test(model, train_iterator, phase="train")
            eval_results, loss_val = validator.test(model, validation_iterator, phase="validation")

            self._save_log(epoch, loss_train, loss_val, eval_results)

            if self.best_eval_result(eval_results):
                torch.save(model, os.path.join(self.save_dir, 'best_model'))
                print("Saved better model selected by validation.")
        print('best f1: {}'.format(self._best_accuracy))


    def _train_step(self, model, data_iterator):

        loss_list = []
        for batch in data_iterator:
            input, label = batch

            self.optimizer.zero_grad()
            logits = model(input)
            loss = model.loss(logits, label.view(len(label)))
            loss_list.append(loss.detach())
            loss.backward()
            self.optimizer.step()
        loss_train = float(abs(np.mean(loss_list)))
        print('train loss: {}'.format(loss_train))

        return loss_train


    def best_eval_result(self, eval_results):
        """Check if the current epoch yields better validation results.

        :param eval_results: dict, format {metrics_name: value}
        :return: bool, True means current results on dev set is the best.
        """
        assert self.eval_metrics in eval_results, \
            "Evaluation doesn't contain metrics '{}'." \
            .format(self.eval_metrics)

        accuracy = eval_results[self.eval_metrics]
        if accuracy >= self._best_accuracy:
            self._best_accuracy = accuracy
            return True
        else:
            return False


    def _save_log(self, epoch, loss_train, loss_val, eval_results):

        result_each_epoch = {}
        result_each_epoch['epoch'] = epoch
        result_each_epoch['loss_train'] = float(loss_train)
        result_each_epoch['loss_validation'] = float(loss_val)
        result_each_epoch['accuracy_validation'] = float(eval_results["accuracy"])
        result_each_epoch['precision_validation'] = float(eval_results["precision"])
        result_each_epoch['recall_validation'] = float(eval_results["recall"])
        result_each_epoch['f1_validation'] = float(eval_results["f1"])
        self.results[epoch] = result_each_epoch
        with open(os.path.join(self.save_dir, 'log'), 'w') as f:
            json.dump(self.results, f, indent=4)



class Tester(object):

    def __init__(self, **kwargs):
        self.save_dir = kwargs['save_dir']

    def test(self, model, data_iter, phase="test"):

        # turn on the testing mode; clean up the history
        model.eval()
        output_list = []
        truth_list = []
        loss_list = []

        # predict
        for batch in data_iter:
            input, label = batch

            with torch.no_grad():
                prediction = model(input)

            output_list.append(prediction.detach())
            truth_list.append(label.detach())
            loss = model.loss(prediction, label.view(len(label)))
            loss_list.append(loss.detach())

        # evaluate
        eval_results = self.evaluate(output_list, truth_list)
        print("[{}] {}, loss: {}".format(phase, self.print_eval_results(eval_results), abs(np.mean(loss_list))))

        return eval_results, abs(np.mean(loss_list))


    def evaluate(self, predict, truth):
        """Compute evaluation metrics.

        :param predict: list of Tensor
        :param truth: list of dict
        :return eval_results: dict, format {name: metrics}.
        """
        y_trues, y_preds = [], []
        for y_true, logit in zip(truth, predict):
            y_true = y_true.cpu().numpy()
            y_pred = [[np.argmax(l).cpu().numpy() for l in logit]]
            y_trues.append(y_true)
            y_preds.append(y_pred)
        y_true = np.concatenate(y_trues, axis=0)
        y_pred = np.concatenate(y_preds, axis=0).reshape(len(y_true), 1)

        accuracy = metrics.accuracy_score(y_true, y_pred)
        precision = metrics.precision_score(y_true, y_pred, pos_label=1)
        recall = metrics.recall_score(y_true, y_pred, pos_label=1)
        f1 = metrics.f1_score(y_true, y_pred, pos_label=1)

        metrics_dict = {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

        print('y_ture: {}'.format(np.array(y_true).reshape(len(y_true))))
        print('y_pred: {}'.format(np.array(y_pred).reshape(len(y_pred))))
        # print(metrics_dict)

        return metrics_dict


    def print_eval_results(self, results):
        """Override this method to support more print formats.
        :param results: dict, (str: float) is (metrics name: value)
        """
        return ", ".join(
            [str(key) + "=" + "{:.4f}".format(value)
             for key, value in results.items()])



class Predictor(object):

    def __init__(self):
        pass


    def predict(self, model, data_iterator):

        # turn on the testing mode; clean up the history
        model.eval()
        batch_output = []

        for batch in data_iterator:
            input = batch

            with torch.no_grad():
                prediction = model(*input)

            batch_output.append(prediction.detach())

        return self._post_processor(batch_output)


    def _post_processor(self, batch_output):
        """Convert logit tensor to label."""
        y_preds = []
        for logit in batch_output:
            y_pred = [[np.argmax(l).cpu().numpy() for l in logit]]
            y_preds.append(y_pred)
        y_pred = np.concatenate(y_preds, axis=0)

        return y_pred
