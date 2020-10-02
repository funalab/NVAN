import time
import numpy as np
from sklearn import metrics

import torch
import torch.nn as nn


class Trainer(object):

    def __init__(self, **kwargs):
        self.optimizer = kwargs['optimizer']
        self.epoch = kwargs['epoch']
        self.save_model_path = kwargs['save_model_path']
        self.eval_metrics = kwargs['eval_metrics']
        self._best_accuracy = 0.0


    def train(self, model, train_iterator, validation_iterator):

        validator = Tester()
        start = time.time()

        for epoch in range(1, self.epoch + 1):
            # turn on network training mode
            model.train()

            # # init iterator
            # train_iterator.init_epoch()

            self._train_step(model, train_iterator)
            eval_results = validator.test(model, validation_iterator)

            if self.best_eval_result(eval_results):
                torch.save(model, self.save_model_path)
                print("Saved better model selected by validation.")


    def _train_step(self, model, data_iterator):

        for batch in data_iterator:
            input, label = batch

            self.optimizer.zero_grad()
            logits = model(input)
            loss = model.loss(logits, label.view(len(label)))
            # print(loss.detach())
            loss.backward()
            self.optimizer.step()


    def best_eval_result(self, eval_results):
        """Check if the current epoch yields better validation results.

        :param eval_results: dict, format {metrics_name: value}
        :return: bool, True means current results on dev set is the best.
        """
        assert self.eval_metrics in eval_results, \
            "Evaluation doesn't contain metrics '{}'." \
            .format(self.eval_metrics)

        accuracy = eval_results[self.eval_metrics]
        if accuracy > self._best_accuracy:
            self._best_accuracy = accuracy
            return True
        else:
            return False



class Tester(object):

    def __init__(self):
        pass


    def test(self, model, validation_iter):

        # turn on the testing mode; clean up the history
        model.eval()
        output_list = []
        truth_list = []

        # predict
        for batch in validation_iter:
            input, label = batch

            with torch.no_grad():
                prediction = model(input)

            output_list.append(prediction.detach())
            truth_list.append(label.detach())

        # evaluate
        eval_results = self.evaluate(output_list, truth_list)
        print("[tester] {}".format(self.print_eval_results(eval_results)))

        return eval_results


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
        y_pred = np.concatenate(y_preds, axis=0)

        precision = metrics.precision_score(y_true, y_pred, pos_label=0)
        recall = metrics.recall_score(y_true, y_pred, pos_label=0)
        f1 = metrics.f1_score(y_true, y_pred, pos_label=0)

        metrics_dict = {"precision": precision, "recall": recall, "f1": f1}

        print('y_ture: {}, y_pred: {}'.format(y_true, y_pred))
        print(metrics_dict)


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
