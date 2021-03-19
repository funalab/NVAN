import os
import json
import time
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt

import torch
import torch.nn as nn


class Trainer(object):

    def __init__(self, **kwargs):
        self.optimizer = kwargs['optimizer']
        self.epoch = kwargs['epoch']
        self.save_dir = kwargs['save_dir']
        self.eval_metrics = kwargs['eval_metrics']
        self._best_accuracy = 0.0
        self.device = kwargs['device']
        self.results = {}


    def train(self, model, train_iterator, validation_iterator):
        tester_args = {
            'save_dir' : self.save_dir,
            'file_list' : None,
            'device' : self.device,
            'criteria_list' : None
            }
        validator = Tester(**tester_args)
        start = time.time()
        best_epoch = 1

        for epoch in range(1, self.epoch + 1):
            print('[epoch {}]'.format(epoch))
            # turn on network training mode
            model.train()

            loss_train = self._train_step(model, train_iterator)
            eval_results, loss_val = validator.test(model, validation_iterator, phase="validation")

            self._save_log(epoch, loss_train, loss_val, eval_results)

            if self.best_eval_result(eval_results):
                torch.save(model.to('cpu'), os.path.join(self.save_dir, 'best_model'))
                model.to(torch.device(self.device))
                best_epoch = epoch
                print("Saved better model selected by validation.")
        with open(os.path.join(self.save_dir, 'best_result'), 'w') as f:
            json.dump({'best f-score': self._best_accuracy,
                        'best epoch': best_epoch}, f, indent=4)
        print('best f1: {}'.format(self._best_accuracy))
        print('best epoch: {}'.format(best_epoch))


    def _train_step(self, model, data_iterator):

        loss_list = []
        for batch in data_iterator:
            input, label = batch

            self.optimizer.zero_grad()
            logits = model(input.to(torch.device(self.device)))
            loss = model.loss(logits, label.to(torch.device(self.device)).view(len(label)))
            loss_list.append(loss.to(torch.device('cpu')).detach())
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
        self.file_list = kwargs['file_list']
        self.device = kwargs['device']
        self.criteria_list = kwargs['criteria_list']

    def test(self, model, data_iter, phase="test"):

        # turn on the testing mode; clean up the history
        model.eval()
        output_list = []
        attn_weights_list = []
        truth_list = []
        loss_list = []

        # predict
        for batch in data_iter:
            input, label = batch

            if phase == 'test':
                with torch.no_grad():
                    prediction, attn_weights = model(input.to(torch.device(self.device)))
                attn_weights_list.append(attn_weights.detach())
            else:
                with torch.no_grad():
                    prediction = model(input.to(torch.device(self.device)))

            output_list.append(prediction.to(torch.device('cpu')).detach())
            truth_list.append(label.detach())
            loss = model.loss(prediction, label.to(torch.device(self.device)).view(len(label)))
            loss_list.append(loss.to(torch.device('cpu')).detach())

        # evaluate
        eval_results = self.evaluate(output_list, truth_list)
        if phase == 'test':
            auroc, aupr = self.evaluate_auc(output_list, truth_list)
            self.vis_attn_weights(attn_weights_list, output_list, truth_list)
            eval_results['AUROC'] = auroc
            eval_results['AUPR'] = aupr

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

        return metrics_dict


    def evaluate_auc(self, predict, truth):
        """Compute evaluation metrics.

        :param predict: list of Tensor
        :param truth: list of dict
        :return eval_results: dict, format {name: metrics}.
        """
        y_trues, y_preds = [], []
        for y_true, logit in zip(truth, predict):
            y_true = y_true.cpu().numpy()
            y_pred = [[l.cpu().numpy()[1] for l in logit]]
            y_trues.append(y_true)
            y_preds.append(y_pred)
        y_true = np.concatenate(y_trues, axis=0).reshape(len(y_trues))
        y_pred = np.concatenate(y_preds, axis=0).reshape(len(y_preds))

        auroc = self._vis_auroc(y_pred, y_true)
        aupr = self._vis_aupr(y_pred, y_true)

        return auroc, aupr


    def _vis_auroc(self, y_pred, y_true):
        fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred)
        auroc = metrics.roc_auc_score(y_true, y_pred)
        plt.figure(figsize=(8, 8))
        plt.plot(fpr, tpr, marker='o', label='AUROC={:.4f}'.format(auroc))
        plt.xlabel('False positive rate', fontsize=18)
        plt.ylabel('True positive rate', fontsize=18)
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.legend(loc=4, fontsize=18)
        plt.savefig(os.path.join(self.save_dir, 'AUROC.pdf'))
        return auroc

    def _vis_aupr(self, y_pred, y_true):
        precision, recall, thresholds = metrics.precision_recall_curve(y_true, y_pred)
        aupr = metrics.average_precision_score(y_true, y_pred)
        precision = np.concatenate([np.array([0.0]), precision])
        recall = np.concatenate([np.array([1.0]), recall])
        plt.figure(figsize=(8, 8))
        plt.plot(precision, recall, marker='o', label='AUPR={:.4f}'.format(aupr))
        plt.xlabel('Recall', fontsize=18)
        plt.ylabel('Precision', fontsize=18)
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.legend(loc=3, fontsize=18)
        plt.savefig(os.path.join(self.save_dir, 'AUPR.pdf'))
        return aupr

    def vis_attn_weights(self, attn_weights_list, predict, truth):
        y_trues, y_preds = [], []
        cnt = 0
        for y_true, logit, aw in zip(truth, predict, attn_weights_list):
            y_true = y_true.cpu().numpy()
            y_pred = [[np.argmax(l).cpu().numpy() for l in logit]]
            aw = aw.squeeze(0).cpu().numpy()
            filename = os.path.join(self.save_dir, 'attention_weight_{}.pdf'.format(self.file_list[cnt]))
            self.make_heatmap(aw, y_pred, y_true, filename)
            cnt += 1


    def make_heatmap(self, aw, y_pred, y_true, filename):
        plt.figure(figsize=(12, 6))
        plt.imshow(aw, interpolation='nearest', vmin=aw.min(), vmax=aw.max(), cmap='jet', aspect=20)
        plt.ylim([-0.5, len(aw)-0.5])
        plt.yticks([i for i in range(len(aw)-1, -1, -1)], self.criteria_list)
        # plt.yticks([i for i in range(0, len(aw))], [i for i in range(len(aw), 0, -1)])
        plt.xlabel('time point')
        plt.title('pred={0}, gt={1}'.format('born' if y_pred[0][0] == 1 else 'abort', 'born' if y_true[0][0] == 1 else 'abort'))
        plt.colorbar()
        plt.savefig(filename)
        plt.close()


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
