# Embryo Classification


### How to train model

1. Make datasets
以下のコマンドを実行してデータセットである多変量時系列データを生成する。
```sh
% python src/tools/extract.py --conf_file confs/datasets/make_dataset.cfg
```
データセットは、`datasets/input` 以下に生成される。

2. Training
以下のコマンドを実行して生成したデータセットを用いてモデルの学習を行う。
```sh
% python src/tools/train.py --conf_file confs/train_muvan.cfg
```
学習の結果は、`results` 以下に生成される `log` ファイルを参照。



### Graph可視化

1. trainのconfig fileにある`Graph`を`True`にしてtrain.pyを実行
2. https://netron.app/ にアクセスして `.onnx`ファイルをアップロード

#### LSTMMultiAttentionClassifier
![LSTM Multi-Attention Classifier](https://gitlab.com/funalab/embryo_classification/-/blob/master/images/graph_LSTMMultiAttentionClassifier.png)