# Kerasでオリジナル画像を学習／予測、分類

## 環境

- Python 3.6.1
- Keras 2.0.6
- Tensorflow 1.2.1

- 追加パッケージ

```
$ pip install h5py
$ pip install Pillow
$ pip install sklearn
```


## 事前準備

- 学習用の素材を準備（imgディレクトリに学習させたい画像をフォルダごとに格納）


### 例：りんご、バナナ、みかんを画像判別させるものを作りたい場合

> `img/apple/フォルダ`に「りんご」の画像をたくさんいれる  
> `img/banana/フォルダ`に「バナナ」の画像をたくさんいれる  
> `img/orange/フォルダ`に「みかん」の画像をたくさんいれる  


## 学習

```
$ python keras_cnn_train.py
```

以下の2ファイルが生成されます

- `keras_cnn_model.h5` 学習済みモデルデータ
- `label.txt` 判別するラベル名が順番に記載されてる

> apple  ← 1行目が判定結果の 0  
> banana ← 2行目が判定結果の 1  
> orange ← 3行目が判定結果の 2  




## 予測・分類

判別したい画像（例：test.jpg）を用意して実行

```
$ python keras_cnn_predict.py test.jpg
Using TensorFlow backend.
2018-01-30 18:15:45.143201: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.2 instructions, but these are available on your machine and could speed up CPU computations.
2018-01-30 18:15:45.143224: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX instructions, but these are available on your machine and could speed up CPU computations.
2018-01-30 18:15:45.143229: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX2 instructions, but these are available on your machine and could speed up CPU computations.
2018-01-30 18:15:45.143233: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use FMA instructions, but these are available on your machine and could speed up CPU computations.
[[  4.40120971e-08   4.55074513e-07   9.99999523e-01]]

各ラベルの確率
0.0000000440
0.0000004551
0.9999995232
一番大きいラベルを予測結果として判定
2
```

この場合「みかん」の可能性が一番高いと判断した。
