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
```

> [[  3.21912841e-04   3.79991432e-15   9.99678135e-01]] ← 各ラベルの予測結果  
> 2  ← 検定結果

この場合「みかん」の可能性が一番高いと判断した。
