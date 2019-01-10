#!/usr/bin/env python

import numpy as np
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from PIL import Image
import sys, json

class ImageClassify:




  def predict(self, x, path):
    '''
    判別する
    '''

    # 学習済みモデルの読み込み
    model = load_model(path)
    # model.summary()

    # 学習したモデルに対して予測を行う
    y = model.predict(x)
    # print(y)

    # print('')

    # for result in y:

      # print('各ラベルの確率')
      # for num in result:
          # print('%.10f' % num) # 指数表記を読めるようにする

      # 一番大きい番号を出力
      # print('一番大きいラベルを予測結果として判定')
      # print(np.argmax(result))
      # print('')

    return y



  def loadLabels(self, path):
    '''
    ラベルを読み込む
    '''
    f = open( path )
    label_str = f.read()
    f.close()

    label_list = label_str.split('\n')
    return label_list


  def result2JSON(self, y, label_list):
    '''
    結果をJSON形式で返す
    '''

    # 辞書型にデータを格納
    json_dict = {}

    item_list = []


    for item in y:
      # print(item)
      item_dict = {}

      for i in range( len(item) ):
        label = label_list[i]   # ラベル名
        score = item[i]         # 確率
        score = '%.10f' % score # 指数表記を読めるようにする
        # print( label, score )

        # 辞書型にデータを格納
        item_dict[label] = score

      # リストに追加
      item_list.append(item_dict)


    json_dict['status'] = 0
    json_dict['result'] = item_list


    # JSON形式に変換
    json_str = json.dumps(json_dict)
    # print(json_str)
    return json_str






  def loadImages(self, imgList):
    '''
    画像を読み込む
    '''

    # 画像サイズ
    image_w = 32
    image_h = 32

    x = []

    for i in range(len(imgList)):
      src = imgList[i]

      img = Image.open(src) # PIL (Pillow(Python Imaging Library))で開く
      img = img.convert('RGB') # RGB
      img = img.resize((image_w, image_h)) # リサイズ
      data = np.asarray(img) # numpyのarrayにする
      x.append(data)

    x = np.array(x) # 1次元の配列にする
    # 0〜255を0〜1.0に変換
    x = x.astype('float32')
    x /= 255

    return x




if __name__ == '__main__':

  # 引数から画像パスを取得
  if len(sys.argv) < 2:
    print('引数に判別する画像のパスを記載してください')
    sys.exit(1)

  fileList = sys.argv[1:]


  _DIR = ''
  _MODEL_PATH = _DIR + 'keras_cnn_model.h5'
  _LABEL_TXT_PATH = _DIR + 'label.txt'

  img = ImageClassify()

  # 画像を読み込んで配列にする
  x = img.loadImages(fileList)

  # 予測
  result = img.predict(x, _MODEL_PATH)

  # ラベル
  label_list = img.loadLabels(_LABEL_TXT_PATH)

  # 結果をJSON形式に変換
  json_str = img.result2JSON(result, label_list)

  # 出力
  print(json_str)

