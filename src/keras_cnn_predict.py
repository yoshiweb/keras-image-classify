#!/usr/bin/env python

import numpy as np
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from PIL import Image
import sys

class ImageClassify:




    def predict(self, x):
        '''
        判別する
        '''

        # 学習済みモデルの読み込み
        model = load_model('keras_cnn_model.h5')
        # model.summary()

        # 学習したモデルに対して予測を行う
        y = model.predict(x)
        print(y)

        print('')

        for result in y:

            print('各ラベルの確率')
            for num in result:
                print('%.10f' % num) # 指数表記を読めるようにする

            # 一番大きい番号を出力
            print('一番大きいラベルを予測結果として判定')
            print(np.argmax(result))
            print('')



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
            img = img.convert("RGB") # RGB
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
        print("引数に判別する画像のパスを記載してください")
        sys.exit(1)

    fileList = sys.argv[1:]

    img = ImageClassify()

    # 画像を読み込んで配列にする
    x = img.loadImages(fileList)

    # 予測
    img.predict(x)


