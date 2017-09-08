import numpy as np
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from PIL import Image
import sys

# 画像サイズ
image_w = 32
image_h = 32

# 引数から画像パスを取得
if len(sys.argv) != 2:
    print("引数に判別する画像のパスを記載してください")
    sys.exit(1)

filename = sys.argv[1]
print('input:', filename)

# 画像を読み込み
img = Image.open(filename) # PIL (Pillow(Python Imaging Library))で開く
img = img.convert("RGB") # RGB
img = img.resize((image_w, image_h)) # リサイズ

# numpyのarrayにする
x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)
x /= 255  # [0-255]の値を[0.0-1.0]に変換


# 学習済みモデルの読み込み
model = load_model('keras_cnn_model.h5')
model.summary()

# 学習したモデルに対して予測を行う
y = model.predict(x)
print(y)

# 一番大きい番号を出力
print(np.argmax(y))
