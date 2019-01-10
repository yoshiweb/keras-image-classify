import numpy as np
import os
import keras
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from sklearn import model_selection
from PIL import Image

batch_size = 10
epochs = 25
image_w = 32
image_h = 32


# 画像パス一覧
pathList = []

# ラベル一覧
labelList = []


# ディレクトリ参照
path = 'img'
files = os.listdir(path)

# ディレクトリ一覧を取得
files_dir = [f for f in files if os.path.isdir(os.path.join(path, f))]
print(files_dir)    # ['apple', 'banana', 'orange']

# クラス数
num_classes = len(files_dir)





f = open('label.txt', 'w') # 書き込みモードで開く

for label_num in range(num_classes):
    dir_name = files_dir[label_num]
    dir_path = os.path.join(path, dir_name)

    f.write(dir_name+'\n')

    # ファイル一覧を取得
    dir_files = os.listdir(dir_path)
    img_files = [fi for fi in dir_files if os.path.isfile(os.path.join(dir_path, fi))]
    # print(img_files)   # ['01.jpg', '02.jpg', '03.jpg']

    for file in img_files:

        if(file[-4:] == '.jpg' or file[-4:] == '.png'):     #ファイル名の後ろ4文字を取り出してそれが.jpgなら
            img_path = os.path.join(dir_path, file)
            pathList.append(img_path)
            labelList.append(label_num)
f.close() # ファイルを閉じる

#print(pathList)



X = []
Y = []

for src in pathList :
    # 画像読み込み
    img = Image.open(src) # PIL (Pillow(Python Imaging Library))で開く
    img = img.convert('RGB') # RGB
    img = img.resize((image_w, image_h)) # リサイズ
    data = np.asarray(img) # numpyのarrayにする
    X.append(data)

for label in labelList :
    Y.append(label)

X = np.array(X)
Y = np.array(Y)



# 訓練データとテストデータに分ける
x_train, x_test, y_train, y_test = model_selection.train_test_split(
    # 入力, 正解, 割合
    X, Y, test_size=0.3
)
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()

model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

# initiate RMSprop optimizer
opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

# Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255


# 学習開始
history = model.fit(x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(x_test, y_test),
            shuffle=True)

# 学習済みのモデルの保存
model.save('keras_cnn_model.h5')







def plot_history(history):
    # print(history.history.keys())

    # 精度の履歴をプロット
    plt.figure()
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(['acc', 'val_acc'], loc='lower right')
    plt.savefig('keras_cnn_train_1.png')
    # plt.show()

    # 損失の履歴をプロット
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['loss', 'val_loss'], loc='upper right')
    plt.savefig('keras_cnn_train_2.png')
    # plt.show()

# 学習履歴をプロット
plot_history(history)
