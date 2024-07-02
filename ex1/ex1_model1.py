import random
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from collections import defaultdict
from PIL import Image
from sklearn.metrics import ConfusionMatrixDisplay
from tensorflow import keras
from keras import layers
import keras
from PIL import Image, ImageFilter
import glob
from sklearn.model_selection import train_test_split
from keras.utils import load_img, img_to_array
import matplotlib.pyplot as plt
from keras.layers import Input, Conv2D, Flatten, Dense, Conv2DTranspose, Reshape, Activation, LeakyReLU, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import backend as K
from keras.optimizers import Adam
import cv2

#読み込む画像をどんなサイズで読み込むか
IMAGE_SIZE = 256

#######################################################################
#画像を配列にする
#Change Image to array

def preprocess(array):
    """
    Normalizes the supplied array and reshapes it into the appropriate format.
    """

    #array = array.astype("float32") / 255.0
    array = np.reshape(array, (len(array), IMAGE_SIZE, IMAGE_SIZE, 3))
    return array

def one_depreprocess(array):
    """
    Normalizes the supplied array and reshapes it into the appropriate format.
    """

    array = array.astype("float32") * 255.0
    array = np.reshape(array, (IMAGE_SIZE, IMAGE_SIZE, 3))
    return array

def depreprocess(array):
    """
    Normalizes the supplied array and reshapes it into the appropriate format.
    """

    #array = array.astype("float32") * 255.0
    array = np.reshape(array, (len(array), IMAGE_SIZE, IMAGE_SIZE, 3))
    return array
#######################################################################


#2 各種設定 https://child-programmer.com/ai/cnn-originaldataset-samplecode/#_CNN_8211_ColaboratoryKerasPython
#######################################################################
#フォルダ別をクラス別とみなして、画像集合を読み込む
#フォルダ名＝クラス名
train_data_path = '/home/takanolab/proglams_python/dataset/kokudo' # ここを変更。Colaboratoryにアップロードしたzipファイルを解凍後の、データセットのフォルダ名を入力
image_size = IMAGE_SIZE # ここを変更。必要に応じて変更してください。「28」を指定した場合、縦28横28ピクセルの画像に変換します。
color_setting = 3  #ここを変更。データセット画像のカラー：「1」はモノクロ・グレースケール。「3」はカラー。
folder = ['hatake','kawa','mori','tatemono'] # ここを変更。データセット画像のフォルダ名（クラス名）を半角英数で入力
class_number = len(folder)
print('今回のデータで分類するクラス数は「', str(class_number), '」です。')

#3 データセットの読み込みとデータ形式の設定・正規化・分割 
#################################################################################

#################################################################################
#訓練用と検証用のデータセットへと分割
X_image = []  
Y_label = []

label_count = np.zeros(class_number)
for index, name in enumerate(folder):
  read_data = train_data_path + '/' + name
  files = glob.glob(read_data + '/*.png') #ここを変更。png形式のファイルを利用する場合のサンプルです。
  print('--- 読み込んだデータセットは', read_data, 'です。')
  num=0
  for i, file in enumerate(files):
    if color_setting == 1:
      img = load_img(file, color_mode = 'grayscale' ,target_size=(image_size, image_size))  
    elif color_setting == 3:
      img = load_img(file, color_mode = 'rgb' ,target_size=(image_size, image_size))
    array = img_to_array(img)
    X_image.append(array)
    num +=1
    Y_label.append(index)
  print('index: ',index,' num:',num)
  label_count[index] = num

X_image = np.array(X_image)
Y_label = np.array(Y_label)

X_image = X_image.astype('float32') / 255
print(len(X_image))
#↓ここの部分をコメントアウトする
#Y_label = keras.utils.to_categorical(Y_label, class_number) #Kerasのバージョンなどにより使えないのでコメントアウト
#Y_label = np_utils.to_categorical(Y_label, class_number) #上記のコードのかわり

train_images, valid_images ,train_labels ,valid_labels = train_test_split(X_image,Y_label,test_size=0.20,shuffle = True)
x_train = train_images
y_train = train_labels
x_test = valid_images
y_test = valid_labels

n=5
for i in range(10):
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_train[[i]].reshape(image_size,image_size,3))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_title(str(y_train[i]))
plt.show()
######################################################################################

######################################################################################
#modelの定義
#definition model

LEARNING_RATE = 0.0003
BATCH_SIZE = 64
Z_DIM = 100
EPOCHS = 800

#https://qiita.com/michelle0915/items/28bc5b844bd0d7ab597b
encoder_input = Input(shape=(IMAGE_SIZE,IMAGE_SIZE,3), name='encoder_input')
x = encoder_input
x = Conv2D(filters=3, kernel_size=3, strides=1, padding='same', name='encoder_conv_0')(x)
x = LeakyReLU()(x)
x = Conv2D(filters=16, kernel_size=3, strides=2, padding='same', name='encoder_conv_1')(x)
x = LeakyReLU()(x)
x = Conv2D(filters=32, kernel_size=3, strides=2, padding='same', name='encoder_conv_2')(x)
x = LeakyReLU()(x)
x = Conv2D(filters=64, kernel_size=3, strides=2, padding='same', name='encoder_conv_3')(x)
x = LeakyReLU()(x)
x = Conv2D(filters=128, kernel_size=3, strides=2, padding='same', name='encoder_conv_4')(x)
x = LeakyReLU()(x)
#encoder_output2=Dense(Z_DIM, name='encoder_output')(x)
shape_before_flattening = K.int_shape(x)[1:]
x = Flatten()(x)
encoder_output = Dense(Z_DIM, name='encoder_output')(x)
encoder = Model(encoder_input, encoder_output)

# デコーダ
decoder_input = Input(shape=(Z_DIM,), name='decoder_input')
x = Dense(np.prod(shape_before_flattening))(decoder_input)
x = Reshape(shape_before_flattening)(x)
x = Conv2DTranspose(filters=128, kernel_size=3, strides=2, padding='same', name='decoder_conv_4')(x)
x = LeakyReLU()(x)
x = Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding='same', name='decoder_conv_3')(x)
x = LeakyReLU()(x)
x = Conv2DTranspose(filters=32, kernel_size=3, strides=2, padding='same', name='decoder_conv_2')(x)
x = LeakyReLU()(x)
x = Conv2DTranspose(filters=16, kernel_size=3, strides=2, padding='same', name='decoder_conv_1')(x)
x = LeakyReLU()(x)
x = Conv2DTranspose(filters=3, kernel_size=3, strides=1, padding='same', name='decoder_conv_0')(x)
x = Activation('sigmoid')(x)
decoder_output = x
decoder = Model(decoder_input, decoder_output)


# エンコーダ/デコーダ連結
model_input = encoder_input
model_output = decoder(encoder_output)
model = Model(model_input, model_output)

# 学習用設定設定（最適化関数、損失関数）
optimizer = Adam(learning_rate=LEARNING_RATE)

#https://keras.io/api/losses/regression_losses/#meansquarederror-class
def r_loss(y_true, y_pred):
  return K.mean(K.square(K.log(y_true + 1) - K.log(y_pred + 1)), axis=-1)
#def r_loss(y_true, y_pred):
#  return K.mean(K.square(y_true - y_pred), axis=[1,2,3])

model.compile(optimizer=optimizer, loss=r_loss, metrics=['accuracy'])
#model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 学習実行
history=model.fit(
    x_train,
    x_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(x_test, x_test),
)

#g_images = model.predict(x_train)
#loss_val=K.mean(K.square(x_train[0]-g_images[0]),axis=[1,2,3])
#print(x_train.shape)
#print(x_train[0].shape)
#loss_val=K.mean(K.square(x_train[0]-g_images[0]),axis=[0,1,2])
#print('=========================')
#print(loss_val)
#print('=========================')

model.summary()

model.save("/home/takanolab/proglams_python/ex1/model/ex1_model1_{}".format(Z_DIM))

feature = model.predict(x_train)
np.savez(
    "/home/takanolab/proglams_python/ex1/feature/ex1_model1_{}_train".format(Z_DIM),  # ファイル名
    feature,
    y_test
)

print(x_train.shape)
print(x_test.shape)