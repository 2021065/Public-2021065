import keras
from keras import layers
from PIL import Image, ImageFilter
from keras.datasets import mnist
import numpy as np
import glob
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.utils import load_img, img_to_array
import matplotlib.pyplot as plt
from keras.layers import Input, Conv2D, Flatten, Dense, Conv2DTranspose, Reshape, Activation, LeakyReLU
from keras.models import Model
from keras import backend as K
from keras.optimizers import Adam
import numpy as np
import cv2

IMAGE_SIZE = 256

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

#2 各種設定 https://child-programmer.com/ai/cnn-originaldataset-samplecode/#_CNN_8211_ColaboratoryKerasPython

train_data_path = 'F:/国土地理院/26/data' # ここを変更。Colaboratoryにアップロードしたzipファイルを解凍後の、データセットのフォルダ名を入力

image_size = IMAGE_SIZE # ここを変更。必要に応じて変更してください。「28」を指定した場合、縦28横28ピクセルの画像に変換します。

color_setting = 3  #ここを変更。データセット画像のカラー：「1」はモノクロ・グレースケール。「3」はカラー。

folder = ['hatake','kawa','mori','tatemono'] # ここを変更。データセット画像のフォルダ名（クラス名）を半角英数で入力

class_number = len(folder)
print('今回のデータで分類するクラス数は「', str(class_number), '」です。')


#3 データセットの読み込みとデータ形式の設定・正規化・分割 

X_image = []  
Y_label = [] 

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

X_image = np.array(X_image)
Y_label = np.array(Y_label)

X_image = X_image.astype('float32') / 255
print(len(X_image))
Y_label = keras.utils.to_categorical(Y_label, class_number) #Kerasのバージョンなどにより使えないのでコメントアウト
#Y_label = np_utils.to_categorical(Y_label, class_number) #上記のコードのかわり

'''
#必要なライブラリのインポート
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator,array_to_img,img_to_array

#画像表示用の関数を定義
#https://www.codexa.net/data_augmentation_python_keras/
def show(datagen, img):
  #表示サイズを設定
  plt.figure(figsize = (10, 5))
  
  #画像をbatch_sizeの数ずつdataに入れる
  #本稿は画像が一枚のため同じ画像がdataに入り続けることになる
  for i, data in enumerate(datagen.flow(img, batch_size = 1, seed = 0)):
    #表示のためnumpy配列からimgに変換する
    show_img = array_to_img(data[0], scale = False)
    #2×3の画像表示の枠を設定＋枠の指定
    plt.subplot(2, 3, i+1)
    #軸を表示しない
    plt.xticks(color = "None")
    plt.yticks(color = "None")
    plt.tick_params(bottom = False, left = False)
    #画像を表示
    plt.imshow(show_img)
    #6回目で繰り返しを強制的に終了
    if i == 5:
      return
#画像配列の形
print(X_image.shape)
#配列に次元を追加
X_image=X_image[np.newaxis, :, :, :]
#次元追加後の配列の形
print(X_image.shape)

rotation_datagen = ImageDataGenerator(
  rotation_range = 45
  brightness_range = [0.3, 0.8]
)
#画像を表示
show(rotation_datagen, X_image)
'''

'''

train_images = glob.glob('F:/国土地理院/org')
train = []
for i in train_images:
    image = cv2.imread(i)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    train.append(image)

train = np.array(train)
x_train = train.astype('float32') / 255
'''
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

x_train = preprocess(x_train)
x_test = preprocess(x_test)

LEARNING_RATE = 0.0003
BATCH_SIZE = 32
Z_DIM = 100
EPOCHS = 100

class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

latent_dim = Z_DIM
#https://keras.io/examples/generative/vae/
encoder_input = keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(encoder_input)
x = layers.Conv2D(64, 3, activation="relu", strides=1, padding="same")(x)
x = layers.Conv2D(128, 3, activation="relu", strides=2, padding="same")(x)
x = layers.Flatten()(x)
x = layers.Dense(16, activation="relu")(x)
z_mean = layers.Dense(latent_dim, name="z_mean")(x)
z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
z = Sampling()([z_mean, z_log_var])
encoder_output=z
encoder = keras.Model(encoder_input, [z_mean, z_log_var, z], name="encoder")
encoder.summary()

# デコーダ
latent_input = keras.Input(shape=(latent_dim,))
x = layers.Dense(64 * 64 * 64, activation="relu")(latent_input)
x = layers.Reshape((64, 64, 64))(x)
x = layers.Conv2DTranspose(128, 3, activation="relu", strides=2, padding="same")(x)
x = layers.Conv2DTranspose(64, 3, activation="relu", strides=1, padding="same")(x)
x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
decoder_output = layers.Conv2DTranspose(3, 3, activation="sigmoid", padding="same")(x)
decoder = keras.Model(latent_input, decoder_output, name="decoder")
decoder.summary()

class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2)
                )
            )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

# エンコーダ/デコーダ連結
model_input = encoder_input
model_output = decoder(encoder_output)
model = VAE(encoder, decoder)

model.compile(optimizer=keras.optimizers.Adam())

# 学習実行
history=model.fit(
    x_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
)


predictions = np.array(encoder.predict(x_test))
predictions = np.array(Sampling()([predictions[0], predictions[1]]))
print(predictions.shape)
print(predictions[[0]])
predictions2 = np.array(decoder.predict(predictions))
print(predictions2.shape)
print(predictions2[[0]])
predictions2 = depreprocess(predictions2)
x_test = depreprocess(x_test)
print(x_test.shape)
print(x_test[[0]])

for i in range(5):
  ax = plt.subplot(2, n, i + 1)
  plt.imshow(x_test[[i]].reshape(image_size, image_size, 3))
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)
  ax.set_title(str(y_test[i]))

  ax = plt.subplot(2, n, i + 1 + n)
  plt.imshow(predictions2[[i]].reshape(image_size, image_size,3))
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)
plt.show()




from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil

n_clusters=4
kmeans_model = KMeans(n_clusters)
predict_clus = kmeans_model.fit_predict(predictions)
print(predict_clus.shape)
print(predict_clus[0])

for i in range(20):
  ax = plt.subplot(4, n, i + 1)
  plt.imshow(x_test[[i]].reshape(image_size, image_size, 3))
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)
  ax.set_title(predict_clus[i])

plt.show()



OUTPUT_DIR = "F:/python_code/VAE_result"
for i in range(n_clusters):
    cluster_dir = OUTPUT_DIR + "/cluster{}".format(i)
    if os.path.exists(cluster_dir):
        shutil.rmtree(cluster_dir)
    os.makedirs(cluster_dir)
# 結果をクラスタごとにディレクトリに保存
print()
i=0
for label in predict_clus:
   im=one_depreprocess(x_test[[i]])
   if label == 0:
      cv2.imwrite(OUTPUT_DIR + '/cluster{}/{}.png'.format(label, i),im)
   elif label == 1:
      cv2.imwrite(OUTPUT_DIR + '/cluster{}/{}.png'.format(label, i),im)
   elif label == 2:
      cv2.imwrite(OUTPUT_DIR + '/cluster{}/{}.png'.format(label, i),im)
   elif label == 3:
      cv2.imwrite(OUTPUT_DIR + '/cluster{}/{}.png'.format(label, i),im)
   elif label == 4:
      cv2.imwrite(OUTPUT_DIR + '/cluster{}/{}.png'.format(label, i),im)
   i+=1
