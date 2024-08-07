import random
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from collections import defaultdict
from PIL import Image
from sklearn.metrics import ConfusionMatrixDisplay
from tensorflow import keras
from keras import layers

from keras.datasets import cifar10
import keras
from keras import layers
from PIL import Image, ImageFilter
from keras.datasets import mnist
import numpy as np
import glob
from sklearn.model_selection import train_test_split
from keras.utils import load_img, img_to_array
import matplotlib.pyplot as plt
from keras.layers import Input, Conv2D, Flatten, Dense, Conv2DTranspose, Reshape, Activation, LeakyReLU, MaxPooling2D, UpSampling2D
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

LEARNING_RATE = 0.0004
BATCH_SIZE = 64
Z_DIM = 100
EPOCHS = 400

encoder_input = Input(shape=(IMAGE_SIZE,IMAGE_SIZE,3), name='encoder_input')
x = encoder_input
x = Conv2D(filters=3, kernel_size=3, strides=1, padding='same', name='encoder_conv_0')(x)
x = LeakyReLU()(x)
x = Conv2D(filters=16, kernel_size=3, strides=2, padding='same', name='encoder_conv_0_1')(x)
x = LeakyReLU()(x)
x = Conv2D(filters=32, kernel_size=3, strides=2, padding='same', name='encoder_conv_1')(x)
x = LeakyReLU()(x)
x = Conv2D(filters=64, kernel_size=3, strides=2, padding='same', name='encoder_conv_3')(x)
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
x = Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding='same', name='decoder_conv_t_0')(x)
x = LeakyReLU()(x)
x = Conv2DTranspose(filters=32, kernel_size=3, strides=2, padding='same', name='decoder_conv_t_2')(x)
x = LeakyReLU()(x)
x = Conv2DTranspose(filters=16, kernel_size=3, strides=2, padding='same', name='decoder_conv_t_2_6')(x)
x = LeakyReLU()(x)
x = Conv2DTranspose(filters=3, kernel_size=3, strides=1, padding='same', name='decoder_conv_t_3')(x)
x = Activation('sigmoid')(x)
decoder_output = x
decoder = Model(decoder_input, decoder_output)


# エンコーダ/デコーダ連結
model_input = encoder_input
model_output = decoder(encoder_output)
model = Model(model_input, model_output)

# 学習用設定設定（最適化関数、損失関数）
optimizer = Adam(learning_rate=LEARNING_RATE)

def r_loss(y_true, y_pred):
  return K.mean(K.square(y_true - y_pred), axis=[1,2,3])

model.compile(optimizer=optimizer, loss=r_loss, metrics=['accuracy'])

# 学習実行
history=model.fit(
    x_train,
    x_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(x_test, x_test),
)

model.summary()

model.save("./model/AE_100")

'''
x_train2 = encoder.predict(x_train)
print(x_train2.shape)
x_test2 = encoder.predict(x_test)
print(x_test2.shape)


#######################################

#y_train = np.squeeze(y_train)
#y_test = np.squeeze(y_test)


# Show a collage of 5x5 random images.
#sample_idxs = np.random.randint(0, 100, size=(5, 5))
#examples = x_train[sample_idxs]
examples = x_train
#show_collage(examples)

class_idx_to_train_idxs = defaultdict(list)
for y_train_idx, y in enumerate(y_train):
    #print(y_train)
    #print(y_train.shape)
    class_idx_to_train_idxs[y].append(y_train_idx)

class_idx_to_test_idxs = defaultdict(list)
for y_test_idx, y in enumerate(y_test):
    class_idx_to_test_idxs[y].append(y_test_idx)


num_classes = 4

class AnchorPositivePairs(keras.utils.Sequence):
    def __init__(self, num_batchs):
        self.num_batchs = num_batchs

    def __len__(self):
        return self.num_batchs

    def __getitem__(self, _idx):
        #x = np.empty((2, num_classes, 64,64,50), dtype=np.float32)
        x = np.empty((2, num_classes, Z_DIM), dtype=np.float32)
        for class_idx in range(num_classes):
            examples_for_class = class_idx_to_train_idxs[class_idx]
            anchor_idx = random.choice(examples_for_class)
            positive_idx = random.choice(examples_for_class)
            while positive_idx == anchor_idx:
                positive_idx = random.choice(examples_for_class)
            x[0, class_idx] = x_train2[anchor_idx]
            x[1, class_idx] = x_train2[positive_idx]
        return x
    
#examples = next(iter(AnchorPositivePairs(num_batchs=1)))

#show_collage(examples)

class EmbeddingModel(keras.Model):
    def train_step(self, data):
        # Note: Workaround for open issue, to be removed.未解決の問題の回避策。削除される予定です。
        if isinstance(data, tuple):
            data = data[0]
        anchors, positives = data[0], data[1]

        with tf.GradientTape() as tape:
            # Run both anchors and positives through model.
            # モデルを通してアンカーとポジティブの両方を実行します。
            anchor_embeddings = self(anchors, training=True)
            positive_embeddings = self(positives, training=True)

            # Calculate cosine similarity between anchors and positives. As they have
            # been normalised this is just the pair wise dot products.
            # アンカーとポジティブの間のコサイン類似度を計算します。彼らがそうしているように
            # 正規化されているため、これは単なるペアごとの内積です。
            similarities = tf.einsum(
                "ae,pe->ap", anchor_embeddings, positive_embeddings
            )

            # Since we intend to use these as logits we scale them by a temperature.
            # This value would normally be chosen as a hyper parameter.
            # これらをロジットとして使用するつもりなので、温度によってスケールします。
            # この値は通常、ハイパーパラメータとして選択されます。
            temperature = 0.2
            similarities /= temperature

            # We use these similarities as logits for a softmax. The labels for
            # this call are just the sequence [0, 1, 2, ..., num_classes] since we
            # want the main diagonal values, which correspond to the anchor/positive
            # pairs, to be high. This loss will move embeddings for the
            # anchor/positive pairs together and move all other pairs apart.
            # これらの類似性をソフトマックスのロジットとして使用します。のラベル
            # この呼び出しはシーケンス [0, 1, 2, ..., num_classes] です。
            # アンカー/ポジティブに対応する主な対角値が必要です
            # ペア、高くなります。この損失により、エンベディングが移動します。
            # アンカー/ポジティブペアを一緒に固定し、他のすべてのペアを離します。
            sparse_labels = tf.range(num_classes)
            loss = self.compiled_loss(sparse_labels, similarities)

        # Calculate gradients and apply via optimizer.
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        # Update and return metrics (specifically the one for the loss value).
        self.compiled_metrics.update_state(sparse_labels, similarities)
        return {m.name: m.result() for m in self.metrics}



inputs2 = layers.Input(shape=(Z_DIM))
x2 = layers.Dense(units=25, activation='relu')(inputs2)
embeddings = layers.Dense(units=8, activation=None)(x2)
embeddings = tf.nn.l2_normalize(embeddings, axis=-1)

model2 = EmbeddingModel(inputs2, embeddings)

model2.summary()

model2.compile(
    optimizer=keras.optimizers.Adam(learning_rate=5e-3),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
)

history = model2.fit(AnchorPositivePairs(num_batchs=64), epochs=300)

model2.save("./model/ML1")

plt.plot(history.history["loss"])
plt.show()

near_neighbours_per_example = 4

embeddings = model2.predict(x_test2)
#print(embeddings.shape)
#print(embeddings[0])
#print(x_test2.shape)
gram_matrix = np.einsum("ae,be->ab", embeddings, embeddings)
near_neighbours = np.argsort(gram_matrix.T)[:, -(near_neighbours_per_example + 1) :]

num_collage_examples = 5

examples = np.empty(
    (
        num_collage_examples,
        near_neighbours_per_example + 1,
        #64,
        #64,
        Z_DIM,
    ),
    dtype=np.float32,
)
for row_idx in range(num_collage_examples):
    examples[row_idx, 0] = x_test2[row_idx]
    anchor_near_neighbours = reversed(near_neighbours[row_idx][:-1])
    for col_idx, nn_idx in enumerate(anchor_near_neighbours):
        examples[row_idx, col_idx + 1] = x_test2[nn_idx]

#show_collage(examples)

confusion_matrix = np.zeros((num_classes, num_classes))

# For each class.
for class_idx in range(num_classes):
    # Consider 10 examples.
    example_idxs = class_idx_to_test_idxs[class_idx][:10]
    for y_test_idx in example_idxs:
        # And count the classes of its near neighbours.
        for nn_idx in near_neighbours[y_test_idx][:-1]:
            nn_class_idx = y_test[nn_idx]
            confusion_matrix[class_idx, nn_class_idx] += 1

# Display a confusion matrix.
labels = [
    "cls1",
    "cls2",
    "cls3",
    "cls4",
]
disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=labels)
disp.plot(include_values=True, cmap="viridis", ax=None, xticks_rotation="vertical")
plt.show()


from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil

n_clusters=4
kmeans_model = KMeans(n_clusters)
predict_clus = kmeans_model.fit_predict(embeddings)
print(predict_clus.shape)
print(predict_clus[0])

for i in range(20):
  ax = plt.subplot(4, n, i + 1)
  plt.imshow(x_test[[i]].reshape(image_size, image_size, 3))
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)
  ax.set_title(predict_clus[i])

plt.show()

OUTPUT_DIR = "F:/python_code/result_VAE_ML_KM"
for i in range(n_clusters):
    cluster_dir = OUTPUT_DIR + "/cluster{}".format(i)
    if os.path.exists(cluster_dir):
        shutil.rmtree(cluster_dir)
    os.makedirs(cluster_dir)
# 結果をクラスタごとにディレクトリに保存

for label, p in zip(predict_clus, x_test):
    shutil.copyfile(p, OUTPUT_DIR + '/cluster{}/{}'.format(label, p.split('/')[-1]))


cls0=[]
cls1=[]
cls2=[]
cls3=[]

print(embeddings.shape)
print(embeddings[[0]])
emb0=[]
emb1=[]
emb2=[]
emb3=[]

i=0
for label in predict_clus:
   im=one_depreprocess(x_test[[i]])
   if label == 0:
      cv2.imwrite(OUTPUT_DIR + '/cluster{}/{}.png'.format(label, i),im)
      cls0.append(x_test[[i]])
      emb0.append(embeddings[[i]])
   elif label == 1:
      cv2.imwrite(OUTPUT_DIR + '/cluster{}/{}.png'.format(label, i),im)
      cls1.append(x_test[[i]])
      emb1.append(embeddings[[i]])
   elif label == 2:
      cv2.imwrite(OUTPUT_DIR + '/cluster{}/{}.png'.format(label, i),im)
      cls2.append(x_test[[i]])
      emb2.append(embeddings[[i]])
   elif label == 3:
      cv2.imwrite(OUTPUT_DIR + '/cluster{}/{}.png'.format(label, i),im)
      cls3.append(x_test[[i]])
      emb3.append(embeddings[[i]])
   elif label == 4:
      cv2.imwrite(OUTPUT_DIR + '/cluster{}/{}.png'.format(label, i),im)
   i+=1

for i in range(5):
  ax = plt.subplot(4, n, i + 1)
  plt.imshow(cls0[i].reshape(image_size, image_size, 3))
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)
  ax.set_title("cluster0")

  ax = plt.subplot(4, n, i + 1 + n)
  plt.imshow(cls1[i].reshape(image_size, image_size, 3))
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)
  ax.set_title("cluster1")

  ax = plt.subplot(4, n, i + 1 + n*2)
  plt.imshow(cls2[i].reshape(image_size, image_size, 3))
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)
  ax.set_title("cluster2")

  ax = plt.subplot(4, n, i + 1 + n*3)
  plt.imshow(cls3[i].reshape(image_size, image_size, 3))
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)
  ax.set_title("cluster3")
plt.show()

emb0=np.array(emb0)
emb1=np.array(emb1)
emb2=np.array(emb2)
emb3=np.array(emb3)

centroids0 = []
centroids1 = []
centroids2 = []
centroids3 = []

df_mean0 = np.average(emb0,axis=0)
np.save('./result_VAE_ML_KM/cls0.npy', df_mean0)
print(df_mean0)

df_mean1 = np.average(emb1,axis=0)
np.save('./result_VAE_ML_KM/cls1.npy', df_mean1)
print(df_mean1)

df_mean2 = np.average(emb2,axis=0)
np.save('./result_VAE_ML_KM/cls2.npy', df_mean2)
print(df_mean2)

df_mean3 = np.average(emb3,axis=0)
np.save('./result_VAE_ML_KM/cls3.npy', df_mean3)
print(df_mean3)

'''