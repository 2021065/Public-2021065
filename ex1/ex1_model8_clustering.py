#https://keras.io/examples/generative/vq_vae/
import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras
import tensorflow as tf

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
import cv2

import random
import matplotlib.pyplot as plt
from collections import defaultdict
import pickle

IMAGE_SIZE = 256
Z_DIM=1000

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

train_data_path = '/home/takanolab/proglams_python/dataset/data_val' # ここを変更。Colaboratoryにアップロードしたzipファイルを解凍後の、データセットのフォルダ名を入力

image_size = IMAGE_SIZE # ここを変更。必要に応じて変更してください。「28」を指定した場合、縦28横28ピクセルの画像に変換します。

color_setting = 3  #ここを変更。データセット画像のカラー：「1」はモノクロ・グレースケール。「3」はカラー。

folder = ['hatake_val','kawa_val','mori_val','tatemono_val'] # ここを変更。データセット画像のフォルダ名（クラス名）を半角英数で入力

class_number = len(folder)
print('今回のデータで分類するクラス数は「', str(class_number), '」です。')


#3 データセットの読み込みとデータ形式の設定・正規化・分割 

X_image = []  
Y_label = []

dir_list = []

for index, name in enumerate(folder):
  read_data = train_data_path + '/' + name
  files = glob.glob(read_data + '/*.png') #ここを変更。png形式のファイルを利用する場合のサンプルです。
  f = open('/home/takanolab/proglams_python/ex3/Image_path/label{}.txt'.format(index), 'wb')
  pickle.dump(files, f)
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

#X_image = X_image.astype('float32') / 255
print(len(X_image))


#↓ここの部分をコメントアウトする
#Y_label = keras.utils.to_categorical(Y_label, class_number) #Kerasのバージョンなどにより使えないのでコメントアウト
#Y_label = np_utils.to_categorical(Y_label, class_number) #上記のコードのかわり
'''
train_images, valid_images ,train_labels ,valid_labels = train_test_split(X_image,Y_label,test_size=0.20,shuffle = True)
x_train = train_images
y_train = train_labels
x_test = valid_images
y_test = valid_labels
'''
#train_images,train_labels = train_test_split(X_image,Y_label,shuffle = True)
x_test = X_image
y_test = Y_label

y_test_count=np.zeros(class_number)
for count,i in enumerate(y_test):
   y_test_count[i] +=1 

latent_dim=16
'''
class VectorQuantizer(layers.Layer):
    def __init__(self, num_embeddings, embedding_dim, beta=0.25, **kwargs):
        super().__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings

        # The `beta` parameter is best kept between [0.25, 2] as per the paper.
        self.beta = beta

        # Initialize the embeddings which we will quantize.
        w_init = tf.random_uniform_initializer()
        self.embeddings = tf.Variable(
            initial_value=w_init(
                shape=(self.embedding_dim, self.num_embeddings), dtype="float32"
            ),
            trainable=True,
            name="embeddings_vqvae",
        )

    def call(self, x):
        # Calculate the input shape of the inputs and
        # then flatten the inputs keeping `embedding_dim` intact.
        input_shape = tf.shape(x)
        flattened = tf.reshape(x, [-1, self.embedding_dim])

        # Quantization.
        encoding_indices = self.get_code_indices(flattened)
        encodings = tf.one_hot(encoding_indices, self.num_embeddings)
        quantized = tf.matmul(encodings, self.embeddings, transpose_b=True)

        # Reshape the quantized values back to the original input shape
        quantized = tf.reshape(quantized, input_shape)

        # Calculate vector quantization loss and add that to the layer. You can learn more
        # about adding losses to different layers here:
        # https://keras.io/guides/making_new_layers_and_models_via_subclassing/. Check
        # the original paper to get a handle on the formulation of the loss function.
        commitment_loss = tf.reduce_mean((tf.stop_gradient(quantized) - x) ** 2)
        codebook_loss = tf.reduce_mean((quantized - tf.stop_gradient(x)) ** 2)
        self.add_loss(self.beta * commitment_loss + codebook_loss)

        # Straight-through estimator.
        quantized = x + tf.stop_gradient(quantized - x)
        return quantized

    def get_code_indices(self, flattened_inputs):
        # Calculate L2-normalized distance between the inputs and the codes.
        similarity = tf.matmul(flattened_inputs, self.embeddings)
        distances = (
            tf.reduce_sum(flattened_inputs ** 2, axis=1, keepdims=True)
            + tf.reduce_sum(self.embeddings ** 2, axis=0)
            - 2 * similarity
        )

        # Derive the indices for minimum distances.
        encoding_indices = tf.argmin(distances, axis=1)
        return encoding_indices
    
def get_encoder(latent_dim):
    encoder_inputs = keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
    x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
    x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
    encoder_outputs = layers.Conv2D(latent_dim, 1, padding="same",name='encoding')(x)
    return keras.Model(encoder_inputs, encoder_outputs, name="encoder")


def get_decoder(latent_dim):
    latent_inputs = keras.Input(shape=get_encoder(latent_dim).output.shape[1:])
    x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(
        latent_inputs
    )
    x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
    decoder_outputs = layers.Conv2DTranspose(3, 3, padding="same")(x)
    return keras.Model(latent_inputs, decoder_outputs, name="decoder")

def get_vqvae(latent_dim=16, num_embeddings=64):
    vq_layer = VectorQuantizer(num_embeddings, latent_dim, name="vector_quantizer")

    inputs = keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
    x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(inputs)
    x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
    encoder_outputs = layers.Conv2D(latent_dim, 3, padding="same",name='encoding')(x)

    quantized_latents = vq_layer(encoder_outputs)

    x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(
        quantized_latents
    )
    x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
    reconstructions = layers.Conv2DTranspose(3, 3, padding="same")(x)
    return keras.Model(inputs, reconstructions, name="vq_vae")

class VQVAETrainer(keras.models.Model):
    def __init__(self, train_variance, latent_dim=32, num_embeddings=128, **kwargs):
        super().__init__(**kwargs)
        self.train_variance = train_variance
        self.latent_dim = latent_dim
        self.num_embeddings = num_embeddings

        self.vqvae = get_vqvae(self.latent_dim, self.num_embeddings)

        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.vq_loss_tracker = keras.metrics.Mean(name="vq_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.vq_loss_tracker,
        ]

    def train_step(self, x):
        with tf.GradientTape() as tape:
            # Outputs from the VQ-VAE.
            reconstructions = self.vqvae(x)

            # Calculate the losses.
            reconstruction_loss = (
                tf.reduce_mean((x - reconstructions) ** 2) / self.train_variance
            )
            total_loss = reconstruction_loss + sum(self.vqvae.losses)

        # Backpropagation.
        grads = tape.gradient(total_loss, self.vqvae.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.vqvae.trainable_variables))

        # Loss tracking.
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.vq_loss_tracker.update_state(sum(self.vqvae.losses))

        # Log results.
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "vqvae_loss": self.vq_loss_tracker.result(),
        }

data_variance = np.var(x_test / 255.0)
#x_train_scaled = (x_test / 255.0) - 0.5
'''
x_test_scaled = (x_test / 255.0) - 0.5

VQVAE_model = keras.models.load_model("/home/takanolab/proglams_python/ex3/model/ex3_model1_{}".format(Z_DIM))
VQVAE_model.summary()

intermediate_layer_model = Model(inputs=VQVAE_model.input,
                                 outputs=VQVAE_model.get_layer('encoding').output)

#x_train2 = intermediate_layer_model.predict(x_train_scaled)
_feature_vector = intermediate_layer_model.predict(x_test_scaled)
print(_feature_vector.shape)


#######################################
#y_train = np.squeeze(y_train)
#y_test = np.squeeze(y_test)


# Show a collage of 5x5 random images.
#sample_idxs = np.random.randint(0, 100, size=(5, 5))
#examples = x_train[sample_idxs]
#examples = x_train2
#show_collage(examples)
'''
class_idx_to_train_idxs = defaultdict(list)
for y_train_idx, y in enumerate(y_test):
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
        x = np.empty((2, num_classes, 64, 64, 16), dtype=np.float32)
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
    #train_step(self, data) メソッドだけをオーバーライドします。
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
            #　ランダムに選択されたものとその対応する同ラベルとの内積
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
        #勾配を計算し、オプティマイザーを介して適用します。
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        # Update and return metrics (specifically the one for the loss value).
        #メトリクス (特に損失値のメトリクス) を更新して返します。
        self.compiled_metrics.update_state(sparse_labels, similarities)
        return {m.name: m.result() for m in self.metrics}

inputs2 = layers.Input(shape=(64, 64, 16))
x2 = layers.Conv2D(filters=16, kernel_size=3, strides=2, activation="relu")(inputs2)
x2 = layers.Conv2D(filters=32, kernel_size=3, strides=2, activation="relu")(x2)
x2 = layers.GlobalAveragePooling2D()(x2)
embeddings = layers.Dense(units=Z_DIM, activation=None)(x2)
embeddings = tf.nn.l2_normalize(embeddings, axis=-1)

model2 = EmbeddingModel(inputs2, embeddings)


model2.compile(
    optimizer=keras.optimizers.Adam(learning_rate=2e-3),
    #one-hot 表現でラベルが作成されている場合は CategoricalCrossentropy を利用する
    #整数でラベルが作成されている場合は、SparseCategoricalCrossentropy を利用する
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
)

history = model2.fit(AnchorPositivePairs(num_batchs=64), epochs=600)
model2.summary()
model2.save("F:\python_code\ex3\model\ex3_model1_{}".format(Z_DIM))
'''

Metric_model = keras.models.load_model("/home/takanolab/proglams_python/ex3/model/ex3_model1M_{}".format(Z_DIM))
Metric_model.summary()

feature_vector = Metric_model.predict(_feature_vector)
#print(_feature_vector[0])
#print(feature_vector[0])

OUTPUT_DIR = "/home/takanolab/proglams_python/ex3/model/ex3_model1M_{}".format(Z_DIM)

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil

n_clusters=4
kmeans_model = KMeans(n_clusters)
predict_clus = kmeans_model.fit_predict(feature_vector)
print(predict_clus)
print(predict_clus[0])

n=5

for i in range(20):
  ax = plt.subplot(4, n, i + 1)
  plt.imshow(one_depreprocess(x_test_scaled[[i]]))
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)
  ax.set_title(predict_clus[i])

plt.show()


for i in range(n_clusters):
    cluster_dir = OUTPUT_DIR + "/cluster{}".format(i)
    if os.path.exists(cluster_dir):
        shutil.rmtree(cluster_dir)
    os.makedirs(cluster_dir)
# 結果をクラスタごとにディレクトリに保存
'''
for label, p in zip(predict_clus, x_test):
    shutil.copyfile(p, OUTPUT_DIR + '/cluster{}/{}'.format(label, p.split('/')[-1]))
'''
count_prediction = np.zeros(class_number)
acc = np.zeros((class_number, class_number))
for i,label in enumerate(predict_clus):
    im=one_depreprocess(x_test[[i]])
    cv2.imwrite(OUTPUT_DIR + '/cluster{}/{}.png'.format(label, i),im)
    count_prediction[label]+=1
    real_label = y_test[i]
    acc[label][real_label] += 1

print(count)
'''
for count2,i in enumerate(acc):
    print('============')
    for j in i:
        if j == 0:
            print('none')
        else:
            print('再現率label{}'.format(j),' =', acc[i][j]/y_test_count[i])
            print('適合率label{}'.format(j),'=', acc[i][j]/count[i])
        print('============')
'''
recall_rate = np.zeros((class_number,class_number))
precision_rate = np.zeros((class_number,class_number))

for count1, i in enumerate(acc):
  print('============')
  for count2, j in enumerate(i):
   if j == 0:
      print('none')
   else:
      recall_rate[count1][count2] = j/y_test_count[count2]
      print('再現率label',count2,'= {:.4f}'.format(recall_rate[count1][count2]))
      #print(j,'/'y_test_count[j]) #確認用
      precision_rate[count1][count2] = j/count_prediction[count1]
      print('適合率label',count2,'= {:.4f}'.format(recall_rate[count1][count2]))
      #print(j,'/',class_count[count]) #確認用
  print('============')


