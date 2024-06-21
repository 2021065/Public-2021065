import tensorflow as tf
from tensorflow import keras
import numpy as np
from keras import backend as K
from keras.models import Model
from keras.utils import load_img, img_to_array

IMAGE_SIZE = 128

def r_loss(y_true, y_pred):
  return K.mean(K.square(y_true - y_pred), axis=[1,2,3])

filename = 'F:/国土地理院/org/1724.png'

# 画像ファイルパスから読み込み
input_img = load_img(filename, color_mode = 'rgb' ,target_size=(IMAGE_SIZE, IMAGE_SIZE))

# numpy配列の取得
input_img = np.asarray(input_img)
img = input_img[np.newaxis ,:, :, :]
print(img.shape)

Encoder_model = keras.models.load_model("./model/ae_encoding1", custom_objects={"r_loss": r_loss })
Encoder_model.summary()

Metric_model = keras.models.load_model("./model/ML1")
Metric_model.summary()

layer_name = 'encoder_output'
intermediate_layer_model = Model(inputs=Encoder_model.input,
                                 outputs=Encoder_model.get_layer(layer_name).output)

_feature_vector = intermediate_layer_model.predict(img)

feature_vector = Metric_model.predict(_feature_vector)

print(feature_vector.shape)
print(feature_vector)

cls0 = np.load('./result_VAE_ML_KM/cls0.npy')
cls1 = np.load('./result_VAE_ML_KM/cls1.npy')
cls2 = np.load('./result_VAE_ML_KM/cls2.npy')
cls3 = np.load('./result_VAE_ML_KM/cls3.npy')

def cos_sim(X,Y):
    return np.dot(X,Y)/(np.linalg.norm(X)*np.linalg.norm(Y))

s = []

s[0] = cos_sim(feature_vector[0],cls0[0])
print(s0)
s[1] = cos_sim(feature_vector[0],cls1[0])
print(s1)
s[2] = cos_sim(feature_vector[0],cls2[0])
print(s2)
s[3] = cos_sim(feature_vector[0],cls3[0])
print(s3)