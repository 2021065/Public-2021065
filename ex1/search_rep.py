import tensorflow as tf
from tensorflow import keras
import numpy as np
from keras import backend as K
from keras.models import Model
from keras.utils import load_img, img_to_array

IMAGE_SIZE = 256
Z_DIM = 100

npz_ = np.load("/home/takanolab/proglams_python/ex1/feature/ex1_model5_{}_test".format(Z_DIM))

def cos_sim(X,Y):
    return np.dot(X,Y)/(np.linalg.norm(X)*np.linalg.norm(Y))

def r_loss(y_true, y_pred):
  return K.mean(K.square(K.log(y_true + 1) - K.log(y_pred + 1)), axis=-1)

Encoder_model = keras.models.load_model("/home/takanolab/proglams_python/ex1/model/ex1_model1_{}".format(Z_DIM), custom_objects={"r_loss": r_loss })
Encoder_model.summary()


Metric_model = keras.models.load_model("/home/takanolab/proglams_python/ex1/model/ex1_model5_{}".format(Z_DIM))
Metric_model.summary()

layer_name = 'leaky_re_lu_3'
intermediate_layer_model = Model(inputs=Encoder_model.input,
                                 outputs=Encoder_model.get_layer(layer_name).output)

_feature_vector = intermediate_layer_model.predict()

feature_vector = Metric_model.predict()
