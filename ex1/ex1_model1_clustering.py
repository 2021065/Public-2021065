import tensorflow as tf
from tensorflow import keras
import numpy as np
import glob
from keras import backend as K
from keras.models import Model
from keras.utils import load_img, img_to_array
from sklearn.model_selection import train_test_split
import cv2

def preprocess(array):
    """
    Normalizes the supplied array and reshapes it into the appropriate format.
    """

    array = array.astype("float32") / 255.0
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

IMAGE_SIZE = 256

#def r_loss(y_true, y_pred):
#  return K.mean(K.square(y_true - y_pred), axis=[1,2,3])
def r_loss(y_true, y_pred):
  return K.mean(K.square(K.log(y_true + 1) - K.log(y_pred + 1)), axis=-1)


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

#train_images,train_labels = train_test_split(X_image,Y_label,shuffle = True)
X_test = X_image
Y_test = Y_label

Z_DIM=1000

Encoder_model = keras.models.load_model("/home/takanolab/proglams_python/ex1/model/ex1_model1_{}".format(Z_DIM), custom_objects={"r_loss": r_loss })
Encoder_model.summary()

layer_name = 'encoder_output'

intermediate_layer_model = Model(inputs=Encoder_model.input,
                                 outputs=Encoder_model.get_layer(layer_name).output)

feature_vector = intermediate_layer_model.predict(X_test)

print(feature_vector.shape)

np.savez(
    "/home/takanolab/proglams_python/ex1/feature/ex1_model1_{}_test".format(Z_DIM),  # ファイル名
    feature_vector,
    Y_test
)

from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil

n_clusters=4
kmeans_model = KMeans(n_clusters)
predict_clus = kmeans_model.fit_predict(feature_vector)
print(predict_clus.shape)
print(predict_clus[0])

n=5

for i in range(20):
  ax = plt.subplot(4, n, i + 1)
  plt.imshow(X_test[[i]].reshape(image_size, image_size, 3))
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)
  ax.set_title(predict_clus[i])

plt.show()



OUTPUT_DIR = "/home/takanolab/proglams_python/result/ex1_result_{}".format(Z_DIM)
for i in range(n_clusters):
    cluster_dir = OUTPUT_DIR + "/cluster{}".format(i)
    if os.path.exists(cluster_dir):
        shutil.rmtree(cluster_dir)
    os.makedirs(cluster_dir)
# 結果をクラスタごとにディレクトリに保存

class_count = np.zeros(class_number)
acc = np.zeros((class_number,class_number))

for i, label in enumerate(predict_clus):
   im=one_depreprocess(X_test[[i]]) #画像データ化
   cv2.imwrite(OUTPUT_DIR + '/cluster{}/{}.png'.format(label, i),im) #画像の保存
   class_count[label]+=1
   real_label = Y_label[i]
   acc[label][real_label]+=1

for i in class_count:
   print(i)

for count, i in enumerate(acc):
  print('============')
  for count2, j in enumerate(i):
   if j == 0:
      print('none')
   else:
      print('再現率label',count2,'= {:.4f}'.format(j/label_count[count2]))
      print(j,'/',label_count[count2])
      print('適合率label',count2,'= {:.4f}'.format(j/class_count[count]))
      print(j,'/',class_count[count])
  print('============')