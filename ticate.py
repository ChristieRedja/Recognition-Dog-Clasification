import os 
import cv2
import glob
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import keras
from PIL import Image

from keras.layers import Dense
from keras.models import Sequential
from keras.preprocessing import image
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Convolution2D,Dense,MaxPool2D,Activation,Dropout,Flatten
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
# Importing Keras libraries and packages
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import BatchNormalization
from tensorflow.keras.preprocessing.image import img_to_array

import os
from keras.applications.vgg16 import VGG16
#from glob import glob

def get_files(directory):
  if not os.path.exists(directory):
    return 0
  count=0
  # crawls inside folders
  for current_path,dirs,files in os.walk(directory):
    for dr in dirs:
      count+= len(glob.glob(os.path.join(current_path,dr+"/*")))
  return count
train_dir ="C:/Users/chris/datasetticate/split_dataset/train"
val_dir="C:/Users/chris/datasetticate/split_dataset/validation"
test_dir="C:/Users/chris/datasetticate/split_dataset/test"
#train file image count
train_samples =get_files(train_dir)
#to get tags
num_classes=len(glob.glob(train_dir+"/*")) 
#test file image count
val_samples=get_files(val_dir)
test_samples=get_files(test_dir)
print(num_classes,"Classes")
print(train_samples,"Train images")
print(val_samples,"Val images")
print(test_samples,"Test images")

train_datagen=ImageDataGenerator(
    rescale=1./255)
val_datagen=ImageDataGenerator(rescale=1./255)
test_datagen=ImageDataGenerator(rescale=1./255)

input_shape=(128,128,3)
train_generator =train_datagen.flow_from_directory(train_dir,target_size=(150, 150),batch_size=10)
val_generator=test_datagen.flow_from_directory(val_dir,shuffle=True,target_size=(150, 150),batch_size=10)
test_generator=test_datagen.flow_from_directory(test_dir,shuffle=True,target_size=(150, 150),batch_size=10)

# Define the CNN model
model = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(128, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(128, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dense(2, activation='sigmoid')
])

import tensorflow as tf
run_opts = tf.compat.v1.RunOptions(report_tensor_allocations_upon_oom = True)
from tensorflow.keras.callbacks import EarlyStopping

earlyStopping = EarlyStopping(
    monitor='val_accuracy',
    mode='auto', 
    baseline=None, 
    restore_best_weights=True, 
    patience = 20,
    verbose = 1
)

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
hist=model.fit(train_generator,
               steps_per_epoch=10,
               epochs=50,
               validation_data=val_generator,
               validation_steps=1,
               callbacks =[earlyStopping]
               )

#save bobot
#model.save("Golden_Husky_Christie_ticate_1.h5")

def predictClasses(predictions):
    valResult = predictions.copy()
    valResult[valResult <= 0.5] = 0
    valResult[valResult > 0.5] = 1
    return valResult

def plotHistory(history):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy history')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss history')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.show()

def plotConfusionMatrix(predictions, labels):
    valResult = predictClasses(predictions)
    confMatrixdf = pd.DataFrame(
        confusion_matrix(labels, valResult),
        index=["Golden", "Husky"],
        columns=["Pred. Golden", "Pred. Husky"]
    )

    plt.title("Confusion Matrix")
    #sn.heatmap(confMatrixdf, annot=True, annot_kws={"size" : "20"})
    sn.heatmap(confMatrixdf,annot=True,cmap='Blues', fmt='g')


plotHistory(hist)

################################################################################################################################
##Untuk membuat confusion matrix akurasi########
# y_actual, y_test = [], []
# import os, cv2
# img_path="C:\\Users\\chris\\datasetticate\\split_dataset\\validation\\Golden"
# img_list= os.listdir(img_path)
# for img in img_list:
#         input_img=cv2.imread(img_path + '/'+ img)
#         img=img_to_array(input_img)
#         img=np.expand_dims(img,axis=0)
#         pred=(model.predict(img)>0.5).astype("int32")
#         y_test.append(pred[0,0])
#         y_actual.append(1)

# img_path="C:\\Users\\chris\\datasetticate\\split_dataset\\validation\\Golden"
# img_list= os.listdir(img_path)
# for img in img_list:
#         input_img=cv2.imread(img_path + '/'+ img)
#         input_img_1=cv2.resize(img,(150,150,3),interpolation=cv2.INTER_AREA)
#         img=img_to_array(input_img_1)
#         img=np.expand_dims(img,axis=0)
#         pred=(model.predict(img)>0.5).astype("int32")
#         y_test.append(pred[0,0])
#         y_actual.append(0)

# y_actual=np.array(y_actual)
# y_test=np.array(y_test)

# from sklearn.metrics import confusion_matrix
# import seaborn as sns
# cn=confusion_matrix(y_test,y_actual)

# ax=plt.subplot()
# cn=confusion_matrix(y_actual,y_test)
# sns.heatmap(cn,cmap="RdPu",annot=True)
# ax.set_xlabel('Predicted'); ax.set_ylabel('True');
# ax.set_title('Confusion Matrix');
# ax.xaxis.set_ticklabels(['Golden','Husky']);
# ax.yaxis.set_ticklabels(['Golden','Husky']);
##########################################################################################################################

def process_and_predict(file, name):
    im = Image.open(file)
    width, height = im.size
    if width == height:
        im = im.resize((150,150), Image.ANTIALIAS)
    else:
        if width > height:
            left = width/2 - height/2
            right = width/2 + height/2
            top = 0
            bottom = height
            im = im.crop((left,top,right,bottom))
            im = im.resize((150,150), Image.ANTIALIAS)
        else:
            left = 0
            right = width
            top = 0
            bottom = width
            im = im.crop((left,top,right,bottom))
            im = im.resize((150,150), Image.ANTIALIAS)
    ar = np.asarray(im)
    ar = ar.astype('float32')
    ar /= 255.0
    ar = ar.reshape(-1, 150, 150, 3)

    # Membuat prediksi
    maxnum = np.argmax(model.predict(ar))
    if maxnum == 0:
        prediction = 'Golden'
    if maxnum == 1:
        prediction = 'Husky'
    print(name + ' is a ' + prediction)

    # Menampilkan gambar
    plt.imshow(im)
    plt.title(prediction)
    plt.axis('off')
    plt.show()

    return im


    

process_and_predict('C:\\Users\\chris\\datasetticate\\split_dataset\\test\\Golden\\9.jpeg', '9')


