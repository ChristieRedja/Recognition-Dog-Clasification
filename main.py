import os
import glob
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import keras
import streamlit as st
from PIL import Image

from keras.layers import Dense
from keras.models import Sequential
from keras.preprocessing import image
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Convolution2D, Dense, MaxPool2D, Activation, Dropout, Flatten
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import BatchNormalization
from tensorflow.keras.preprocessing.image import img_to_array

def get_files(directory):
    if not os.path.exists(directory):
        return 0
    count = 0
    for current_path, dirs, files in os.walk(directory):
        for dr in dirs:
            count += len(glob.glob(os.path.join(current_path, dr + "/*")))
    return count

train_dir = "datasetticate/split_dataset/train"
val_dir = "datasetticate/split_dataset/validation"
test_dir = "datasetticate/split_dataset/test"

train_samples = get_files(train_dir)
num_classes = len(glob.glob(train_dir + "/*"))
val_samples = get_files(val_dir)
test_samples = get_files(test_dir)

train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(train_dir, target_size=(150, 150), batch_size=10)
val_generator = val_datagen.flow_from_directory(val_dir, shuffle=True, target_size=(150, 150), batch_size=10)
test_generator = test_datagen.flow_from_directory(test_dir, shuffle=True, target_size=(150, 150), batch_size=10)

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

earlyStopping = keras.callbacks.EarlyStopping(
    monitor='val_accuracy',
    mode='auto',
    baseline=None,
    restore_best_weights=True,
    patience=20,
    verbose=1
)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

hist = model.fit(train_generator, steps_per_epoch=10, epochs=50, validation_data=val_generator,
                 validation_steps=1, callbacks=[earlyStopping])

model.save("model.h5")


def process_and_predict(file):
    im = file
    width, height = im.size
    if width == height:
        im = im.resize((150, 150), Image.ANTIALIAS)
    else:
        if width > height:
            left = width / 2 - height / 2
            right = width / 2 + height / 2
            top = 0
            bottom = height
            im = im.crop((left, top, right, bottom))
            im = im.resize((150, 150), Image.ANTIALIAS)
        else:
            left = 0
            right = width
            top = 0
            bottom = width
            im = im.crop((left, top, right, bottom))
            im = im.resize((150, 150), Image.ANTIALIAS)
    ar = np.asarray(im)
    ar = ar.astype('float32')
    ar /= 255.0
    ar = ar.reshape(-1, 150, 150, 3)

    maxnum = np.argmax(model.predict(ar))
    if maxnum == 0:
        prediction = 'Golden'
    if maxnum == 1:
        prediction = 'Husky'
    if maxnum != 0 and maxnum != 1:
        prediction ='Other'
    st.write(' is a ' + prediction)

    st.image(im, caption=prediction, use_column_width=True)

if __name__ == "__main__":
    st.sidebar.subheader('Data Testing')
    is_training_mode = st.sidebar.checkbox('Training Mode', value=False)
    image = Image.open("test.jpeg")
    file = st.sidebar.file_uploader(label='Pilih data Testing', type=('JPEG'))
      
    if file is not None:
        st.sidebar.write('File Uploaded')
        try:
            image = Image.open(file)
        except Exception as e:
            print(e)
            image = Image.open("test.jpeg")
    
    st.header("Face Recognition using Keras")
   
    if (st.button('Testing dengan Keras', key=3)):
        st.write('Sedang melakukan testing')
        if image is not None:
            process_and_predict(image)
        else:
            st.write('File testing belum diunggah')
    else:
        st.write('')
