import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
# more info on callbakcs: https://keras.io/callbacks/ model saver is cool too.
from tensorflow.keras.callbacks import TensorBoard
import pickle
import time
import numpy as np
import os
from tensorflow.python.summary.writer.writer import FileWriter
import cv2

pickle_in = open("x.pickle", "rb")
X = pickle.load(pickle_in)
print(X[1])
pickle_in = open("y.pickle", "rb")
y = pickle.load(pickle_in)
y = np.array(y)
print(y[1])

X = X / 255.0

try:
    model = tf.keras.models.load_model("64x3-CNN.model")


except:

    dense_layers = [2]
    layer_sizes = [64]
    conv_layers = [3]

    for dense_layer in dense_layers:
        for layer_size in layer_sizes:
            for conv_layer in conv_layers:
                NAME = "{}-conv-{}-nodes-{}-dense-{}".format(conv_layer, layer_size, dense_layer, int(time.time()))
                print(NAME)

                model = Sequential()

                model.add(Conv2D(layer_size, (3, 3), input_shape=X.shape[1:]))
                model.add(Activation('relu'))
                model.add(MaxPooling2D(pool_size=(2, 2)))

                for l in range(conv_layer - 1):
                    model.add(Conv2D(layer_size, (3, 3)))
                    model.add(Activation('relu'))
                    model.add(MaxPooling2D(pool_size=(2, 2)))

                model.add(Flatten())

                for _ in range(dense_layer):
                    model.add(Dense(layer_size))
                    model.add(Activation('relu'))

                model.add(Dense(1))
                model.add(Activation('sigmoid'))

                tboard_log_dir = os.path.join("logs", NAME)
                tensorboard = TensorBoard(log_dir=tboard_log_dir)

                model.compile(loss='binary_crossentropy',
                              optimizer='adam',
                              metrics=['accuracy'],
                              )

                model.fit(X, y,
                          batch_size=32,
                          epochs=20,
                          validation_split=0.2,
                          callbacks=[tensorboard])

                model.save('64x3-CNN.model')


def prepare(file):
    IMG_SIZE = 50
    img_array = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    img_array = img_array / 255.0
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
