import os
from typing import Tuple
import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout
from PIL import Image
import numpy as np
import pandas as pd
import re


class AutoIRAD:
    def __init__(self, resolution: Tuple[int, int] = (100, 100)):
        base_model = InceptionV3(weights='imagenet', include_top=False,
                                 input_shape=(resolution[0], resolution[1], 3), classifier_activation='None')

        for layer in base_model.layers:
            layer.trainable = False

        model = Sequential()

        model.add(base_model)
        model.add(Flatten())
        model.add(Dense(2048, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(17, activation='softmax'))
        model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=1e-4),
                      loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        model.summary()

        self.model = model

    def train(self, X_train, y_train, epochs: int = 80):
        self.model.fit(X_train, y_train, epochs=epochs)

    def predict(self, X_test):
        return np.argmax(self.model.predict(X_test), axis=1)


if __name__ == '__main__':
    img_size = (256, 256)

    df = pd.read_csv('data/best_family.csv')
    df.set_index('dataset', inplace=True)

    with open(os.path.join('data', 'family_rma.csv'), 'r') as f:
        scores = pd.read_csv(f, index_col=0)

    images_dir = os.path.join('autoIRAD', 'images')
    images = []
    ys = []
    datasets = []
    for root, dirs, files in os.walk(images_dir):
        for file in files:
            if file.endswith(".png"):
                with open(os.path.join(root, file), 'rb') as f:
                    image = Image.open(f)
                    image = np.asarray(image)
                    images.append(image)
                y = df.loc[int(file.split('_')[0]), 'family']
                ys.append(y)
                datasets.append('_'.join(file.split('_')[1:-1]))
    images = np.array(images)
    images = np.resize(images, (len(images), *img_size, 3))
    ys = np.array(ys)
    auto = AutoIRAD(resolution=img_size)
    auto.train(images, ys)
    chosen = np.array(auto.predict(images))
    chosen = np.array([scores.columns[i] for i in chosen])
    print(np.array([scores.loc[datasets[i], chosen[i]] for i in range(len(datasets))]).mean())
