import json
import numpy as np
import time

import keras
from keras.datasets import mnist

from Layers import Dense, Conv2D, GlobalAveragePooling2D, MaxPool2D, BatchNorm2D
from Activations import ReLU, SoftMax, Tanh
from Models import Sequential

(x_train, y_train), (x_test, y_test) = mnist.load_data()

with open('../nets/numpy_mnist.json', 'r') as f:
    net_weights = json.load(f)

model = Sequential()
model.add(Conv2D(net_weights[0], net_weights[1], padding=3))
model.add(ReLU())
model.add(MaxPool2D(2))
model.add(BatchNorm2D(net_weights[2], net_weights[3], net_weights[4], net_weights[5]))
model.add(Conv2D(net_weights[6], net_weights[7]))
model.add(ReLU())
model.add(GlobalAveragePooling2D())
# model.add(Dense(net_weights[6], net_weights[7]))
# model.add(Tanh())
model.add(Dense(net_weights[8], net_weights[9]))
model.add(SoftMax())

img_rows, img_cols = 28, 28
x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)[:1000]

num_classes = 10

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)[:1000]

def acc(y, y_pred):
    y_l = np.argmax(y, axis=1)
    y_tl = np.argmax(y_pred, axis=1)
    return sum((y_l == y_tl).astype(float))/ len(y)

t = time.time()
y_pred = model(x_test)
t = time.time() - t
print(acc(y_test, y_pred))
print(f'\t{t:.3f} sec')
#print(y_pred)