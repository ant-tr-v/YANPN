import json
import numpy as np

import keras
from keras.models import Sequential
from keras.layers import Dense

from keras import backend as K

model = Sequential()
model.add(Dense(20, activation='tanh', input_shape=(1,)))
model.add(Dense(30, activation='tanh'))
model.add(Dense(30, activation='tanh'))
model.add(Dense(1))

model.load_weights('../nets/keras_sin.h5')

weights = []
for w in model.weights:
    l = K.eval(w).tolist()
    print(l)
    weights.append(l)

with open('../nets/numpy_sin.json', 'w') as f:
    json.dump(weights, f)