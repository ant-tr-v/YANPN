import json

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv2D, GlobalAveragePooling2D
from keras import backend as K

data_format = 'channels_first'
img_rows, img_cols = 28, 28
input_shape = (1, img_rows, img_cols)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape,
                 data_format=data_format, padding='same'))
model.add(Conv2D(128, (3, 3), activation='relu', data_format=data_format))
model.add(Conv2D(128, (3, 3), activation='relu', data_format=data_format))
model.add(GlobalAveragePooling2D(data_format=data_format))
model.add(Dense(256, activation='tanh'))
model.add(Dense(10, activation='softmax'))

model.load_weights('../nets/keras_mnist.h5')

weights = []
for w in model.weights:
    l = K.eval(w)
    print(l.shape)
    weights.append(l.tolist())

with open('../nets/numpy_mnist.json', 'w') as f:
    json.dump(weights, f)