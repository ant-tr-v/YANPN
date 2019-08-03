import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv2D, GlobalAveragePooling2D, MaxPool2D, BatchNormalization


import keras.backend.tensorflow_backend


#https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py

img_rows, img_cols = 28, 28

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)[:1000]
input_shape = (1, img_rows, img_cols)

num_classes = 10

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)[:1000]

data_format = 'channels_first'  # because we need it io other projects


model = Sequential()
model.add(Conv2D(32, kernel_size=(7, 7),
                 activation='relu',
                 input_shape=input_shape,
                 data_format=data_format, padding='same'))
model.add(MaxPool2D(pool_size=(2,2), data_format=data_format))
model.add(BatchNormalization(axis=1))
model.add(Conv2D(128, (3, 3), activation='relu', data_format=data_format))
model.add(GlobalAveragePooling2D(data_format=data_format))
#model.add(Dense(256, activation='tanh'))
model.add(Dense(10, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(lr=1e-3),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=256,
          epochs=1,
          verbose=1)

model.save('../nets/keras_mnist.h5')
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print(f'Test accuracy:{score[1]:.7f}')

