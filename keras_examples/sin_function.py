import numpy as np
import matplotlib.pyplot as plt

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

X = np.random.uniform(-3.14, 3.14, 50)
y = np.sin(X)

model = Sequential()
model.add(Dense(20, activation='tanh', input_shape=(1,)))
model.add(Dense(30, activation='tanh'))
model.add(Dense(30, activation='tanh'))
model.add(Dense(1))

model.summary()
model.compile(loss='mse', optimizer=Adam(lr=3e-3), metrics=['mse'])

model.fit(X, y, batch_size=10, epochs=500)
x0 = np.linspace(-3.14, 3.14, 1000)
y0 = model.predict(x0)
plt.plot(X, y, '.', linewidth=3.0)
plt.plot(x0, y0)
plt.show()

model.save('../nets/keras_sin.h5')