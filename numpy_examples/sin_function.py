import json
import numpy as np
import matplotlib.pyplot as plt

from Layers import Dense
from Activations import Tanh
from Models import Sequential

with open('../nets/numpy_sin.json', 'r') as f:
    net_weights = json.load(f)

model = Sequential()
model.add(Dense(net_weights[0], net_weights[1]))
model.add(Tanh())
model.add(Dense(net_weights[2], net_weights[3]))
model.add(Tanh())
model.add(Dense(net_weights[4], net_weights[5]))
model.add(Tanh())
model.add(Dense(net_weights[6], net_weights[7]))


X = np.linspace(-3.14, 3.14, 1000)
y = model(X)

print(y)
plt.plot(X, y)
plt.show()