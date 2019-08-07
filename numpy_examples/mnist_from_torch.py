import json
import numpy as np
import time

import torch
from torchvision import datasets, transforms

from Layers import Dense, Conv2D, GlobalAveragePooling2D, MaxPool2D, BatchNorm2D
from Activations import ReLU, SoftMax, Tanh
from Models import Sequential


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

num_classes = 10

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=2500)


def acc(y, y_pred):
    y_tl = np.argmax(y_pred, axis=1)
    y_l = y
    return sum((y_l == y_tl).astype(float))/ len(y)


x_test, y_test = None, None

for x0, y0 in test_loader:
    x_test, y_test = x0, y0
    break

x_test, y_test = x_test.numpy(), y_test.numpy()


t = time.time()
y_pred = model(x_test)
t = time.time() - t
print(acc(y_test, y_pred))
print(f'\t{t:.3f} sec')
#print(y_pred)