import json
import torch
from torchsummary import summary

from MnistModel import MnistNet


with open('../nets/torch_mnist.net', 'rb') as f:
    model = torch.load(f)

summary(model, (1, 28, 28))

weights = []
for l in model.dump():
    print(l.shape)
    weights.append(l.tolist())

with open('../nets/numpy_mnist.json', 'w') as f:
    json.dump(weights, f)
