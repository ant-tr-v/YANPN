import json
import torch
from torchsummary import summary

from SinModel import TorchSin


with open('../nets/torch_sin.net', 'rb') as f:
    model = torch.load(f)

summary(model, (1,))

weights = []
for l in model.dump():
    print(l.shape)
    weights.append(l.tolist())

with open('../nets/numpy_sin.json', 'w') as f:
    json.dump(weights, f)
