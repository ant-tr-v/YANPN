import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

class TorchSin(nn.Module):
    def __init__(self):
        super().__init__()
        self.dense0 = nn.Linear(1, 20)
        self.dense1 = nn.Linear(20, 30)
        self.dense2 = nn.Linear(30, 30)
        self.dense3 = nn.Linear(30, 1)

    def forward(self, x):
        x = torch.tanh(self.dense0(x))
        x = torch.tanh(self.dense1(x))
        x = torch.tanh(self.dense2(x))
        return self.dense3(x)

    def dump(self):
        return [self.dense0.weight.detach().cpu().numpy().T,
                self.dense0.bias.detach().cpu().numpy(),
                self.dense1.weight.detach().cpu().numpy().T,
                self.dense1.bias.detach().cpu().numpy(),
                self.dense2.weight.detach().cpu().numpy().T,
                self.dense2.bias.detach().cpu().numpy(),
                self.dense3.weight.detach().cpu().numpy().T,
                self.dense3.bias.detach().cpu().numpy()]