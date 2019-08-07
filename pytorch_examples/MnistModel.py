import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class MnistNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 7, padding=3)
        self.bn = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 128, 3)
        self.fc = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = self.bn(x)
        x = F.relu(self.conv2(x))
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(-1, 128)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)

    def dump(self):
        return [np.transpose(self.conv1.weight.detach().cpu().numpy(), (2, 3, 1, 0)),
                self.conv1.bias.detach().cpu().numpy(),
                self.bn.weight.detach().cpu().numpy(),
                self.bn.bias.detach().cpu().numpy(),
                self.bn.running_mean.detach().cpu().numpy(),
                self.bn.running_var.detach().cpu().numpy(),
                np.transpose(self.conv2.weight.detach().cpu().numpy(), (2, 3, 1, 0)),
                self.conv2.bias.detach().cpu().numpy(),
                self.fc.weight.detach().cpu().numpy().T,
                self.fc.bias.detach().cpu().numpy(),
                ]