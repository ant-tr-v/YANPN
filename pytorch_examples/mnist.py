import numpy as np
import matplotlib.pyplot as plt

import torch.optim as optim
import time
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

from MnistModel import MnistNet
from torchvision import datasets, transforms


def train(model, device, train_loader, optimizer, n_epochs=1, verbose=True):
    model.train()
    for epoch in range(n_epochs):
        if verbose:
            print(f'Epoch #{epoch}')
        t = time.time()
        av_loss = []
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = F.cross_entropy(output, target)
            av_loss.append(loss.detach().cpu().numpy())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        dt = time.time() - t
        if verbose:
            print(f'\t took: {dt:.2f}s, loss {np.mean(av_loss)}')


def test_model(model, device, test_loader):
    model.eval()
    corr = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            corr += pred.eq(target.view_as(pred)).sum().item()

    acc = corr / len(test_loader.dataset)
    return acc


batch_size = 128

train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=batch_size)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = MnistNet().to(device)


opt = optim.Adam(model.parameters(), lr=3e-4)
train(model, device, train_loader, opt, 3)

acc = test_model(model, device, test_loader)

print('test_acc: ', acc)

with open('../nets/torch_mnist.net', 'wb') as f:
    torch.save(model, f)

