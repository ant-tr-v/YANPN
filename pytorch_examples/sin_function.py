import numpy as np
import matplotlib.pyplot as plt

import torch.optim as optim
import time
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

from SinModel import TorchSin

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
            loss = F.mse_loss(output, target)
            av_loss.append(loss.detach().cpu().numpy())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        dt = time.time() - t
        if verbose:
            print(f'\t took: {dt:.2f}s, loss {np.mean(av_loss)}')


X = np.random.uniform(-3.14, 3.14, 50).astype(np.float32)
y = np.sin(X).astype(np.float32)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = TorchSin().to(device)

batch_size = 10
loader = torch.utils.data.DataLoader(list(zip(X.reshape((-1, 1)), y.reshape((-1, 1)))), batch_size=batch_size)
opt = optim.Adam(model.parameters(), lr=1e-3)

train(model, device, loader, opt, 500)

with torch.no_grad():
    x0 = np.linspace(-3.14, 3.14, 1000, dtype=np.float32)
    y0 = model(torch.tensor(x0.reshape((-1, 1))).to(device)).cpu().numpy()

    plt.plot(X, y, '.', linewidth=3.0)
    plt.plot(x0, y0)
    plt.show()

with open('../nets/torch_sin.net', 'wb') as f:
    torch.save(model, f)