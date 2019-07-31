import numpy as np
from collections.abc import Iterable

class Dense:
    def __init__(self, weights, biases=None):
        self.W = np.array(weights)
        if self.W.ndim != 2:
            raise ValueError
        self.W = self.W.T
        self.bias = np.array(biases) if biases is not None else None
        if self.bias is not None and (self.bias.ndim != 1 or len(self.bias) != self.W.shape[0]):
            raise ValueError

    def __call__(self, x, *args, **kwargs):
        if not isinstance(x, Iterable):
            x = np.array([x])
        else:
            x = np.array(x)
        x.reshape((-1,))
        if len(x) != self.W.shape[1]:
            raise ValueError
        x = self.W @ x
        if self.bias is not None:
            x += self.bias
        return x