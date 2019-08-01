import numpy as np
from scipy.special import softmax


class Tanh:
    def __call__(self, x, *args, **kwargs):
        return np.tanh(x)


class ReLU:
    def __call__(self, x, *args, **kwargs):
        return np.maximum(x, 0)


class SoftMax:
    def __call__(self, x, *args, **kwargs):
        return softmax(x)
