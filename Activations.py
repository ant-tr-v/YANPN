import numpy as np


class Tanh:
    def __call__(self, x, *args, **kwargs):
        return np.tanh(x)