from collections.abc import Iterable, Callable
import numpy as np


class Sequential:
    def __init__(self):
        self._forvard = []

    def add(self, f: Callable):
        self._forvard.append(f)

    def __call__(self, X: Iterable,*args, **kwargs):
        y = []
        for x in X:
            for f in self._forvard:
                x = f(x)
            y.append(x)
        return np.array(y)