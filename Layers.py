import numpy as np
from collections.abc import Iterable

from im2col import im2col_indices



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


class Conv2D:
    #https://wiseodd.github.io/techblog/2016/07/16/convnet-conv-layer/
    def __init__(self, weights, biases=None, padding=0, stride=1, data_format='channels_first'):
        if data_format not in ['channels_first', 'channels_last']:
            raise ValueError('data_format should be ether "channels_first" or "channels_last"')
        self.data_format = data_format
        self.W = np.array(weights)
        self.padding=padding
        self.stride = stride
        if self.W.ndim != 4:
            raise ValueError("weights.ndim != 4")
        self.bias = np.array(biases) if biases is not None else None
        if self.bias is not None and (self.bias.ndim != 1 or len(self.bias) != self.W.shape[-1]):
            raise ValueError("Incorrect bias")
        self.W = np.transpose(self.W, (3, 2, 0, 1))
        self.n_filters, self.d_filters, self.h_filter, self.w_filter = self.W.shape
        self.W = self.W.reshape(self.n_filters, -1)

    def __call__(self, x, *args, **kwargs):
        x = np.array(x)
        if x.ndim == 2:
            x = np.array([x])
        if x.ndim != 3:
            raise Exception('Image should be ether 2d or 3d')
        if self.data_format == 'channels_last':
            x = np.transpose(x, (2, 0, 1))
        d_x, h_x, w_x = x.shape
        if d_x != self.d_filters:
            raise Exception('different number of channels and filter depth')

        h_out = (h_x - self.h_filter + 2 * self.padding) / self.stride + 1
        w_out = (w_x - self.w_filter + 2 * self.padding) / self.stride + 1

        if not h_out.is_integer() or not w_out.is_integer():
            raise Exception('Invalid output dimension!')
        h_out, w_out = int(h_out), int(w_out)

        X_col = im2col_indices(x, self.h_filter, self.w_filter, padding=self.padding, stride=self.stride)

        out = self.W @ X_col
        out = out.reshape(self.n_filters, h_out, w_out)
        if self.bias is not None:
            for i, b in enumerate(self.bias):
                out[i] += b
        return out


class GlobalAveragePooling2D:
    def __init__(self, data_format='channels_first'):
        if data_format not in ['channels_first', 'channels_last']:
            raise ValueError('data_format should be ether "channels_first" or "channels_last"')
        self.data_format = data_format

    def __call__(self, x, *args, **kwargs):
        x = np.array(x)
        if x.ndim == 2:
            x = np.array([x])
        if x.ndim != 3:
            raise ValueError('Image should be ether 2d or 3d')
        if self.data_format == 'channels_last':
            x = np.transpose(x, (2, 0, 1))

        y = []
        for ch in x:
            y.append(np.mean(ch))
        return np.array(y)
