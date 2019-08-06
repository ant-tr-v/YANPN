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
            raise ValueError('Incorrect shape')

    def __call__(self, x, *args, **kwargs):
        if not isinstance(x, Iterable):
            x = np.array([x])
        else:
            x = np.array(x)
        x.reshape((-1,))
        if len(x) != self.W.shape[1]:
            raise ValueError('Incorrect input shape')
        x = self.W @ x
        if self.bias is not None:
            x += self.bias
        return x


class Conv2D:
    #https://wiseodd.github.io/techblog/2016/07/16/convnet-conv-layer/
    def __init__(self, weights, biases=None, padding=0, stride=1, data_format='channels_first'):
        if data_format not in ['channels_first', 'channels_last']:
            raise AttributeError('data_format should be ether "channels_first" or "channels_last"')
        self.data_format = data_format
        self.W = np.array(weights)
        self.padding=padding
        self.stride = stride
        if self.W.ndim != 4:
            raise AttributeError("weights.ndim != 4")
        self.bias = np.array(biases) if biases is not None else None
        if self.bias is not None and (self.bias.ndim != 1 or len(self.bias) != self.W.shape[-1]):
            raise AttributeError("Incorrect bias")
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
        if self.data_format == 'channels_last':
            out = np.transpose(out, (1, 2, 0))
        return out


class GlobalAveragePooling2D:
    def __init__(self, data_format='channels_first'):
        if data_format not in ['channels_first', 'channels_last']:
            raise AttributeError('data_format should be ether "channels_first" or "channels_last"')
        self.data_format = data_format

    def __call__(self, x, *args, **kwargs):
        x = np.array(x)
        if x.ndim == 2:
            x = np.array([x])
        if x.ndim != 3:
            raise AttributeError('Image should be ether 2d or 3d')
        if self.data_format == 'channels_last':
            x = np.transpose(x, (2, 0, 1))

        y = []
        for ch in x:
            y.append(np.mean(ch))
        return np.array(y)


class MaxPool2D:
    def __init__(self, size=2, stride=None, data_format='channels_first'):
        if data_format not in ['channels_first', 'channels_last']:
            raise AttributeError('data_format should be ether "channels_first" or "channels_last"')
        self.data_format = data_format
        self.size = size
        self.stride = stride or size

    def __call__(self, x, *args, **kwargs):
        # https://github.com/wiseodd/hipsternet/blob/master/hipsternet/layer.py
        x = np.array(x)
        if x.ndim == 2:
            x = np.array([x])
        if x.ndim != 3:
            raise Exception('Image should be ether 2d or 3d')
        if self.data_format == 'channels_last':
            x = np.transpose(x, (2, 0, 1))
        d, h, w = x.shape

        h_out = (h - self.size) / self.stride + 1
        w_out = (w - self.size) / self.stride + 1

        size = self.size
        stride = self.stride

        if not w_out.is_integer() or not h_out.is_integer():
            raise Exception('Invalid output dimension!')

        h_out, w_out = int(h_out), int(w_out)

        #we modified im2col so now we have to do som things again

        i0 = np.repeat(np.arange(size), size)
        i1 = stride * np.repeat(np.arange(h_out), w_out)
        j0 = np.tile(np.arange(size), size)
        j1 = stride * np.tile(np.arange(w_out), h_out)
        i = i0.reshape(-1, 1) + i1.reshape(1, -1)
        j = j0.reshape(-1, 1) + j1.reshape(1, -1)

        cols = x[:, i, j]

        X_col = cols.transpose(1, 2, 0).reshape(size*size, -1)
        max_idx = np.argmax(X_col, axis=0)
        out = X_col[max_idx, range(max_idx.size)]
        out = out.reshape(h_out, w_out, d)
        out = out.transpose(2, 0, 1)
        if self.data_format == 'channels_last':
            out = np.transpose(out, (1, 2, 0))
        return out


class BatchNorm2D:
    def __init__(self, gamma, betta, mean, var,  data_format='channels_first'):
        if data_format not in ['channels_first', 'channels_last']:
            raise AttributeError('data_format should be ether "channels_first" or "channels_last"')
        self.data_format = data_format
        self.mean = np.array(mean)
        self.std = 1.0 / (np.sqrt(np.array(var) + 1e-3))
        self.gamma = np.array(gamma)
        self.betta = np.array(betta)

    def __call__(self, x, *args, **kwargs):
        x = np.array(x)
        if x.ndim == 2:
            x = np.array([x])
        if x.ndim != 3:
            raise Exception('Image should be ether 2d or 3d')

        if self.data_format == 'channels_last':
            x = np.transpose(x, (2, 0, 1))

        for i, m, s, g, b in zip(list(range(x.shape[0])), self.mean, self.std, self.gamma, self.betta):
            x[i] = (g * (x[i] - m) *s + b)
        if self.data_format == 'channels_last':
            x = np.transpose(x, (1, 2, 0))
        return x
