import numpy as np
import torch
import torch.nn.functional as F


class GazeMSEError(object):

    _to_degrees = 180. / np.pi

    def _to_vector(self, a):
        if a.shape[1] == 2:
            sin = torch.sin(a)
            cos = torch.cos(a)
            v = torch.stack([
                cos[:, 0] * sin[:, 1],
                -sin[:, 0],
                cos[:, 0] * cos[:, 1],
            ], dim=1)
        elif a.shape[1] == 3:
            v = F.normalize(a)
        else:
            raise ValueError('Do not know how to convert tensor of size %s' % a.shape)
        return v

    def __call__(self, a, b):
        a = self._to_vector(a)
        b = self._to_vector(b)
        return torch.mean((a - b) ** 2)
