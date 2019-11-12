import numpy as np
import torch
import torch.nn.functional as F


class AngularError(object):

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
        sim = F.cosine_similarity(a, b, dim=1, eps=1e-8)
        sim = F.hardtanh_(sim, min_val=-1+1e-8, max_val=1-1e-8)
        return torch.acos(sim) * self._to_degrees
