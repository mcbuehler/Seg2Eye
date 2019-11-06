"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.nn.utils.spectral_norm as spectral_norm
from models.networks.normalization import SPADE, ApplyStyle, SPADE_STYLE_Block

# ResNet block that uses SPADE and StyleGAN AdaIN.
class SPADE_STYLE_ResnetBlock(nn.Module):
    def __init__(self, fin, fout, opt):
        super().__init__()
        # Attributes
        self.learned_shortcut = (fin != fout)
        fmiddle = min(fin, fout)

        # create conv layers
        self.conv_0 = nn.Conv2d(fin, fmiddle, kernel_size=3, padding=1)
        self.conv_1 = nn.Conv2d(fmiddle, fout, kernel_size=3, padding=1)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(fin, fout, kernel_size=1, bias=False)

        # apply spectral norm if specified
        if 'spectral' in opt.norm_G:
            self.conv_0 = spectral_norm(self.conv_0)
            self.conv_1 = spectral_norm(self.conv_1)
            if self.learned_shortcut:
                self.conv_s = spectral_norm(self.conv_s)

        self.norm_0 = SPADE_STYLE_Block(fin, opt)
        self.norm_1 = SPADE_STYLE_Block(fmiddle, opt)

        if self.learned_shortcut:
            self.norm_s = SPADE_STYLE_Block(fin, opt)

    # note the resnet block with SPADE also takes in |seg|,
    # the semantic segmentation map as input
    def forward(self, x, seg, latent_style):
        x_s = self.shortcut(x, seg, latent_style)

        dx = self.conv_0(self.actvn(self.norm_0(x, seg, latent_style)))
        dx = self.conv_1(self.actvn(self.norm_1(dx, seg, latent_style)))

        out = x_s + dx

        return out

    def shortcut(self, x, seg, latent_style):
        if self.learned_shortcut:
            x_s = self.conv_s(self.norm_s(x, seg, latent_style))
        else:
            x_s = x
        return x_s

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)