"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import torch.nn as nn
import torch.nn.functional as F

from models.networks.architecture import SPADE_STYLE_ResnetBlock
from models.networks.base_network import BaseNetwork


class SPADESTYLEGenerator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--num_upsampling_layers',
                            choices=('normal', 'more', 'most'), default='normal',
                            help="If 'more', adds upsampling layer between the two middle resnet blocks. If 'most', also add one more upsampling + resnet layer at the end of the generator")

        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        nf = opt.ngf

        self.sw, self.sh = self.compute_latent_vector_size(opt)

        # We start with a downsampled segmentation map
        self.fc = nn.Conv2d(self.opt.semantic_nc, 16 * nf, 3, padding=1)

        self.head_0 = SPADE_STYLE_ResnetBlock(16 * nf, 16 * nf, opt)

        self.G_middle_0 = SPADE_STYLE_ResnetBlock(16 * nf, 16 * nf, opt)
        self.G_middle_1 = SPADE_STYLE_ResnetBlock(16 * nf, 16 * nf, opt)

        self.up_0 = SPADE_STYLE_ResnetBlock(16 * nf, 8 * nf, opt)
        self.up_1 = SPADE_STYLE_ResnetBlock(8 * nf, 4 * nf, opt)
        self.up_2 = SPADE_STYLE_ResnetBlock(4 * nf, 2 * nf, opt)
        self.up_3 = SPADE_STYLE_ResnetBlock(2 * nf, 1 * nf, opt)

        final_nc = nf

        if opt.num_upsampling_layers == 'most':
            self.up_4 = self._get_resnet_block(1 * nf, nf // 2, opt)
            final_nc = nf // 2

        self.conv_img = nn.Conv2d(final_nc, opt.output_nc, 3, padding=1)

        self.up = nn.Upsample(scale_factor=2)

    def compute_latent_vector_size(self, opt):
        # Size for the starting feature map
        if opt.num_upsampling_layers == 'normal':
            num_up_layers = 5
        elif opt.num_upsampling_layers == 'more':
            num_up_layers = 6
        elif opt.num_upsampling_layers == 'most':
            num_up_layers = 7
        else:
            raise ValueError('opt.num_upsampling_layers [%s] not recognized' %
                             opt.num_upsampling_layers)

        sw = opt.crop_size // (2**num_up_layers)
        sh = round(sw / opt.aspect_ratio)

        return sw, sh

    def forward(self, input, w=None):
        seg = input
        # we downsample the segmap and run convolution
        x = F.interpolate(seg, size=(self.sh, self.sw))
        x = self.fc(x)

        x = self.head_0(x, seg, w)

        x = self.up(x)
        x = self.G_middle_0(x, seg, w)

        if self.opt.num_upsampling_layers == 'more' or \
           self.opt.num_upsampling_layers == 'most':
            x = self.up(x)

        x = self.G_middle_1(x, seg, w)

        x = self.up(x)
        x = self.up_0(x, seg, w)
        x = self.up(x)
        x = self.up_1(x, seg, w)
        x = self.up(x)
        x = self.up_2(x, seg, w)
        x = self.up(x)
        x = self.up_3(x, seg, w)

        if self.opt.num_upsampling_layers == 'most':
            x = self.up(x)
            x = self.up_4(x, seg, w)

        x = self.conv_img(F.leaky_relu(x, 2e-1))
        x = F.tanh(x)

        return x