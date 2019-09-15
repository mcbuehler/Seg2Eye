"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import torch
import torch.nn as nn
from data.postprocessor import ImagePostprocessor
from models.networks import MSECalculator

from models.networks.architecture import SPADEResnetBlock

from RAdam.radam import RAdam
from models.networks.base_network import BaseNetwork
from util import util


class Refiner(BaseNetwork):
    """ Same architecture as the image discriminator """
    name = 'refiner'

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.FloatTensor = torch.cuda.FloatTensor if self.use_gpu() \
            else torch.FloatTensor

        ndf = 64
        kw = 3

        self.head = nn.Conv2d(opt.input_nc, ndf, kw, stride=2, padding=1)

        self.block1 = self.get_block(mode='spade', fin=ndf, fout=ndf*4)
        self.block2 = self.get_block(mode='spade', fin=ndf*4, fout=ndf)

        # self.block3 = self.get_block(mode='conv', fin=ndf, fout=ndf*4)
        # self.block4 = self.get_block(mode='conv', fin=ndf*4, fout=ndf)

        self.tail = nn.Conv2d(ndf, 1, kw, stride=1, padding=1)
        self.up = nn.Upsample(scale_factor=2)

        self.actvn = nn.LeakyReLU(False)
        self.criterionL2 = nn.MSELoss()

    def get_block(self, mode, fin, fout):
        if mode == 'spade':
            block = SPADEResnetBlock(fin, fout, self.opt)
        elif mode == 'conv':
            fmiddle = min(fin, fout)
            block = nn.Sequential(
                nn.Conv2d(fin, fmiddle, kernel_size=3, padding=1),
                nn.BatchNorm2d(fmiddle, affine=True),
                nn.Conv2d(fmiddle, fout, kernel_size=3, padding=1)
            )
        else:
            raise ValueError(f"Invalid mode: {mode}")
        return block

    def use_gpu(self):
        return len(self.opt.gpu_ids) > 0

    def forward(self, x, target_mask, mode='train'):
        if mode == 'train':
            self.train()
        else:
            self.eval()

        x = self.preprocess_input(x)
        seg = self.preprocess_label(target_mask)

        x = self.head(x)
        x = self.actvn(x)

        x_spade = self.block1(x, seg)
        x_spade = self.up(x_spade)
        x_spade = self.actvn(x_spade)
        x_spade = self.block2(x_spade, seg)

        # x_conv = self.block3(x)
        # x_conv = self.actvn(x_conv)
        # x_conv = self.block4(x_conv)
        #
        # x = torch.cat([x_spade, x_conv], dim=1)
        x = x_spade
        x = self.actvn(x)
        x = self.tail(x)

        return x

    def get_losses(self, style_image, target_mask, target_image, mode='train'):
        out_pred = self.forward(style_image, target_mask, mode=mode)

        target_image = self.preprocess_input(target_image)
        loss = self.criterionL2(target_image, out_pred)

        if not torch.isnan(loss) and mode == 'train':
            loss.backward()

        out_pred_clamped = out_pred.clamp(-1, 1)

        losses = {f'{mode}/mse': loss,
                      f'{mode}/mse_relative':
                          torch.sum(MSECalculator.calculate_mse_for_images(
                              ImagePostprocessor.to_255resized_imagebatch(out_pred_clamped),
                          ImagePostprocessor.to_255resized_imagebatch(target_image))) * 1471 / out_pred.shape[0]
                  }
        return losses, out_pred

    def preprocess_input(self, image):
        if len(self.opt.gpu_ids) > 0:
            image = image.cuda()
        return image

    def preprocess_label(self, mask):
        mask = mask.long()
        if len(self.opt.gpu_ids) > 0:
            mask = mask.cuda()

        # create one-hot label map
        label_map = mask
        if len(label_map.shape) == 3:
            label_map = label_map.unsqueeze(0)
        bs, _, h, w = label_map.size()
        nc = self.opt.label_nc
        input_label = self.FloatTensor(bs, nc, h, w).zero_()
        input_semantics = input_label.scatter_(1, label_map, 1.0)

        return input_semantics

    def save(self, epoch):
        util.save_network(self, self.name, epoch, self.opt)

    def create_optimizer(self, params, opt):
        if opt.use_radam:
            optimizer_class = RAdam
        else:
            optimizer_class = torch.optim.Adam
        print(f"Creating {optimizer_class} for {len(params)} parameters.")
        optimizer = optimizer_class(params, lr=opt.lr, betas=(opt.beta1, opt.beta2), weight_decay=opt.weight_decay)
        return optimizer

