"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import torch
import torch.nn as nn

from RAdam.radam import RAdam
from models.networks.base_network import BaseNetwork
from util import util


class PupilLocator(BaseNetwork):
    """ Same architecture as the image discriminator """

    VAL_PUPIL = 3
    name = 'pupil_locator'

    def __init__(self, opt):
        super().__init__()

        kw = 3
        # pw = int(np.ceil((kw - 1.0) / 2))
        pw = 0
        # ndf = opt.ngf
        ndf = 16
        # norm_layer = get_nonspade_norm_layer(opt, 'batch')

        self.layer1 = nn.Conv2d(opt.input_nc, ndf, kw, stride=2, padding=pw)
        self.bn1 = nn.BatchNorm2d(ndf, affine=True)

        self.layer2 = nn.Conv2d(ndf * 1, ndf * 2, kw, stride=2, padding=pw)
        self.bn2 = nn.BatchNorm2d(ndf * 2, affine=True)

        self.layer3 = nn.Conv2d(ndf * 2, ndf * 4, kw, stride=2, padding=pw)
        self.bn3 = nn.BatchNorm2d(ndf * 4, affine=True)

        self.layer4 = nn.Conv2d(ndf * 4, ndf * 8, kw, stride=2, padding=pw)
        self.bn4 = nn.BatchNorm2d(ndf * 8, affine=True)
        # self.layer3 = norm_layer(nn.Conv2d(ndf * 2, ndf * 4, kw, stride=2, padding=pw))
        # self.layer4 = norm_layer(nn.Conv2d(ndf * 4, ndf * 8, kw, stride=2, padding=pw))
        # self.layer5 = norm_layer(nn.Conv2d(ndf * 8, ndf * 8, kw, stride=2, padding=pw))
        # if opt.crop_size >= 256:
        #     self.layer6 = norm_layer(nn.Conv2d(ndf * 8, ndf * 8, kw, stride=2, padding=pw))

        self.maxpool = nn.MaxPool2d(2)

        self.so = 4

        # self.fc_out = nn.Linear(ndf * 8 * s0 * s0, 2)
        # self.fc_out = nn.Linear(4032, 2)
        # self.fc_out = nn.Linear(159264, 2)
        # len_flat = 18240
        # len_flat = 36480
        # self.fc_down = nn.Linear(len_flat, int(len_flat/2))
        self.fc_out = nn.Linear(8064, 2)

        self.actvn = nn.LeakyReLU(False)
        self.opt = opt

        # self.net = models.resnet50(pretrained=True)
        # num_ftrs = self.net.fc.in_features
        # self.net.fc = nn.Linear(num_ftrs, 2)

        # self.net = resnet18(num_classes=2)

        self.criterionL2 = nn.MSELoss()

    def forward(self, x, mode='train'):
        if mode == 'train':
            self.train()
        else:
            self.eval()

        x = self.preprocess_input(x)

        x = self.layer1(x)
        x = self.bn1(x)
        # x = self.bn1(x)
        # x = self.maxpool(x)
        x = self.layer2(self.actvn(x))
        x = self.bn2(x)
        x = self.maxpool(x)
        x = self.layer3(self.actvn(x))
        # x = self.maxpool(x)
        x = self.bn3(x)
        x = self.layer4(self.actvn(x))
        x = self.bn4(x)

        # x = self.layer4(self.actvn(x))
        # x = self.layer5(self.actvn(x))
        # if self.opt.crop_size >= 256:
        #     x = self.layer6(self.actvn(x))
        x = self.actvn(x)

        x = x.view(x.size(0), -1)
        # x = self.fc_down(x)
        # x = self.actvn(x)
        out = self.fc_out(x)

        # out = self.net(x)
        return out

    def get_losses(self, image, mask, mode='train'):
        out_pred = self.forward(image, mode=mode)

        mask = self.preprocess_label(mask)
        out_true = self.get_pupil_location(mask)
        loss = self.criterionL2(out_true, out_pred)
        losses = {f'{mode}/mse': loss}
        if not torch.isnan(loss) and mode == 'train':
            loss.backward()
        return losses, out_pred, out_true

    def preprocess_input(self, image):
        if len(self.opt.gpu_ids) > 0:
            image = image.cuda()
        return image

    def preprocess_label(self, mask):
        mask = mask.long()
        if len(self.opt.gpu_ids) > 0:
            mask = mask.cuda()
        return mask

    @classmethod
    def get_pupil_location(cls, mask_in):
        if len(mask_in.shape) == 4:
            locations = [cls.get_pupil_location(m[0]) for m in mask_in]
            return torch.stack(locations)
        else:
            # absolute location
            if len(mask_in.shape) == 3:
                mask = mask_in[0]
            else:
                mask = mask_in
            pupil_mask = mask == cls.VAL_PUPIL
            location = torch.nonzero(pupil_mask).float()
            x = (torch.mean(location[:, 0]) / mask.shape[-2]) - 0.5
            y = (torch.mean(location[:, 1]) / mask.shape[-1]) - 0.5
            return torch.stack([x, y])

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

