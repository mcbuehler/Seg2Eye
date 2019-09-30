"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from data.postprocessor import ImagePostprocessor

from models.networks.architecture import VGG19


# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input
class GANLoss(nn.Module):
    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor, opt=None):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_tensor = None
        self.fake_label_tensor = None
        self.zero_tensor = None
        self.Tensor = tensor
        self.gan_mode = gan_mode
        self.opt = opt
        if gan_mode == 'ls':
            pass
        elif gan_mode == 'original':
            pass
        elif gan_mode == 'w':
            pass
        elif gan_mode == 'hinge':
            pass
        else:
            raise ValueError('Unexpected gan_mode {}'.format(gan_mode))

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            if self.real_label_tensor is None:
                self.real_label_tensor = self.Tensor(1).fill_(self.real_label)
                self.real_label_tensor.requires_grad_(False)
            return self.real_label_tensor.expand_as(input)
        else:
            if self.fake_label_tensor is None:
                self.fake_label_tensor = self.Tensor(1).fill_(self.fake_label)
                self.fake_label_tensor.requires_grad_(False)
            return self.fake_label_tensor.expand_as(input)

    def get_zero_tensor(self, input):
        if self.opt.use_apex:
            return torch.zeros_like(input)
        else:
            if self.zero_tensor is None:
                self.zero_tensor = self.Tensor(1).fill_(0)
                self.zero_tensor.requires_grad_(False)
            return self.zero_tensor.expand_as(input)

    def loss(self, input, target_is_real, for_discriminator=True):
        if self.gan_mode == 'original':  # cross entropy loss
            target_tensor = self.get_target_tensor(input, target_is_real)
            loss = F.binary_cross_entropy_with_logits(input, target_tensor)
            return loss
        elif self.gan_mode == 'ls':
            target_tensor = self.get_target_tensor(input, target_is_real)
            return F.mse_loss(input, target_tensor)
        elif self.gan_mode == 'hinge':
            if for_discriminator:
                if target_is_real:
                    minval = torch.min(input - 1, self.get_zero_tensor(input))
                    loss = -torch.mean(minval)
                else:
                    minval = torch.min(-input - 1, self.get_zero_tensor(input))
                    loss = -torch.mean(minval)
            else:
                assert target_is_real, "The generator's hinge loss must be aiming for real"
                loss = -torch.mean(input)
            return loss
        else:
            # wgan
            if target_is_real:
                return -input.mean()
            else:
                return input.mean()

    def __call__(self, input, target_is_real, for_discriminator=True):
        # computing loss is a bit complicated because |input| may not be
        # a tensor, but list of tensors in case of multiscale discriminator
        if isinstance(input, list):
            loss = 0
            for pred_i in input:
                if isinstance(pred_i, list):
                    pred_i = pred_i[-1]
                loss_tensor = self.loss(pred_i, target_is_real, for_discriminator)
                bs = 1 if len(loss_tensor.size()) == 0 else loss_tensor.size(0)
                new_loss = torch.mean(loss_tensor.view(bs, -1), dim=1)
                loss += new_loss
            return loss / len(input)
        else:
            return self.loss(input, target_is_real, for_discriminator)


# Perceptual loss that uses a pretrained VGG network
class VGGLoss(nn.Module):
    def __init__(self, gpu_ids):
        super(VGGLoss, self).__init__()
        self.vgg = VGG19().cuda()
        self.criterion = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss


# KL Divergence loss used in VAE with an image encoder
class KLDLoss(nn.Module):
    def forward(self, mu, logvar):
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())


def openEDSaccuracy(produced, target):
    produced = produced.float()
    target = target.float()
    diff = produced - target
    squared_diff = torch.mul(diff, diff)
    h, w = squared_diff.shape[-2:]
    sum = torch.sum(squared_diff).float()
    accuracy = torch.sqrt(sum)
    accuracy = accuracy / (h * w)
    return accuracy


class MSECalculator:
    @classmethod
    def calculate_mse_for_images(cls, produced, target):
        assert produced.shape == target.shape
        assert torch.min(produced) >= 0 and torch.max(produced) <= 255, f"Min: {torch.min(produced)}, max: {torch.max(produced)}"
        assert torch.min(target) >= 0 and torch.max(target) <= 255
        assert produced.shape[-2:] == (640, 400), f"Invalid shape: {produced.shape}"
        assert len(produced.shape) == 4, "Please feed 4D tensors"

        mse_error = list()
        # We compute the norm for each image and then normalise it
        batch_size = produced.shape[0]
        for i in range(batch_size):
            produced_i = produced[i]
            target_i = target[i]
            # diff_i = torch.add(produced_i, torch.mul(target_i, -1)).float()
            norm_i = openEDSaccuracy(produced_i, target_i)
            mse_error.append(norm_i)
        mse_error = torch.stack(mse_error)
        return mse_error


    @classmethod
    def calculate_mse_for_tensors(cls, produced, target):
        assert produced.shape == target.shape
        assert torch.min(produced) >= -1 and torch.max(produced) <= 1, f"Min: {torch.min(produced)}, max: {torch.max(produced)}"
        assert torch.min(target) >= -1 and torch.max(target) <= 1
        # assert produced.shape[-2:] == (640, 400), f"Invalid shape: {produced.shape}"
        assert len(produced.shape) == 4, "Please feed 4D tensors"

        produced = ImagePostprocessor.to_255imagebatch(produced)
        target = ImagePostprocessor.to_255imagebatch(target)

        mse_error = list()
        # We compute the norm for each image and then normalise it
        batch_size = produced.shape[0]
        for i in range(batch_size):
            produced_i = produced[i]
            target_i = target[i]
            # diff_i = torch.add(produced_i, torch.mul(target_i, -1)).float()
            norm_i = openEDSaccuracy(produced_i, target_i)
            mse_error.append(norm_i)
        mse_error = torch.stack(mse_error)
        return mse_error

    @classmethod
    def calculate_error_statistics(cls, all_errors, mode, dataset_key):
        """

        Args:
            all_errors:
            mode: full, partial

        Returns:

        """
        all_errors_sum = np.sum(all_errors)
        relative_errors_sum = all_errors_sum / len(all_errors) * 1471
        errors_dict = {
            # f'mse/{dataset_key}/{mode}/sum': all_errors_sum,
                       f'mse/{dataset_key}/{mode}/relative': relative_errors_sum}
        return errors_dict


def gram_matrix(input):
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)


class StyleLoss(nn.Module):

    def __init__(self):
        super(StyleLoss, self).__init__()


    def forward(self, predicted_feature, target_feature):
        G_predicted = gram_matrix(predicted_feature)
        G_target = gram_matrix(target_feature).detach()
        loss = F.mse_loss(G_predicted, G_target)
        return loss