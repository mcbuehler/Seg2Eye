"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import torch
from torch import nn
import models.networks as networks
import util.util as util
from RAdam.radam import RAdam
from data.postprocessor import ImagePostprocessor


class Pix2PixModel(torch.nn.Module):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        networks.modify_commandline_options(parser, is_train)
        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.FloatTensor = torch.cuda.FloatTensor if self.use_gpu() \
            else torch.FloatTensor
        self.ByteTensor = torch.cuda.ByteTensor if self.use_gpu() \
            else torch.ByteTensor

        self.netG, self.netD, self.netE = self.initialize_networks(opt)

        # Should handle input_ns 1 and > 1
        self.style_aggregation_method = self._get_style_aggregation_method()

        # set loss functions
        if opt.isTrain:
            self.criterionGAN = networks.GANLoss(
                opt.gan_mode, tensor=self.FloatTensor, opt=self.opt)
            self.criterionFeat = nn.L1Loss()
            self.criterionL2 = nn.MSELoss()
            if not opt.no_vgg_loss:
                self.criterionVGG = networks.VGGLoss(self.opt.gpu_ids)
            if opt.use_vae:
                self.KLDLoss = networks.KLDLoss()
            self.reset_loss_log()

    def get_loss_log(self):
        loss_log = {key: torch.mean(torch.stack(self.loss_log[key])) for key in self.loss_log if len(self.loss_log[key])}
        return loss_log

    def reset_loss_log(self):
        self.loss_log = {"L1/raw": list(), "L2/raw": list(), "KLD/raw": list(), "train/mse": list()}

    # Entry point for all calls involving forward pass
    # of deep networks. We used this approach since DataParallel module
    # can't parallelize custom functions, we branch to different
    # routines based on |mode|.
    def forward(self, data, mode):
        input_semantics, style_image, target_image = self.preprocess_input(data)

        if mode == 'generator':
            g_loss, generated = self.compute_generator_loss(
                input_semantics, style_image, target_image)
            return g_loss, generated
        elif mode == 'discriminator':
            d_loss = self.compute_discriminator_loss(
                input_semantics, style_image, target_image)
            return d_loss
        elif mode == 'encode_only':
            z, mu, logvar = self.encode_z(style_image)
            return mu, logvar
        elif mode == 'inference':
            with torch.no_grad():
                fake_image, _ = self.generate_fake(input_semantics, style_image)
                # We don't want to track our losses here as it could confound with training losses
                self.reset_loss_log()
            return fake_image
        else:
            raise ValueError("|mode| is invalid")

    def create_optimizers(self, opt):
        G_params = list(self.netG.parameters())
        if opt.use_vae or opt.spadeStyleGen:
            G_params += list(self.netE.parameters())
        if opt.isTrain:
            D_params = list(self.netD.parameters())

        if opt.no_TTUR:
            beta1, beta2 = opt.beta1, opt.beta2
            G_lr, D_lr = opt.lr, opt.lr
        else:
            beta1, beta2 = 0, 0.9
            G_lr, D_lr = opt.lr / 2, opt.lr * 2

        if self.opt.use_radam:
            optimizer_class = RAdam
        else:
            optimizer_class = torch.optim.Adam

        optimizer_G = optimizer_class(G_params, lr=G_lr, betas=(beta1, beta2), weight_decay=self.opt.weight_decay)
        optimizer_D = optimizer_class(D_params, lr=D_lr, betas=(beta1, beta2), weight_decay=self.opt.weight_decay)

        return optimizer_G, optimizer_D

    def save(self, epoch):
        util.save_network(self.netG, 'G', epoch, self.opt)
        util.save_network(self.netD, 'D', epoch, self.opt)
        if self.opt.use_vae or self.opt.spadeStyleGen:
            util.save_network(self.netE, 'E', epoch, self.opt)

    ############################################################################
    # Private helper methods
    ############################################################################

    def initialize_networks(self, opt):
        netG = networks.define_G(opt)
        netD = networks.define_D(opt) if opt.isTrain else None
        netE = networks.define_E(opt) if opt.use_vae or opt.spadeStyleGen else None

        if not opt.isTrain or opt.continue_train:
            netG = util.load_network(netG, 'G', opt.which_epoch, opt)
            if opt.isTrain:
                netD = util.load_network(netD, 'D', opt.which_epoch, opt)
            if opt.use_vae or opt.spadeStyleGen:
                netE = util.load_network(netE, 'E', opt.which_epoch, opt)

        return netG, netD, netE

    # preprocess the input, such as moving the tensors to GPUs and
    # transforming the label map to one-hot encoding
    # |data|: dictionary of the input data

    def preprocess_input(self, data):
        # move to GPU and change data types
        data['label'] = data['label'].long()
        if self.use_gpu():
            data['label'] = data['label'].cuda()
            data['style_image'] = data['style_image'].cuda()

        # create one-hot label map
        label_map = data['label']
        if len(label_map.shape) == 3:
            label_map = label_map.unsqueeze(0)
        bs, _, h, w = label_map.size()
        nc = self.opt.label_nc
        input_label = self.FloatTensor(bs, nc, h, w).zero_()
        if torch.max(label_map) > 3:
            print(f"torch.max(label_map) is more than 3: {torch.max(label_map)} here: {data['filename']}")
        input_semantics = input_label.scatter_(1, label_map, 1.0)

        if "target" in data:
            if self.use_gpu():
                data['target'] = data['target'].cuda()
            return input_semantics, data['style_image'], data['target']
        return input_semantics, data['style_image'], None

    def compute_generator_loss(self, input_semantics, style_image, target_image):
        G_losses = {}

        fake_image, KLD_loss = self.generate_fake(
            input_semantics, style_image, compute_kld_loss=self.opt.use_vae)

        if self.opt.use_vae:
            G_losses['KLD'] = KLD_loss

        pred_fake, pred_real = self.discriminate(input_semantics, fake_image, target_image)

        G_losses['GAN'] = self.criterionGAN(pred_fake, True,
                                            for_discriminator=False)

        if self.opt.lambda_l2:
            l2_loss = self.criterionL2(fake_image, target_image)
            G_losses['L2/weighted'] = l2_loss * self.opt.lambda_l2
            # Only for logging
            self.loss_log['L2/raw'].append(l2_loss)
        if self.opt.lambda_l1:
            l1_loss = self.criterionL2(fake_image, target_image)
            G_losses['L1/weighted'] = l1_loss * self.opt.lambda_l1
            # Only for logging
            self.loss_log['L1/raw'].append(l1_loss)

        if not self.opt.no_ganFeat_loss:
            num_D = len(pred_fake)
            GAN_Feat_loss = self.FloatTensor(1).fill_(0)
            for i in range(num_D):  # for each discriminator
                # last output is the final prediction, so we exclude it
                num_intermediate_outputs = len(pred_fake[i]) - 1
                for j in range(num_intermediate_outputs):  # for each layer output
                    unweighted_loss = self.criterionFeat(
                        pred_fake[i][j], pred_real[i][j].detach())
                    GAN_Feat_loss += unweighted_loss * self.opt.lambda_feat / num_D
            G_losses['GAN_Feat'] = GAN_Feat_loss

        if not self.opt.no_vgg_loss:
            G_losses['VGG'] = self.criterionVGG(fake_image, style_image) \
                * self.opt.lambda_vgg

        return G_losses, fake_image

    def compute_discriminator_loss(self, input_semantics, real_image, target_image):
        D_losses = {}
        with torch.no_grad():
            fake_image, _ = self.generate_fake(input_semantics, real_image)
            fake_image = fake_image.detach()
            fake_image.requires_grad_()

        pred_fake, pred_real = self.discriminate(
            input_semantics, fake_image, target_image)

        D_losses['D/Fake'] = self.criterionGAN(pred_fake, False,
                                               for_discriminator=True)
        D_losses['D/real'] = self.criterionGAN(pred_real, True,
                                               for_discriminator=True)

        return D_losses

    def encode_z(self, real_image):
        mu, logvar = self.netE(real_image)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def _get_style_aggregation_method(self):
        if self.opt.style_aggr_method == 'mean':
            return lambda tensor: torch.mean(tensor, dim=1)
        elif self.opt.style_aggr_method == 'max':
            # torch.max returns indices and values, but we only want values
            return lambda tensor: torch.max(tensor, dim=1).values
        else:
            raise ValueError(f"Aggregation method not found: {self.opt.style_aggr_method}")

    def _compute_multiple_netE(self, real_image):
        # We have several style images per input sample
        # for batch_i in range(real_image.shape[0]):
        #     # now we have n style images that we treat as a batch for one sample
        #     # mu will have shape (opt.input_ns, opt.dim_z)
        outputs_netE_tensor = torch.stack([self.netE(real_image[batch_i])[0]
                                           for batch_i in range(real_image.shape[0])], dim=0)

        # batchSize, input_ns, z_dim if opt.use_z else w_dim
        if self.opt.use_z:
            assert outputs_netE_tensor.shape == (*real_image.shape[:2], self.opt.z_dim)
        else:
            assert outputs_netE_tensor.shape == (*real_image.shape[:2], self.opt.w_dim)
        return outputs_netE_tensor

    def _compute_aggregated_z(self, real_image):
        assert self.opt.use_z, "You need to set use_z to True"
        # We have several style images per input sample
        multiple_z = self._compute_multiple_netE(real_image)
        z = self.style_aggregation_method(multiple_z)
        assert z.shape == (real_image.shape[0], self.opt.z_dim)
        return z

    def _compute_aggregated_w(self, real_image):
        # We have several style images per input sample
        if self.opt.use_z:
            multiple_z = self._compute_multiple_netE(real_image)
            # multiple_w has shape (bs, input_ns, w_dim)
            multiple_w = torch.stack([self.netE(z, mode='downscale') for z in multiple_z])
        else:
            multiple_w = self._compute_multiple_netE(real_image)
        # w has shape (bs, w_dim)
        w = self.style_aggregation_method(multiple_w)
        assert w.shape == (real_image.shape[0], self.opt.w_dim)
        return w

    def encode_w(self, real_image):
        if len(real_image.shape) == 5:  #shape[1] is input_ns
            if self.opt.use_z and self.opt.style_aggr_space == 'z':
                # We aggregate in z space
                z = self._compute_aggregated_z(real_image)
                w = self.netE(z, mode='downscale')
            elif self.opt.style_aggr_space == 'w':
                # We compute input_ns mu vectors for each sample. multiple_z.shape = (bs, input_ns, z_dim)
                # This is the way we should be taking if we do not use a z space
                w = self._compute_aggregated_w(real_image)
        else:
            # netE outputs a mu of dim opt.z_dim
            # This is legacy code for backward compatibility
            # and not used any more in more recent experiments (3.9.19 and newer)
            z, _ = self.netE(real_image)
            w = self.netE(z, mode='downscale')
        return w

    def generate_fake(self, input_semantics, style_image, compute_kld_loss=False):
        latent_style = None
        KLD_loss = None
        # TODO: be careful not to overwrite values once we have a VAE for w as well
        if self.opt.use_vae:
            latent_style, mu, logvar = self.encode_z(style_image)
            if compute_kld_loss:
                kld_loss_raw = self.KLDLoss(mu, logvar)
                KLD_loss = kld_loss_raw * self.opt.lambda_kld
                self.loss_log['KLD/raw'].append(kld_loss_raw)

        if self.opt.spadeStyleGen:
            latent_style = self.encode_w(style_image)

        fake_image = self.netG(input_semantics, latent_style)

        assert (not compute_kld_loss) or self.opt.use_vae, \
            "You cannot compute KLD loss if opt.use_vae == False"

        return fake_image, KLD_loss

    # Given fake and real image, return the prediction of discriminator
    # for each fake and real image.

    def discriminate(self, input_semantics, fake_image, real_image):
        fake_concat = torch.cat([input_semantics, fake_image], dim=1)
        real_concat = torch.cat([input_semantics, real_image], dim=1)

        # In Batch Normalization, the fake and real images are
        # recommended to be in the same batch to avoid disparate
        # statistics in fake and real images.
        # So both fake and real images are fed to D all at once.
        fake_and_real = torch.cat([fake_concat, real_concat], dim=0)

        discriminator_out = self.netD(fake_and_real)

        pred_fake, pred_real = self.divide_pred(discriminator_out)

        return pred_fake, pred_real

    # Take the prediction of fake and real images from the combined batch
    def divide_pred(self, pred):
        # the prediction contains the intermediate outputs of multiscale GAN,
        # so it's usually a list
        if type(pred) == list:
            fake = []
            real = []
            for p in pred:
                fake.append([tensor[:tensor.size(0) // 2] for tensor in p])
                real.append([tensor[tensor.size(0) // 2:] for tensor in p])
        else:
            fake = pred[:pred.size(0) // 2]
            real = pred[pred.size(0) // 2:]

        return fake, real

    def get_edges(self, t):
        edge = self.ByteTensor(t.size()).zero_()
        edge[:, :, :, 1:] = edge[:, :, :, 1:] | (t[:, :, :, 1:] != t[:, :, :, :-1])
        edge[:, :, :, :-1] = edge[:, :, :, :-1] | (t[:, :, :, 1:] != t[:, :, :, :-1])
        edge[:, :, 1:, :] = edge[:, :, 1:, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
        edge[:, :, :-1, :] = edge[:, :, :-1, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
        return edge.float()

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std) + mu

    def use_gpu(self):
        return len(self.opt.gpu_ids) > 0
