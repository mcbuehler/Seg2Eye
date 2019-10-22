"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import torch
from util.tester import MSECalculator

from torch import nn
import models.networks as networks
import util.util as util



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

        # set loss functions
        if opt.isTrain:
            self.criterionGAN = networks.GANLoss(
                opt.gan_mode, tensor=self.FloatTensor, opt=self.opt)
            self.criterionFeat = nn.L1Loss()
            self.criterionL1 = nn.L1Loss()
            self.criterionL2 = nn.MSELoss()
            self.criterionOpenEDS = MSECalculator.calculate_mse_for_tensors
            if not opt.no_vgg_loss:
                self.criterionVGG = networks.VGGLoss(self.opt.gpu_ids)
            if opt.use_vae:
                self.KLDLoss = networks.KLDLoss()
            if opt.lambda_style_feat > 0:
                # loss on style feature maps
                self.criterion_style_feat = nn.MSELoss()
            if opt.lambda_style_w > 0:
                # Loss on latent style code
                self.criterion_style_w = nn.MSELoss()
            if opt.lambda_gram > 0:
                self.criterion_gram = networks.StyleLoss()
            self.reset_loss_log()

    def get_loss_log(self):
        loss_log = {key: torch.mean(torch.stack(self.loss_log[key])) for key in self.loss_log if len(self.loss_log[key])}
        return loss_log

    def add_to_loss_log(self, key, value):
        if not key in self.loss_log:
            self.loss_log[key] = list()
        self.loss_log[key].append(value)

    def reset_loss_log(self):
        self.loss_log = {}

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
            w, features = self.encode_w(style_image)
            return w
        elif mode == 'inference':
            with torch.no_grad():
                if 'latent_style' in data:
                    print("Using given latent style...")
                    fake_image = self.generate_fake_from_stylecode(input_semantics, data['latent_style'])
                else:
                    fake_image, _, _, _ = self.generate_fake(input_semantics, style_image)
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

    def _compute_style_feature_loss(self, features_fake, features_real):
        n_feature_maps = len(features_fake[0])
        n_batch = len(features_fake)
        assert n_feature_maps == len(features_real[0])
        losses = list()
        for i in range(n_feature_maps):
            feature_map_batch_fake = torch.stack([features_fake[b][i] for b in range(n_batch)])
            feature_map_batch_real = torch.stack([features_real[b][i] for b in range(n_batch)])
            feature_map_batch_fake.detach()
            losses.append(self.criterion_style_feat(feature_map_batch_fake, feature_map_batch_real))
        return torch.sum(torch.stack(losses))

    def _compute_gram_loss(self, features_fake, features_real):
        n_feature_maps = len(features_fake[0])
        n_batch = len(features_fake)
        assert n_feature_maps == len(features_real[0])
        losses = list()
        for i in range(n_feature_maps):
            feature_map_batch_fake = torch.stack([features_fake[b][i] for b in range(n_batch)])
            feature_map_batch_real = torch.stack([features_real[b][i] for b in range(n_batch)])
            feature_map_batch_fake.detach()
            losses.append(self.criterion_gram(feature_map_batch_fake, feature_map_batch_real))
        return torch.sum(torch.stack(losses))

    def compute_generator_loss(self, input_semantics, style_image, target_image):
        G_losses = {}

        fake_image, KLD_loss, latent_style_real, style_features_real = self.generate_fake(
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
            self.add_to_loss_log('L2/raw', l2_loss)
        if self.opt.lambda_l1:
            l1_loss = self.criterionL1(fake_image, target_image)
            G_losses['L1/weighted'] = l1_loss * self.opt.lambda_l1
            # Only for logging
            self.add_to_loss_log('L1/raw', l1_loss)
        if self.opt.lambda_openeds:
            openeds_loss = self.criterionOpenEDS(fake_image, target_image)
            G_losses['openeds/weighted'] = openeds_loss * self.opt.lambda_openeds
            # Only for logging
            self.add_to_loss_log('openeds/raw', openeds_loss)

        if self.opt.spadeStyleGen and \
                (self.opt.lambda_style_feat or self.opt.lambda_style_w or self.opt.lambda_gram):
            # We have some style consistency loss
            latent_style_fake, style_features_fake = self.encode_w(fake_image.unsqueeze(1))
            if self.opt.lambda_style_w > 0:
                # We need to expand the produced image to simulate several style images
                # It is important to detach the fake latent style, otherwise we learn the wrong thing.
                latent_style_fake.detach()
                style_w_loss_raw = self.criterion_style_w(latent_style_fake, latent_style_real)
                G_losses['style_w/weighted'] = style_w_loss_raw * self.opt.lambda_style_w
                self.add_to_loss_log('style_w/raw', style_w_loss_raw)
            if self.opt.lambda_style_feat > 0:
                style_feat_loss_raw = self._compute_style_feature_loss(style_features_fake, style_features_real)
                G_losses['style_feat/weighted'] = style_feat_loss_raw * self.opt.lambda_style_feat
                self.add_to_loss_log('style_feat/raw', style_feat_loss_raw)
            if self.opt.lambda_gram > 0:
                gram_loss_raw = self._compute_gram_loss(style_features_fake, style_features_real)
                G_losses['gram/weighted'] = gram_loss_raw * self.opt.lambda_gram
                self.add_to_loss_log('gram/raw', gram_loss_raw)

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
            fake_image, _ , _, _ = self.generate_fake(input_semantics, real_image)
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
        mu, logvar, features = self.netE(real_image)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar, features

    def _aggregate_tensor(self, tensor, dim=1):
        if self.opt.style_aggr_method == 'mean':
            return torch.mean(tensor, dim=dim)
        elif self.opt.style_aggr_method == 'max':
            # torch.max returns indices and values, but we only want values
            return torch.max(tensor, dim=dim).values
        else:
            raise ValueError(f"Aggregation method not found: {self.opt.style_aggr_method}")

    def _compute_multiple_netE(self, real_image):
        # We have several style images per input sample
        # for batch_i in range(real_image.shape[0]):
        #     # now we have n style images that we treat as a batch for one sample
        #     # mu will have shape (opt.input_ns, opt.dim_z)
        result = [self.netE(real_image[batch_i]) for batch_i in range(real_image.shape[0])]
        mu, logvar, features = zip(*result)
        outputs_netE_tensor = torch.stack(mu, dim=0)

        # batchSize, input_ns, z_dim if opt.use_z else w_dim
        if self.opt.use_z:
            assert outputs_netE_tensor.shape == (*real_image.shape[:2], self.opt.z_dim)
        else:
            assert outputs_netE_tensor.shape == (*real_image.shape[:2], self.opt.w_dim)
        return outputs_netE_tensor, features

    def _compute_aggregated_z(self, real_image):
        assert self.opt.use_z, "You need to set use_z to True"
        # We have several style images per input sample
        multiple_z, features = self._compute_multiple_netE(real_image)
        z = self._aggregate_tensor(multiple_z)
        assert z.shape == (real_image.shape[0], self.opt.z_dim)
        return z, features

    def _compute_aggregated_w(self, real_image):
        # We have several style images per input sample
        multiple_w, features = self._compute_multiple_netE(real_image)
        # w has shape (bs, w_dim)
        w = self._aggregate_tensor(multiple_w)

        features_aggregated = list()
        for batch_i in range(real_image.shape[0]):
            # Keep track of aggregated feature maps for all samples in batch
            for f in features[batch_i]:
                self._aggregate_tensor(f, dim=0)
            features_aggregated.append([self._aggregate_tensor(f, dim=0) for f in features[batch_i]])
        assert w.shape == (real_image.shape[0], self.opt.w_dim)
        return w, features_aggregated

    def encode_w(self, real_image):
        if len(real_image.shape) == 5:  #shape[1] is input_ns
            if self.opt.use_z and self.opt.style_aggr_space == 'z':
                # We aggregate in z space
                z, features = self._compute_aggregated_z(real_image)
                w = self.netE(z, mode='downscale')
            elif self.opt.style_aggr_space == 'w':
                # We compute input_ns mu vectors for each sample. multiple_z.shape = (bs, input_ns, z_dim)
                # This is the way we should be taking if we do not use a z space
                w, features = self._compute_aggregated_w(real_image)
        else:
            raise ValueError("real_image should have 5 dimensions")
        return w, features

    def generate_fake_from_stylecode(self, input_semantics, latent_style):
        fake_image = self.netG(input_semantics, latent_style)
        return fake_image

    def generate_fake(self, input_semantics, style_image, compute_kld_loss=False):
        latent_style = None
        KLD_loss = None
        features = None
        # TODO: be careful not to overwrite values once we have a VAE for w as well
        if self.opt.use_vae:
            latent_style, mu, logvar, features = self.encode_z(style_image)
            if compute_kld_loss:
                kld_loss_raw = self.KLDLoss(mu, logvar)
                KLD_loss = kld_loss_raw * self.opt.lambda_kld
                self.add_to_loss_log('KLD/raw', kld_loss_raw)

        if self.opt.spadeStyleGen:
            latent_style, features = self.encode_w(style_image)

        fake_image = self.generate_fake_from_stylecode(input_semantics, latent_style)

        assert (not compute_kld_loss) or self.opt.use_vae, \
            "You cannot compute KLD loss if opt.use_vae == False"

        return fake_image, KLD_loss, latent_style, features

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



class Pix2PixRefiner(Pix2PixModel):

    def forward(self, data, mode):
        start_tensor, input_semantics, style_image, target_image = self.preprocess_input(data)

        if mode == 'generator':
            g_loss, generated = self.compute_generator_loss(start_tensor,
                input_semantics, style_image, target_image)
            return g_loss, generated
        elif mode == 'discriminator':
            d_loss = self.compute_discriminator_loss(start_tensor,
                input_semantics, style_image, target_image)
            return d_loss
        elif mode == 'inference':
            with torch.no_grad():
                fake_image, _, _, _ = self.generate_fake(start_tensor, input_semantics, style_image)
                # We don't want to track our losses here as it could confound with training losses
                self.reset_loss_log()
            return fake_image
        else:
            raise ValueError("|mode| is invalid")

    def _preprocess_label_map(self, label_map):
        if len(label_map.shape) == 3:
            label_map = label_map.unsqueeze(0)
        bs, _, h, w = label_map.size()
        nc = self.opt.label_nc
        input_label = self.FloatTensor(bs, nc, h, w).zero_()
        input_semantics = input_label.scatter_(1, label_map, 1.0)
        return input_semantics

    def preprocess_input(self, data):
        # move to GPU and change data types
        data['label'] = data['label'].long()
        if self.use_gpu():
            data['label'] = data['label'].cuda()
            data['style_image'] = data['style_image'].cuda()
            data['start_tensor'] = data['start_tensor'].cuda()

        # create one-hot label map
        input_semantics = self._preprocess_label_map(data['label'])

        if "target" in data:
            if self.use_gpu():
                data['target'] = data['target'].cuda()
            return data['start_tensor'], input_semantics, data['style_image'], data['target']
        return data['start_tensor'], input_semantics, data['style_image'], None

    def generate_fake(self, start_tensor, input_semantics, style_image, compute_kld_loss=False):
        latent_style = None
        KLD_loss = None
        features = None
        if self.opt.spadeStyleGen:
            latent_style, features = self.encode_w(style_image)

        fake_image = self.netG(start_tensor, input_semantics, latent_style)

        assert (not compute_kld_loss) or self.opt.use_vae, \
            "You cannot compute KLD loss if opt.use_vae == False"

        return fake_image, KLD_loss, latent_style, features

    def compute_discriminator_loss(self, start_tensor, input_semantics, real_image, target_image):
        D_losses = {}
        with torch.no_grad():
            fake_image, _ , _, _ = self.generate_fake(start_tensor, input_semantics, real_image)
            fake_image = fake_image.detach()
            fake_image.requires_grad_()

        pred_fake, pred_real = self.discriminate(
            input_semantics, fake_image, target_image)

        D_losses['D/Fake'] = self.criterionGAN(pred_fake, False,
                                               for_discriminator=True)
        D_losses['D/real'] = self.criterionGAN(pred_real, True,
                                               for_discriminator=True)

        return D_losses

    def compute_generator_loss(self, start_tensor, input_semantics, style_image, target_image):
        G_losses = {}

        fake_image, KLD_loss, latent_style_real, style_features_real = self.generate_fake(start_tensor,
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
            self.add_to_loss_log('L2/raw', l2_loss)
        if self.opt.lambda_l1:
            l1_loss = self.criterionL1(fake_image, target_image)
            G_losses['L1/weighted'] = l1_loss * self.opt.lambda_l1
            # Only for logging
            self.add_to_loss_log('L1/raw', l1_loss)
        if self.opt.lambda_openeds:
            openeds_loss = self.criterionOpenEDS(fake_image, target_image)
            G_losses['openeds/weighted'] = openeds_loss * self.opt.lambda_openeds
            # Only for logging
            self.add_to_loss_log('openeds/relative', openeds_loss * 1471)

        if self.opt.spadeStyleGen and \
                (self.opt.lambda_style_feat or self.opt.lambda_style_w):
            # We have some style consistency loss
            latent_style_fake, style_features_fake = self.encode_w(fake_image.unsqueeze(1))
            if self.opt.lambda_style_w > 0:
                # We need to expand the produced image to simulate several style images
                # It is important to detach the fake latent style, otherwise we learn the wrong thing.
                latent_style_fake.detach()
                style_w_loss_raw = self.criterion_style_w(latent_style_fake, latent_style_real)
                G_losses['style_w/weighted'] = style_w_loss_raw * self.opt.lambda_style_w
                self.add_to_loss_log('style_w/raw', style_w_loss_raw)
            if self.opt.lambda_style_feat > 0:
                style_feat_loss_raw = self._compute_style_feature_loss(style_features_fake, style_features_real)
                G_losses['style_feat/weighted'] = style_feat_loss_raw * self.opt.lambda_style_feat
                self.add_to_loss_log('style_feat/raw', style_feat_loss_raw )

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

        return G_losses, fake_image

    def initialize_networks(self, opt):
        netG, netD, netE = super().initialize_networks(opt)
        if opt.pretrainD and opt.isTrain and not opt.continue_train:
            netD = util.load_network(netD, 'D', opt.which_epoch, opt, opt.pretrained_path)
        return netG, netD, netE
