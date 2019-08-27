"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import os
import re
from collections import OrderedDict

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch import functional

import data
from data.base_dataset import __resize
from data.preprocessor import ImagePreprocessor
from models.networks import openEDSaccuracy
from options.test_options import TestOptions
from models.pix2pix_model import Pix2PixModel
from util.validation import calculate_mse_for_images
from util.visualizer import Visualizer
from util import html

opt = TestOptions().parse()
# opt.serial_batches = False
dataloader = data.create_dataloader(opt)

model = Pix2PixModel(opt)
model.eval()


visualizer = Visualizer(opt)

base_path = os.getcwd()
if opt.checkpoints_dir.startswith("./"):
    opt.checkpoints_dir = os.path.join(base_path, opt.checkpoints_dir[2:])
else:
    opt.checkpoints_dir = os.path.join(base_path, opt.checkpoints_dir)


results_dir = os.path.join(opt.checkpoints_dir, opt.name, opt.results_dir)
# create a webpage that summarizes the all results
web_dir = os.path.join(opt.checkpoints_dir, opt.name, "web",
                       '%s_%s' % (opt.phase, opt.which_epoch))
if not os.path.exists(results_dir):
    os.mkdir(results_dir)

webpage = html.HTML(web_dir,
                    'Experiment = %s, Phase = %s, Epoch = %s' %
                    (opt.name, opt.phase, opt.which_epoch))

# We have different behaviour for validation and test
is_validation = opt.dataset_key == "validation"

# squared_losses_per_sample = {}
filepaths = list()
errors = {}

# opt.how_many = 10

for i, data_i in enumerate(dataloader):
    if i * opt.batchSize >= opt.how_many:
        break

    print(f"Processing batch {i} / {int(dataloader.dataset.N / opt.batchSize)}")

    generated = model(data_i, mode='inference')

    img_filename = data_i['filename']
    # The test file names are only 12 characters long, so we have dot to remove
    img_filename = [re.sub(r'\.', '', f) for f in img_filename]

    fake_images = ImagePreprocessor.unnormalize(np.copy(generated.detach().cpu()))

    for b in range(generated.shape[0]):
        # print('process image... %s' % img_filename[b])
        # visuals = OrderedDict([('input_label', data_i['label'][b]),
        #                        ('synthesized_image', generated[b])])
        # visualizer.save_images(webpage, visuals, img_filename[b:b + 1])

        # target_tensor = ImagePreprocessor.unnormalize(data_i['image'][b])

        # take the first channel
        # fake_np = __resize(fake_images[b][0], 400, 640, cv2.INTER_LINEAR)
        fake_np = __resize(fake_images[b][0], 400, 640, cv2.INTER_NEAREST)
        # fake_np = fake_images[b][0]
        if is_validation:
            true_np = np.copy(data_i["image_original"][b].detach().cpu())

            if False:
                label_np = np.copy(data_i["label"][b][0].detach().cpu())
                # label_np = __resize(label_np.astype(np.float), 400, 640, cv2.INTER_LINEAR).astype(np.uint)
                label_np = __resize(label_np.astype(np.float), 400, 640, cv2.INTER_NEAREST).astype(np.uint)
                label_np = label_np / np.max(label_np) * 255
                # cat = np.concatenate([label_np, true_np, fake_np], axis=1)
                # figure = Image.fromarray(cat)
                # plt.imshow(figure)
                # plt.show()

            errors[img_filename[b]] = calculate_mse_for_images(fake_np, true_np)
            # errors[img_filename[b]] = calculate_mse_for_images(fake_np, label_np)

        else:
            # We are testing
            result_path = os.path.join(results_dir, img_filename[b] + ".npy")
            np.save(result_path, fake_np.astype(np.uint8))
            filepaths.append(result_path)

        # plt.imshow(fake_np, cmap='gray')
        # if 1:
        #     plt.show()
        # result_path_img = os.path.join(results_dir, img_filename[b] + ".png")
        # plt.savefig(result_path_img)

        # l_x = fake_tensor.shape[2]
        # l_y = fake_tensor.shape[1]


        # acc = openEDSaccuracy(fake_tensor, target_tensor)
        # print(acc)
        # squared_losses_per_sample[img_filename[b]] = acc

        # nn.MSELoss()
        # target = np.copy(data_i['image'][b][0].cpu())
        # plt.imshow(np.copy(fake_tensor.detach().cpu())[0])
        # plt.show()
        # plt.imshow()
        # plt.show()

if is_validation:
    print(errors)
    print(f"Total error: {np.sum([e for e in errors.values()])}")
else:
    # We are testing
    print(filepaths)
    path_filepaths = os.path.join(results_dir, "pred_npy_list.txt")
    with open(path_filepaths, 'w') as f:
        for line in filepaths:
            f.write(line)
            f.write(os.linesep)
# estimated_error = np.sum(squared_losses_per_sample.values())
# print(f"Estimated error: {estimated_error}")
webpage.save()

