"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import os
import re
from collections import OrderedDict

import cv2
import h5py
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


results_dir = os.path.join(opt.checkpoints_dir, opt.name, opt.results_dir, opt.dataset_key)
if not os.path.exists(results_dir):
    os.mkdir(results_dir)


# We have different behaviour for validation and test
is_validation = opt.dataset_key in [ "validation", "train"]

# squared_losses_per_sample = {}
filepaths = list()

if is_validation and opt.write_error_log:
    error_log = h5py.File(os.path.join(results_dir, f"error_log_{opt.dataset_key}.h5"), "w")
    N = dataloader.dataset.N
    error_log.create_dataset("error", shape=(N,), dtype=np.float)
    error_log.create_dataset("user", shape=(N,), dtype='S4')
    error_log.create_dataset("filename", shape=(N,), dtype='S13')
    error_log.create_dataset("visualisation", shape=(N, 640, 1200), dtype=np.uint8)


print("Running test script.")
print(f"Is validation: {is_validation}. Dataset_key: {opt.dataset_key}")
print(f"write error log: {opt.write_error_log}")
# opt.how_many = 250


total_error = 0
for i, data_i in enumerate(dataloader):
    if i * opt.batchSize >= opt.how_many:
        break

    print(f"Processing batch {i} / {int(min(dataloader.dataset.N, opt.how_many) / opt.batchSize)}")

    generated = model(data_i, mode='inference')

    img_filename = data_i['filename']
    # The test file names are only 12 characters long, so we have dot to remove
    img_filename = [re.sub(r'\.', '', f) for f in img_filename]

    fake_images = ImagePreprocessor.unnormalize(np.copy(generated.detach().cpu()))

    visualisations = list()
    errors = list()
    for b in range(generated.shape[0]):
        log_index = i * opt.batchSize + b
        fake_np = __resize(fake_images[b][0], 400, 640, cv2.INTER_NEAREST)
        # fake_np = fake_images[b][0]
        if is_validation:
            true_np = np.copy(data_i["image_original"][b].detach().cpu())

            if opt.write_error_log:
                # We create visualisations
                label_np = np.copy(data_i["label"][b][0].detach().cpu())
                # label_np = __resize(label_np.astype(np.float), 400, 640, cv2.INTER_LINEAR).astype(np.uint)
                label_np = __resize(label_np.astype(np.float), 400, 640, cv2.INTER_NEAREST).astype(np.uint)
                label_np = label_np / np.max(label_np) * 255
                cat = np.concatenate([label_np, true_np, fake_np], axis=1)
                visualisations.append(cat)

            error = calculate_mse_for_images(fake_np, true_np)
            errors.append(error)

        else:
            # We are testing
            result_path = os.path.join(results_dir, img_filename[b] + ".npy")
            np.save(result_path, fake_np.astype(np.uint8))
            filepaths.append(result_path)

    if is_validation and opt.write_error_log:
        # We add the entire batch to the output file
        error_log["user"][i * opt.batchSize: i * opt.batchSize + opt.batchSize] = np.array(data_i["user"], dtype='S4')
        error_log["filename"][i * opt.batchSize: i * opt.batchSize + opt.batchSize] = np.array(data_i["filename"], dtype='S13')
        error_log["error"][i * opt.batchSize: i * opt.batchSize + opt.batchSize] = errors
        error_log["visualisation"][i * opt.batchSize: i * opt.batchSize + opt.batchSize] = visualisations
    total_error += np.sum(errors)

if is_validation:
    error_log.create_dataset("error_relative_n1471", data=np.multiply(error_log["error"], 1471), dtype=np.float)
    N_actual = min(i * opt.batchSize + opt.batchSize, dataloader.dataset.N)
    print(f"Total error calculated on {N_actual} / {dataloader.dataset.N} samples: {total_error}")
    # number in test set, this is the relative number we have to use in order to compare to the leaderboard
    N_test = 1471
    ratio = N_test / N_actual
    print(f"Total error (relative to test set): {total_error * ratio}")
    if opt.write_error_log:
        error_log.close()
else:
    # We are testing
    print(filepaths)
    path_filepaths = os.path.join(results_dir, "pred_npy_list.txt")
    with open(path_filepaths, 'w') as f:
        for line in filepaths:
            f.write(line)
            f.write(os.linesep)
    print("Done")