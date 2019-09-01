"""
Starting command:
python test.py --name $name --dataset_mode openeds \
    --dataroot $DATAROOT  --aspect_ratio 0.8 --no_instance \
    --load_size 256 --crop_size 256 --preprocess_mode fixed --batchSize 24 \
    --netG spade
    --write_error_log \
    --dataset_key train
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
from data.postprocessor import ImagePostprocessor
from data.preprocessor import ImagePreprocessor
from models.networks import openEDSaccuracy
from options.test_options import TestOptions
from models.pix2pix_model import Pix2PixModel
from util.validation import calculate_mse_for_images
from util.visualizer import Visualizer, visualize_sidebyside

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
    os.makedirs(results_dir)


# We have different behaviour for validation and test
is_validation = opt.dataset_key in ["validation", "train"]

# squared_losses_per_sample = {}
filepaths = list()

if is_validation and opt.write_error_log:
    error_log = h5py.File(os.path.join(results_dir, f"error_log_{opt.dataset_key}.h5"), "w")
    N = dataloader.dataset.N
    error_log.create_dataset("error", shape=(N,), dtype=np.float)
    error_log.create_dataset("user", shape=(N,), dtype='S4')
    error_log.create_dataset("filename", shape=(N,), dtype='S13')
    error_log.create_dataset("visualisation", shape=(N, 1, 320, 800), dtype=np.uint8)


print("Running test script.")
print(f"Is validation: {is_validation}. Dataset_key: {opt.dataset_key}")
print(f"write error log: {opt.write_error_log}")
# opt.how_many = 140


all_errors = list()
for i, data_i in enumerate(dataloader):
    if i * opt.batchSize >= opt.how_many:
        break
    # data_i["target_original"] = data_i["target_original"].unsqueeze(1)

    print(f"Processing batch {i} / {int(min(dataloader.dataset.N, opt.how_many) / opt.batchSize)}")

    fake = model.forward(data_i, mode="inference").cpu()

    # generated = model(data_i, mode='inference')

    img_filename = data_i['filename']
    # The test file names are only 12 characters long, so we have dot to remove
    img_filename = [re.sub(r'\.', '', f) for f in img_filename]

    # fake_images = ImagePreprocessor.unnormalize(np.copy(generated.detach().cpu()))

    # generated = ImagePreprocessor.fake_to_target(result_data["fake"])
    fake_resized = ImagePostprocessor.to_255resized_imagebatch(fake, as_tensor=True)

    if not is_validation:
    # We are testing
        for b in range(len(img_filename)):
            result_path = os.path.join(results_dir, img_filename[b] + ".npy")
            assert torch.min(fake_resized[b]) >= 0 and torch.max(fake_resized[b]) <= 255
            np.save(result_path, np.copy(fake_resized[b]).astype(np.uint8))
            filepaths.append(result_path)

    if is_validation:
        target_image = ImagePostprocessor.as_batch(data_i["target_original"], as_tensor=True)
        errors = np.array(calculate_mse_for_images(fake_resized, target_image))
        all_errors += list(errors)
        if opt.write_error_log:
            visualisation_data = {**data_i, "fake": fake}
            visuals = visualize_sidebyside(visualisation_data, visualizer, log=False)

            # We add the entire batch to the output file
            error_log["user"][i * opt.batchSize: i * opt.batchSize + opt.batchSize] = np.array(data_i["user"], dtype='S4')
            error_log["filename"][i * opt.batchSize: i * opt.batchSize + opt.batchSize] = np.array(data_i["filename"], dtype='S13')
            error_log["error"][i * opt.batchSize: i * opt.batchSize + opt.batchSize] = errors
            # error_log["visualisation"][i * opt.batchSize: i * opt.batchSize + opt.batchSize] = visualisations
            # vis = np.array([v for k, v in visuals.items()])
            vis = visuals
            vis = np.array([np.copy(v) for k, v in visuals.items()])
            error_log["visualisation"][i * opt.batchSize: i * opt.batchSize + opt.batchSize] = vis

if is_validation:
    N_actual = min(i * opt.batchSize + opt.batchSize, dataloader.dataset.N)
    print(f"Error calculated on {N_actual} / {dataloader.dataset.N} samples{os.linesep}"
          f"  sum: {np.sum(all_errors):.2f}  {os.linesep}"
          f"  sum relative to n=1471: {np.sum(all_errors)/len(all_errors)*1471:.2f} {os.linesep}"
          f"  mean: {np.mean(all_errors):.6f} (std: {np.std(all_errors):.4f})"
          f"  dataset_key: {opt.dataset_key}, model: {opt.name}")
    if opt.write_error_log:
        error_log.create_dataset("error_relative_n1471", data=np.multiply(error_log["error"], 1471), dtype=np.float)
        error_log.close()
else:
    # We are testing
    path_filepaths = os.path.join(results_dir, "pred_npy_list.txt")
    with open(path_filepaths, 'w') as f:
        for line in filepaths:
            f.write(line)
            f.write(os.linesep)
    print(f"Written {len(filepaths)} files. Filepath: {path_filepaths}")