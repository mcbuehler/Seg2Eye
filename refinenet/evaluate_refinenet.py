"""
Runs inference on datasets
"""

import argparse
import logging
import os
import time

import cv2 as cv
import numpy as np
import torch
from torch.utils.data import DataLoader

from core.checkpoint_manager import CheckpointManager
#import train_refinenet
from model import RefineNet
from dataset import OpenEDSDataset


input_size = (400, 640)
output_stride = 16


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logger = logging.getLogger(__name__)

# Parse argument
parser = argparse.ArgumentParser(description='Evaluate a trained RefineNet model.')
parser.add_argument('--input_dir', default='res/refinenet/', type=str, help='Saved model path')
parser.add_argument('--dataroot', default= 'res/openeds.h5', type=str, help='Root to dataset (h5)')
parser.add_argument('--segmentations_generative', default='res/segmentations_generative.h5', type=str, help='Segmentation mask predictions for unlabeled generative dataset')
parser.add_argument('--segmentations_sequence', default='res/segmentations_sequence.h5', type=str, help='Segmentation mask predictions for unlabeled sequence dataset')
parser.add_argument('--distances_and_indices', default='res/distances_and_indices.h5', type=str, help='File referencing nearest neighbour images')
args = parser.parse_args()
assert os.path.isdir(args.input_dir)

# Whether to show images during inference
show = False

# Create output handle
base_output_dir = os.path.join(args.input_dir, 'refinenet_submission_%s' % time.strftime('%y%m%d_%H%M%S'))
# (Un-)Comment out splits that you do not want to run inference on.
dataset_specs = [
    # 'train',
#    'validation',
    'test',
]

datasets = [
    OpenEDSDataset(args, split, pick1=True)
    for split in dataset_specs
]
dataloaders = [
    DataLoader(dataset,
               batch_size=32,
               shuffle=False,
               drop_last=False,
               num_workers=2,
               pin_memory=True,
               )
    for dataset in datasets
]

# Build Model
model = RefineNet(
    num_classes=1,
    backbone='resnet',
    output_stride=output_stride,
    sync_bn=False,
    freeze_bn=False,
)

model = model.to(device)
model.eval()

# Load checkpoint
model.output_dir = args.input_dir
checkpoint_manager = CheckpointManager(model)
checkpoint_manager.load_last_checkpoint()

# Iterate through
with torch.no_grad():
    for d, (dataset, dataloader) in enumerate(zip(datasets, dataloaders)):
        split = dataset_specs[d]
        output_dir = '%s/%s' % (base_output_dir, split)
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)

        all_ofpaths = []

        for b, input_dict in enumerate(dataloader):
            print(f"Processing batch {b}")
            input_dict_cuda = {}
            for k, v in input_dict.items():
                if isinstance(v, torch.Tensor):
                    input_dict_cuda[k] = v.detach().to(device, dtype=torch.float32,
                                                       non_blocking=True)
                else:
                    input_dict_cuda[k] = v

            # Inference
            output_dict = model(input_dict_cuda)
            predictions = output_dict['prediction'].detach().cpu().numpy()
            predictions = (255. / 2. * (predictions + 1.0)).astype(np.uint8)

            for i, prediction in enumerate(predictions):
                prediction = prediction[0, :]  # remove redundant dimension
                key = input_dict['person_id'][i]
                fname = input_dict['fname'][i]

                # Show sample image
                if show and i == 0 and b % 2 == 0: 
                    input_image = (255. / 2. * (input_dict['input'][i, 0, :].numpy() + 1.)).astype(np.uint8)  # noqa
                    cv.imshow('sample', cv.hconcat([input_image, prediction]))
                    cv.waitKey(1)

                # Save individual npy file
                ofpath = '%s/%s.npy' % (output_dir, fname)
                np.save(ofpath, prediction)
                all_ofpaths.append(ofpath)

        with open('%s/pred_npy_list.txt' % output_dir, 'w') as f:  # noqa
            for line in all_ofpaths:
                f.write(line)
                f.write(os.linesep)

        print('Wrote split: %s' % split)
