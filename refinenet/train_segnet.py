#!/usr/bin/env python3
"""Main script for training a model for eye shape segmentation."""
import logging
import sys

import cv2 as cv
import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from core import DefaultConfig
import core.training as training

sys.path.insert(0, './deeplab')
from deeplab.modeling.deeplab import DeepLab  # noqa
del sys.path[0]

input_size = (400, 640)
output_stride = 16

config = DefaultConfig()
logger = logging.getLogger(__name__)


class OpenEDSDataset(Dataset):

    def __init__(self, dataset_path: str, split: str, augment: bool = None):
        self.path = dataset_path
        self.split = split
        hdf = h5py.File(self.path, 'r')
        self.hdf = None

        # Construct mapping from full-data index to key and person-specific index
        self.idx_to_kv = []
        for person_id in list(hdf[split].keys()):
            n = hdf[split][person_id]['images_ss'].shape[0]
            self.idx_to_kv += [(person_id, i) for i in range(n)]

        logger.info('Loaded HDF dataset at: %s' % self.path)

    def __len__(self):
        return len(self.idx_to_kv)

    def preprocess_image(self, image):
        image = image.astype(np.float32)
        image = cv.resize(image, input_size, interpolation=cv.INTER_AREA)
        image *= 2.0 / 255.0
        image -= 1.0
        image = np.expand_dims(image, 0)  # HW -> CHW
        image = np.repeat(image, 3, axis=0)
        return image

    def __getitem__(self, idx):
        if self.hdf is None:
            self.hdf = h5py.File(self.path, 'r')

        key, idx = self.idx_to_kv[idx]
        person_data = self.hdf[self.split][key]

        x_key = 'images_ss'
        y_key = None
        if 'labels_ss' in person_data:  # NOT TEST SPLIT
            y_key = 'labels_ss'

        # Get images, face (64x64) and eyes (256x64)
        x = self.preprocess_image(np.copy(person_data[x_key][idx, :]))
        entry = {'image': x}

        # Get labels and other data
        if y_key is not None:
            y = np.copy(person_data[y_key][idx, :])
            # y[y == 0] = 0
            # y[(y == 0) | (y == 1) | (y == 2)] = 1
            y = cv.resize(y, dsize=input_size, interpolation=cv.INTER_AREA)
            entry['segmentation'] = y

        # Convert arrays to tensors
        return dict([
            (k, torch.from_numpy(a))
            for k, a in entry.items()
        ])


class MyDeepLab(DeepLab):
    def forward(self, input_dict):
        x = None
        if self.training:
            x = input_dict['train']['image']
            y_true = input_dict['train']['segmentation']
        else:
            x = input_dict['image']
            y_true = input_dict['segmentation'] if 'segmentation' in input_dict else None

        output_dict = {'input': x}

        # Make prediction
        y_pred = super().forward(x)
        # y_pred = torch.sigmoid(y_pred)
        output_dict['prediction'] = torch.argmax(y_pred, dim=1)

        # Loss calculation
        if y_true is not None:
            output_dict['bce_loss'] = F.cross_entropy(y_pred, y_true.long())
            output_dict['groundtruth'] = y_true

        return output_dict


if __name__ == '__main__':
    config, device, dataset_splits = training.script_init_common()
    datasrc = '/big/marcel/all.h5'
    train_data, test_data = training.init_datasets(
        # Train
        [
            ('train', OpenEDSDataset, datasrc, 'train'),
        ],
        # Validation
        [
            ('val', OpenEDSDataset, datasrc, 'validation'),
            ('test', OpenEDSDataset, datasrc, 'test'),
        ],
    )

    # Define model
    model = MyDeepLab(
        num_classes=4,
        backbone='resnet',
        output_stride=output_stride,
        sync_bn=False,
        freeze_bn=False,
    )
    print(model)
    model = model.to(device)

    # Optimizer
    optimizers = [
        torch.optim.SGD(
            model.parameters(),
            lr=config.learning_rate,
            momentum=0.9, nesterov=True, weight_decay=config.weight_decay,
        ),
    ]

    # Setup
    model, optimizers, tensorboard = training.setup_common(model, optimizers)

    # Training
    for current_step, loss_terms, outputs, images_to_log_to_tensorboard \
            in training.main_loop_iterator(model, optimizers, train_data, test_data, tensorboard):

        # Train with angular error
        loss_terms.append(outputs['bce_loss'])

        # Log some training images
        if training.step_modulo(current_step, config.tensorboard_images_every_n_steps):
            num_images = 4
            all_images = [
                (255. / 2. * (outputs['input'][:num_images, 0, :].cpu().numpy() + 1.0)).astype(np.uint8),  # noqa
                (255. / 3. * outputs['prediction'][:num_images, :].cpu().numpy()).astype(np.uint8),  # noqa
            ]

            if 'groundtruth' in outputs:
                all_images.append((255. / 3. * outputs['groundtruth'][:num_images, :].cpu().numpy()).astype(np.uint8))  # noqa

            for i in range(1, num_images + 1):
                images_to_log_to_tensorboard['predictions/%d' % i] = \
                    np.expand_dims(cv.hconcat([
                        cv.resize(imgs[i - 1, :], (400, 640))
                        for imgs in all_images
                    ]), 0)

    # Do final test on full test sets (without subsampling)
    training.do_final_full_test(model, test_data, tensorboard)

    # Exit without hanging
    sys.exit(0)
