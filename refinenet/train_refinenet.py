#!/usr/bin/env python3
"""Main script for training a model for eye shape segmentation."""
import logging
import sys

import cv2 as cv
import h5py
import numpy as np
import torch
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

    def __init__(self, paths, split: str, augment: bool = None,
                 pick1: bool = False):
        self.path = paths.dataroot

        self.path_segmentations_train = paths.segmentations_train
        # These are segmentations for the validation set
        self.path_segmentations_generative = paths.segmentations_generative
        self.path_segmentations_sequence = paths.segmentations_sequence

        self.path_distances_and_indices = paths.distances_and_indices
        self.split = split
        self.pick1 = pick1
        if self.split == 'test':
            assert self.pick1 is True
        hdf = h5py.File(self.path, 'r')
        self.hdf = None

        # Construct mapping from full-data index to key and person-specific index
        self.idx_to_kv = []
        all_person_ids = list(hdf[split].keys())
        if split == 'train':
            all_person_ids.remove('U111')
        for person_id in all_person_ids:
            if split == 'test':
                n = hdf[split][person_id]['labels_gen'].shape[0]
            else:
                n = hdf[split][person_id]['labels_ss'].shape[0]
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
        return image

    def colorize_segmap(self, segmap):
        all_means = np.array([
            125.73929,  # Eyelid area
            103.19314,  # Sclera area
            76.50751,  # Iris area
            34.1294,  # Pupil
        ])
        out = np.empty(segmap.shape, dtype=np.uint8)
        for j, v in enumerate(all_means):
            out[segmap == j] = v
        return out

    def __getitem__(self, idx):
        if self.hdf is None:
            self.hdf = h5py.File(self.path, 'r')
            self.idx_hdf = h5py.File(self.path_distances_and_indices, 'r')  # noqa
            if self.split == 'validation' or self.split == 'test':
                self.map_hdf = h5py.File(self.path_segmentations_generative, 'r')  # noqa: val+test/images only
                self.map_hdf_seq = h5py.File(self.path_segmentations_sequence, 'r')  # noqa: val+test/images_seq only
            else:
                # 'deeplab_predictions_190912_161737.h5'
                self.map_hdf = h5py.File(self.path_segmentations_train, 'r')  # noqa: train/images only

        key, idx = self.idx_to_kv[idx]
        person_data = self.hdf[self.split][key]

        if self.split != 'test':
            x = person_data['labels_ss'][idx, :]
            y = person_data['images_ss'][idx, :]
            fname = person_data['labels_ss_filenames'][idx].decode('utf-8').replace('.', '')

            nn_data = self.idx_hdf[self.split][key][fname]
            candidate_indices = nn_data['index']
            rpos = (
                np.random.randint(0, len(candidate_indices))
                if self.pick1 is False else 0
            )
            ridx = candidate_indices[rpos]
            rfrom = nn_data['subset'][rpos]

            if rfrom == b'g':
                rim = person_data['images_gen'][ridx, :]
                rss = self.map_hdf[self.split][key][ridx, :]
            elif rfrom == b's':
                num_generative = person_data['images_gen'].shape[0]
                rim = person_data['images_seq'][ridx - num_generative, :]
                rss = self.map_hdf_seq[self.split][key][ridx - num_generative, :]
            else:
                raise ValueError('Unknown subset source: %s' % rfrom)
        else:
            x = person_data['labels_gen'][idx, :]
            y = None
            fname = person_data['labels_gen_filenames'][idx].decode('utf-8').replace('.', '')

            nn_data = self.idx_hdf[self.split][key][fname]
            candidate_indices = nn_data['index']
            rpos = 0
            ridx = candidate_indices[rpos]
            rfrom = nn_data['subset'][rpos]

            if rfrom == b'g':
                rim = person_data['images_ss'][ridx, :]
                rss = self.map_hdf[self.split][key][ridx, :]
            elif rfrom == b's':
                num_generative = person_data['images_ss'].shape[0]
                rim = person_data['images_seq'][ridx - num_generative, :]
                rss = self.map_hdf_seq[self.split][key][ridx - num_generative, :]
            else:
                raise ValueError('Unknown subset source: %s' % rfrom)

        # Colorize all segmentation maps
        x = self.colorize_segmap(x)
        rss = self.colorize_segmap(rss)

        # Form input image
        x = np.concatenate([self.preprocess_image(img) for img in [x, rim, rss]], axis=0)
        entry = {'input': x, 'person_id': key, 'fname': fname}

        # Get labels and other data
        if y is not None:
            entry['target'] = self.preprocess_image(np.copy(y))

        # Convert arrays to tensors
        return dict([
            (k, torch.from_numpy(a) if isinstance(a, np.ndarray) else a)
            for k, a in entry.items()
        ])


class RefineNet(DeepLab):
    def forward(self, input_dict):
        x = None
        if self.training:
            x = input_dict['train']['input']
            y_true = input_dict['train']['target']
            input_dict = input_dict['train']
        else:
            x = input_dict['input']
            y_true = input_dict['target'] if 'target' in input_dict else None

        output_dict = {'input': x}

        # Make prediction
        network_output = super().forward(x)
        output_dict['residual'] = network_output

        # Alpha-blend style image and new pixels
        reference_image = x[:, 1, :, :].unsqueeze(1)
        y_pred = torch.clamp(network_output + reference_image,
                             min=-1.0, max=1.0)
        output_dict['prediction'] = y_pred

        # Copy through some strings
        output_dict['person_id'] = input_dict['person_id']
        output_dict['fname'] = input_dict['fname']

        # Loss calculation
        if y_true is not None:
            output_dict['l1_loss'] = torch.mean(torch.abs(y_pred - y_true))
            output_dict['groundtruth'] = y_true

            output_dict['per_image_score'] = torch.sqrt(
                torch.sum(
                    (255. / 2. * (y_pred - y_true)) ** 2,
                    dim=[1, 2, 3],
                )
            ) / float(np.prod(y_true.shape[2:]))  # at this point, 1 scalar per entry

            output_dict['score'] = 1471 * torch.sum(
                output_dict['per_image_score']
            ) / float(y_true.shape[0])  # at this point, 1 scalar

            # Allow direct optimizing of what we believe is the OpenEDS loss
            output_dict['eds_loss'] = torch.mean(output_dict['per_image_score'])

        return output_dict


if __name__ == '__main__':
    config, device, dataset_splits = training.script_init_common()
    datasrc = '/big/marcel/190910_all.h5'
    train_data, test_data = training.init_datasets(
        # Train
        [
            ('train', OpenEDSDataset, datasrc, 'train'),
        ],
        # Validation
        [
            ('val', OpenEDSDataset, datasrc, 'validation'),
            ('val/pick1', OpenEDSDataset, datasrc, 'validation', {'pick1': True}),
            ('test', OpenEDSDataset, datasrc, 'test', {'pick1': True}),
        ],
    )

    # Define model
    model = RefineNet(
        num_classes=1,
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
            weight_decay=config.weight_decay,
            momentum=0.99, nesterov=True,
        ),
    ]

    # Setup
    model, optimizers, tensorboard = training.setup_common(model, optimizers)

    # Training
    for current_step, loss_terms, outputs, images_to_log_to_tensorboard \
            in training.main_loop_iterator(model, optimizers, train_data, test_data, tensorboard):

        # Train with angular error
        loss_terms.append(outputs['eds_loss'])

        # Log some training images
        if training.step_modulo(current_step, config.tensorboard_images_every_n_steps):
            num_images = 4
            def convert_to_uint_rgb(tensor, shift=True):  # noqa
                cpu_tensor = tensor.detach().cpu().numpy()
                tensor_shifted = cpu_tensor + 1.0 if shift else cpu_tensor
                tensor_scaled = 255. / 2. * tensor_shifted
                tensor_clipped = np.clip(tensor_scaled, 0.0, 255.)
                return tensor_clipped.astype(np.uint8)
            all_images = [
                convert_to_uint_rgb(outputs['input'][:num_images, 0, :]),
                convert_to_uint_rgb(outputs['input'][:num_images, 1, :]),
                convert_to_uint_rgb(outputs['input'][:num_images, 2, :]),
                convert_to_uint_rgb(outputs['residual'][:num_images, 0, :]),
                convert_to_uint_rgb(outputs['prediction'][:num_images, 0, :]),
            ]

            if 'groundtruth' in outputs:
                all_images.append(
                    convert_to_uint_rgb(outputs['groundtruth'][:num_images, 0, :]))
                all_images.append(np.abs(
                    all_images[-2].astype(np.float32) -
                    all_images[-1].astype(np.float32)).astype(np.uint8))

            for i in range(1, num_images + 1):
                images_concatenated = cv.hconcat([
                    cv.resize(imgs[i - 1, :], (400, 640)) for imgs in all_images
                ])
                foot = np.zeros((130, images_concatenated.shape[1]), dtype=np.uint8)
                text_to_put = '%s / %s' % (outputs['person_id'][i], outputs['fname'][i])
                if 'per_image_score' in outputs:
                    text_to_put += ' (err: %.2f)' % (1471 * outputs['per_image_score'][i])
                cv.putText(foot, text_to_put, (20, 88), cv.FONT_HERSHEY_DUPLEX, fontScale=3,
                           color=(255, 255, 255), thickness=3, lineType=cv.LINE_AA)
                images_to_log_to_tensorboard['predictions/%d' % i] = \
                    np.expand_dims(cv.vconcat([images_concatenated, foot]), 0)

    # Do final test on full test sets (without subsampling)
    training.do_final_full_test(model, test_data, tensorboard)

    # Exit without hanging
    sys.exit(0)
