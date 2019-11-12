"""Main script for training a model for eye shape segmentation."""
import logging
import sys

import cv2 as cv
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from core import DefaultConfig

input_size = (400, 640)

config = DefaultConfig()
logger = logging.getLogger(__name__)


class OpenEDSDataset(Dataset):

    def __init__(self, paths, split: str, augment: bool = None,
                 pick1: bool = False):
        self.path = paths.dataroot
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
            self.map_hdf = h5py.File(self.path_segmentations_generative, 'r')
            self.map_hdf_seq = h5py.File(self.path_segmentations_sequence, 'r')

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
