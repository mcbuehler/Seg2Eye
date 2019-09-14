"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import json
import re

import h5py
import numpy as np
import torch
from cv2 import cv2

from data.base_dataset import BaseDataset, get_params, get_transform, flip


class OpenEDSDataset(BaseDataset):
    style_image_refs = None
    h5_in_file = None

    def __init__(self):
        super().__init__()

    def _setup_data_file(self):
        if self.h5_in_file is None:
            self.h5_in_file = h5py.File(self.root, 'r')  # , libver='latest')
            self.h5_in = self.h5_in_file[self.dataset_key]

        if 'ref' in self.opt.style_sample_method and self.style_image_refs is None:
            assert self.opt.style_ref != '', "You need to provide a h5 file for style references."
            self.style_image_refs = h5py.File(self.opt.style_ref, 'r')

    @staticmethod
    def modify_commandline_options(parser, is_train):
        # parser.add_argument('--no_pairing_check', action='store_true',
        #                     help='If specified, skip sanity check of correct label-image file pairing')
        return parser

    def initialize(self, opt):  #TODO: print style sampling method
        self.opt = opt
        print(f"Style sampling method: {self.opt.style_sample_method}")

        self.root = opt.dataroot
        self.dataset_key = opt.dataset_key
        print(f"Dataset key: {self.dataset_key}")
        self.key_style_images = "images_ss" if self.dataset_key == "test" else "images_gen"
        self.label_key = "labels_ss" if self.dataset_key != "test" else "labels_gen"
        self.key_filenames = "labels_gen_filenames" if self.dataset_key == "test" else "images_ss_filenames"

        self._setup_data_file()

        self.user_ids = list(self.h5_in.keys())

        self.N = 0
        self.N_start = list()
        for user in self.user_ids:
            self.N_start.append(self.N)
            keys = ['labels_gen_filenames'] if self.dataset_key == "test" else [
                'images_ss_filenames']  # TODO: leverage 'images_gen_filenames'
            # for k in self.h5_in[user].keys():
            #     print(k, self.h5_in[user][k].shape)
            for key in keys:
                if key in self.h5_in[user]:
                    self.N += self.h5_in[user][key].shape[0]
        self.h5_in_file.close()
        self.h5_in_file = None

    def _get_tuple_identifier_from_index(self, index):
        idx_user = 0
        for i in range(len(self.user_ids)):
            if index >= self.N_start[i]:
                idx_user = i
            else:
                break
        within_index = index - self.N_start[idx_user]
        return self.user_ids[idx_user], within_index

    def __getitem__(self, index):
        self._setup_data_file()

        # Input to Generator: 1+ style and 1 content image (segmentation mask)
        user, idx_target_image = self._get_tuple_identifier_from_index(index)
        # style_image = np.random.choice(self.h5_in[user]["images_gen"])

        mask = self.h5_in[user][self.label_key][idx_target_image]
        # It is important to use nearest neighbour interpolation for resizing the mask. Otherwise we might get
        # values that are out of the allowed range.
        params = get_params(self.opt, mask.shape)  # Only give h and w
        transform_mask = get_transform(self.opt, params,  method=cv2.INTER_NEAREST, normalize=False)
        # the toTensor method in transform will convert uint8 [0, 255] to foat [-1, 1], so we need to revert this.
        mask_tensor = transform_mask(mask) * 255.0

        filename = self.h5_in[user][self.key_filenames][idx_target_image].decode('utf-8')
        filename = re.sub(f'\.', '', filename)
        # Get input_ns style images (already preprocessed to tensor)
        style_image_tensor = self.get_style_images(user, self.opt.input_ns, filename)

        if torch.max(mask_tensor) > 3:
            print(user, idx_target_image, filename)
            print(np.max(mask))
            print(torch.max(mask_tensor))

        input_dict = {'label': mask_tensor,
                      'instance': 0,
                      'filename': filename,
                      'user': user,
                      'style_image': style_image_tensor
                      }
        if self.dataset_key != "test":
            transform_image = get_transform(self.opt, params)
            target_image = np.array(self.h5_in[user]["images_ss"][idx_target_image])
            target_image_tensor = transform_image(target_image)
            # Only flip the target image now, otherwise it might be flipped twice
            target_image = flip(target_image, params['flip'])
            input_dict = {**input_dict,
                          'target': target_image_tensor,
                          'target_original': torch.from_numpy(np.expand_dims(target_image, axis=0)).int()
                          }
        return input_dict

    def get_particular(self, idx):
        batch = self[idx]
        return self.unsqueeze_batch(batch)

    def postprocess(self, input_dict):
        return input_dict

    def __len__(self):
        return self.N

    def _get_transform(self, params, **kwargs):
        transform_image = get_transform(self.opt, params, **kwargs)
        return transform_image, params

    def get_validation_indices(self):
        # All first indices of  a person
        indices = self.N_start
        # All last indices of a person
        indices += [idx - 1 for idx in self.N_start[1:]] + [self.N - 1]
        return indices

    def get_random_indices(self, n):
        indices = np.random.choice(list(range(self.N)), n)
        return indices

    def _sample_style_idx(self, n_images, n, user_id=None, filename=None):
        subsets = None
        if self.opt.style_sample_method == 'random':
            indices = np.random.choice(list(range(n_images)), n)
            # We should give the subset key as well
        elif self.opt.style_sample_method == 'first':
            indices = list(range(min(n, n_images)))
            # We should give the subset key as well
            indices = [(self.key_style_images, idx) for idx in indices]
        elif 'ref' in self.opt.style_sample_method:
            use_sequence_data = 'subset' in list(self.style_image_refs[self.opt.dataset_key][user_id][filename].keys())
            all_indices = self.style_image_refs[self.opt.dataset_key][user_id][filename]['index']
            if use_sequence_data:
                all_subsets = self.style_image_refs[self.opt.dataset_key][user_id][filename]['subset']
            if 'random' in self.opt.style_sample_method:
                reduced_n = re.sub(r"[^\d]", "", self.opt.style_sample_method)
                # e.g. 0.3 for taking the nearest 40%
                if reduced_n:
                    reduced_n = int(reduced_n)
                else:
                    reduced_n = 40
                # reduced_indices = all_indices[:reduced_n]
                # to_select only contains indices from 0 to reduced_n
                to_select = np.random.choice(list(range(reduced_n)), n)
                indices = [all_indices[to_select[i]] for i in range(n)]
                if use_sequence_data:
                    subsets = [all_subsets[to_select[i]] for i in range(n)]
            else:
                # Best first
                # We should give the subset key as well
                indices = all_indices[:n]
                if use_sequence_data:
                    subsets = all_subsets[:n]
        else:
            raise ValueError(f"Invalid style sampling method: {self.opt.style_sample_method}")
        return indices, subsets

    def get_style_images(self, user_id, n, filename=None):
        # user_idx = self.user_ids.index(user_id)
        # n_user = self.N_start[user_idx + 1] - self.N_start[user_idx]
        # within_idx = np.random.choice(list(range(n_user)), size=n)
        # selected_idx = [self.N_start[user_idx] + i for i in within_idx]
        # return selected_idx
        n_images = self.h5_in[user_id][self.key_style_images].shape[0]
        selected_idx, subsets = self._sample_style_idx(n_images, n, user_id=user_id, filename=filename)

        subset_keys = {b'g': self.key_style_images, b's': 'images_seq'}
        style_images = list()
        for i, sel_i in enumerate(selected_idx):
            subset_key = subset_keys[subsets[i]]
            # The indices for the seq dataset are too big (they were appended to the number in the gen dataset)
            sel_i = sel_i - n_images
            style_images.append(self.h5_in[user_id][subset_key][sel_i])

        # Preprocessing
        size = style_images[0].shape[-2:]
        # Convert to tensor
        params = get_params(self.opt, size)  # Only give h and w
        transform_image = get_transform(self.opt, params)
        tensors = [transform_image(img) for img in style_images]
        style_image_tensor = torch.stack(tensors)
        return style_image_tensor

    @classmethod
    def unsqueeze_batch(cls, batch):
        # Create a first batch size dimension
        for key in ["style_image", "target", "target_original", "label"]:
        # for key in ["target_original"]:
            batch[key] = batch[key].unsqueeze(0)
        return batch

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.h5_in_file is not None:
            self.h5_in_file.close()
            self.h5_in_file = None
            self.h5_in = None
