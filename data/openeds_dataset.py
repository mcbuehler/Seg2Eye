"""
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import re

import h5py
import numpy as np
import torch
from cv2 import cv2

from data.base_dataset import BaseDataset, get_params, get_transform, flip


class OpenEDSDataset(BaseDataset):
    style_image_refs = None
    h5_in_file = None
    pred_seg_file = None

    def __init__(self):
        super().__init__()

    def _setup_data_file(self):
        if self.h5_in_file is None:
            self.h5_in_file = h5py.File(self.root, 'r')  # , libver='latest')
            self.h5_in = self.h5_in_file[self.dataset_key]

        if 'ref' in self.opt.style_sample_method and self.style_image_refs is None:
            assert self.opt.style_ref != '', "You need to provide a h5 file for style references."
            self.style_image_refs = h5py.File(self.opt.style_ref, 'r')

        if self.opt.netG == 'spaderefiner':
            assert self.opt.seg_file != ''
            self.pred_seg_file = h5py.File(self.opt.seg_file, 'r')

    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def initialize(self, opt):  #TODO: print style sampling method
        self.opt = opt

        self.root = opt.dataroot
        self.dataset_key = opt.dataset_key
        # The keys of the style images, labels and filenlames differ for the test set
        self.key_style_images = "images_ss" if self.dataset_key == "test" else "images_gen"
        self.label_key = "labels_ss" if self.dataset_key != "test" else "labels_gen"
        self.key_filenames = "labels_gen_filenames" if self.dataset_key == "test" else "images_ss_filenames"

        self._setup_data_file()

        self.user_ids = list(self.h5_in.keys())

        self.N = 0
        # N_start will hold the start indices for persons in the dataset
        self.N_start = list()
        for user in self.user_ids:
            self.N_start.append(self.N)
            keys = ['labels_gen_filenames'] if self.dataset_key == "test" else [
                'images_ss_filenames']
            for key in keys:
                if key in self.h5_in[user]:
                    self.N += self.h5_in[user][key].shape[0]
        self.h5_in_file.close()
        self.h5_in_file = None

    def _get_tuple_identifier_from_index(self, index):
        """
        Returns the user id and the within index for given index.
        :param index: index of the overall (flattened) dataset
        :return:
        """
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
        mask = self.h5_in[user][self.label_key][idx_target_image]

        params = get_params(self.opt, mask.shape)
        transform_mask = get_transform(self.opt, params,  method=cv2.INTER_NEAREST, normalize=False)
        # the toTensor method in transform will convert uint8 [0, 255]
        # to foat [-1, 1], so we need to revert this.
        mask_tensor = transform_mask(mask) * 255.0

        filename = self.h5_in[user][self.key_filenames][idx_target_image].decode('utf-8')
        # Some filenames contain an additional dot, which we want to remove.
        filename = re.sub(f'\.', '', filename)
        # Get style images (already preprocessed to tensor)
        transform_image = get_transform(self.opt, params)
        style_image_tensor, selected_idx, subsets = self.get_style_images(user, self.opt.input_ns, transform_image, filename)

        input_dict = {'label': mask_tensor,
                      'filename': filename,
                      'user': user,
                      'style_image': style_image_tensor
                      }

        if self.dataset_key != "test":
            # We have ground truth images
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
        """
        Returns the entry for the given index
        :param idx: int
        :return: dict with data for given idx
        """
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
        indices = self.N_start
        # The first index for each person in the dataset
        indices += [idx - 1 for idx in self.N_start[1:]] + [self.N - 1]
        return indices

    def get_random_indices(self, n):
        indices = np.random.choice(list(range(self.N)), n)
        return indices

    def _sample_style_idx(self, n_images, n, user_id=None, filename=None):
        subsets = None
        if self.opt.style_sample_method == 'random':
            # The style images are randomly sampled for the give person
            indices = np.random.choice(list(range(n_images)), n)
        elif self.opt.style_sample_method == 'first':
            indices = list(range(min(n, n_images)))
            # We take the style images in given order
            # (yields a deterministic output)
            indices = [idx for idx in indices]
        elif 'ref' in self.opt.style_sample_method:
            # We have a pre-created similarity ranking of style images
            # The style images can come from the generative or the sequence
            # dataset
            use_sequence_data = 'subset' in list(self.style_image_refs[self.opt.dataset_key][user_id][filename].keys())
            all_indices = self.style_image_refs[self.opt.dataset_key][user_id][filename]['index']
            if use_sequence_data:
                all_subsets = self.style_image_refs[self.opt.dataset_key][user_id][filename]['subset']
            if 'random' in self.opt.style_sample_method:
                # opt.style_sample_method is something like "ref_random40"
                # In that case we randomly sample from the most similar 40 images.
                reduced_n = re.sub(r"[^\d]", "", self.opt.style_sample_method)
                if reduced_n:
                    reduced_n = int(reduced_n)
                else:
                    reduced_n = 40
                to_select = np.random.choice(list(range(reduced_n)), n)
                indices = [all_indices[to_select[i]] for i in range(n)]
                if use_sequence_data:
                    subsets = [all_subsets[to_select[i]] for i in range(n)]
            else:
                # Most similar images first
                indices = all_indices[:n]
                if use_sequence_data:
                    subsets = all_subsets[:n]
        else:
            raise ValueError(f"Invalid style sampling method: {self.opt.style_sample_method}")
        return indices, subsets

    def get_style_images(self, user_id, n, transform_image, filename=None):
        n_images = self.h5_in[user_id][self.key_style_images].shape[0]
        selected_idx, subsets = self._sample_style_idx(n_images, n, user_id=user_id, filename=filename)
        # Subsets are either the generative or sequence dataset
        subset_keys = {b'g': self.key_style_images, b's': 'images_seq'}
        style_images = list()
        for i, sel_i in enumerate(selected_idx):
            if subsets is not None:
                subset_key = subset_keys[subsets[i]]
            else:
                subset_key = self.key_style_images
            # The indices for the seq dataset are too big (they were appended to the number in the gen dataset)
            if subset_key == 'images_seq':
                sel_i = sel_i - n_images
                selected_idx[i] = sel_i
            style_images.append(self.h5_in[user_id][subset_key][sel_i])

        tensors = [transform_image(img) for img in style_images]
        style_image_tensor = torch.stack(tensors)
        return style_image_tensor, selected_idx, subsets

    def unsqueeze_batch(self, batch):
        # Create a first batch size dimension
        keys = ["style_image", "target", "target_original", "label"]
        if self.opt.netG == 'spaderefiner':
            keys.append("start_tensor")
        for key in keys:
            batch[key] = batch[key].unsqueeze(0)
        return batch

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.h5_in_file is not None:
            self.h5_in_file.close()
            self.h5_in_file = None
            self.h5_in = None
        self.style_image_refs.close()
        self.pred_seg_file.close()
