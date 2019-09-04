"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import h5py
import numpy as np
import torch
from cv2 import cv2

from data.base_dataset import BaseDataset, get_params, get_transform, flip


class OpenEDSDataset(BaseDataset):
    def __init__(self):
        super().__init__()

    def _setup_data_file(self):
        if self.h5_in_file is None:
            self.h5_in_file = h5py.File(self.root, 'r')  # , libver='latest')
            self.h5_in = self.h5_in_file[self.dataset_key]

    @staticmethod
    def modify_commandline_options(parser, is_train):
        # parser.add_argument('--no_pairing_check', action='store_true',
        #                     help='If specified, skip sanity check of correct label-image file pairing')
        return parser

    def initialize(self, opt):  #TODO: print style sampling method
        self.opt = opt

        self.root = opt.dataroot
        self.dataset_key = opt.dataset_key
        print(f"Dataset key: {self.dataset_key}")
        self.key_style_images = "images_ss" if self.dataset_key == "test" else "images_gen"
        self.label_key = "labels_ss" if self.dataset_key != "test" else "labels_gen"
        self.key_filenames = "labels_gen_filenames" if self.dataset_key == "test" else "images_ss_filenames"

        self.h5_in_file = None
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
        #
        # assert len(list(self.h5_in.keys())) > 0
        # # e.g. {'p01': 2500, ...}
        # self.N_person_ids = {group: self.h5_in[group]["image"].shape[0] for group in self.h5_in}
        # # e.g. ['p01', 'p02',...]
        # self.person_ids = [group for group in self.h5_in.keys()]
        # # e.g. [0, 2500, 7033,...]
        # self.person_startindex = [0]
        # for person_id in self.h5_in:
        #     self.person_startindex += [self.person_startindex[-1] + self.N_person_ids[person_id]]
        # self.N = np.sum(list(self.N_person_ids.values()))
        #
        self.h5_in_file.close()
        self.h5_in_file = None

    # def get_paths(self, opt):
    #     label_paths = []
    #     image_paths = []
    #     instance_paths = []
    #     assert False, "A subclass of Pix2pixDataset must override self.get_paths(self, opt)"
    #     return label_paths, image_paths, instance_paths

    # def paths_match(self, path1, path2):
    #     filename1_without_ext = os.path.splitext(os.path.basename(path1))[0]
    #     filename2_without_ext = os.path.splitext(os.path.basename(path2))[0]
    #     return filename1_without_ext == filename2_without_ext
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

        # Get input_ns style images (already preprocessed to tensor)
        style_image_tensor = self.get_style_images(user, self.opt.input_ns)

        filename = self.h5_in[user][self.key_filenames][idx_target_image].decode('utf-8')

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

    def _sample_style_idx(self, n_images, n):
        if self.opt.style_sample_method == 'random':
            indices = np.random.choice(list(range(n_images)), n)
        elif self.opt.style_sample_method == 'first':
            indices = list(range(min(n,n_images)))
        else:
            raise ValueError(f"Invalid style sampling method: {self.opt.style_sample_method}")
        return indices

    def get_style_images(self, user_id, n):
        # user_idx = self.user_ids.index(user_id)
        # n_user = self.N_start[user_idx + 1] - self.N_start[user_idx]
        # within_idx = np.random.choice(list(range(n_user)), size=n)
        # selected_idx = [self.N_start[user_idx] + i for i in within_idx]
        # return selected_idx
        n_images = self.h5_in[user_id][self.key_style_images].shape[0]
        selected_idx = self._sample_style_idx(n_images, n)

        style_images = [self.h5_in[user_id][self.key_style_images][i] for i in selected_idx]

        # Preprocessing
        size = style_images[0].shape[-2:]
        # Convert to tensor
        params = get_params(self.opt, size)  # Only give h and w
        transform_image =  get_transform(self.opt, params)
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

# for dataset_key in f:
#     data = f[dataset_key]
#     print("n users: ", len(data.keys()))
#     for user in data:
#         for key in data[user]:
#             print(data[user][key].shape)
