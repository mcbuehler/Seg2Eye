"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import h5py

from data.base_dataset import BaseDataset, get_params, get_transform


class OpenEDSGenerativeDataset(BaseDataset):
    def __init__(self):
        super().__init__()

    def _setup_data_file(self):
        if self.h5_in_file is None:
            self.h5_in_file = h5py.File(self.root, 'r')  # , libver='latest')
            self.data_user = self.h5_in_file[self.dataset_key][self.person_id]

    def initialize(self, opt):
        self.opt = opt
        self.person_id = self.opt.person_id

        self.root = opt.dataroot
        self.dataset_key = opt.dataset_key
        self.key_style_images = "images_ss" if self.dataset_key == "test" else "images_gen"
        self.key_filenames = "images_ss_filenames" if self.dataset_key == "test" else "images_gen_filenames"

        self.h5_in_file = None
        self._setup_data_file()

        self.N = self.data_user[self.key_style_images].shape[0]

        assert self.N == self.data_user[self.key_filenames].shape[0]

        self.h5_in_file.close()
        self.h5_in_file = None

    def __getitem__(self, index):
        self._setup_data_file()
        filename = self.data_user[self.key_filenames][index].decode('utf-8')
        image = self.data_user[self.key_style_images][index]

        params = get_params(self.opt, image.shape)  # Only give h and w
        transform_image = get_transform(self.opt, params)
        # the toTensor method in transform will convert uint8 [0, 255] to foat [-1, 1], so we need to revert this.
        image_tensor = transform_image(image)

        input_dict = {'image': image_tensor,
                      'index': index,
                      'filename': filename
                      }
        return input_dict

    def __len__(self):
        return self.N

    def _get_transform(self, params, **kwargs):
        transform_image = get_transform(self.opt, params, **kwargs)
        return transform_image, params

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.h5_in_file is not None:
            self.h5_in_file.close()
            self.h5_in_file = None
            self.data_user = None
