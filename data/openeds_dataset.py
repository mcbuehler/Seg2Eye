"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import torch
from cv2 import cv2

from data.base_dataset import BaseDataset, get_params, get_transform
from PIL import Image
import util.util as util
import os
import numpy as np
import h5py


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

    def initialize(self, opt):
        self.opt = opt

        self.root = opt.dataroot
        self.dataset_key = opt.dataset_key
        print(f"Dataset key: {self.dataset_key}")

        self.h5_in_file = None
        self._setup_data_file()

        self.user_ids = list(self.h5_in.keys())

        self.N = 0
        self.N_start = list()
        for user in self.user_ids:
            self.N_start.append(self.N)
            keys = ['labels_gen_filenames'] if self.dataset_key == "test" else ['images_ss_filenames']  # TODO: leverage 'images_gen_filenames'
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
        user, idx_image = self._get_tuple_identifier_from_index(index)
        # style_image = np.random.choice(self.h5_in[user]["images_gen"])

        label_key = "labels_ss" if self.dataset_key != "test" else "labels_gen"
        mask = self.h5_in[user][label_key][idx_image]

        params = get_params(self.opt, mask.shape)
        # It is important to use nearest neighbour interpolation for resizing the mask. Otherwise we might get
        # values that are out of the allowed range.
        transform_mask = get_transform(self.opt, params, method=cv2.INTER_NEAREST, normalize=False)
        # the toTensor method in transform will convert uint8 [0, 255] to foat [-1, 1], so we need to revert this.
        mask_tensor = transform_mask(mask) * 255.0
        # This is the image that we should be producing. We feed it to D.
        if self.dataset_key == "test" or self.dataset_key == "validation":
            # We take a random image from the user as input
            style_img_idx = np.random.choice(list(range(self.h5_in[user]["images_ss"].shape[0])))
            target_image = self.h5_in[user]["images_ss"][style_img_idx]
            filename = self.h5_in[user]["labels_gen_filenames"][idx_image].decode('utf-8')
        else:
            target_image = self.h5_in[user]["images_ss"][idx_image]
            filename = self.h5_in[user]["images_ss_filenames"][idx_image].decode('utf-8')
        transform_image = get_transform(self.opt, params)
        target_image_tensor = transform_image(target_image)

        if torch.max(mask_tensor) > 3:
            print(user, idx_image, filename)
            print(np.max(mask))
            print(torch.max(mask_tensor))

        # Real input to Discriminator
        # random_user = np.random.choice(self.user_ids)
        # real = np.random.choice(self.h5_in[random_user]["images_gen"])

        input_dict = {'label': mask_tensor,
                      'instance': 0,
                      'filename': filename,
                      'user': user,
                      'image': target_image_tensor,
                      'image_original': torch.from_numpy(target_image)
                      }

        # if self.keep_original:
        #     input_dict['original'] = target_image
        # print(input_dict)
        return input_dict

    # def get_particular(self, user_id_content, image_idx_content, image_idx_style):

    def postprocess(self, input_dict):
        return input_dict

    def __len__(self):
        return self.N

    def get_validation_indices(self):
        return self.N_start

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