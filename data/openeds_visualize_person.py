import os

import PIL
import h5py
import torch
from PIL import Image
from torchvision.utils import make_grid
from matplotlib import pyplot as plt
import numpy as np


def get_grid(images):
    tensor = torch.from_numpy(images).unsqueeze(1)
    grid = make_grid(tensor, padding=0, pad_value=1)
    grid_np = grid.numpy().transpose(1, 2, 0)
    return grid_np

def normalize_masks(masks):
    masks = masks.astype(float)
    masks = masks / 3
    return masks

def normalize_images(masks):
    masks = masks.astype(float)
    masks = masks / 255
    return masks


def count_black(img, ignore_top=False):
    if ignore_top:
        h, w = img.shape
        img = img[int(h/2):, :]
    if isinstance(img, torch.Tensor):
        return torch.sum(img > 250)
    return np.sum(img >= 250)


def get_sort_key(mode, **kwargs):
    if mode == 'blackcount':
        return lambda img: count_black(img, True)
    elif mode == 'id':
        return lambda img, id: int(id)


def resize(grid_np):
    pil_img = Image.fromarray(grid_np)
    maxsize = (1280, 800)
    pil_img.thumbnail(maxsize, PIL.Image.ANTIALIAS)
    return pil_img


def generate_grid_for_person(h5_in, dataset_key, subset_keys, person_id, shuffle=False, limit=-1, path_out=''):
        person_data = h5_in[dataset_key][person_id]

        grids = list()
        for subset_key in subset_keys:
            # print(h5_in[dataset_key][person_id][subset_key+"_filenames"][3])
            if shuffle:
                idx = np.random.choice(list(range(person_data[subset_key].shape[0])), size=limit)
                images = np.array([person_data[subset_key][i] for i in idx])
            else:
                images =person_data[subset_key][:limit]
            images = torch.from_numpy(images)
            grid = make_grid(images.unsqueeze(1), nrow=4)
            grids.append(grid)
        grid_np = torch.cat(grids, dim=-1).numpy()[0]

        path_out = os.path.join(path_out, person_id + '.png')
        image = resize(grid_np)
        image.save(path_out)



if __name__ == '__main__':
    path_h5 = '/home/marcel/projects/data/openeds/190910_all.h5'
    h5_in = h5py.File(path_h5, 'r')
    path_out = '/home/marcel/projects/data/openeds/person_analysis/index_order'

    limit = 12
    shuffle = True
    # shuffle = False
    np.random.seed(1234)
    dataset_key = 'validation'
    person_id = 'U218'

    generate_grid_for_person(h5_in, dataset_key, ['images_ss', 'images_gen', 'images_seq'], person_id, shuffle=shuffle, limit=limit, path_out=path_out)
    exit()

    all_person_ids = list(h5_in[dataset_key].keys())
    for i, person_id in enumerate(all_person_ids):
        print(f"Processing {i} / {len(all_person_ids)}")
        person_data = h5_in[dataset_key][person_id]

        n = person_data['images_ss'].shape[0]
        # print(f"Taking {limit} out of {n} images for person {person_id}")
        # if shuffle:
        #     idx = np.random.choice(list(range(n)), size=limit)
        #     images = np.array([person_data['images_gen'][i] for i in idx])
        # else:
        #     # images = np.copy(person_data['images_gen'][:limit])
        #     # images = np.copy(person_data['images_ss'][:limit])
        #     masks = np.copy(person_data['labels_gen'][:limit])
        #     masks = masks.astype(float)
        #     masks = masks / np.max(masks)

        print(person_data.keys())
        ids = person_data['filenames']
        print(ids)
        exit()



        # images = sorted(images, key=lambda img: count_black(img, True), reverse=True)

        # black_pixels = [count_black(img, True) for img in images]
        # black_median = np.median(black_pixels)
        # debug
        # black_median = 0
        # if black_median < 1:
        #     grid_np = get_grid(images)
        # else:
        #
        #     # hist, bins = np.histogram(images)
        #
        #     low = np.array([img for img in images if count_black(img, True) < black_median])
        #     high = np.array([img for img in images if count_black(img, True) >= black_median])
        #
        #     grid_low = get_grid(low)
        #     grid_high = get_grid(high)
        #
        #     # hist_low = plt.hist(low.reshape(-1))
        #     # plt.show()
        #
        #     grid_np = np.concatenate([grid_low, grid_high], axis=0)

        # pil_img = resize(grid_np)

        grid_np_mask = get_grid(masks)
        pil_mask = resize(grid_np_mask)
        path_out_person_mask = os.path.join(path_out, person_id + '_mask.png')
        pil_mask.save(path_out_person_mask)

        # path_out_person = os.path.join(path_out, person_id + '.png')
        # pil_img.save(path_out_person)


        # plt.hist(images.reshape(-1))
        # plt.show()

    h5_in.close()