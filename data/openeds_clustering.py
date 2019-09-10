import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision.utils import make_grid

from sklearn.cluster import KMeans
from data.openeds_visualize_person import get_grid, normalize_masks, normalize_images, count_black


def relative_count_black(window):
    return np.sum(window < 10) / window.size


def var_region(img, mask, region_value):
    img_selected = img[mask == region_value]
    return np.var(img_selected)


def get_features(img, mask):
    h, w = img.shape

    black_pixels = np.abs(relative_count_black(img[int(h/2):,:]) - relative_count_black(mask[int(h/2):,:]))
    features = [black_pixels]
    for region_value in [1, 2, 3]:
        features.append(var_region(img, mask, region_value))
    return features


def save_as_img(a, person_id, key):
    out = Image.fromarray(a)
    out.save(os.path.join('checkpoints/clustering/', f'{person_id}_{key}.png'))


def mask_to_255(images):
    images = np.array(images)
    return (images.astype(np.float) / 3 * 255).astype(np.uint8)

path_h5 = '/home/marcel/projects/data/openeds/all.h5'
h5_in = h5py.File(path_h5, 'r')

limit = 10
save_images = False
save_images=True
filenames = [f.decode('utf-8') for f in h5_in['validation']['U216']['images_ss_filenames'][:]]
idx = filenames.index('000000289857')


results = {}
ratios = list()
all_keys = ['images_ss']

for person_id in h5_in['validation'].keys():
    results[person_id] = {}
    print('Processing ', person_id)
    for mask_idx in range(h5_in['validation'][person_id]['labels_ss'].shape[0]):
        mask = h5_in['validation'][person_id]['labels_ss'][mask_idx]
        # print(h5_in['validation'][person_id].keys())

    # img = h5_in['validation']['U216']['images_ss'][idx]
    # print(img.shape)

        images = h5_in['validation'][person_id]['images_ss'][:]
        images = images[:limit]
        features = np.array([get_features(img, mask) for img in images])
        f_max = np.max(features, axis=0)
        features = np.array([f / f_max for f in features])

        features = np.sum(features, axis=-1)
        selected_img_idx = np.argmin(features)
        selected_img = images[selected_img_idx]

        true_img = h5_in['validation'][person_id]['images_ss'][mask_idx]
        top = np.concatenate([selected_img, true_img], axis=-1)
        bottom = np.concatenate([mask_to_255(mask), mask_to_255(mask)], axis=-1)
        combo = np.concatenate([top, bottom], axis=-2)
        plt.imshow(combo, cmap='gray')
        plt.show()

        if mask_idx > 0:
            break

        # cluster_idx = KMeans(n_clusters=2, random_state=1234).fit_predict(features)

        # cluster1 = [images[i] for i in range(len(cluster_idx)) if cluster_idx[i]]
        # cluster2 = [images[i] for i in range(len(cluster_idx)) if not cluster_idx[i]]
        # features1 = [features[i] for i in range(len(cluster_idx)) if cluster_idx[i]]
        # features2 = [features[i] for i in range(len(cluster_idx)) if not cluster_idx[i]]

        # print(np.mean(features1), np.mean(features2))
        # if np.mean(features1) > np.mean(features2):
        #     tmp = cluster1
        #     cluster1 = cluster2
        #     cluster2 = tmp
        #     # Switch 0 to 1 and 1 to 0
        #     cluster_idx = [(i-1)*-1 for i in cluster_idx]

        # results[person_id][key] = cluster_idx

        # masks_cluster1 = [h5_in['validation'][person_id]['labels_ss'][i] for i in range(len(cluster_idx)) if cluster_idx[i]]
        # masks_cluster2 = [h5_in['validation'][person_id]['labels_ss'][i] for i in range(len(cluster_idx)) if not
        #                   cluster_idx[i]]

        # if save_images:
        #     height = min(len(cluster1), len(cluster2), limit)
        #     print('height:', height)
        #     left_img = np.concatenate(cluster1[:height], axis=-2)
        #     left_masks = np.concatenate(mask_to_255(masks_cluster1[:height]), axis=-2)
        #     right_img = np.concatenate(cluster2[:height], axis=-2)
        #     right_masks = np.concatenate(mask_to_255(masks_cluster2[:height]), axis=-2)
        #     all = np.concatenate([left_masks, left_img, right_img, right_masks], axis=-1)
        #     # all = np.concatenate([left_img, right_img], axis=-1)
        #     #
        #     a, b = np.min(all), np.max(all)
        #     plt.imshow(all)
        #     plt.show()
            # save_as_img(all, person_id, 'both')

            # cluster1_mean = np.mean(np.array(cluster1), axis=0).astype(np.uint8)
            # cluster2_mean = np.mean(np.array(cluster2), axis=0).astype(np.uint8)
            # cluster_mean = np.concatenate([cluster1_mean, cluster2_mean], axis=-1)
            # save_as_img(cluster_mean, person_id, f'{key}_mean')

    # n = min(len(results[person_id][all_keys[0]]), len(results[person_id][all_keys[1]]))
    # inconsistent = [i for i in range(n)
    #                 if results[person_id][all_keys[0]][i] != results[person_id][all_keys[1]][i]]
    # n_inconsistent = len(inconsistent)
    # ratio_inconsistent = n_inconsistent / n
    #
    # print(f'Person {person_id}: {n_inconsistent} / {n} are inconsistent. Ratio = {ratio_inconsistent:.2f}')
    # ratios.append(ratio_inconsistent)
# print(relative_count_black(img))

# print(relative_count_black(img))
h5_in.close()

# plt.hist(ratios)
# plt.show()
exit()


path_h5 = '/home/marcel/projects/data/openeds/all.h5'
h5_in = h5py.File(path_h5, 'r')

limit = 31
# limit = 60
np.random.seed(1234)
dataset_key = 'train'
data_in = h5_in[dataset_key]

person_ids = list(data_in.keys())

person_id = person_ids[11]
data_person = data_in[person_id]

images_ss = np.copy(data_person['images_ss'])
masks_ss = np.copy(data_person['labels_ss'])

images = images_ss[:limit]
masks = masks_ss[:limit]

# print(torch.min(mask), torch.max(mask))



# idx_iris = masks == 3
# idx_pupil = masks == 1
# idx_eye = masks == 2
# idx_bg = masks == 0
# masks[idx_iris] = 0
# masks[idx_pupil] = 3
# masks[idx_bg] = 0
# masks[idx_eye] = 2

def get_pupil_location(mask):
    location = torch.nonzero((mask == 3)).float()
    x = torch.mean(location[:,0]) / mask.shape[-2]
    y = torch.mean(location[:, 1]) / mask.shape[-1]
    return torch.Tensor(np.array([x, y])).long()

print(masks.shape)
masks = masks[0]
masks = torch.from_numpy(masks)
location = torch.nonzero((masks == 3)).float()
x = torch.mean(location[:,0])
y = torch.mean(location[:, 1])

x = int(x)
y = int(y)
print(x,y)
p = np.zeros(masks.shape)
p[x,y] = 1
plt.imshow(p[x-20:x+20, y-20:y+20])
plt.show()
print()

plt.imshow(masks[x-20:x+20, y-20:y+20])
plt.show()
input()


exit()

masks = normalize_masks(masks)
images = normalize_images(images)

mask = torch.from_numpy(masks).unsqueeze(1)

image = torch.from_numpy(images).unsqueeze(1)

ssim_loss = SSIM(size_average=False)

mask = mask[22]
#
#

mse = nn.MSELoss(reduce=False)
similarities = torch.mean(torch.mean(mse(mask, image), dim=-1), dim=-1)
similarities = torch.stack([torch.abs(torch.sum(img[0,320:] <= 0.01) - torch.sum(mask[0,320:] <= 0.01)) for img in image])
similarities = torch.max(similarities) - similarities
print(similarities)

image = torch.stack([image[i] for i in torch.argsort(similarities)])

# s = [mask[:,:,i] for i in idx_nonzero]
# nonzero_mask = torch.stack(s)
# similarities = ssim_loss.forward(mask[idx_nonzero], image[idx_nonzero])
#
# print(similarities)
# grid_images = get_grid(image)
# grid_masks = get_grid(mask)

image = torch.cat([image, mask.unsqueeze(0)], dim=0)
grid_images = make_grid(image)
# grid_masks = make_grid(mask)

# grid_np = np.concatenate([grid_images, grid_masks], axis=-2)
# grid_np = grid_masks
grid_np = grid_images.numpy()
plt.imshow(grid_np.transpose(1, 2, 0))
plt.show()



