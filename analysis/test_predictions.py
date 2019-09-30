import re
import shutil

import h5py
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from matplotlib import cm

from models.networks.loss import openEDSaccuracy


def get_best_image(ref_person, data_person, filename, n_best=1, unlabeled_key_gen='images_gen', unlabeled_key_seq='images_seq', shuffle=False):
    if shuffle:
        all_indices = np.copy(ref_person[filename]['index'][:200])
        random_indices = np.array(np.random.choice(range(all_indices.shape[0]), n_best))
        best_indices = [all_indices[i] for i in random_indices]
        subsets = [ref_person[filename]['subset'][i] for i in random_indices]
    else:
        best_indices = ref_person[filename]['index'][:n_best]
        subsets = ref_person[filename]['subset'][:n_best]
    best_indices_out = list()
    all_subsets = list()
    if 'subset' in ref_person[filename].keys():# TODO: remove this bloody hack
        subset_keys = {b's': data_person[unlabeled_key_seq], b'g': data_person[unlabeled_key_gen]}
        images = list()
        for current_i, best_i in enumerate(best_indices):
            b_or_s = subsets[current_i]
            all_subsets.append(b_or_s)
            dataset = subset_keys[b_or_s]
            if b_or_s == b's':
                # Fixing inconsistency when generating the h5 file
                best_i = best_i - data_person[unlabeled_key_gen].shape[0]

            best_indices_out.append(best_i)
            image = dataset[best_i]
            images.append(image)
    else:
        best_indices = [k for k in ref_person[filename]['index'][:n_best] if k < data_person[unlabeled_key_gen].shape[0]]
        images = [data_person[unlabeled_key_gen][k] for k in best_indices]
    # images = [subset_keys[subsets[current_i]][best_i] for current_i, best_i in enumerate(best_indices)]
    best_img = np.mean(images, axis=0)
    # plt.imshow(best_img)
    # plt.show()
    return best_img, best_indices_out, all_subsets


def get_true(data_true, idx):
        return data_true['images_ss'][idx]



# base_path = "datasets"
out_path = "/home/marcel/projects/Seg2Eye/predictions190914/"

# predictions_filename = "deeplab_predictions_190912_100418.h5"
# predictions_filename = "deeplab_predictions_190912_100418.h5"
predictions_filename = "deeplab_predictions_190913_210218.h5"


path_true = '/home/marcel/projects/data/openeds/190910_all.h5'
# path_ref = '/home/marcel/projects/Seg2Eye/datasets/distances_and_indices.h5'
path_ref = '/home/marcel/projects/data/openeds/datasets/distances_and_indices.h5'
path_seg = "/home/marcel/projects/data/openeds/datasets/190914_deeplab_seg_predictions.h5"

# gan_model_name = "190912_size512_bs1_lr.0002_l1_0_l2_15_lambda_w_0.5_lambda_feat_0.001_w1024___spadeStyleGen_cmadd_ns3_SAMmax_wc0_SSMref_random6"
gan_model_name = "190915_refiner_size256_bs1_lr.0002_l1_10_l2_0_lopeneds_0_lambda_w_0.5_lambda_feat_0.001_w1024___spadeStyleGen_cmadd_ns3_SAMmax_wc0_SSMref_random6"
# gan_model_name = "190915_refiner_size256_bs1_lr.00004_l1_0_l2_15_lopeneds_0_lambda_w_0.5_lambda_feat_0.001_w1024___spadeStyleGen_cmadd_ns3_SAMmax_wc0_SSMref_random6"
# folder_gan_preds = os.path.join("/home/marcel/projects/Seg2Eye/checkpoints/", gan_model_name, 'results/validation/')
folder_gan_preds = os.path.join('/home/marcel/projects/Seg2Eye/checkpoints/', gan_model_name, 'results/validation')

h5_segmap = h5py.File(path_seg, 'r')
h5_true = h5py.File(path_true, 'r')
h5_ref = h5py.File(path_ref, 'r')

copy_true_mask = False
# copy_true_mask = False
split = 'validation'
# unlabeled_key = 'images_ss'

threshold = 5.02
write_output = False
do_visualize = False
shuffle = False
limit = -1
is_test = split == 'test'

key_unlabaled = 'images_gen' if not is_test else 'images_ss'
key_labels = 'labels_ss' if not is_test else 'labels_gen'
key_filenames = key_labels+'_filenames'

filepaths = list()
all_errors = {}
# for person_i, person_id in enumerate(h5_segmap[split].keys()):
for person_i, person_id in enumerate(h5_true[split].keys()):
    print(person_i, person_id)
    all_errors[person_id] = list()
    data_true = h5_true[split][person_id]
    data_ref = h5_ref[split][person_id]
    data_seg = {k: h5_segmap[split][k][person_id] for k in ["gen", "seq"]}
    filenames = list(data_true[key_filenames][:])
    for i, identifier in enumerate(filenames):

        # This is the identifier for the input
        identifier_clean = re.sub(r"\.", "", identifier.decode('utf-8'))
        best_img, best_indices, subsets = get_best_image(data_ref, data_true, identifier_clean, n_best=1, unlabeled_key_gen=key_unlabaled, shuffle=True)
        best_idx = best_indices[0]
        subset = subsets[0]

        # best_seg = get_best_image(data_seg, data_true, identifier_clean, n_best=1, unlabeled_key_gen='gen', unlabeled_key_seq='seq')
        if subset == b's':
            best_seg = data_seg['seq'][best_idx]
        elif subset == b'g':
            best_seg = data_seg['gen'][best_idx]
        else:
            raise ValueError(f"invalid subset {subset}")

        # Index in our base dataset
        assert identifier == data_true[key_filenames][i]
        true_seg = data_true[key_labels][i]
        # idx = data_true['labels_gen_filenames'][:] == (identifier+'.').encode()
        # idx = np.nonzero(idx[0])[0][0]
        # true_seg = data_true['labels_gen'][idx]
        #
        overlap_mask = true_seg == best_seg
        error_seg = (1 - np.sum(overlap_mask) / np.size(true_seg)) * 100

        gan_img = np.load(os.path.join(folder_gan_preds, identifier_clean+'.npy'))[0]
        gan_img_orig = gan_img
        if copy_true_mask:
            gan_img[overlap_mask] = best_img[overlap_mask]

        if not is_test:
            true_img = get_true(data_true, i)

            error_mse_nn = openEDSaccuracy(torch.from_numpy(true_img), torch.from_numpy(best_img)).numpy() * 1471

            error_gan = openEDSaccuracy(torch.from_numpy(true_img), torch.from_numpy(gan_img)).numpy() * 1471
            all_errors[person_id].append(
                {'mse_nn': error_mse_nn, 'seg': error_seg, 'identifier': identifier, 'mse_gan': error_gan})
        # plt.imshow(np.concatenate([true_img, best_img, gan_img[0]], axis=-1), cmap='gray')
        # plt.show()
        # print(error_mse_nn)

        if write_output:
            filepaths.append(os.path.join(out_path, f"{identifier_clean}.npy"))
            if error_seg < threshold:
                best_img = best_img.astype(np.uint8)
                np.save(filepaths[-1], best_img)
            else:
                path_from = os.path.join(folder_gan_preds, f"{identifier_clean}.npy")
                shutil.copy(path_from, filepaths[-1])

        if do_visualize:
            if copy_true_mask:
                cat = np.concatenate([true_seg.astype(np.float)/3*255, best_seg.astype(np.float)/3*255, gan_img_orig, gan_img, best_img, np.abs(best_img-gan_img_orig)], axis=-1)
            else:
                cat = np.concatenate(
                    [true_seg.astype(np.float) / 3 * 255, best_seg.astype(np.float) / 3 * 255, gan_img,
                     best_img ], axis=-1)
            plt.imshow(cat, cmap='gray')
            plt.show()


        if i > limit > 0:
            break
        # del true_img, best_img, true_seg, best_seg, gan_img_orig, gan_img

if write_output:
    path_filepaths = os.path.join(out_path, "pred_npy_list.txt")
    with open(path_filepaths, 'w') as f:
        for line in filepaths:
            f.write(line)
            f.write(os.linesep)
        print(f"Written to {path_filepaths}")

errors_flat = {'mse_nn': list(), 'seg': list(), 'mse_gan': list(), 'color': list(), 'mse_gan': list()}
colors = cm.rainbow(np.linspace(0, 1, len(list(all_errors.keys()))))
for i, person_id in enumerate(all_errors):
    c = colors[i]
    for e in all_errors[person_id]:
        errors_flat['mse_nn'].append(e['mse_nn'])
        errors_flat['mse_gan'].append(e['mse_gan'])
        errors_flat['seg'].append(e['seg'])
        errors_flat['color'].append(c)


n_people = len(errors_flat['mse_nn'])
print(n_people)
for key in errors_flat:
    print(key, np.mean(errors_flat[key]))

# errors_flat = [e for key, val in all_errors.items() for e in val]
#
#
# for q in [0.75, 0.8, 0.85, 0.9]:
#     print(q, np.nanquantile(errors_flat, q=q))

plt.scatter(errors_flat['seg'], errors_flat['mse_gan'], color='black')
plt.scatter(errors_flat['seg'], errors_flat['mse_nn'], color=errors_flat['color'])

# plt.title('Error in overlap of predicted vs. true segmentation mask (in %)')
plt.title(f'Error')
plt.xlabel('segmentation error')
plt.ylabel('relative rmse')
# plt.hist(errors_flat, density=False, bins=20)
# plt.savefig(f'segmap_errors_{predictions_filename[:-3]}.png')
plt.savefig('validation_raw.png')
plt.show()
