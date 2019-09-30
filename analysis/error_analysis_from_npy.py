import h5py
import numpy as np
import torch
from PIL import Image
import os
import pandas as pd

# Number of samples to consider
from fid.fid_score import FIDCalculator
from util.visualizer import visualize_sidebyside

from util.tester import MSECalculator

from options.test_options import TestOptions
from util.files import listdir, create_folder_if_not_exists
# n = 5
n = 5


h5_in = h5py.File("/home/marcel/projects/data/openeds/all.h5", 'r')
data_in = h5_in["validation"]

# path = "/home/marcel/projects/Seg2Eye/checkpoints/190911_wookie_nn_validation"
path = "/home/marcel/Downloads/refinenet_code/res/refinenet/refinenet_submission_190923_133232/test"
image_files = listdir(path, postfix='.npy')
image_ids = [f[:-4] for f in image_files]

if n > 0:
    image_ids = image_ids[:n]
print(len(image_files))

FIDCalculator()

def find_person_and_idx(image_id):
    # image_id = image_id.encode('utf-8')
    for person_id in data_in.keys():
        filenames = [f.decode('utf-8') for f in data_in[person_id]['images_ss_filenames'][:]]
        # contains = data_in[person_id]['images_gen_filenames'] == image_id
        contains = image_id in filenames
        if contains:
            idx = filenames.index(image_id)
            return person_id, idx
    raise ValueError(f"Not found: {image_id}")

results = {
    'image_id': list(),
    'error': list(),
    'person_id': list(),
    'index': list(),
}

# image_ids is a list of all npy filenames in the target folder
for i, image_id in enumerate(image_ids):
    print(f"{i} / {len(image_ids)}")
    person_id, idx = find_person_and_idx(image_id)
    image_true = data_in[person_id]['images_ss'][idx]
    image_pred = np.load(os.path.join(path, image_id+".npy"))

    image_true = torch.from_numpy(image_true).unsqueeze(0).unsqueeze(0)
    image_pred = torch.from_numpy(image_pred).unsqueeze(0).unsqueeze(0)
    mse_error = MSECalculator.calculate_mse_for_images(image_pred, image_true)

    results['error'].append(float(mse_error))
    results['person_id'].append(person_id)
    results['image_id'].append(image_id)
    results['index'].append(idx)

print("Error: ", np.mean(results['error']))

def get_person_id(r_idx):
    return results['person_id'][r_idx]

def get_image_idx(r_idx):
    return results['index'][r_idx]

sorted_errors = np.argsort(np.array(results['error']))

for mode, r_indices in {'worst': sorted_errors[-100:], 'best': sorted_errors[:100]}.items():
    folder = os.path.join(path, mode)
    create_folder_if_not_exists(folder)

    errors = [results['error'][i] for i in r_indices]

    image_true = np.array([data_in[get_person_id(i)]['images_ss'][get_image_idx(i)] for i in r_indices])
    content_true = np.array([data_in[get_person_id(i)]['labels_ss'][get_image_idx(i)] for i in r_indices])
    image_pred = np.array([np.load(os.path.join(path, results['image_id'][i]+".npy")) for i in r_indices])
    image_style = np.zeros(image_true.shape)
    person_ids = np.array([get_person_id(i) for i in r_indices])
    image_ids = np.array([results['image_id'][i] for i in r_indices])

    data = {
        'label': torch.from_numpy(content_true).unsqueeze(1),
        'fake': torch.from_numpy(image_pred).unsqueeze(1),
        'style_image': torch.from_numpy(image_style).unsqueeze(1),
        'target_original': torch.from_numpy(image_true).unsqueeze(1),
        'user': person_ids,
        'filename': image_ids
    }
    visuals = visualize_sidebyside(data, w=200, h=320, error_list=errors)

    for i, v in enumerate(visuals.values()):
        img_np = (v.numpy() + 1) * 255 / 2
        img_np = img_np[0].astype(np.uint8)
        img = Image.fromarray(img_np)
        out_path = os.path.join(folder, f'{i}.png')
        img.save(out_path)


exit()



opt = TestOptions().parse()

# path_results = "/home/marcel/projects/SPADE_custom/checkpoints/190823_spade__use_vae/results/"

# dataset_key = "train"
# dataset_key = "validation"
# name = "190823_spade"
# checkpoints_dir = "/home/marcel/projects/SPADE_custom/checkpoints/"
path_results = os.path.join(opt.checkpoints_dir, opt.name, "results", opt.dataset_key)
path_error_log = os.path.join(path_results, f"error_log_{opt.dataset_key}.h5")
folder_validation_results = os.path.join(path_results, "visualisations")
if not os.path.exists(folder_validation_results):
    os.makedirs(folder_validation_results)

error_log = h5py.File(path_error_log, 'r')


def select_n(best=True, n=20):
    print(f"Selecting {n}. Best: {best}")
    if best:
        idx_selected = (np.copy(error_log["error"])).argsort()[:n]
        folder_out = os.path.join(folder_validation_results, "best")
    else:
        idx_selected = (np.copy(error_log["error"])).argsort()[-n:]
        folder_out = os.path.join(folder_validation_results, "worst")
    if not os.path.exists(folder_out):
        os.mkdir(folder_out)

    keys = ["filename", "user", "error"]  # error_relative_n1471
    df = pd.DataFrame({
        key: [error_log[key][idx] for idx in idx_selected] for key in keys
    })
    df.to_csv(os.path.join(folder_out, "metadata.csv"))

    for idx in idx_selected:
        visualisation = error_log["visualisation"][idx].squeeze()
        visualisation = Image.fromarray(visualisation)
        filename = error_log['filename'][idx].decode('utf-8')
        visualisation.save(os.path.join(folder_out, f"{filename}.png"), format='png')
    print(f"Saved visualisations to {folder_out}")


def calculate_error():
    error = np.sum(error_log["error"])
    print(f"Total error: {error}")
    relative_error = error / len(error_log["error"]) * 1471
    print(f"Relative error: {relative_error}")
    return error, relative_error


def error_histogram():
    print("Creating histogram")
    import matplotlib.pyplot as plt
    relative_error = np.multiply(error_log['error'], 1471)
    n, bins, patches = plt.hist(relative_error, bins=30, density=True)
    plt.title(f"Error distribution for {opt.dataset_key} \n(n={len(error_log['error'])}, mu: {np.mean(relative_error):.2f}, std: {np.std(relative_error):.2f})")
    plt.xlabel("Relative error (n=1471)")
    plt.ylabel("Ratio")
    path_out = os.path.join(path_results, "histogram.png")
    plt.savefig(path_out)
    plt.show()
    print(f"Saved histogram to {path_out}")


select_n(best=True, n=n)
select_n(best=False, n=n)
total_error, relative_error = calculate_error()
error_histogram()


error_log.close()