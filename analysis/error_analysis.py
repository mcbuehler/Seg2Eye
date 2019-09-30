import h5py
import numpy as np
from PIL import Image
import os
import pandas as pd

# Number of samples to consider
from options.test_options import TestOptions

# n = 5
n = 100
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