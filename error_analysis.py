import h5py
import numpy as np
from PIL import Image
import os
import pandas as pd

# Number of samples to consider
n = 100

# path_results = "/home/marcel/projects/SPADE_custom/checkpoints/190823_spade__use_vae/results/"
path_results = "/home/marcel/projects/SPADE_custom/checkpoints/190823_spade/results/"
path_error_log = os.path.join(path_results, "error_log.h5")
folder_validation_results = os.path.join(path_results, "validation_visualisations")
if not os.path.exists(folder_validation_results):
    os.mkdir(folder_validation_results)

error_log = h5py.File(path_error_log, 'r')


def select_n(best=True, n=20):
    if best:
        idx_selected = (np.copy(error_log["error"])).argsort()[:n]
        folder_out = os.path.join(folder_validation_results, "best")
    else:
        idx_selected = (np.copy(error_log["error"])).argsort()[-n:]
        folder_out = os.path.join(folder_validation_results, "worst")
    if not os.path.exists(folder_out):
        os.mkdir(folder_out)

    keys = ["filename", "user", "error"]
    df = pd.DataFrame({
        key: [error_log[key][idx] for idx in idx_selected] for key in keys
    })
    df.to_csv(os.path.join(folder_out, "metadata.csv"))

    for idx in idx_selected:
        visualisation = error_log["visualisation"][idx]
        visualisation = Image.fromarray(visualisation)
        filename = error_log['filename'][idx].decode('utf-8')
        visualisation.save(os.path.join(folder_out, f"{filename}.png"))


def print_total_error():
    error = np.sum(error_log["error"])
    print(f"Total error: {error}")


# select_n(best=True, n=n)
# select_n(best=False, n=n)
print_total_error()

error_log.close()