"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import json
import re
import sys
from copy import deepcopy

import h5py
import numpy as np

import torch

import data
from models import networks
from models.networks import PupilLocator
from options.pupillocator_options import PupilLocatorOptions
from util import util

# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
# parse options
opt = PupilLocatorOptions().parse()
opt.batchSize = 1

# opt.display_freq = 500
# opt.print_freq = 100
# opt.validation_limit = 50

# Max number of relations to save (take best)
max_save = 100

limit = -1
# max_save = 5
# opt.display_freq = 1
# opt.print_freq = 100
# opt.validation_limit = 50

# opt.full_val_freq = 100
# full_val_limit = 1

# load the dataset

model = networks.define_pupil_locator(opt)
model_one_gpu = model.module if len(opt.gpu_ids) > 0 else model
util.load_network(model, model_one_gpu.name, opt.which_epoch, opt)

# path_out = 'datasets/style_images.json'
path_out = 'datasets/style_images.h5'
h5_out = h5py.File(path_out, 'w')


# For each dataset
for dataset_key in ('train', 'validation', 'test'):
    dl = data.create_inference_dataloader(opt, dataset_key=dataset_key, shuffle=False)
    print(f"Processing dataset '{dataset_key}'")

    h5_out.create_group(dataset_key)
    seen_user_ids = list()
    current_user_id = ''

    for i, data_i in enumerate(dl):
        # We assume batch size 0
        new_user_id = data_i['user'][0]
        if new_user_id != current_user_id:
            assert new_user_id not in seen_user_ids,"User has already been processed"
            print(f"Processing user {new_user_id}")
            # keep track
            seen_user_ids.append(new_user_id)
            opt_gen = deepcopy(opt)
            opt_gen.person_id = new_user_id
            opt_gen.batchSize = 1  # Might be slow, but we have to make sure we are not missing any images
            opt_gen.dataset_mode = 'openedsgenerative'
            current_user_id = new_user_id

            h5_out[dataset_key].create_group(current_user_id)
            # Data loader for a specific person
            # Ideally we could compute the predictions for all generative images for that person in one shot.
            dl_gen = data.create_inference_dataloader(opt_gen, dataset_key=dataset_key, shuffle=False, full_batch=True)
            predictions = list()
            for gen_i, gen_data_i in enumerate(dl_gen):
                pred = model(gen_data_i['image'], mode='test')
                predictions.append({
                    # Using detach frees the memory after finishing the loop
                    'pred': pred.detach().cpu()[0],
                    'index': gen_data_i['index'][0],
                    # 'filename': gen_data_i['filename'][0]
                })
                if gen_i > limit > 0:
                    break
            gen_pupil_positions = {
                key: [predictions[i][key] for i in range(len(predictions))] for key in predictions[0]
            }

        mask = data_i['label']
        loc_pupil_mask = PupilLocator.get_pupil_location(mask)[0]

        filename = data_i['filename'][0]
        # Test filenames have this dot
        filename = re.sub(r'\.', '', filename)

        def calculate_distance(true, pred):
            assert true.shape == pred.shape
            diff = (true - pred)
            squared = diff.pow(2)
            return squared.sum()

        distances = torch.stack([
            calculate_distance(loc_pupil_mask, gen_pupil_positions['pred'][i])
            for i in range(len(gen_pupil_positions['pred']))]
        ).squeeze()
        idx_sorted = torch.argsort(distances, descending=False)

        data_sorted = [{'dist': distances[i].numpy(),
                        'index': gen_pupil_positions['index'][i].numpy(),
                        # 'filename': gen_pupil_positions['filename'][i]
                        }
                        for i in idx_sorted]

        h5_out[dataset_key][current_user_id].create_group(filename)
        for key in data_sorted[0]:
            d = np.array([data_sorted[i][key] for i in range(len(data_sorted))])
            if max_save > 0:
                d = d[:max_save]
            h5_out[dataset_key][current_user_id][filename].create_dataset(
                key, data=d)
        if limit > 0 and len(seen_user_ids) > 2:
            break


h5_out.close()
# with open(path_out, 'w') as file_out:
#     json.dump(data_out, file_out)
print(f"Written results to {path_out}")


