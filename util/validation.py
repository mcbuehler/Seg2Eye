from collections import OrderedDict

import torch
import numpy as np

from data.base_dataset import __resize
from data.preprocessor import ImagePreprocessor
from models.networks import openEDSaccuracy
from data.postprocessor import ImagePostprocessor
from util.image_annotate import get_text_image
# from util.util import reformat_data_from_loader
from util.visualizer import visualize_sidebyside


def calculate_mse_for_images(produced, target):
    assert produced.shape == target.shape
    assert torch.min(produced) >= 0 and torch.max(produced) <= 255
    assert torch.min(target) >= 0 and torch.max(target) <= 255
    assert produced.shape[-2:] == (640, 400), f"Invalid shape: {produced.shape}"
    assert len(produced.shape) == 4, "Please feed 4D tensors"

    mse_error = list()
    # We compute the norm for each image and then normalise it
    batch_size = produced.shape[0]
    for i in range(batch_size):
        produced_i = produced[i]
        target_i = target[i]
        # diff_i = torch.add(produced_i, torch.mul(target_i, -1)).float()
        norm_i = openEDSaccuracy(produced_i, target_i)
        mse_error.append(norm_i)
    mse_error = torch.stack(mse_error)
    return mse_error


def calculate_relative_sum_mse(data):
    fake = ImagePostprocessor.to_255resized_imagebatch(data['fake'], as_tensor=True)
    real = ImagePostprocessor.as_batch(data["target_original"], as_tensor=True)

    mse_error = calculate_mse_for_images(fake, real)
    mse_error = torch.mean(mse_error) * 1471
    return mse_error


def run_validation(dataloader, pix2pix_model, visualizer, epoch, iter_counter, limit=500, visualisation_limit=4, log_key='val'):
    validation_indices = dataloader.dataset.get_validation_indices()[:limit]
    result_list = list()
    for i_val in validation_indices:
        data_i = dataloader.dataset.get_particular(i_val)
        data_i['fake'] = pix2pix_model.forward(data_i, mode="inference").cpu()
        result_list.append(data_i)
    result = {k: [rl[k] for rl in result_list] for k in result_list[0].keys()}
    for key in ["style_image", "target", "target_original", "fake", "label"]:
        result[key] = torch.cat(result[key], dim=0)

    print(f"Running logging for dataset '{log_key}' on {limit} images")
    # data = get_validation_data(dataloader, pix2pix_model, limit=limit)
    mse_errors = calculate_relative_sum_mse(result)

    errors_dict = {f'mse/{log_key}': mse_errors}
    visualizer.print_current_errors(epoch, iter_counter.total_steps_so_far, errors_dict, t=0)
    visualizer.plot_current_errors(errors_dict, iter_counter.total_steps_so_far)

    visualize_sidebyside(result, visualizer, epoch, iter_counter.total_steps_so_far, limit=visualisation_limit, log_key=log_key)
    # calculate_mse(result, visualizer, epoch, iter_counter.total_steps_so_far, limit=limit, log_key=log_key)