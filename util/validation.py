from collections import OrderedDict

import torch
import numpy as np

from data.base_dataset import __resize
from data.preprocessor import ImagePreprocessor
from util.image_annotate import get_text_image
from util.visualizer import visualize_sidebyside


def get_validation_data(dataloader, pix2pix_model, limit=4):
    result = {"fake": list(), "content": list(), "target": list(), "target_original": list()}
    for i_val, data_val in enumerate(dataloader):
        # 1st component: generated image
        result["fake"].append(pix2pix_model.forward(data_val, mode="inference").cpu())
        # 2nd component: input segmentation mask
        result["content"].append(torch.div(data_val['label'].float(), 3))
        # 3rd component: ground truth image
        result["target"].append(data_val['image'])
        result["target_original"].append(data_val['image_original'])
        if i_val > limit > 0:
            break
    return result


def calculate_mse_for_images(produced, target, simulate_n=-1):
    assert produced.shape[-2:] == (640, 400)
    assert target.shape[-2:] == (640, 400)
    mse_error = 0
    if len(produced.shape) == 2:
        # We want it to have three dimensions
        produced = np.array([produced])
        target = np.array([target])

    # We compute the norm for each image and then normalise it
    batch_size = produced.shape[0]
    for i in range(batch_size):
        fake_vector = produced[i].reshape(-1).astype(np.int16)
        real_vector = target[i].reshape(-1).astype(np.int16)
        mse_error += np.linalg.norm(fake_vector - real_vector)
    mse_error = mse_error / (640 * 400)
    # We always want to work with 1471 samples to have comparable errors

    if simulate_n > 0:
        mse_error = mse_error * simulate_n / batch_size
    return mse_error


def plot_mse(data, visualizer, epoch, total_steps_so_far, limit=-1):
    n = len(data["fake"])
    if limit > 0:
        n = min(n, limit)

    result = [(ImagePreprocessor.unnormalize(np.copy(data["fake"][i].detach().cpu())),
        np.copy(data["target_original"][i])) for i in range(n)]
    fake = np.array([r[0] for r in result]).squeeze()
    real = np.array([r[1] for r in result]).squeeze()
    # fake_vector = fake_vector[:2]
    # real_vector = real_vector[:2]
    fake_resized = np.array([__resize(img, 400, 640) for img in fake])
    mse_error = calculate_mse_for_images(fake_resized, real, simulate_n=1471)
    mse_error = torch.Tensor([mse_error])
    errors = {'val/mse': mse_error}
    visualizer.print_current_errors(epoch, total_steps_so_far, errors, t=0)
    visualizer.plot_current_errors(errors, total_steps_so_far)


def run_validation(dataloader, pix2pix_model, visualizer, epoch, iter_counter, limit=500, visualisation_limit=4):
    print(f"Running validation on {limit} images")
    data = get_validation_data(dataloader, pix2pix_model, limit=limit)
    visualize_sidebyside(data, visualizer, epoch, iter_counter.total_steps_so_far, limit=visualisation_limit, log_key='val')
    plot_mse(data, visualizer, epoch, iter_counter.total_steps_so_far, limit=limit)