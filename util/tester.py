"""
Starting command:
python test.py --name $name --dataset_mode openeds \
    --dataroot $DATAROOT  --aspect_ratio 0.8 --no_instance \
    --load_size 256 --crop_size 256 --preprocess_mode fixed --batchSize 24 \
    --netG spade
    --write_error_log \
    --dataset_key train
"""

import os
import re
from collections import OrderedDict
from copy import deepcopy

import h5py
import numpy as np
import torch

import data
from data.postprocessor import ImagePostprocessor
from models.networks import openEDSaccuracy
from util.visualizer import Visualizer, visualize_sidebyside


class Tester:
    def __init__(self, opt, g_logger, dataset_key, visualize=False, visualizer=None, limit=-1, write_error_log=False):
        self.opt = deepcopy(opt)
        self.g_logger = g_logger

        self.opt.serial_batches = True
        self.opt.no_flip = True
        self.opt.phase = 'test'
        self.opt.style_sample_method = 'first'
        self.opt.isTrain = False
        self.opt.dataset_key = dataset_key

        if 'write_error_log' not in self.opt:
            self.opt.write_error_log = write_error_log

        self.dataloader = data.create_dataloader(self.opt)

        if visualize:
            self.visualizer = Visualizer(self.opt) if visualizer is None else visualizer

        base_path = os.getcwd()
        if self.opt.checkpoints_dir.startswith("./"):
            self.opt.checkpoints_dir = os.path.join(base_path,self.opt.checkpoints_dir[2:])
        else:
            self.opt.checkpoints_dir = os.path.join(base_path,self.opt.checkpoints_dir)

        self.is_validation =self.opt.dataset_key in ["validation", "train"]
        self.N = self.dataloader.dataset.N

        if self.opt.write_error_log or not self.is_validation:
            self.results_dir = os.path.join(opt.checkpoints_dir,self.opt.name,self.opt.results_dir,self.opt.dataset_key)
            if not os.path.exists(self.results_dir):
                os.makedirs(self.results_dir)

        self.limit = min(limit, self.N) if limit > 0 else self.N

        if self.is_validation and self.opt.write_error_log:
            error_log = h5py.File(os.path.join(self.results_dir, f"error_log_{opt.dataset_key}.h5"), "w")
            error_log.create_dataset("error", shape=(self.N,), dtype=np.float)
            error_log.create_dataset("user", shape=(self.N,), dtype='S4')
            error_log.create_dataset("filename", shape=(self.N,), dtype='S13')
            error_log.create_dataset("visualisation", shape=(self.N, 1, 380, 1000), dtype=np.uint8)

    def get_fake_and_resized(self, model, data_i):
        fake = model.forward(data_i, mode="inference").cpu()
        fake_resized = ImagePostprocessor.to_255resized_imagebatch(fake, as_tensor=True)
        return fake, fake_resized

    @staticmethod
    def run_partial_validation(dataloader, pix2pix_model, visualizer, epoch, iter_counter, limit=500,
                               visualisation_limit=4,
                               log_key='val', g_logger=None):
        if 'rand' in log_key:
            validation_indices = dataloader.dataset.get_random_indices(limit)
        else:
            # Use fixed validation indices
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
        mean_errors_relative, std_errors_relative, error_list = MSECalculator.calculate_relative_sum_mse(result)

        errors_dict = {f'mse/{log_key}/mean/relative': mean_errors_relative, f'mse/{log_key}/std/relative': std_errors_relative}
        visualizer.print_current_errors(epoch, iter_counter.total_steps_so_far, errors_dict, t=0)
        visualizer.plot_current_errors(errors_dict, iter_counter.total_steps_so_far)

        visualize_sidebyside(result, visualizer, epoch, iter_counter.total_steps_so_far, limit=visualisation_limit,
                             log_key=log_key, error_list=error_list)
        errors_dict = OrderedDict(
            [('epoch', epoch), ('n_steps', iter_counter.total_steps_so_far)] + [(k, np.copy(v.cpu())) for k, v in
                                                                                errors_dict.items()])
        g_logger.update_or_append_row(errors_dict)
        return errors_dict

    def run_full_validation(self, model):
        print(f"write error log: {self.opt.write_error_log}")
        assert self.is_validation, "Must be in validation mode"
        if self.opt.write_error_log:
            error_log = h5py.File(os.path.join(self.results_dir, f"error_log_{self.opt.dataset_key}.h5"), "w")
            error_log.create_dataset("error", shape=(self.N,), dtype=np.float)
            error_log.create_dataset("user", shape=(self.N,), dtype='S4')
            error_log.create_dataset("filename", shape=(self.N,), dtype='S13')
            error_log.create_dataset("visualisation", shape=(self.N, 1, 380, 1000), dtype=np.uint8)

        all_errors = list()

        for i, data_i in enumerate(self.dataloader):
            if i * self.opt.batchSize >= self.limit:
                break
            print(f"Processing batch {i} / {int(self.limit / self.opt.batchSize)}")
            fake, fake_resized = self.get_fake_and_resized(model, data_i)

            target_image = ImagePostprocessor.as_batch(data_i["target_original"], as_tensor=True)
            errors = np.array(MSECalculator.calculate_mse_for_images(fake_resized, target_image))
            all_errors += list(errors)

            if self.opt.write_error_log:
                visualisation_data = {**data_i, "fake": fake}
                visuals = visualize_sidebyside(visualisation_data, self.visualizer, log=False)

                # We add the entire batch to the output file
                idx_from, idx_to = i * self.opt.batchSize, i * self.opt.batchSize + self.opt.batchSize
                error_log["user"][idx_from:idx_to] = np.array(data_i["user"],
                                                              dtype='S4')
                error_log["filename"][idx_from:idx_to] = np.array(data_i["filename"],
                                                                  dtype='S13')
                error_log["error"][idx_from:idx_to] = errors
                vis = np.array([np.copy(v) for k, v in visuals.items()])
                # vis are all floats in [-1, 1]
                vis = (vis + 1) * 128
                error_log["visualisation"][idx_from:idx_to] = vis

        N_actual = min(i * self.opt.batchSize + self.opt.batchSize, self.dataloader.dataset.N)
        all_errors_sum = np.sum(all_errors)
        relative_errors_sum = all_errors_sum / len(all_errors) * 1471

        print(f"Error calculated on {N_actual} / {self.dataloader.dataset.N} samples{os.linesep}"
              f"  sum: {all_errors_sum:.2f}  {os.linesep}"
              f"  sum relative to n=1471: {relative_errors_sum:.2f} {os.linesep}"
              f"  mean: {np.mean(all_errors):.6f} (std: {np.std(all_errors):.4f})"
              f"  dataset_key: {self.opt.dataset_key}, model: {self.opt.name}")
        if self.opt.write_error_log:
            error_log.create_dataset("error_relative_n1471", data=np.multiply(error_log["error"], 1471),
                                     dtype=np.float)
            error_log.close()

        errors_dict = {f'mse/{self.opt.dataset_key}/full/sum': all_errors_sum, f'mse/{self.opt.dataset_key}/full/relative': relative_errors_sum}
        self.g_logger.update_or_append_row(errors_dict)
        return all_errors_sum, relative_errors_sum

    def run_test(self, model):
        filepaths = list()

        for i, data_i in enumerate(self.dataloader):
            if i * self.opt.batchSize >= self.limit:
                break

            img_filename = data_i['filename']
            # The test file names are only 12 characters long, so we have dot to remove
            img_filename = [re.sub(r'\.', '', f) for f in img_filename]
            # We are testing
            for b in range(len(img_filename)):
                result_path = os.path.join(self.results_dir, img_filename[b] + ".npy")
                fake, fake_resized = self.get_fake_and_resized(model, data_i)
                assert torch.min(fake_resized[b]) >= 0 and torch.max(fake_resized[b]) <= 255
                np.save(result_path, np.copy(fake_resized[b]).astype(np.uint8))
                filepaths.append(result_path)

        # We are testing
        path_filepaths = os.path.join(self.results_dir, "pred_npy_list.txt")
        with open(path_filepaths, 'w') as f:
            for line in filepaths:
                f.write(line)
                f.write(os.linesep)
        print(f"Written {len(filepaths)} files. Filepath: {path_filepaths}")

    def run(self, model):
        print(f"Is validation: {self.is_validation}. Dataset_key: {self.opt.dataset_key}")
        if self.is_validation:
            self.run_full_validation(model)
        else:
            self.run_test(model)


class MSECalculator:
    @classmethod
    def calculate_mse_for_images(cls, produced, target):
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

    @classmethod
    def calculate_relative_sum_mse(cls, data):
        fake = ImagePostprocessor.to_255resized_imagebatch(data['fake'], as_tensor=True)
        real = ImagePostprocessor.as_batch(data["target_original"], as_tensor=True)

        mse_error = cls.calculate_mse_for_images(fake, real)
        relative_mse_error = mse_error * 1471
        mu = torch.mean(relative_mse_error)
        std = torch.std(relative_mse_error)
        return mu, std, mse_error
