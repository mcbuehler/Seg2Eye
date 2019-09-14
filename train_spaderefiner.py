"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import os
import sys
import traceback

import numpy as np
import torch

from options.refiner_options import RefinerOptions
from options.train_options import TrainOptions
from torch.nn.utils import clip_grad_norm_

import data
from models import networks
from options.pupillocator_options import PupilLocatorOptions
from util import util
from util.files import copy_src
from util.iter_counter import IterationCounter
from util.visualizer import Visualizer
from util.visualizer import annotate_pupil

# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
# parse options
opt = RefinerOptions().parse()


opt.display_freq = 500
opt.print_freq = 100
opt.validation_limit = 250

# opt.display_freq = 10
# opt.print_freq = 5
# opt.validation_limit = 10

# opt.full_val_freq = 100
# full_val_limit = 1

copy_src(path_from='./', path_to=os.path.join(opt.checkpoints_dir, opt.name))

# print options to help debugging
print(' '.join(sys.argv))

# load the dataset
dataloader = data.create_dataloader(opt)

# Validation
dataloader_val = data.create_inference_dataloader(opt, dataset_key='validation', shuffle=True)
dataloader_test = data.create_inference_dataloader(opt, dataset_key='test', shuffle=True)

model = networks.define_refiner(opt)
model_one_gpu = model.module if len(opt.gpu_ids) > 0 else model
if not opt.isTrain or opt.continue_train:
    util.load_network(model, model_one_gpu.name, opt.which_epoch, opt)

optimizer = model_one_gpu.create_optimizer(list(model.parameters()), opt)

# create tool for counting iterations
iter_counter = IterationCounter(opt, len(dataloader))

# create tool for visualization
visualizer = Visualizer(opt)


try:
    for epoch in iter_counter.training_epochs():
        if iter_counter.current_epoch != epoch:
            # They are only equal at the very beginning and after loading a model
            iter_counter.record_epoch_start(epoch)

        for i, data_i in enumerate(dataloader, start=iter_counter.epoch_iter):
            iter_counter.record_one_iteration()

            optimizer.zero_grad()

            style_image, target_mask, target_image = data_i['style_image'], data_i['label'], data_i['target']

            # Take best style image for each sample in batch
            style_image = torch.stack([style_image[i][0] for i in range(style_image.shape[0])])

            losses, out_pred = model_one_gpu.get_losses(style_image, target_mask, target_image)

            clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()

            # Visualizations
            if iter_counter.needs_printing():
                visualizer.print_current_errors(epoch, iter_counter.total_steps_so_far,
                                                losses, iter_counter.time_per_iter)
                visualizer.plot_current_errors(losses, iter_counter.total_steps_so_far)

            if iter_counter.needs_displaying():

                visualizer.print_current_errors(epoch, iter_counter.total_steps_so_far,
                                                losses, iter_counter.time_per_iter)
                visualizer.plot_current_errors(losses, iter_counter.total_steps_so_far)
                out_pred_cpu = out_pred.cpu()
                visuals = {
                    'train': torch.cat([
                        style_image,
                        target_mask / 3,
                        target_image,
                        out_pred_cpu,
                        torch.abs(out_pred_cpu - target_image)
                    ], dim=-1)
                }
                visualizer.display_current_results(visuals, epoch, iter_counter.total_steps_so_far)

                with torch.no_grad():
                    losses_list = list()
                    for i, data_i in enumerate(dataloader_val):
                        style_image, target_mask, target_image = data_i['style_image'], data_i['label'], data_i[
                            'target']
                        style_image = torch.stack([style_image[i][0] for i in range(style_image.shape[0])])
                        losses, out_pred = model_one_gpu.get_losses(style_image, target_mask, target_image, mode='validation')
                        out_pred_cpu = out_pred.cpu()

                        if i > opt.validation_limit:
                            break
                        losses_list.append(losses)
                    losses = {key: torch.stack([losses_list[i][key] for i in range(len(losses_list))]) for key in losses_list[0]}
                    # Filtering out nans
                    # losses = {key: torch.stack([v for v in losses[key] if not np.isnan(v.cpu().detach().numpy())]) for key in losses}
                    visualizer.print_current_errors(epoch, iter_counter.total_steps_so_far,
                                                    losses, iter_counter.time_per_iter)
                    visualizer.plot_current_errors(losses, iter_counter.total_steps_so_far)
                    visuals = {
                    'validation': torch.cat([
                        style_image,
                        target_mask / 3,
                        target_image,
                        out_pred_cpu,
                        torch.abs(out_pred_cpu - target_image)
                    ], dim=-1)
                }
                    visualizer.display_current_results(visuals, epoch, iter_counter.total_steps_so_far)

                    visuals = {}
                    for i, data_i in enumerate(dataloader_test):
                        style_image, target_mask = data_i['style_image'], data_i['label']
                        style_image = torch.stack([style_image[i][0] for i in range(style_image.shape[0])])
                        out_pred = model_one_gpu.forward(style_image, target_mask, mode='test')
                        out_pred_cpu = out_pred.cpu()

                        visuals[f'test/{i}'] = torch.cat([
                        style_image,
                        target_mask / 3,
                        out_pred_cpu,
                        torch.abs(out_pred_cpu - target_image)
                        ], dim=-1)
                        break

                    visualizer.display_current_results(visuals, epoch, iter_counter.total_steps_so_far)

            if iter_counter.needs_saving():
                print('saving the latest model (epoch %d, total_steps %d)' %
                      (epoch, iter_counter.total_steps_so_far))
                model_one_gpu.save('latest')
                iter_counter.record_current_iter()

        # trainer.update_learning_rate(epoch)
        iter_counter.record_epoch_end()

        if epoch % opt.save_epoch_freq == 0 or \
           epoch == iter_counter.total_epochs:
            iter_counter.record_current_iter()
            print('saving the model at the end of epoch %d, iters %d' %
                  (epoch, iter_counter.total_steps_so_far))
            model_one_gpu.save('latest')
            model_one_gpu.save(epoch)
    print('Training was successfully finished.')

except (KeyboardInterrupt, SystemExit):
        print("KeyboardInterrupt. Shutting down.")
        print(traceback.format_exc())
except Exception as e:
    print(traceback.format_exc())
finally:
    print('saving the model before quitting')
    model_one_gpu.save('latest')
    iter_counter.record_current_iter()
    del dataloader_val
    del dataloader


