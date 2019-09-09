"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import os
import sys
import traceback

import numpy as np
import torch
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
opt = PupilLocatorOptions().parse()


opt.display_freq = 500
opt.print_freq = 100
opt.validation_limit = 50

# opt.display_freq = 1
# opt.print_freq = 100
# opt.validation_limit = 50

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

model = networks.define_pupil_locator(opt)
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

            model.train()
            optimizer.zero_grad()

            image, mask = data_i['target'], data_i['label']
            losses, _, _ = model_one_gpu.get_losses(image, mask)

            clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()

            # Visualizations
            if iter_counter.needs_printing():
                visualizer.print_current_errors(epoch, iter_counter.total_steps_so_far,
                                                losses, iter_counter.time_per_iter)
                visualizer.plot_current_errors(losses, iter_counter.total_steps_so_far)

            if iter_counter.needs_displaying():
                mask = data_i['label'][:4]
                image = data_i['target'][:4]

                losses, out_pred, out_true = model_one_gpu.get_losses(data_i['target'], data_i['label'],
                                                                      mode='train')

                visualizer.print_current_errors(epoch, iter_counter.total_steps_so_far,
                                                losses, iter_counter.time_per_iter)
                visualizer.plot_current_errors(losses, iter_counter.total_steps_so_far)
                visuals = {
                    'train/label': mask / 3,
                    'train/target': annotate_pupil(image, out_true),
                    'train/pred': annotate_pupil(image, out_pred)
                }
                visualizer.display_current_results(visuals, epoch, iter_counter.total_steps_so_far)

                with torch.no_grad():
                    losses_list = list()
                    for i, data_i in enumerate(dataloader_val):
                        losses, out_pred, out_true = model_one_gpu.get_losses(data_i['target'], data_i['label'], mode='validation')
                        if i > opt.validation_limit:
                            break
                        losses_list.append(losses)
                    losses = {key: torch.stack([losses_list[i][key] for i in range(len(losses_list))]) for key in losses_list[0]}
                    # Filtering out nans
                    losses = {key: torch.stack([v for v in losses[key] if not np.isnan(v.cpu().detach().numpy())]) for key in losses}
                    visualizer.print_current_errors(epoch, iter_counter.total_steps_so_far,
                                                    losses, iter_counter.time_per_iter)
                    visualizer.plot_current_errors(losses, iter_counter.total_steps_so_far)
                    visuals = {
                        'validation/label': data_i['label'][:8] / 3,
                        'validation/target': annotate_pupil(data_i['target'][:8], out_true),
                        'validation/pred': annotate_pupil(data_i['target'][:8], out_pred)
                    }
                    visualizer.display_current_results(visuals, epoch, iter_counter.total_steps_so_far)

                    for i, data_i in enumerate(dataloader_test):
                        style_images = torch.stack(
                            # We take the first style image for each sample in the back
                            [data_i['style_image'][b][0] for b in range(min(16, len(data_i['style_image'])))]
                        )
                        out_pred = model_one_gpu.forward(style_images, mode='test')
                        break
                    visuals = {
                        'test/pred': annotate_pupil(style_images, out_pred)
                    }
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


