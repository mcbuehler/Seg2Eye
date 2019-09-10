"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import os
import ntpath
import time
from collections import OrderedDict

import torch
from PIL import Image

from data.postprocessor import ImagePostprocessor
from data.preprocessor import ImagePreprocessor
from util.image_annotate import get_text_image
from . import util
import scipy.misc
import numpy as np
from torchvision.utils import make_grid
try:
    from StringIO import StringIO  # Python 2.7
except ImportError:
    from io import BytesIO         # Python 3.x


class Visualizer():
    def __init__(self, opt):
        self.opt = opt
        self.tf_log = opt.isTrain and opt.tf_log
        self.name = opt.name
        if self.tf_log:
            import tensorflow as tf
            self.tf = tf
            self.log_dir = os.path.join(opt.checkpoints_dir, opt.name, 'logs')
            self.writer = tf.summary.FileWriter(self.log_dir)

        if opt.isTrain:
            self.log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
            with open(self.log_name, "a") as log_file:
                now = time.strftime("%c")
                log_file.write('================ Training Loss (%s) ================\n' % now)

    # |visuals|: dictionary of images to display or save
    def display_current_results(self, visuals, epoch, step):

        ## convert tensors to numpy arrays
        visuals = self.convert_visuals_to_numpy(visuals)
                
        if self.tf_log: # show images in tensorboard output
            img_summaries = []
            for label, image_numpy in visuals.items():
                # Write the image to a string
                try:
                    s = StringIO()
                except:
                    s = BytesIO()
                if len(image_numpy.shape) >= 4:
                    image_numpy = image_numpy[0]
                if len(image_numpy.shape) == 3 and image_numpy.shape[0] == self.opt.batchSize:
                    # We have a 2d image where first dimension is still batch size
                    image_numpy = image_numpy[0, :, :, np.newaxis]
                    image_numpy = np.repeat(image_numpy, 3, 2)

                Image.fromarray(image_numpy).save(s, format="jpeg")
                # scipy.misc.toimage(image_numpy).save(s, format="jpeg")
                # Create an Image object
                img_sum = self.tf.Summary.Image(encoded_image_string=s.getvalue(), height=image_numpy.shape[0], width=image_numpy.shape[1])
                # Create a Summary value
                img_summaries.append(self.tf.Summary.Value(tag=label, image=img_sum))

            # Create and write Summary
            summary = self.tf.Summary(value=img_summaries)
            self.writer.add_summary(summary, step)

    # errors: dictionary of error labels and values
    def plot_current_errors(self, errors, step):
        if self.tf_log:
            for tag, value in errors.items():
                value = value.mean()
                summary = self.tf.Summary(value=[self.tf.Summary.Value(tag=tag, simple_value=value)])
                self.writer.add_summary(summary, step)

    # errors: same format as |errors| of plotCurrentErrors
    def print_current_errors(self, epoch, i, errors, t):
        message = '(epoch: %d, iters: %d, time: %.3f) ' % (epoch, i, t)
        for k, v in errors.items():
            #print(v)
            #if v != 0:
            v = v.mean()
            message += '%s: %.3f ' % (k, v)

        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)

    def convert_visuals_to_numpy(self, visuals):
        for key, t in visuals.items():
            tile = self.opt.batchSize > 8
            if 'input_label' == key:
                t = util.tensor2label(t, self.opt.label_nc + 2, tile=tile)
            else:
                t = util.tensor2im(t, tile=tile)
            visuals[key] = t
        return visuals

    # save image to the disk
    def save_images(self, webpage, visuals, image_path):        
        visuals = self.convert_visuals_to_numpy(visuals)        
        
        image_dir = webpage.get_image_dir()
        short_path = ntpath.basename(image_path[0])
        name = os.path.splitext(short_path)[0]

        webpage.add_header(name)
        ims = []
        txts = []
        links = []

        for label, image_numpy in visuals.items():
            image_name = os.path.join(label, '%s.png' % (name))
            save_path = os.path.join(image_dir, image_name)
            util.save_image(image_numpy, save_path, create_dir=True)

            ims.append(image_name)
            txts.append(label)
            links.append(image_name)
        webpage.add_images(ims, txts, links, width=self.win_size)


def visualize_sidebyside(data, limit=-1, key_fake='fake', key_content='label', key_target='target_original', key_style='style_image', log_key='', w=200, h=320, error_list=None):
    # Validation results
    visuals_val = list()

    if limit > 0:
        # Apply the limit
        for key in data:
            data[key] = data[key][:limit]

    content = ImagePostprocessor.to_1resized_imagebatch(data[key_content], w, h, as_tensor=True)
    fake = ImagePostprocessor.to_1resized_imagebatch(data[key_fake], w, h, as_tensor=True)
    target = ImagePostprocessor.to_1resized_imagebatch(data[key_target], w, h, as_tensor=True)
    if len(data[key_style].shape) == 5:
        # We have multiple style images. Let's take max 4 per sample and create a grid for each sample
        style_grids = [make_grid(data[key_style][i, :4], nrow=2, padding=0) for i in range(data[key_style].shape[0])]
        data[key_style] = torch.mean(torch.stack(style_grids, dim=0), dim=1).unsqueeze(1)
    style = ImagePostprocessor.to_1resized_imagebatch(data[key_style], w, h, as_tensor=True)
    # error_heatmap = torch.pow(fake - target, 2)
    error_heatmap = ImagePostprocessor.get_error_map(fake, target)
    for i in range(len(data[key_fake])):
        cat_val = torch.cat((style[i], content[i], target[i], fake[i], error_heatmap[i]), dim=-1)

        # # 4th component: text annotation with metadata
        text = f'{data["user"][i]} / {data["filename"][i]}'
        if error_list is not None:
            err = error_list[i] * 1471
            text += f' (err: {err:.2f})'
        text_val = get_text_image(text, dim=(60, cat_val.shape[2])).unsqueeze(0)#f'{data_val["user"][0]}/{data_val["filename"][0]}', dim=(cat_val.shape[3], 50)
        cat_val = torch.cat((cat_val, text_val), dim=-2)
        #
        visuals_val.append((f'{log_key}/{i}', cat_val))
        if i > limit > 0:
            break

    visuals_val = OrderedDict(visuals_val)
    return visuals_val


def annotate_pupil(tensor, pupil_location, col=(0, 255, 0), offset=10):
    annotated_tensor = list()
    pupil_location[:,0] = (pupil_location[:, 0] + 0.5) * tensor.shape[-2]
    pupil_location[:,1] = (pupil_location[:, 1] + 0.5) * tensor.shape[-1]
    pupil_location = pupil_location.int()
    for i, t in enumerate(tensor):
        t = torch.cat([t,t,t], dim=0)
        loc_x = pupil_location[i][0]
        loc_x = [max(0, loc_x - offset), min(tensor.shape[-2], loc_x + offset)]
        loc_y = pupil_location[i][1]
        loc_y = [max(0, loc_y - offset), min(tensor.shape[-1], loc_y + offset)]

        for x in range(loc_x[0], loc_x[1]):
            for y in range(loc_y[0], loc_y[1]):
                for c in range(len(col)):
                    t[c, x, y] = col[c]
        annotated_tensor.append(t)
    return torch.stack(annotated_tensor)