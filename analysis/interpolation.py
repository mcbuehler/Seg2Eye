import os
from datetime import datetime
import subprocess
import PIL

from data.postprocessor import ImagePostprocessor

from models.pix2pix_model import Pix2PixModel
from options.test_options import TestOptions

from util.files import create_folder_if_not_exists

import data
import cv2
import numpy as np
from PIL import Image

import torch

opt = TestOptions().parse()
dataset_class = data.find_dataset_using_name(opt.dataset_mode)
dataset = dataset_class()
dataset.initialize(opt)

model = Pix2PixModel(opt)

# Combo that shows background issue
# index_start = 500
# index_end = 7234

def style_image_to_grid(data):
    grid = torch.cat((torch.cat((data['style_image'][0][0], data['style_image'][0][1]), dim=-1),

                             torch.cat((data['style_image'][0][2], data['style_image'][0][3]), dim=-1))
                            , dim=-2)
    return grid


def run_interpolation(index_start, index_end, n=5, write_cat=False, show=False, write_single=False, out_folder=None):
    if write_single:
        create_folder_if_not_exists(out_folder)
        print(f"Outputting single png files to {out_folder}")

    if np.abs(index_start - index_end) < 100:
        index_end += 200

    data_start = dataset.get_particular(index_start)
    # data_start['style_image'] = data_start['style_image'][0]
    data_end = dataset.get_particular(index_end)

    # data_mask= dataset.get_particular(index_mask)
    data_mask = data_end
    # data_end['style_image'] = data_end['style_image'][0]

    latent_style_start = model.forward(data_start, mode="encode_only")
    latent_style_end = model.forward(data_end, mode="encode_only")

    # Interpolate style
    # target_label = data_start['label']
    results = list()
    # shuffled_indices = np.random.choice(list(range(1000,8000)), 2000, replace=False)
    # diffs = list()
    # for i, idx in enumerate(shuffled_indices):
    #     print("processing, ", i)
    #     data_i = dataset.get_particular(idx)
    #     # diff = (target_label.float().cuda() - data_i['label'].float().cuda())**2
    #     diff = torch.sum(target_label.cpu().float() != data_i['label'].cpu().float()).numpy()
    #     diffs.append(np.sum(diff))
    # idx_sorted = np.argsort(diffs)

    # for idx in idx_sorted[:5]:
    #     print(idx)
    #     data_i = dataset.get_particular(shuffled_indices[idx])
    #     fake = model.forward(data_i, mode="inference").detach().cpu()
    #     fake_resized = ImagePostprocessor.to_255resized_imagebatch(fake, as_tensor=False)[0][0].astype(np.uint8)
    #     results.append(fake_resized)

    results = list()

    border_width = 20

    # mask = data_mask['label'].float() /3 * 255
    # mask = ImagePostprocessor.resize(mask, as_tensor=False)[0][0]

    data_i = data_mask
    for i, alpha in enumerate(np.linspace(0, 1, n)):
        data_i['latent_style'] = latent_style_start * (1 - alpha) + latent_style_end * alpha
        # data_i_end['latent_style'] = latent_style_start * (1 - alpha) + latent_style_end * alpha

        fake = model.forward(data_i, mode="inference").detach().cpu()
        fake_resized = ImagePostprocessor.to_255resized_imagebatch(fake, as_tensor=False)[0][0].astype(np.uint8)


        results.append(fake_resized)

        if write_single:
            path_out = f'{out_folder}/{i:05d}.png'
            Image.fromarray(fake_resized).save(path_out)


        # fake = model.forward(data_i_end, mode="inference").detach().cpu()
        # fake_resized = ImagePostprocessor.to_255resized_imagebatch(fake, as_tensor=False)[0][0].astype(np.uint8)
        # results_end.append(fake_resized)

    if write_cat or show:
        # results.append(np.zeros(mask.shape[0], 10))

        # results.append(np.zeros((mask.shape[0], border_width), dtype=np.uint8))
        # results.append(original_end)
        # data_i['latent_style'] = latent_style_start
        # for i in [0, 100, 300, 600]:
        #     data_new = dataset.get_particular(i)
        #     data_i['label'] = data_new['label']
        #     fake = model.forward(data_new, mode="inference").detach().cpu()
        #     fake_resized = ImagePostprocessor.to_255resized_imagebatch(fake, as_tensor=False)[0][0].astype(np.uint8)
        #     results.append(fake_resized)
        cat = np.concatenate(results, axis=-1)
        # cat = np.concatenate([np.concatenate(results_start, axis=-1), np.concatenate(results_end, axis=-1)], axis=-2)
        # import matplotlib.pyplot as plt
        # plt.imshow(cat, cmap='gray')
        # plt.show()

        image = Image.fromarray(cat)
        # current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        if write_cat:
            path_out = f'/home/marcel/Documents/paper/interpolation/selected/{index_start}_{index_end}.png'
            image.save(path_out)
            print(f"Saved to {path_out}")
        if show:
            cv2.imshow("wind", cat)
            cv2.waitKey(1000)
        return cat
    return path_out


def create_random_figure(n=5):
    # index_start = 100
    # index_end = 800
    # index_end = 7500
    indices = np.random.choice(list(range(2400)), 3)
    # r = [run_interpolation(si, ei) for (si, ei) in zip(start_indices, end_indices)]
    r = run_interpolation(*list(indices), n)

    # out = np.concatenate(r, axis=-2)
    out = r


def run_single(idx):
    data_idx = dataset.get_particular(idx)
    fake = model.forward(data_idx, mode="inference").detach().cpu()
    fake_resized = ImagePostprocessor.to_255resized_imagebatch(fake, as_tensor=False)[0][0].astype(np.uint8)
    label = ImagePostprocessor.to_255resized_imagebatch(data_idx['label'], as_tensor=False)[0][0].astype(np.uint8)
    original = ImagePostprocessor.to_255resized_imagebatch(data_idx['target'], as_tensor=False)[0][0].astype(np.uint8)
    data = {'fake': fake_resized, 'label': label, 'original': original}
    for i in range(data_idx['style_image'].shape[1]):
        img = ImagePostprocessor.to_255resized_imagebatch(data_idx['style_image'][0][i], as_tensor=False)[0][
            0].astype(np.uint8)
        data[f'style{i}'] = img

    base_path = "/home/marcel/Documents/paper/images/"
    path = os.path.join(base_path, str(idx))
    for key, image in data.items():
        if not os.path.exists(path):
            os.mkdir(path)
        image = Image.fromarray(image)
        image.save(os.path.join(path, f'{key}.png'))

    cat = [
        data[f'style0'],
        data['label'],
        np.ones((data['fake'].shape[0], 20)).astype(np.uint8) * 255,
        data['fake'],
        np.ones((data['fake'].shape[0], 20)).astype(np.uint8) * 255,
        data['original']
    ]
    image_cat = np.concatenate(cat, axis=-1)
    image = Image.fromarray(image_cat)
    image.save(os.path.join(base_path, f'cat{idx}.png'))


def stack(combos):
    images = [Image.open(f'/home/marcel/Documents/paper/interpolation/selected/{c[0]}_{c[1]}.png') for c in combos]
    stacked = np.concatenate(images, axis=-2)
    Image.fromarray(stacked).save(f'/home/marcel/Documents/paper/interpolation/selected/stacked.png')


def create_random_single(n):
    indices = np.random.choice(list(range(2400)), n)
    for i in indices:
        run_single(i)


def interpolation_video(index_start, index_end, n=50):
    data_start = dataset.get_particular(index_start)
    data_end = dataset.get_particular(index_end)

    data_mask = data_end

    latent_style_start = model.forward(data_start, mode="encode_only")
    latent_style_end = model.forward(data_end, mode="encode_only")

    data_i = data_mask

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    path_out = f'/home/marcel/Documents/paper/interpolation/selected/{index_start}_{index_end}/'
    from util.files import create_folder_if_not_exists
    create_folder_if_not_exists(path_out)

    out = cv2.VideoWriter(path_out, fourcc, 20.0, (640, 400), False)
    for i, alpha in enumerate(np.linspace(0, 1, n)):
        data_i['latent_style'] = latent_style_start * (1 - alpha) + latent_style_end * alpha
        # data_i_end['latent_style'] = latent_style_start * (1 - alpha) + latent_style_end * alpha

        fake = model.forward(data_i, mode="inference").detach().cpu()
        fake_resized = ImagePostprocessor.to_255resized_imagebatch(fake, as_tensor=False)[0][0].astype(np.uint8)
        Image.fromarray(fake_resized).save(os.path.join(path_out, f"frame{i:04d}.jpg"))

        cv2.imshow('frame', fake_resized)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break



    #
    # for combo in combos:
    #
    #     if ret==True:
    #         # write the flipped frame
    #         out.write(frame)
    #
    #         cv2.imshow('frame',frame)
    #         if cv2.waitKey(1) & 0xFF == ord('q'):
    #             break
    #     else:
    #         break

# Release everything if job is finished

#     create_random_figure()
#     cv2.waitKey(500)

# for i in range(25):
#     create_random_figure(n=5)


# good combos:
# combos = [(448, 1404)]
# for (index_start, index_end) in combos:
#     run_interpolation(index_start, index_end, index_end, n=5)

combos = (
    (234, 2097),
    (2060, 1209),
    (604, 1006),
    (830, 1097),
    #(1097, 2097),
    #(323, 1340),
)
# stack(combos)

# Create fake images and video for single images
def video_single(combos, n):
    for combo in combos:
        index_start, index_end = combo[0], combo[1]
        print(f"Running combo {index_start} {index_end}")
        out_folder = f'/home/marcel/Documents/paper/interpolation/{index_start}_{index_end}'
        run_interpolation(index_start, index_end, n=n, write_single=True, out_folder=out_folder)
        subprocess.run(f"ffmpeg -f image2 -r 20 -i {out_folder}/%05d.png -vcodec mpeg4 -y {out_folder}/{index_start}_{index_end}_{n}.mp4", shell=True)


def stack_and_video(combos, n):
    path_tmp = f'/home/marcel/Documents/paper/interpolation/stacked'
    create_folder_if_not_exists(path_tmp)
    for i in range(n):
        images = list()
        for combo in combos:
            in_file = f'/home/marcel/Documents/paper/interpolation/{combo[0]}_{combo[1]}/{i:05d}.png'
            img = Image.open(in_file)
            images.append(img)
        stacked = np.concatenate(images, axis=-1)
        Image.fromarray(stacked).save(os.path.join(path_tmp, f"{i:05d}.png"))
    subprocess.run(
            f"ffmpeg -r 16 -i {path_tmp}/%05d.png -vcodec mpeg4 -q:v 2  -y {path_tmp}/stacked_{n}.avi",
            #f"avconv -s hd720 -r 20 -i {path_tmp}/%05d.png -vcodec mpeg4 -y {path_tmp}/stacked_{n}.mp4",
            shell=True)


n = 150
video_single(combos, n)
stack_and_video(combos, n)

# create_random_single(25)
# run_single(800)
# for i in range(25):
# create_random_single(50)


# run_interpolation(1097, 2097, None, n=5)