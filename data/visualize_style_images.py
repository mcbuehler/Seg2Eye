import h5py
import torch
import numpy as np

import matplotlib.pyplot as plt
from torchvision.utils import make_grid

style_image_refs = h5py.File("datasets/0910_deeplab_top_image_indices_for_marcel.h5", 'r')

dataset_key = 'train'
user_id = 'U116'
limit = 50


h5_in_file = h5py.File('/home/marcel/projects/data/openeds/all.h5', 'r')  # , libver='latest')
h5_in = h5_in_file[dataset_key]
keys = list(style_image_refs[dataset_key][user_id].keys())
filename = keys[0]


idx = style_image_refs[dataset_key][user_id][filename]['index'][:limit]
dist = style_image_refs[dataset_key][user_id][filename]['dist'][:limit]
style_images = [h5_in[user_id]['images_gen'][i] for i in idx]
from util.image_annotate import get_text_image

for i in range(len(style_images)):
    txt = get_text_image(text=str(dist[i]), dim=(100, style_images[i].shape[-1]))
    style_images[i] = torch.cat([
        torch.from_numpy(np.array(style_images[i])).float(), txt.float()
    ], dim=-2).numpy()


out = torch.from_numpy(np.array(style_images)).unsqueeze(1)
grid = make_grid(out)

plt.imshow(grid[0], cmap='gray')
plt.show()
print(dist)

