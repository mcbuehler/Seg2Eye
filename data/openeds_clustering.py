import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torchvision.utils import make_grid

from ssim.pytorch_ssim import SSIM
from data.openeds_visualize_person import get_grid, normalize_masks, normalize_images, count_black

col = np.array((0, 255, 0))
col = np.array([col for i in range(20*20)])
col = col.reshape((3, 20, 20))

print(col[:, 0, 0])
col = torch.Tensor(0, 255, 0)
orig_shape = col.shape
new_shape = (20, 3)

input = col.unsqueeze(1) # [100, 1, 1024, 14, 14]
input = input.expand(*new_shape) # [100, 10, 1024, 14, 14]
input = input.transpose(0, 1).contiguous() # [10, 100, 1024, 14, 14]
input = input.view(-1, *orig_shape[1:]) # [1000, 1024, 14, 14]

print(input.shape)

col = torch.ones((3, 20, 20))
a = torch.mul(col, torch.Tensor((10,10,0)))
exit()


path_h5 = '/home/marcel/projects/data/openeds/all.h5'
h5_in = h5py.File(path_h5, 'r')

limit = 31
# limit = 60
np.random.seed(1234)
dataset_key = 'train'
data_in = h5_in[dataset_key]

person_ids = list(data_in.keys())

person_id = person_ids[11]
data_person = data_in[person_id]

images_ss = np.copy(data_person['images_ss'])
masks_ss = np.copy(data_person['labels_ss'])

images = images_ss[:limit]
masks = masks_ss[:limit]

# print(torch.min(mask), torch.max(mask))



# idx_iris = masks == 3
# idx_pupil = masks == 1
# idx_eye = masks == 2
# idx_bg = masks == 0
# masks[idx_iris] = 0
# masks[idx_pupil] = 3
# masks[idx_bg] = 0
# masks[idx_eye] = 2

def get_pupil_location(mask):
    location = torch.nonzero((mask == 3)).float()
    x = torch.mean(location[:,0]) / mask.shape[-2]
    y = torch.mean(location[:, 1]) / mask.shape[-1]
    return torch.Tensor(np.array([x, y])).long()

print(masks.shape)
masks = masks[0]
masks = torch.from_numpy(masks)
location = torch.nonzero((masks == 3)).float()
x = torch.mean(location[:,0])
y = torch.mean(location[:, 1])

x = int(x)
y = int(y)
print(x,y)
p = np.zeros(masks.shape)
p[x,y] = 1
plt.imshow(p[x-20:x+20, y-20:y+20])
plt.show()
print()

plt.imshow(masks[x-20:x+20, y-20:y+20])
plt.show()
input()


exit()

masks = normalize_masks(masks)
images = normalize_images(images)

mask = torch.from_numpy(masks).unsqueeze(1)

image = torch.from_numpy(images).unsqueeze(1)

ssim_loss = SSIM(size_average=False)

mask = mask[22]
#
#

mse = nn.MSELoss(reduce=False)
similarities = torch.mean(torch.mean(mse(mask, image), dim=-1), dim=-1)
similarities = torch.stack([torch.abs(torch.sum(img[0,320:] <= 0.01) - torch.sum(mask[0,320:] <= 0.01)) for img in image])
similarities = torch.max(similarities) - similarities
print(similarities)

image = torch.stack([image[i] for i in torch.argsort(similarities)])

# s = [mask[:,:,i] for i in idx_nonzero]
# nonzero_mask = torch.stack(s)
# similarities = ssim_loss.forward(mask[idx_nonzero], image[idx_nonzero])
#
# print(similarities)
# grid_images = get_grid(image)
# grid_masks = get_grid(mask)

image = torch.cat([image, mask.unsqueeze(0)], dim=0)
grid_images = make_grid(image)
# grid_masks = make_grid(mask)

# grid_np = np.concatenate([grid_images, grid_masks], axis=-2)
# grid_np = grid_masks
grid_np = grid_images.numpy()
plt.imshow(grid_np.transpose(1, 2, 0))
plt.show()



