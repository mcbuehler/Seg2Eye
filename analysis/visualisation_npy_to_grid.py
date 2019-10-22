from util.files import listdir
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os


path_refiner = "/usr2/home/marcello/mnt/lh/submission_res/refinenet/refinenet_submission_191002_144708/test/"
path_seg2eye = "/usr2/home/marcello/mnt/lh/wookie_spade_checkpoints/190925_spadestyle_size256_l2_15_lambda_w_0.5_lambda_feat_0.001_lambda_gram_100000_w16___spadeStyleGen_cmadd_ns4_SAMmax_SSMref_random100/results/test"

nrow = 4
ncol = 6

n = nrow * ncol

files = listdir(path_seg2eye, postfix=".npy")
selection = np.random.choice(files, size=n)
print(selection)

refiner = np.array([np.load(os.path.join(path_refiner, f)) for f in selection])
seg2eye = np.array([np.load(os.path.join(path_seg2eye, f))[0] for f in selection])

sep = np.ones((640, 20), dtype=np.uint8) * 255
sep_black = np.zeros((640, 10))

# side by side
l = list()
for row in range(nrow):
    i_start, i_end = row*ncol, row*ncol+ncol
    row_img = list()
    for i in range(i_start, i_end):
        row_img.append(refiner[i])
        row_img.append(sep_black)
        row_img.append(seg2eye[i])
        if i < i_end-1:
            row_img.append(sep)
    cat_row = np.concatenate(row_img, axis=-1)
    l.append(cat_row)

cat = np.concatenate(l, axis=-2).astype(np.uint8)
Image.fromarray(cat).save("/usr2/home/marcello/studies/2018-3_herbstsemester/research-in-datascience/wookie/continuation/seg2eye_archive/paper/illustrations/refiner_vs_seg2eye.png")
plt.imshow(cat, cmap="gray")
plt.show()


# blocks

