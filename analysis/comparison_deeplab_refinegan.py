from util.files import listdir
import numpy as np
import os
import matplotlib.pyplot as plt

path_deeplab = "/home/marcel/projects/openeds_submission/res/refinenet/refinenet_submission_190924_130403/validation"
path_refinegan = "/home/marcel/projects/Seg2Eye/checkpoints/190916_refiner_size256_bs1_lr_l1_0_l2_15_lopeneds_0_lambda_w_0.5_lambda_feat_0.001_w1024___spadeStyleGen__pretrainD_cmadd_ns3_SAMmax_wc0_SSMref_random50/results/validation/"

image_files = listdir(path_deeplab, postfix='.npy')
image_ids = [f[:-4] for f in image_files]

for i, image_id in enumerate(image_ids):
    print(f"{i} / {len(image_ids)}")
    image_deeplab = np.load(os.path.join(path_deeplab, image_files[i]))
    image_refinegan = np.load(os.path.join(path_refinegan, image_files[i]))[0]

    diff = np.square(image_deeplab - image_refinegan)
    # diff = diff / np.max(diff) * 255
    cat = np.concatenate([image_deeplab, image_refinegan, diff], axis=-1)
    plt.imshow(cat, cmap='gray')
    plt.show()
    print()
