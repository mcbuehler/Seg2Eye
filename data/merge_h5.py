import h5py

p1 = "/home/marcel/projects/moe-gaze/outputs/MyDeepLab/190913_025402.e72b33/deeplab_predictions_190914_123838.h5"
p2 = "/home/marcel/projects/moe-gaze/outputs/MyDeepLab/190913_025402.e72b33/deeplab_predictions_190913_210218.h5"

# p1 = "/home/marcel/projects/moe-gaze/outputs/MyDeepLab/190913_025402.e72b33/deeplab_predictions_190914_123714.h5"
# p2 = "/home/marcel/projects/moe-gaze/outputs/MyDeepLab/190913_025402.e72b33/deeplab_predictions_190914_123714.h5"
p_out = "/home/marcel/projects/data/openeds/190914_deeplab_seg_predictions.h5"
f1 = h5py.File(p1, 'r')
f2 = h5py.File(p2, 'r')
f_out = h5py.File(p_out, 'w')
# f_out = h5py.File(p_out, 'r')


for f in [f2, f1]:
    print(f)
    for key in f:
        print(key)
        f[key].copy(f[key], f_out, key)


#
# # Filter out seq in training
# p = "datasets/distances_and_indices.h5"
# f_out = h5py.File('datasets/190914_distances_and_indices.h5', 'w')
# f_in = h5py.File(p, 'r')
#
# print(f_in.keys())
# exit()
# for split in f_in.keys():
#     f_out.copy(f_in[split], f_out, split)
# n_updated = 0
# import numpy as np
# for person_id in f_out['train']:
#     for filename in f_out['train'][person_id]:
#         print(f_out['train'][person_id][filename].keys())
#         print(f_out['train'][person_id][filename]['subset'][:])
#         all_not_seq = f_out['train'][person_id][filename]['subset'][:] != b's'
#         for key in f_out['train'][person_id][filename]:
#
#             old_shape = f_out['train'][person_id][filename][key].shape
#
#             new_data = np.copy(f_out['train'][person_id][filename][key][all_not_seq])
#             del f_out['train'][person_id][filename][key]
#             new_shape = new_data.shape
#             f_out['train'][person_id][filename][key] = new_data
#
#             if old_shape != new_shape:
#                 n_updated += 1
# f_out.close()
# f_in.close()
#
# print(n_updated)
