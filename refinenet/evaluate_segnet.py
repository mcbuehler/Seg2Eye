import argparse
import logging
import os
import time

import cv2 as cv
import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

from core import CheckpointManager
import train_segnet
from train_segnet import MyDeepLab, OpenEDSDataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logger = logging.getLogger(__name__)

# Parse argument
parser = argparse.ArgumentParser(description='Evaluate a trained DeepLab model.')
parser.add_argument('input_dir', type=str, help='Saved model path')
args = parser.parse_args()
assert os.path.isdir(args.input_dir)


class OpenEDSDataset_Eval(OpenEDSDataset):

    def __init__(self, dataset_path: str, split: str, augment: bool = None, x_key: str = None):
        self.path = dataset_path
        self.split = split
        self.x_key = x_key
        if self.x_key is None:
            self.x_key = 'images_ss' if split == 'test' else 'images_gen'
        hdf = h5py.File(self.path, 'r')
        self.hdf = None

        # Construct mapping from full-data index to key and person-specific index
        self.idx_to_kv = []
        for person_id in list(hdf[split].keys()):
            n = hdf[split][person_id][self.x_key].shape[0]
            self.idx_to_kv += [(person_id, i) for i in range(n)]

        logger.info('Loaded HDF dataset at: %s' % self.path)

    def __getitem__(self, idx):
        if self.hdf is None:
            self.hdf = h5py.File(self.path, 'r')

        key, idx = self.idx_to_kv[idx]
        person_data = self.hdf[self.split][key]

        # Get images, face (64x64) and eyes (256x64)
        x = self.preprocess_image(np.copy(person_data[self.x_key][idx, :]))
        entry = {
            'person_id': key,
            'image': x,
        }

        # Convert arrays to tensors
        return dict([
            (k, torch.from_numpy(a))
            if isinstance(a, np.ndarray)
            else (k, a)
            for k, a in entry.items()
        ])


dataset_specs = [
    'train',
    'validation',
    # 'test',
]
datasrc = '/big/marcel/190910_all.h5'
datasets = [
    OpenEDSDataset_Eval(datasrc, split, x_key='images_seq')
    for split in dataset_specs
]
dataloaders = [
    DataLoader(dataset,
               batch_size=80,
               shuffle=False,
               drop_last=False,
               num_workers=2,
               pin_memory=True,
               )
    for dataset in datasets
]

# Build Model
model = MyDeepLab(
    num_classes=4,
    backbone='resnet',
    output_stride=train_segnet.output_stride,
    sync_bn=False,
    freeze_bn=False,
)
print(model)
model = model.to(device)
model.eval()

# Load checkpoint
model.output_dir = args.input_dir
checkpoint_manager = CheckpointManager(model)
checkpoint_manager.load_last_checkpoint()

# Create output handle
of = h5py.File('/big/marcel/deeplab_predictions_%s.h5' % time.strftime('%y%m%d_%H%M%S'), 'w')  # noqa

# Iterate through
with torch.no_grad():
    for d, (dataset, dataloader) in enumerate(zip(datasets, dataloaders)):
        split = dataset_specs[d]
        og = of.create_group(split)
        all_predictions = {}
        previous_key = None
        for b, input_dict in enumerate(dataloader):
            input_dict_cuda = {}
            for k, v in input_dict.items():
                if isinstance(v, torch.Tensor):
                    input_dict_cuda[k] = v.detach().to(device, dtype=torch.float32,
                                                       non_blocking=True)

            # Inference
            output_dict = model(input_dict_cuda)
            predictions = output_dict['prediction'].detach().cpu().numpy().astype(np.uint8)

            for i, prediction in enumerate(predictions):
                key = input_dict['person_id'][i]
                if key not in all_predictions:
                    all_predictions[key] = []
                    if key != previous_key:
                        if previous_key is not None:
                            og.create_dataset(previous_key,
                                              data=np.asarray(all_predictions[previous_key]))
                            print('Stored %d entries to %s/%s' %
                                  (len(all_predictions[previous_key]), split, previous_key))
                            del all_predictions[previous_key]
                        previous_key = key
                all_predictions[key].append(prediction)

                if i == 0 and b % 2 == 0:
                    input_image = (255. / 2. * (input_dict['image'][i, 0, :].numpy() + 1.)).astype(np.uint8)  # noqa
                    prediction = (255. / 3. * np.copy(prediction)).astype(np.uint8)
                    cv.imshow('sample', cv.hconcat([input_image, prediction]))
                    cv.waitKey(1)

        for key, prediction_list in all_predictions.items():
            all_predictions[key] = np.asarray(prediction_list)
            og.create_dataset(key, data=np.asarray(all_predictions[key]))

        print('Wrote split: %s' % split)
