import logging
import sys

import numpy as np
import torch

from core import DefaultConfig

sys.path.insert(0, './deeplab')
from deeplab.modeling.deeplab import DeepLab  # noqa
del sys.path[0]

input_size = (400, 640)
output_stride = 16

config = DefaultConfig()
logger = logging.getLogger(__name__)


class RefineNet(DeepLab):
    def forward(self, input_dict):
        x = None
        if self.training:
            x = input_dict['train']['input']
            y_true = input_dict['train']['target']
            input_dict = input_dict['train']
        else:
            x = input_dict['input']
            y_true = input_dict['target'] if 'target' in input_dict else None

        output_dict = {'input': x}

        # Make prediction
        network_output = super().forward(x)
        output_dict['residual'] = network_output

        # Alpha-blend style image and new pixels
        reference_image = x[:, 1, :, :].unsqueeze(1)
        y_pred = torch.clamp(network_output + reference_image,
                             min=-1.0, max=1.0)
        output_dict['prediction'] = y_pred

        # Copy through some strings
        output_dict['person_id'] = input_dict['person_id']
        output_dict['fname'] = input_dict['fname']

        # Loss calculation
        if y_true is not None:
            output_dict['l1_loss'] = torch.mean(torch.abs(y_pred - y_true))
            output_dict['groundtruth'] = y_true

            output_dict['per_image_score'] = torch.sqrt(
                torch.sum(
                    (255. / 2. * (y_pred - y_true)) ** 2,
                    dim=[1, 2, 3],
                )
            ) / float(np.prod(y_true.shape[2:]))  # at this point, 1 scalar per entry

            output_dict['score'] = 1471 * torch.sum(
                output_dict['per_image_score']
            ) / float(y_true.shape[0])  # at this point, 1 scalar

            # Allow direct optimizing of what we believe is the OpenEDS loss
            output_dict['eds_loss'] = torch.mean(output_dict['per_image_score'])

        return output_dict
