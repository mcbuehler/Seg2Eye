[![License CC BY-NC-SA 4.0](https://img.shields.io/badge/license-CC4.0-blue.svg)](https://raw.githubusercontent.com/nvlabs/SPADE/master/LICENSE.md)
![Python 3.6](https://img.shields.io/badge/python-3.6-green.svg)

# Content-Consistent Generation of Realistic Eyes with Style
[Project page](https://ait.ethz.ch/projects/2019/seg2eye/) |  [Paper](https://arxiv.org/abs/1911.03346)

### Abstract
Accurately labeled real-world training data can be scarce, and hence recent
 works adapt, modify or generate images to boost target datasets.
 However, retaining relevant details from input data in the generated images
 is challenging and failure could be critical to the performance on the final
 task. In this work, we synthesize person-specific eye images that satisfy a
 given semantic segmentation mask (content), while following the style of a
 specified person from only a few reference images. We introduce two
 approaches, (a) one used to win the
 [OpenEDS Synthetic Eye Generation Challenge](https://research.fb.com/programs/openeds-challenge)
 at [ICCVW 2019](http://iccv2019.thecvf.com/), and (b) a
 principled approach to solving the problem involving simultaneous injection
 of style and content information at multiple scales.

![Style Interpolation Demo](https://github.com/mcbuehler/Seg2Eye/raw/clean/docs/interpolation_single.gif)

[Marcel C. Bühler](http://mcbuehler.ch), [Seonwook Park](https://ait.ethz.ch/people/spark/), [Shalini De Mello](https://research.nvidia.com/person/shalini-gupta),
[Xucong Zhang](https://ait.ethz.ch/people/zhang/), [Otmar Hilliger](https://ait.ethz.ch/people/hilliges/) \\

[ICCV Workshop 2019](http://iccv2019.thecvf.com/) \\

[VR and AR Workshop](https://research.fb.com/programs/the-2019-openeds-workshop-eye-tracking-for-vr-and-ar/)



## Dataset Preparation

1. You need access to the OpenEDS Dataset. Please find more information [here](https://research.fb.com/programs/openeds-challenge).

2. Unzip all folders and set the `base_path` to the root folder containing the unpacked subfolders. This folder should also contain the json files with the mappings of file to users (_OpenEDS_{train,validation,test}_userID_mapping_to_images.json_).

In 'data/prepare_openeds.py', update the 'base_path = "..."' with the path to the unzipped OpenEDS Dataset. Then run
```
python data/prepare_openeds.py
```
This will produce an H5 file that you can use to train or test Seg2Eye models.

## Training New Models
Run
```
python train.py --dataroot PATH_TO_H5_FILE
```

Please note:

* This implementation currently does not support multi-GPU training.

## Testing
```
python test.py --dataroot PATH_TO_H5_FILE --name CHECKPOINT_NAME \
    --dataset_key VALIDATION|TEST --load_from_opt_file
```

## Code Structure

- `train.py`, `test.py`: the entry point for training and testing.
- `trainers/pix2pix_trainer.py`: harnesses and reports the progress of training.
- `models/pix2pix_model.py`: creates the networks, and compute the losses
- `models/networks/`: defines the architecture of all models
- `options/`: creates option lists using `argparse` package. More individuals are dynamically added in other files as well. Please see the section below.
- `data/`: defines the class for loading and processing data.

### [License](https://raw.githubusercontent.com/mcbuehler/Seg2Eye/master/LICENSE.md)

Copyright (C) 2019 NVIDIA Corporation.

All rights reserved.
Licensed under the [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode) (**Attribution-NonCommercial-ShareAlike 4.0 International**)

The code is released for academic research use only.


## Citation
If you reference our work, please cite our paper.
```
@inproceedings{Buehler2019ICCVW,
  author    = {Marcel C. Buehler and Seonwook Park and Shalini De Mello and Xucong Zhang and Otmar Hilliges},
  title     = {Content-Consistent Generation of Realistic Eyes with Style},
  year      = {2019},
  booktitle = {International Conference on Computer Vision Workshops (ICCVW)},
  location  = {Seoul, Korea}
}
```


## Acknowledgments
This repository is a fork of the original [SPADE](https://github.com/NVlabs/SPADE) implementation.

This work was supported in part by the ERC Grant OPTINT (StG-2016-717054).


