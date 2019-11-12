[![License CC BY-NC-SA 4.0](https://img.shields.io/badge/license-CC4.0-blue.svg)](https://raw.githubusercontent.com/nvlabs/SPADE/master/LICENSE.md)
![Python 3.6](https://img.shields.io/badge/python-3.6-green.svg)

# Please note: 
# This README is outdated



This repository contains the code for Seg2Eye.


# Semantic Image Synthesis with SPADE
![GauGAN demo](https://nvlabs.github.io/SPADE//images/ocean.gif)

### [Project page](https://nvlabs.github.io/SPADE/) |   [Paper](https://arxiv.org/abs/1903.07291) | [Online Interactive Demo of GauGAN](https://www.nvidia.com/en-us/research/ai-playground/) | [GTC 2019 demo](https://youtu.be/p5U4NgVGAwg) | [Youtube Demo of GauGAN](https://youtu.be/MXWm6w4E5q0)

Semantic Image Synthesis with Spatially-Adaptive Normalization.<br>
[Taesung Park](http://taesung.me/),  [Ming-Yu Liu](http://mingyuliu.net/), [Ting-Chun Wang](https://tcwang0509.github.io/),  and [Jun-Yan Zhu](http://people.csail.mit.edu/junyanz/).<br>
In CVPR 2019 (Oral).

### [License](https://raw.githubusercontent.com/nvlabs/SPADE/master/LICENSE.md)

Copyright (C) 2019 NVIDIA Corporation.

All rights reserved.
Licensed under the [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode) (**Attribution-NonCommercial-ShareAlike 4.0 International**)

The code is released for academic research use only. For commercial use, please contact [researchinquiries@nvidia.com](researchinquiries@nvidia.com).

## Installation

Clone this repo.
```bash
git clone https://github.com/NVlabs/SPADE.git
cd SPADE/
```

This code requires PyTorch 1.0 and python 3+. Please install dependencies by
```bash
pip install -r requirements.txt
```

This code also requires the Synchronized-BatchNorm-PyTorch rep.
```
cd models/networks/
git clone https://github.com/vacancy/Synchronized-BatchNorm-PyTorch
cp -rf Synchronized-BatchNorm-PyTorch/sync_batchnorm .
cd ../../
```



## Dataset Preparation

1. You need access to the OpenEDS Dataset as described [here](https://research.fb.com/programs/openeds-challenge).

2. Unzip all folders and set the `base_path` to the root folder containing the unpacked subfolders. This folder should also contain the json files with the mappings of file to users (_OpenEDS_{train,validation,test}_userID_mapping_to_images.json_).
Run
```data/prepare_openeds.py```

## Training New Models

TODO

## Testing


## Code Structure

- `train.py`, `test.py`: the entry point for training and testing.
- `trainers/pix2pix_trainer.py`: harnesses and reports the progress of training.
- `models/pix2pix_model.py`: creates the networks, and compute the losses
- `models/networks/`: defines the architecture of all models
- `options/`: creates option lists using `argparse` package. More individuals are dynamically added in other files as well. Please see the section below.
- `data/`: defines the class for loading and processing data.

## Options

This code repo contains many options. Some options belong to only one specific model, and some options have different default values depending on other options. To address this, the `BaseOption` class dynamically loads and sets options depending on what model, network, and datasets are used. This is done by calling the static method `modify_commandline_options` of various classes. It takes in the`parser` of `argparse` package and modifies the list of options. For example, since COCO-stuff dataset contains a special label "unknown", when COCO-stuff dataset is used, it sets `--contain_dontcare_label` automatically at `data/coco_dataset.py`. You can take a look at `def gather_options()` of `options/base_options.py`, or `models/network/__init__.py` to get a sense of how this works.

## VAE-Style Training with an Encoder For Style Control and Multi-Modal Outputs

To train our model along with an image encoder to enable multi-modal outputs as in Figure 15 of the [paper](https://arxiv.org/pdf/1903.07291.pdf), please use `--use_vae`. The model will create `netE` in addition to `netG` and `netD` and train with KL-Divergence loss.

### Citation
If you use this code for your research, please cite our papers.
```
@inproceedings{park2019SPADE,
  title={Semantic Image Synthesis with Spatially-Adaptive Normalization},
  author={Park, Taesung and Liu, Ming-Yu and Wang, Ting-Chun and Zhu, Jun-Yan},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year={2019}
}
```

## Acknowledgments
This repository is a fork of the original [SPADE implementation](https://github.com/NVlabs/SPADE), which borrows heavily from pix2pixHD. We thank Jiayuan Mao for his Synchronized Batch Normalization code.



Please note:

* This implementation currently does not support multi-GPU training.
