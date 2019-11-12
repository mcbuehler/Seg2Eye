## Setup
Please use python 3.5 or above.

1) Install requirements

```pip install -r requirements.txt```

2) Download the required files and the dataset

```sh downloads.sh```

3) Add the directory to PYTHONPATH:

```export PYTHONPATH=$PYTHONPATH:$(pwd)```

## Inference
There are two steps for running inference on our model. The first step is already
pre-computed and the corresponding files can be downloaded (please see "Setup" above).

1) We identify the most similar images in the unlabeled test set for each test segmentation mask:
A segmentation network predicts the semantic eye regions for the unlabeled test set images
and we rank these images by their L2 distance to the target segmentation mask.
This ranking is already pre-generated and stored in h5 files.

2) For each target segmentation mask in the test set, we then select this "most similar eye image" 
(one could consider this a nearest neighbour), and refine it via a fully convolutional architecture 
in order to minimize the final root mean squared error based objective.


### Running inference
In order to run inference and produce a npy file for each test segmentation mask,
please run the following command:

```python evaluate_refinenet.py```

This will produce the submission files in a sub-folder of `res/refinenet`.



