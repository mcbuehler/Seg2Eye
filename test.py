"""
Example usage:
python test.py --dataroot PATH_TO_H5_FILE --name CHECKPOINT_NAME \
    --dataset_key VALIDATION|TEST --load_from_opt_file

"""

import data
from models.pix2pix_model import Pix2PixModel
from options.test_options import TestOptions
from util.tester import Tester

if __name__ == "__main__":
    opt = TestOptions().parse()
    # load the dataset
    dataloader = data.create_dataloader(opt)

    tester = Tester(opt, dataset_key=opt.dataset_key)

    model = Pix2PixModel(opt)
    # model = Pix2PixModel(opt)
    if opt.dataset_key in ['validation', 'train'] and not opt.produce_npy:
        # Iterates through the entire dataset and computes MSE error
        tester.run(model, mode='full', write_error_log=opt.write_error_log)
    else:
        # Produces npy files with predictions
        print("Running inference")
        tester.run_test(model)

