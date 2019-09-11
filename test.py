"""
export name=190910_size256_bs1_lr.0002_l1_0_l2_15_w32___spadeStyleGen_cmadd_ns7_SAMmax_wc0_SSMref_random40
export DK=validation
bsub -n 1 -W 4:00 -o "lsf_"$DK"_""$name" -R "rusage[mem=32048, ngpus_excl_p=1]" -R "select[gpu_mtotal0>=10240]" \
python test.py --dataroot $DR --name $name --dataset_key $DK --batchSize 1 --load_from_opt_file

"""

import data
from models.pix2pix_model import Pix2PixModel
from options.test_options import TestOptions
from util.gsheet import GoogleSheetLogger
from util.tester import Tester

if __name__ == "__main__":

    opt = TestOptions().parse()
    # opt.dataset_key = 'test'
    g_logger = GoogleSheetLogger(opt)
    # load the dataset
    dataloader = data.create_dataloader(opt)

    tester = Tester(opt, g_logger, dataset_key=opt.dataset_key)

    model = Pix2PixModel(opt)
    if opt.dataset_key in ['validation', 'train']:
        epoch = 2
        n_steps = 18000
        tester.run(model, mode='full', epoch=epoch, n_steps=n_steps, write_error_log=opt.write_error_log)
    else:
        tester.run_test(model)

