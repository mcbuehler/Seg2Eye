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
        tester.run(model, mode='full', epoch=epoch, n_steps=n_steps, write_error_log=True)
    else:
        tester.run_test(model)

