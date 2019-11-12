import glob
import logging
import os

import torch

from core import DefaultConfig

config = DefaultConfig()
logger = logging.getLogger(__name__)


class CheckpointManager(object):

    __model = None
    __suffix = '.pt'

    def __init__(self, model):
        self.__model = model

    def __save(self, instance, ofpath):
        assert not os.path.isfile(ofpath)
        if hasattr(instance, 'module'):  # case where nn.DataParallel was used
            state_dict = instance.module.state_dict()
        else:
            state_dict = instance.state_dict()
        ofdir = os.path.dirname(ofpath)  # create folder if not exists
        if not os.path.isdir(ofdir):
            os.makedirs(ofdir)
        torch.save(state_dict, ofpath)
        logger.info('> Saved parameters to: %s' % ofpath)

    def __load(self, instance, ifpath):
        assert os.path.isfile(ifpath)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        instance.load_state_dict(torch.load(ifpath, map_location=device))
        logger.info('> Loaded parameters from: %s' % ifpath)

        step = int(os.path.split(ifpath)[-1][:-3])
        return step

    def __output_dir(self, instance):
        return os.path.relpath(os.path.join(
            instance.output_dir,
            'checkpoints',
        ))

    def __output_fpath(self, instance, current_step):
        return os.path.relpath(os.path.join(
            self.__output_dir(instance),
            ('%07d' % current_step) + self.__suffix,
        ))

    def save_at_step(self, current_step):
        self.__save(self.__model, self.__output_fpath(self.__model, current_step))
        self.__only_keep_n_checkpoints(self.__model)

    def __get_available_checkpoints(self, instance):
        output_dir = self.__output_dir(instance)
        return sorted([
            (int(os.path.split(fn)[-1].split('.')[0]), fn)
            for fn in glob.glob(os.path.join(output_dir, '*' + self.__suffix))
            if fn.endswith(self.__suffix) and os.path.isfile(fn)
        ])

    def __only_keep_n_checkpoints(self, instance):
        # TODO: Log validation loss and keep only n best
        available = self.__get_available_checkpoints(instance)
        if len(available) > config.checkpoints_keep_n:
            for step, fpath in available[:-config.checkpoints_keep_n]:
                os.remove(fpath)
                logger.info('> Removing parameters file at: %s' % fpath)

    def load_last_checkpoint(self):
        return self.__load_last_checkpoint(self.__model)

    def __load_last_checkpoint(self, instance):
        available = self.__get_available_checkpoints(instance)
        if len(available) > 0:
            return self.__load(self.__model, available[-1][1])
