import glob
import json
import os
import sys

import logging
logger = logging.getLogger(__name__)


class DefaultConfig(object):

    # Training
    fully_reproducible = False  # enable with possible penalty of performance
    use_apex = True

    batch_size = 32
    weight_decay = 0.0001
    num_epochs = 50
    num_warmup_epochs = 5  # No. of epochs to warmup LR from base to target

    train_data_workers = 8

    log_every_n_steps = 20  # NOTE: Every other interval has to be a multiple of this!!!
    tensorboard_scalars_every_n_steps = 20
    tensorboard_images_every_n_steps = 200
    tensorboard_learning_rate_every_n_steps = 200

    # Learning rate
    base_learning_rate = 0.0004
    @property
    def learning_rate(self):
        return self.batch_size * self.base_learning_rate
    # Available strategies:
    #     'exponential': step function with exponential decay
    #     'cyclic':      spiky down-up-downs (with exponential decay of peaks)
    lr_decay_strategy = 'exponential'
    lr_decay_factor = 0.5
    lr_decay_epoch_interval = 5
    gradient_norm_clip = 0.0

    # Evaluation
    test_num_samples = 10000
    test_batch_size = 64
    test_data_workers = 4
    test_every_n_steps = 1000

    # Model configuration (MoE)
    moe_ensembled_predictions = False
    track_running_stats = True

    moe_classify_attributes = False

    pretraining_epochs = 0
    pretraining_batch_size = 1024
    pretraining_learning_rate = 0.005

    eyes_densenet_growth_rate = 8
    eyes_densenet_block_configuration = [5, 5, 5, 5]

    face_densenet_growth_rate = 8
    face_densenet_block_configuration = [5, 5, 5, 5]

    num_experts = 40
    experts_densenet_growth_rate = 8
    experts_densenet_block_configuration = [5, 5, 5, 5]

    # Checkpoints management
    checkpoints_keep_n = 3
    resume_from = ''

    # Google Sheets related
    gsheet_secrets_json_file = ''
    gsheet_workbook_key = ''

    # Below lie necessary methods for working configuration tracking

    __instance = None

    # Make this a singleton class
    def __new__(cls):
        if cls.__instance is None:
            cls.__instance = super().__new__(cls)
            cls.__filecontents = cls.__get_config_file_contents()
            cls.__pycontents = cls.__get_python_file_contents()
            cls.__immutable = True
        return cls.__instance

    def import_json(self, json_path):
        """Import JSON config to over-write existing config entries."""
        assert os.path.isfile(json_path)
        assert not hasattr(self.__class__, '__imported_json_path')
        logger.info('Loading ' + json_path)
        with open(json_path, 'r') as f:
            json_string = f.read()
        self.import_dict(json.loads(json_string))
        self.__class__.__imported_json_path = json_path
        self.__class__.__filecontents[os.path.basename(json_path)] = json_string

    def import_dict(self, dictionary):
        """Import a set of key-value pairs from a dict to over-write existing config entries."""
        self.__class__.__immutable = False
        for key, value in dictionary.items():
            if not hasattr(self, key):
                raise ValueError('Unknown configuration key: ' + key)
            assert type(getattr(self, key)) is type(value)
            setattr(self, key, value)
        self.__class__.__immutable = True

    def __get_config_file_contents():
        """Retrieve and cache default and user config file contents."""
        out = {}
        for relpath in ['config_default.py']:
            path = os.path.relpath(os.path.dirname(__file__) + '/' + relpath)
            assert os.path.isfile(path)
            with open(path, 'r') as f:
                out[os.path.basename(path)] = f.read()
        return out

    def __get_python_file_contents():
        """Retrieve and cache default and user config file contents."""
        out = {}
        base_path = os.path.relpath(os.path.dirname(__file__) + '/../')
        source_fpaths = [
            p for p in glob.glob(base_path + '/**/*.py')
            if not p.startswith('./3rdparty/')
        ]
        source_fpaths += [os.path.relpath(sys.argv[0])]
        for fpath in source_fpaths:
            assert os.path.isfile(fpath)
            with open(fpath, 'r') as f:
                out[fpath[2:]] = f.read()
        return out

    def get_all_key_values(self):
        return dict([
            (key, getattr(self, key))
            for key in dir(self)
            if not key.startswith('_DefaultConfig')
            and not key.startswith('__')
            and not callable(getattr(self, key))
        ])

    def get_full_json(self):
        return json.dumps(self.get_all_key_values())

    def write_file_contents(self, target_base_dir):
        """Write cached config file contents to target directory."""
        assert os.path.isdir(target_base_dir)

        # Write config file contents
        target_dir = target_base_dir + '/configs'
        if not os.path.isdir(target_dir):
            os.makedirs(target_dir)
        outputs = {  # Also output flattened config
            'combined.json': self.get_full_json(),
        }
        outputs.update(self.__class__.__filecontents)
        for fname, content in outputs.items():
            fpath = os.path.relpath(target_dir + '/' + fname)
            with open(fpath, 'w') as f:
                f.write(content)
                logger.info('Written %s' % fpath)

        # Write Python file contents
        target_dir = target_base_dir + '/src'
        for fname, content in self.__pycontents.items():
            fpath = os.path.relpath(target_dir + '/' + fname)
            dpath = os.path.dirname(fpath)
            if not os.path.isdir(dpath):
                os.makedirs(dpath)
            with open(fpath, 'w') as f:
                f.write(content)
        logger.info('Written %d source files to %s' %
                    (len(self.__pycontents), os.path.relpath(target_dir)))

    def __setattr__(self, name, value):
        """Initial configs should not be overwritten!"""
        if self.__class__.__immutable:
            raise AttributeError('DefaultConfig instance attributes are immutable.')
        else:
            super().__setattr__(name, value)

    def __delattr__(self, name):
        """Initial configs should not be removed!"""
        if self.__class__.__immutable:
            raise AttributeError('DefaultConfig instance attributes are immutable.')
        else:
            super().__delattr__(name)
