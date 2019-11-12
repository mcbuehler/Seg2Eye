import argparse
from collections import OrderedDict
import functools
import gc
import hashlib
import json
import logging
import os
import sys
import time

from apex import amp
import cv2 as cv
import coloredlogs
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import torchvision.utils as vutils

from core import DefaultConfig, CheckpointManager, GoogleSheetLogger, Tensorboard
# from core.gaze import mean_angular_error

config = DefaultConfig()

# Setup logger
logger = logging.getLogger(__name__)

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def script_init_common():
    parser = argparse.ArgumentParser(description='Train a gaze estimation model.')
    parser.add_argument('-v', type=str, help='Desired logging level.', default='info',
                        choices=['debug', 'info', 'warning', 'error', 'critical'])
    parser.add_argument('config_json', type=str, nargs='*',
                        help=('Path to config in JSON format. '
                              'Multiple configs will be parsed in the specified order.'))
    for key in dir(config):
        if key.startswith('_DefaultConfig') or key.startswith('__'):
            continue
        if key in vars(DefaultConfig) and isinstance(vars(DefaultConfig)[key], property):
            continue
        value = getattr(config, key)
        value_type = type(value)
        if callable(value):
            continue
        parser.add_argument('--' + key.replace('_', '-'), type=value_type, metavar=value,
                            help='Expected type is `%s`.' % value_type.__name__)
    args = parser.parse_args()

    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Set logger format and verbosity level
    coloredlogs.install(
        datefmt='%d/%m %H:%M',
        fmt='%(asctime)s %(levelname)s %(message)s',
        level=args.v.upper(),
    )

    # Parse configs in order specified by user
    for json_path in args.config_json:
        config.import_json(json_path)

    # Apply configs passed through command line
    config.import_dict(dict([
        (key.replace('-', '_'), type(getattr(config, key))(value))
        for key, value in vars(args).items()
        if value is not None and hasattr(config, key)
    ]))

    # Improve reproducibility
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    if config.fully_reproducible:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    np.random.seed(0)

    # Load dataset splits
    with open('gazecapture_split.json', 'r') as f:
        gc_split = json.load(f)
    with open('celeba_split.json', 'r') as f:
        celeba_split = json.load(f)
    with open('eyediap_split.json', 'r') as f:
        eyediap_split = json.load(f)

    # Collect all splits
    all_splits = {}
    for prefix, split in [('gc', gc_split), ('celeba', celeba_split),
                          ('eyediap', eyediap_split)]:
        for k, v in split.items():
            all_splits[prefix + '/' + k] = v

    return config, device, all_splits


def init_datasets(train_specs, test_specs):

    # Initialize training datasets
    train_data = OrderedDict()
    for tag, dataset_class, path, keys in train_specs:
        dataset = dataset_class(path, keys, augment=True)
        dataset.original_full_dataset = dataset
        dataloader = DataLoader(dataset,
                                batch_size=config.batch_size,
                                shuffle=True,
                                drop_last=True,
                                num_workers=config.train_data_workers,
                                pin_memory=True,
                                )
        train_data[tag] = {
            'dataset': dataset,
            'dataloader': dataloader,
        }
        logger.info('> Ready to use training dataset: %s' % tag)
        logger.info('         with number of entries: %d' % len(dataset))

    # Initialize test datasets
    test_data = OrderedDict()
    for i, row in enumerate(test_specs):
        if len(row) == 4:
            test_specs[i] = tuple(list(row) + [{}])
    for tag, dataset_class, path, keys, kwargs in test_specs:
        # Get the full dataset
        dataset = dataset_class(path, keys, **kwargs)
        dataset.original_full_dataset = dataset
        # then subsample datasets for quicker testing
        num_subset = config.test_num_samples
        if len(dataset) > num_subset:
            subset = Subset(dataset, sorted(np.random.permutation(len(dataset))[:num_subset]))
            subset.original_full_dataset = dataset
            dataset = subset
        dataloader = DataLoader(dataset,
                                batch_size=config.test_batch_size,
                                shuffle=True,
                                num_workers=config.test_data_workers,
                                pin_memory=True,
                                )
        test_data[tag] = {
            'dataset': dataset,
            'dataloader': dataloader,
        }
        logger.info('> Ready to use evaluation dataset: %s' % tag)
        logger.info('           with number of entries: %d' % len(dataset.original_full_dataset))
        logger.info('     of which we evaluate on just: %d' % len(dataset))

    return train_data, test_data


def setup_common(model, optimizers):
    identifier = (model.__class__.__name__ + '/' +
                  time.strftime('%y%m%d_%H%M%S') + '.' +
                  hashlib.md5(config.get_full_json().encode('utf-8')).hexdigest()[:6]
                  )

    if len(config.resume_from) > 0:
        identifier = config.resume_from.split('/')[-1]
        output_dir = config.resume_from

    else:
        output_dir = '../outputs/' + identifier

    # Initialize tensorboard
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    tensorboard = Tensorboard(output_dir)

    # Write source code to output dir
    config.write_file_contents(output_dir)

    # Log messages to file
    root_logger = logging.getLogger()
    file_handler = logging.FileHandler(output_dir + '/messages.log')
    file_handler.setFormatter(root_logger.handlers[0].formatter)
    for handler in root_logger.handlers[1:]:  # all except stdout
        root_logger.removeHandler(handler)
    root_logger.addHandler(file_handler)

    # Print model details
    num_params = sum([
        np.prod(p.size())
        for p in filter(lambda p: p.requires_grad, model.parameters())
    ])
    logger.info('\nThere are %d trainable parameters.\n' % num_params)

    # Wrap optimizer with AMP if required
    if config.use_apex:
        model, optimizers = amp.initialize(
            model, optimizers,
            opt_level='O1',
            num_losses=len(optimizers),
            # loss_scale=1.0
        )
        amp._amp_state.loss_scalers[0]._loss_scale = 1.0

    # Cache base and target learning rate for each optimizer
    for optimizer in optimizers:
        optimizer.target_lr = optimizer.param_groups[0]['lr']
        optimizer.base_lr = optimizer.target_lr / config.batch_size

    # Sneak in some extra information into the model class instance
    model.identifier = identifier
    model.output_dir = output_dir
    model.checkpoint_manager = CheckpointManager(model)
    model.gsheet_logger = GoogleSheetLogger(model)
    model.last_epoch = 0.0
    model.last_step = 0

    # Load pre-trained model weights if available
    if len(config.resume_from) > 0:
        model.last_step = model.checkpoint_manager.load_last_checkpoint()

    return model, optimizers, tensorboard


def salvage_memory():
    """Try to free whatever memory that can be freed."""
    torch.cuda.empty_cache()
    gc.collect()


def get_training_batches(train_data_dicts):
    """Get training batches of data from all training data sources."""
    out = {}
    for tag, data_dict in train_data_dicts.items():
        if 'data_iterator' not in data_dict:
            data_dict['data_iterator'] = iter(data_dict['dataloader'])
        # Try to get data
        while True:
            try:
                out[tag] = next(data_dict['data_iterator'])
                break
            except StopIteration:
                del data_dict['data_iterator']
                salvage_memory()
                data_dict['data_iterator'] = iter(data_dict['dataloader'])

        # Move tensors to GPU
        for k, v in out[tag].items():
            if isinstance(v, torch.Tensor):
                out[tag][k] = v.detach().to(device, dtype=torch.float32, non_blocking=True)
    return out


def test_model_on_all(model, test_data_dicts, current_step, tensorboard=None,
                      log_key_prefix='test'):
    """Get training batches of data from all training data sources."""
    model.eval()
    salvage_memory()
    final_out = {}
    for tag, data_dict in test_data_dicts.items():
        with torch.no_grad():
            num_entries = len(data_dict['dataset'])
            # preds = []
            # trues = []
            for i, input_data in enumerate(data_dict['dataloader']):
                batch_size = next(iter(input_data.values())).shape[0]

                # Move tensors to GPU
                for k, v in input_data.items():
                    if isinstance(v, torch.Tensor):
                        input_data[k] = v.detach().to(device, dtype=torch.float32,
                                                      non_blocking=True)

                # Inference
                batch_out = model(input_data)
                weighted_batch_out = dict([
                    (k, v.detach().cpu().numpy() * (batch_size / num_entries))
                    for k, v in batch_out.items()
                    if isinstance(v, torch.Tensor) and v.dim() == 0
                ])
                if tag not in final_out:
                    final_out[tag] = dict([(k, 0.0) for k in weighted_batch_out.keys()])
                for k, v in weighted_batch_out.items():
                    final_out[tag][k] += v

                # # Cache gaze values (predictions and ground-truth)
                # if 'gaze_direction' in input_data:
                #     trues += list(input_data['gaze_direction'].cpu().numpy())
                #     preds += list(batch_out['gaze_prediction'].detach().cpu().numpy())

                """
                if i == 0:  # For Semantic Segmentation
                    assert tensorboard
                    num_images = 4
                    all_images = [
                        (255. / 2. * (batch_out['input'][:num_images, 0, :].cpu().numpy() + 1.0)).astype(np.uint8),  # noqa
                        (255. / 3. * batch_out['prediction'][:num_images, :].cpu().numpy()).astype(np.uint8),  # noqa
                    ]

                    if 'groundtruth' in batch_out:
                        all_images.append((255. / 3. * batch_out['groundtruth'][:num_images, :].cpu().numpy()).astype(np.uint8))  # noqa

                    for i in range(1, num_images + 1):
                        tensorboard.add_image(
                            log_key_prefix + '_%s/predictions/%d' % (tag, i),
                            np.expand_dims(cv.hconcat([
                                cv.resize(imgs[i - 1, :], (400, 640))
                                for imgs in all_images
                            ]), 0)
                        )
                """

                if i == 0:  # For RefineNet
                    num_images = 4
                    def convert_to_uint_rgb(tensor, shift=True):  # noqa
                        cpu_tensor = tensor.detach().cpu().numpy()
                        tensor_shifted = cpu_tensor + 1.0 if shift else cpu_tensor
                        tensor_scaled = 255. / 2. * tensor_shifted
                        tensor_clipped = np.clip(tensor_scaled, 0.0, 255.)
                        return tensor_clipped.astype(np.uint8)
                    all_images = [
                        convert_to_uint_rgb(batch_out['input'][:num_images, 0, :]),
                        convert_to_uint_rgb(batch_out['input'][:num_images, 1, :]),
                        convert_to_uint_rgb(batch_out['input'][:num_images, 2, :]),
                        convert_to_uint_rgb(batch_out['residual'][:num_images, 0, :]),
                        convert_to_uint_rgb(batch_out['prediction'][:num_images, 0, :]),
                    ]

                    if 'groundtruth' in batch_out:
                        all_images.append(
                            convert_to_uint_rgb(batch_out['groundtruth'][:num_images, 0, :]))
                        all_images.append(np.abs(
                            all_images[-2].astype(np.float32) -
                            all_images[-1].astype(np.float32)).astype(np.uint8))

                    for i in range(1, num_images + 1):
                        images_concatenated = cv.hconcat([
                            cv.resize(imgs[i - 1, :], (400, 640)) for imgs in all_images
                        ])
                        foot = np.zeros((130, images_concatenated.shape[1]), dtype=np.uint8)
                        text_to_put = '%s / %s' % (batch_out['person_id'][i], batch_out['fname'][i])
                        if 'per_image_score' in batch_out:
                            text_to_put += ' (err: %.2f)' % (1471 * batch_out['per_image_score'][i])
                        cv.putText(foot, text_to_put, (20, 88), cv.FONT_HERSHEY_DUPLEX, fontScale=3,
                                   color=(255, 255, 255), thickness=3, lineType=cv.LINE_AA)
                        tensorboard.add_image(
                            log_key_prefix + '_%s/predictions/%d' % (tag, i),
                            np.expand_dims(cv.vconcat([images_concatenated, foot]), 0),
                        )

        # # In particular for MoE, the final prediction error is going to be different
        # # to just getting the average of weighted-sum-of-gaze-errors
        # if len(trues) > 0 and len(preds) > 0:
        #     final_out[tag]['gaze_ang_from_final_predictions'] = \
        #             mean_angular_error(np.array(trues), np.array(preds))

        # Calculate mean error over whole dataset
        logger.info('%10s test: %s' % ('[%s]' % tag,
                                       ', '.join(['%s: %.4g' % (k, final_out[tag][k])
                                                  for k in sorted(final_out[tag].keys())])))

    # Write to tensorboard
    if tensorboard:
        tensorboard.update_current_step(current_step)
        for tag, out in final_out.items():
            for k, v in out.items():
                tensorboard.add_scalar(log_key_prefix + '_%s/%s' % (tag, k), v)

    # Log training metrics to Google Sheets
    for_gsheet = None
    if model.gsheet_logger.ready:
        for_gsheet = {}
        for tag, out in final_out.items():
            for k, v in out.items():
                for_gsheet[log_key_prefix + '/%s/%s' % (tag, k)] = v

    # Free up memory
    salvage_memory()

    return final_out, for_gsheet


def log_images_with_uncertainties(model, train_data_dicts, test_data_dicts, current_step,
                                  tensorboard=None):
    model.eval()
    for tag, data_dict in (list(train_data_dicts.items()) + list(test_data_dicts.items())):
        with torch.no_grad():
            batch_size = 49
            dataloader = DataLoader(data_dict['dataset'].original_full_dataset,
                                    batch_size=batch_size,
                                    shuffle=True,
                                    num_workers=config.test_data_workers,
                                    )
            # Move tensors to GPU
            input_data = next(iter(dataloader))
            input_data_cuda = {}
            for k, v in input_data.items():
                input_data_cuda[k] = v.detach().to(device, dtype=torch.float32, non_blocking=True)

            # Inference
            batch_out = model(input_data_cuda)
            for k, v in batch_out.items():
                batch_out[k] = v.detach().cpu().numpy()

            all_data = sorted(list(zip(
                list(batch_out['aleatoric_uncertainty']),
                list(batch_out['gaze_ang']),
                list(input_data['eye']),
            )))

            # Annotate with errors and uncertainties
            annotated = []
            for std, err, img in all_data:
                img = (img.numpy() + 1.0) * (255. / 2.)
                img = img.astype(np.uint8).transpose(1, 2, 0)
                img = cv.resize(img, (120, 72))
                img = np.pad(img, pad_width=((0, 50), (0, 0), (0, 0)), mode='constant',
                             constant_values=255)
                cv.putText(img, 'mean: %.3f' % err, org=(2, 92),
                           fontFace=cv.FONT_HERSHEY_DUPLEX, fontScale=0.5, color=(0, 0, 0),
                           lineType=cv.LINE_AA)
                cv.putText(img, 'std: %.3f' % std, org=(22, 112),
                           fontFace=cv.FONT_HERSHEY_DUPLEX, fontScale=0.5, color=(0, 0, 0),
                           lineType=cv.LINE_AA)
                annotated.append(torch.from_numpy(img.transpose(2, 0, 1)))

            # Write
            grid = vutils.make_grid(annotated, nrow=int(np.sqrt(batch_size)), padding=10,
                                    pad_value=255.0)
            tensorboard.add_image('%s/uncertainties' % tag, grid)

            # Cleanup
            del dataloader, input_data, batch_out
            salvage_memory()


def do_final_full_test(model, test_data, tensorboard):
    for k, v in test_data.items():
        # Get the full dataset
        if 'dataloader' in test_data[k]:
            del v['dataloader']
        test_data[k]['dataloader'] = DataLoader(v['dataset'].original_full_dataset,
                                                batch_size=config.test_batch_size,
                                                shuffle=False,
                                                num_workers=config.test_data_workers,
                                                pin_memory=True,
                                                )
        logger.info('> Ready to do full test on dataset: %s' % k)

    logger.info('# Now beginning full test on all evaluation sets.')
    logger.info('# Hold on tight, this might take a while.')
    logger.info('#')
    _, for_gsheet = test_model_on_all(model, test_data, model.last_step + 2,
                                      tensorboard=tensorboard,
                                      log_key_prefix='full_test')

    # Clean up dataloaders
    for k, v in test_data.items():
        del v['dataloader']

    # Log training metrics to Google Sheets
    if for_gsheet is not None:
        model.gsheet_logger.update_or_append_row(for_gsheet)

    # Free memory
    salvage_memory()


def learning_rate_schedule(optimizer, epoch_len, tensorboard_log_func, step):
    num_warmup_steps = int(epoch_len * config.num_warmup_epochs)
    selected_lr = optimizer.target_lr
    if step < num_warmup_steps:
        b = optimizer.base_lr
        a = (optimizer.target_lr - b) / float(num_warmup_steps)
        selected_lr = a * step + b
    else:
        # Decay learning rate with step function and exponential decrease?
        new_step = step - num_warmup_steps
        epoch = new_step / float(epoch_len)
        current_interval = int(epoch / config.lr_decay_epoch_interval)
        if config.lr_decay_strategy == 'exponential':
            # Step function decay
            selected_lr = optimizer.target_lr * np.power(config.lr_decay_factor, current_interval)
        elif config.lr_decay_strategy == 'cyclic':
            # Note, we start from the up state (due to previous warmup stage)
            # so each period consists of down-up (not up-down)
            peak_a = optimizer.target_lr * np.power(config.lr_decay_factor, current_interval)
            peak_b = peak_a * config.lr_decay_factor
            half_interval = 0.5 * config.lr_decay_epoch_interval
            current_interval_start = current_interval * config.lr_decay_epoch_interval
            current_interval_half = current_interval_start + half_interval
            if epoch < current_interval_half:
                # negative slope (down from peak_a)
                slope = -(peak_a - optimizer.base_lr) / half_interval
            else:
                # positive slope (up to peak_b)
                slope = (peak_b - optimizer.base_lr) / half_interval
            selected_lr = slope * (epoch - current_interval_half) + optimizer.base_lr

    # Log to Tensorboard and return
    if step_modulo(step, config.tensorboard_learning_rate_every_n_steps):
        tensorboard_log_func(selected_lr)
    return selected_lr


def step_modulo(current, interval_size):
    return current % interval_size == (interval_size - 1)


def main_loop_iterator(model, optimizers, train_data, test_data, tensorboard=None):
    assert tensorboard is not None  # We assume this exists in LR schedule logging
    initial_step = model.last_step  # Allow resuming
    max_dataset_len = np.amax([len(data_dict['dataset']) for data_dict in train_data.values()])
    num_steps_per_epoch = int(max_dataset_len / config.batch_size)
    num_training_steps = int(config.num_epochs * num_steps_per_epoch)
    lr_schedulers = [
        torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            functools.partial(learning_rate_schedule, optimizer, num_steps_per_epoch,
                              functools.partial(tensorboard.add_scalar, 'lr/optim_%d' % i)),
        ) for i, optimizer in enumerate(optimizers)
    ]
    model.train()
    current_step = 0
    for current_step in range(initial_step, num_training_steps):
        current_epoch = (current_step * config.batch_size) / max_dataset_len  # fractional value
        tensorboard.update_current_step(current_step + 1)
        input_data = get_training_batches(train_data)

        # Set correct states before training iteration
        model.train()
        for optimizer in optimizers:
            optimizer.zero_grad()

        # Forward pass and yield
        loss_terms = []
        images_to_log_to_tensorboard = {}
        outputs = model(input_data)
        yield current_step, loss_terms, outputs, images_to_log_to_tensorboard

        # There should be as many loss terms as there are optimizers!
        assert len(loss_terms) == len(optimizers)

        losses_and_optimizers = [(l, o) for l, o in zip(loss_terms, optimizers) if l is not None]

        # Perform gradient calculations for each loss term
        for i, (loss, optimizer) in enumerate(losses_and_optimizers):
            not_last = i < (len(losses_and_optimizers) - 1)
            if config.use_apex:
                with amp.scale_loss(loss, optimizer, delay_unscale=not_last) as scaled_loss:
                    scaled_loss.backward(retain_graph=not_last)
            else:
                loss.backward(retain_graph=not_last)

        # Clip gradients
        if config.gradient_norm_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_norm_clip)

        # Apply gradients
        for _, optimizer in losses_and_optimizers:
            optimizer.step()

        # Print outputs
        if step_modulo(current_step, config.log_every_n_steps):
            metrics = dict([(k, v.detach().cpu().numpy())
                            for k, v in outputs.items()
                            if isinstance(v, torch.Tensor) and v.dim() == 0])
            for i, loss in enumerate(loss_terms):  # Add loss terms
                if loss is not None:
                    metrics['loss_%d' % (i + 1)] = loss.detach().cpu().numpy()

            log = ('Step %d, Epoch %.2f> ' % (current_step + 1, current_epoch)
                   + ', '.join(['%s: %.4g' % (k, metrics[k]) for k in sorted(metrics.keys())]))
            logger.info(log)

            # Log to Tensorboard
            if step_modulo(current_step, config.tensorboard_scalars_every_n_steps):
                for key, metric in metrics.items():
                    tensorboard.add_scalar('train/%s' % key, metric)

                tensorboard.add_scalar('lr/epoch', current_epoch)

                if step_modulo(current_step, config.tensorboard_images_every_n_steps):
                    for k, img in images_to_log_to_tensorboard.items():
                        tensorboard.add_image(k, img)

                    # # Log images with aleatoric error values
                    # log_images_with_uncertainties(
                    #     model, train_data, test_data, current_step + 1, tensorboard=tensorboard,
                    # )

        # We're done with the previous outputs
        del input_data, outputs, loss_terms, images_to_log_to_tensorboard

        # Full test over all evaluation datasets
        if step_modulo(current_step, config.test_every_n_steps):
            # Do test on subset of validation datasets
            _, for_gsheet = test_model_on_all(model, test_data, current_step + 1,
                                              tensorboard=tensorboard)

            # Log training metrics to Google Sheets
            if for_gsheet is not None:
                for_gsheet['Step'] = current_step + 1
                for_gsheet['Epoch'] = current_epoch
                for k, v in metrics.items():
                    for_gsheet['train/' + k] = v
                model.gsheet_logger.update_or_append_row(for_gsheet)

            # Save checkpoint
            model.checkpoint_manager.save_at_step(current_step + 1)

            # Free memory
            salvage_memory()

        # Remember what the last step/epoch were
        model.last_epoch = current_epoch
        model.last_step = current_step

        # Update learning rate
        # NOTE: should be last
        tensorboard.update_current_step(current_step + 2)
        for lr_scheduler in lr_schedulers:
            lr_scheduler.step(current_step + 1)

    # We're out of the training loop now, make a checkpoint
    current_step += 1
    model.checkpoint_manager.save_at_step(current_step + 1)

    # Close all dataloaders
    for k, v in list(train_data.items()) + list(test_data.items()):
        if 'data_iterator' in v:
            v['data_iterator'].__del__()
            del v['data_iterator']
        v['dataloader']
        del v['dataloader']

    # Clear memory where possible
    salvage_memory()


def cleanup_and_quit(train_data, test_data, tensorboard):
    # Close tensorboard
    if tensorboard:
        tensorboard.__del__()

    # Close all dataloaders and datasets
    for k, v in list(train_data.items()) + list(test_data.items()):
        if 'data_iterator' in v:
            v['data_iterator'].__del__()
        if 'dataset' in v:
            v['dataset'].__del__()
        for item in ['data_iterator', 'dataloader', 'dataset']:
            if item in v:
                del v[item]

    # Finally exit
    sys.exit(0)
