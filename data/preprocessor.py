"""
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import cv2
import numpy as np

import cv2 as cv
import torch
from PIL import Image
from torchvision import transforms


class ImagePreprocessor(object):
    vgg_transform = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

    @staticmethod
    def equalize(image):  # Proper colour image intensity equalization
        image = image.astype('uint8')
        if len(image.shape) == 2:
            # We have a b/w image
            output = cv.equalizeHist(image)
        else:
            ycrcb = cv.cvtColor(image, cv.COLOR_RGB2YCrCb)
            ycrcb[:, :, 0] = cv.equalizeHist(ycrcb[:, :, 0])
            output = cv.cvtColor(ycrcb, cv.COLOR_YCrCb2RGB)
        return output

    @staticmethod
    def bgr2rbg(image):
        # without copy there is an error because of negative strides
        return np.array(image[:, :, ::-1])

    @classmethod
    def rgb2bgr(cls, image):
        # Calling this method twice inverts its effect.
        return cls.bgr2rbg(image)

    @staticmethod
    def normalize(image):
        image = image.astype(np.float32)
        image = image * 2 / 255.0
        image = image - 1
        return image

    @staticmethod
    def unnormalize(image):
        image = image + 1
        image = image * 255 / 2
        if isinstance(image, torch.Tensor):
            return image
        else:
            return image.astype(np.uint16)

    @classmethod
    def unnormalize_tensor(cls, tensor):
        if len(tensor.shape) == 3:
            out = torch.add(tensor, 1)
            out = torch.div(torch.mul(out, 255), 2)
            return out
        else:
            return torch.stack([cls.unnormalize_tensor(t) for t in tensor], dim=0)

    @classmethod
    def toRange01(cls, tensor):
        # We expect tensor to be in range [-1, 1]
        if not (torch.min(tensor) >= -1 and torch.max(tensor) <= 1):
            message = "Tensor not in range [-1, 1]. Min: {}, max: {}".format(torch.min(tensor), torch.max(tensor))
            print(message)
        #     raise ValueError(message)
        out = torch.div(torch.add(tensor, 1), 2)
        # assert torch.min(out) >= 0 and torch.max(out) <= 1
        return out

    @classmethod
    def vgg_normalize(cls, tensor):
        vgg_normalized = torch.stack([cls.vgg_transform(t) for t in tensor])
        return vgg_normalized

    @classmethod
    def hwc2chw(cls, image):
        if len(image.shape) == 3:
            return image.transpose((2, 0, 1))
        return np.stack([cls.hwc2chw(img) for img in image])


    @classmethod
    def chw2hwc_tensor(cls, image):
        if len(image.shape) == 4:
            return image.permute(0, 2, 3, 1)
        else:
            return image.permute(1, 2, 0)

    @classmethod
    def chw2hwc(cls, image):
        if isinstance(image, torch.Tensor):
            return cls.chw2hwc_tensor(image)
        if len(image.shape) == 3:
                return image.transpose((1, 2, 0))
        return np.stack([cls.chw2hwc(img) for img in image])

    @classmethod
    def resize(cls, img, w, h, method=Image.BICUBIC):
        if isinstance(img, torch.Tensor):
            img_np = np.copy(img.cpu().detach())
            return cls.resize(img_np, w, h, method)
        if isinstance(img, np.ndarray):
            return cv2.resize(img, (w, h), interpolation=method)
        return img.resize((w, h), method)

    @staticmethod
    def rescale(image, w, h):
        image = cv.resize(image, dsize=(w, h), interpolation=cv.INTER_CUBIC)
        return image

    @staticmethod
    def gray2rgb(image):
        """
        Expands the dimensions of a gray-scale image such that it has three
            dimensions.
        Args:
            image: a single image

        Returns: image with three channels
        """
        image = np.expand_dims(image, axis=2)
        return np.repeat(image, 3, 2)

    @classmethod
    def rgb2gray(cls, image):
        """
        Converts an RGB image to gray-scale
        Args:
            image: a single image

        Returns: gray-scale image (single channel)
        """
        image = np.mean(image, axis=2)
        return cls.gray2rgb(image)

    @classmethod
    def preprocess(cls, image, bgr2rbg=False, width=60, height=36):
        # if isinstance(image.dtype, tuple):
        #     print(image)
        #     exit()
        if bgr2rbg:
            image = cls.bgr2rbg(image)
        if image.shape != (height, width):
            image = cls.rescale(image, width, height)
        # We need to equalize before normalizing
        # TODO: find out whether this makes sense
        image = cls.equalize(image)
        image = cls.normalize(image)
        image = cls.hwc2chw(image)
        return image.astype(np.float32)


class Preprocessor:
    """
    Base class for preprocessors
    """

    def __init__(self,
                 do_augmentation,
                 eye_image_shape=(72, 120),
                 difficulty=1,
                 kappa_augment_labels=False):
        self.do_augmentation = do_augmentation
        self._eye_image_shape = eye_image_shape

        # Define bounds for noise values for different augmentation types
        self._difficulty = difficulty
        self._augmentation_ranges = {  # (easy, hard)
            'translation': (2.0, 10.0),
            'intensity': (0.5, 20.0),
            'blur': (0.1, 1.0),
            'scale': (0.01, 0.1),
            'rescale': (1.0, 0.2),
        }
        self.kappa_augment_labels = kappa_augment_labels
        # difference between visual and optical axis in degrees
        # (diff_pitch, diff_yaw)
        # positive pitch: looking further up
        # positive yaw: looking further left
        self.kappa_tuning = (2, 5)

    @staticmethod
    def bgr2rgb(image):
        # BGR to RGB conversion
        image = image[..., ::-1]
        return image

    @staticmethod
    def equalize(image):  # Proper colour image intensity equalization
        if len(image.shape) == 2:
            # We have a b/w image
            output = cv.equalizeHist(image)
        else:
            ycrcb = cv.cvtColor(image, cv.COLOR_RGB2YCrCb)
            ycrcb[:, :, 0] = cv.equalizeHist(ycrcb[:, :, 0])
            output = cv.cvtColor(ycrcb, cv.COLOR_YCrCb2RGB)
        return output

    def _headpose_to_radians(self, json_data):
        h_pitch, h_yaw, _ = eval(json_data['head_pose'])
        if h_pitch > 180.0:  # Need to correct pitch
            h_pitch -= 360.0
        h_yaw -= 180.0  # Need to correct yaw

        h_pitch = -h_pitch
        h_yaw = -h_yaw
        return np.asarray([np.radians(h_pitch), np.radians(h_yaw)],
                          dtype=np.float32)

    def _rescale(self, eye, ow, oh):
        # Rescale image if required
        rescale_max = self._value_from_type('rescale')
        if rescale_max < 1.0:
            rescale_noise = np.random.uniform(low=rescale_max, high=1.0)
            interpolation = cv.INTER_CUBIC
            eye = cv.resize(eye, dsize=(0, 0), fx=rescale_noise,
                            fy=rescale_noise,
                            interpolation=interpolation)

            eye = self.equalize(eye)
            eye = cv.resize(eye, dsize=(oh, ow), interpolation=interpolation)
        return eye

    def _rgb_noise(self, eye):
        # Add rgb noise to eye image
        intensity_noise = int(self._value_from_type('intensity'))
        if intensity_noise > 0:
            eye = eye.astype(np.int16)
            eye += np.random.randint(low=-intensity_noise,
                                     high=intensity_noise,
                                     size=eye.shape, dtype=np.int16)
            cv.normalize(eye, eye, alpha=0, beta=255, norm_type=cv.NORM_MINMAX)
            eye = eye.astype(np.uint8)
        return eye

    def _blur(self, eye):
        # Add blur to eye image
        blur_noise = self._noisy_value_from_type('blur')
        if blur_noise > 0:
            eye = cv.GaussianBlur(eye, (7, 7), 0.5 + np.abs(blur_noise))
        return eye

    def augment(self, eye):
        oh, ow = self._eye_image_shape

        eye = self._rescale(eye, oh, ow)
        eye = self._rgb_noise(eye)
        eye = self._blur(eye)

        return eye

    def _value_from_type(self, augmentation_type):
        # Scale to be in range
        easy_value, hard_value = self._augmentation_ranges[augmentation_type]
        value = (hard_value - easy_value) * self._difficulty + easy_value
        value = (np.clip(value, easy_value, hard_value)
                 if easy_value < hard_value
                 else np.clip(value, hard_value, easy_value))
        return value

    def _noisy_value_from_type(self, augmentation_type):
        random_multipliers = []
        # Get normal distributed random value
        if len(random_multipliers) == 0:
            random_multipliers.extend(
                list(np.random.normal(size=(len(self._augmentation_ranges),))))
        return random_multipliers.pop() * self._value_from_type(
            augmentation_type)
