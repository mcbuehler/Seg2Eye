import numpy as np

import cv2 as cv
import torch
from torchvision import transforms

# from src.util import gaze as gaze_func


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
    def chw2hwc(cls, image):
        if len(image.shape) == 3:
            return image.transpose((1, 2, 0))
        return np.stack([cls.chw2hwc(img) for img in image])

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
    #
    # @staticmethod
    # def look_vec_to_gaze_vec(json_data):
    #     look_vec = np.array(eval(json_data['eye_details']['look_vec']))[:3]
    #     look_vec[0] = -look_vec[0]
    #
    #     original_gaze = gaze_func.vector_to_pitchyaw(
    #         look_vec.reshape((1, 3))).flatten()
    #     rotate_mat = np.asmatrix(np.eye(3))
    #     look_vec = rotate_mat * look_vec.reshape(3, 1)
    #
    #     gaze = gaze_func.vector_to_pitchyaw(look_vec.reshape((1, 3))).flatten()
    #     if gaze[1] > 0.0:
    #         gaze[1] = np.pi - gaze[1]
    #     elif gaze[1] < 0.0:
    #         gaze[1] = -(np.pi + gaze[1])
    #     gaze = gaze.astype(np.float32)
    #     return gaze, original_gaze
    #
    # def do_kappa_augment_labels(self, gaze):
    #     """
    #     Modifies input gaze
    #     Args:
    #         gaze: 2D vector pitch/yaw
    #
    #     Returns:
    #
    #     """
    #     pitch_degree, yaw_degree = [gaze_func.radians2degree(v) for v in gaze]
    #     pitch_degree_augmented = pitch_degree + self.kappa_tuning[0]
    #     yaw_degree_augmented = yaw_degree + self.kappa_tuning[1]
    #     # TODO: make sure the gaze is in the allowed range
    #     result = np.array([gaze_func.degree2radians(pitch_degree_augmented), gaze_func.degree2radians(yaw_degree_augmented)])
    #     return result.astype(np.float32)

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


class UnityPreprocessor(Preprocessor):
    def __init__(self,
                 do_augmentation,
                 eye_image_shape=(72, 120),
                 difficulty=1,
                 kappa_augment_labels=False):
        super().__init__(do_augmentation=do_augmentation,
                         eye_image_shape=eye_image_shape,
                         difficulty=difficulty,
                         kappa_augment_labels=kappa_augment_labels
                         )
        self.do_augmentation = do_augmentation

    def preprocess(self, full_image, json_data):
        """Use annotations to segment eyes and calculate gaze direction."""
        result_dict = dict()

        # Convert look vector to gaze direction in polar angles
        gaze, original_gaze = self.look_vec_to_gaze_vec(json_data)
        if self.kappa_augment_labels:
            gaze = self.do_kappa_augment_labels(gaze)
        result_dict['gaze'] = gaze

        # image might have 2 or 3 channels
        ih, iw = int(full_image.shape[0]), int(full_image.shape[1])
        iw_2, ih_2 = 0.5 * int(iw), 0.5 * int(ih)
        oh, ow = self._eye_image_shape

        def process_coords(coords_list):
            coords = [eval(l) for l in coords_list]
            return np.array([(x, ih - y, z) for (x, y, z) in coords])

        result_dict['head'] = self._headpose_to_radians(json_data)

        interior_landmarks = process_coords(json_data['interior_margin_2d'])
        caruncle_landmarks = process_coords(json_data['caruncle_2d'])
        iris_landmarks = process_coords(json_data['iris_2d'])

        # Prepare to segment eye image
        left_corner = np.mean(caruncle_landmarks[:, :2], axis=0)
        right_corner = interior_landmarks[8, :2]
        eye_width = 1.5 * abs(left_corner[0] - right_corner[0])
        eye_middle = np.mean([np.amin(interior_landmarks[:, :2], axis=0),
                              np.amax(interior_landmarks[:, :2], axis=0)],
                             axis=0)

        # Centre axes to eyeball centre
        translate_mat = np.asmatrix(np.eye(3))
        translate_mat[:2, 2] = [[-iw_2], [-ih_2]]

        # Scale image to fit output dimensions (with a little bit of noise)
        scale_mat = np.asmatrix(np.eye(3))
        scale = 1. + self._noisy_value_from_type('scale')
        scale_inv = 1. / scale
        np.fill_diagonal(scale_mat, ow / eye_width * scale)
        original_eyeball_radius = 71.7593
        eyeball_radius = original_eyeball_radius * scale_mat[
            0, 0]  # See: https://goo.gl/ZnXgDE
        result_dict['radius'] = np.float32(eyeball_radius)

        # Re-centre eye image such that eye fits (based on determined `eye_middle`)
        recentre_mat = np.asmatrix(np.eye(3))
        recentre_mat[0, 2] = iw / 2 - eye_middle[
            0] + 0.5 * eye_width * scale_inv
        recentre_mat[1, 2] = ih / 2 - eye_middle[
            1] + 0.5 * oh / ow * eye_width * scale_inv
        recentre_mat[0, 2] += self._noisy_value_from_type('translation')  # x
        recentre_mat[1, 2] += self._noisy_value_from_type('translation')  # y

        # Apply transforms
        rotate_mat = np.asmatrix(np.eye(3))
        transform_mat = recentre_mat * scale_mat * rotate_mat * translate_mat
        eye = cv.warpAffine(full_image, transform_mat[:2, :3], (ow, oh))

        # Store "clean" eye image before adding noises
        clean_eye = np.copy(eye)
        clean_eye = self.equalize(clean_eye)
        clean_eye = clean_eye.astype(np.float32)
        clean_eye *= 2.0 / 255.0
        clean_eye -= 1.0

        result_dict['clean_eye'] = clean_eye

        # Start augmentation
        if self.do_augmentation:
            eye = self.augment(eye)

        # Histogram equalization and preprocessing for NN
        eye = self.equalize(eye)
        eye = eye.astype(np.float32)
        eye *= 2.0 / 255.0
        eye -= 1.0

        result_dict['eye'] = eye

        # Select and transform landmark coordinates
        iris_centre = np.asarray([
            iw_2 + original_eyeball_radius * -np.cos(
                original_gaze[0]) * np.sin(
                original_gaze[1]),
            ih_2 + original_eyeball_radius * -np.sin(original_gaze[0]),
        ])
        landmarks = np.concatenate([interior_landmarks[::2, :2],  # 8
                                    iris_landmarks[::4, :2],  # 8
                                    iris_centre.reshape((1, 2)),
                                    [[iw_2, ih_2]],  # Eyeball centre
                                    ])  # 18 in total
        landmarks = np.asmatrix(np.pad(landmarks, ((0, 0), (0, 1)), 'constant',
                                       constant_values=1))
        landmarks = np.asarray(landmarks * transform_mat.T)
        landmarks = landmarks[:, :2]  # We only need x, y
        result_dict['landmarks'] = landmarks.astype(np.float32)

        return_keys = ['clean_eye', 'eye', 'gaze', 'landmarks', 'head']
        return [result_dict[k] for k in return_keys]


class MPIIPreprocessor(Preprocessor):
    def __init__(self,
                 eye_image_shape=(36, 60)):
        super().__init__(do_augmentation=False,
                         eye_image_shape=eye_image_shape)

    def preprocess(self, image):
        if len(image.shape) == 2:
            # b/w image
            pass
        else:
            image = self.bgr2rgb(image)

        if self._eye_image_shape is not None:
            oh, ow = self._eye_image_shape
            image = cv.resize(image, (ow, oh))
        image = self.equalize(image)
        image = image.astype(np.float32)
        image *= 2.0 / 255.0
        image -= 1.0
        return image


class RefinedPreprocessor(Preprocessor):
    def __init__(self,
                 do_augmentation,
                 eye_image_shape=(72, 120)):
        super().__init__(do_augmentation, eye_image_shape=eye_image_shape)

    def preprocess(self, full_image, json_data=None):
        """Use annotations to segment eyes and calculate gaze direction."""
        result_dict = dict()

        if self._eye_image_shape is not None:
            oh, ow = self._eye_image_shape
            full_image = cv.resize(full_image, (ow, oh))

        if json_data:
            result_dict['head'] = json_data['head']
            # Convert look vector to gaze direction in polar angles
            # gaze, original_gaze = self._look_vec_to_gaze_vec(json_data)
            result_dict['gaze'] = np.array(json_data['gaze']).astype(
                np.float32)

        eye = full_image

        # Store "clean" eye image before adding noises
        clean_eye = np.copy(eye)
        clean_eye = self.equalize(clean_eye)
        clean_eye = clean_eye.astype(np.float32)
        clean_eye *= 2.0 / 255.0
        clean_eye -= 1.0

        result_dict['clean_eye'] = clean_eye

        # Start augmentation
        if self.do_augmentation:
            eye = self.augment(eye)

        # Histogram equalization and preprocessing for NN
        eye = self.equalize(eye)
        eye = eye.astype(np.float32)
        eye *= 2.0 / 255.0
        eye -= 1.0

        result_dict['eye'] = eye
        return_keys = ['clean_eye', 'eye', 'gaze']
        return [result_dict[k] for k in return_keys if k in result_dict.keys()]