import numpy as np
import torch
import cv2


class ImagePostprocessor:
    eps = 1e-6

    @classmethod
    def as_batch(cls, image, as_tensor=True):
        image = cls.to_tensor(image)
        # Now are always dealing with tensors
        shape_len = len(image.shape)
        if shape_len == 3:
            image = image.unsqueeze(0)
        if shape_len > 4:
            raise ValueError(f"Image has too many dimensions: {shape_len}")
        return cls.return_as(image, as_tensor=as_tensor)

    @classmethod
    def to_numpy(cls, tensor):
        if isinstance(tensor, np.ndarray):
            return tensor
        if isinstance(tensor, torch.Tensor):
            return np.copy(tensor.cpu().detach())
        else:
            raise ValueError(f"Invalid data type: {type(tensor)}")

    @classmethod
    def to_tensor(cls, array):
        if isinstance(array, np.ndarray):
            return torch.from_numpy(array)
        if isinstance(array, torch.Tensor):
            return array
        else:
            raise ValueError(f"Invalid data type: {type(array)}")

    @classmethod
    def return_as(cls, image, as_tensor):
        if as_tensor:
            if isinstance(image, torch.Tensor):
                return image
            else:
                return cls.to_tensor(image)
        else:
            if isinstance(image, np.ndarray):
                return image
            else:
                return cls.to_numpy(image)

    @classmethod
    def unnormalize(cls, image, as_tensor=True):
        image = cls.return_as(image, as_tensor=True)
        min_val, max_val = torch.min(image), torch.max(image)
        eps = 1e-6  # Small epsilon that allows for small deviations (e.g. when resizing)
        if min_val >= -1-eps and max_val <= 1+eps:
            image = torch.add(image, 1)
            image = torch.div(torch.mul(image, 255), 2)
        elif min_val >= 0 and max_val < 4:
            # Label map
            image = torch.div(image, 3) * 255
        elif min_val >= 0 and max_val <= 255:
            pass
        else:
            raise ValueError(f"Invalid ranges for image. Min: {min_val}, max: {max_val}")
        return cls.return_as(image, as_tensor).int()

    @classmethod
    def normalize(cls, image, as_tensor=True):
        image = cls.return_as(image, as_tensor=True)
        min_val, max_val = torch.min(image), torch.max(image)
        eps = 1e-6  # Small epsilon that allows for small deviations (e.g. when resizing)
        if min_val >= -1-eps and max_val <= 1+eps:
            pass
            # We assume we don't have images with range [0, 1]
        elif min_val >= 0:
            image = torch.div(image, torch.max(image))
            image = torch.mul(image, 2)
            image = torch.add(image, -1)
        else:
            raise ValueError(f"Invalid ranges for image. Min: {min_val}, max: {max_val}")
        return cls.return_as(image, as_tensor)

    @classmethod
    def to_255imagebatch(cls, image, as_tensor=True):
        image = cls.as_batch(image, as_tensor=True)
        image = cls.unnormalize(image, as_tensor=True)
        return cls.return_as(image, as_tensor)

    @classmethod
    def to_255resized_imagebatch(cls, image, w=400, h=640, as_tensor=True):
        image = cls.resize(image, w, h, as_tensor=True)
        image = cls.to_255imagebatch(image)
        return cls.return_as(image, as_tensor)

    @classmethod
    def to_1resized_imagebatch(cls, image, w=400, h=640, as_tensor=True):
        image = cls.resize(image, w, h, as_tensor=True)
        image = cls.normalize(image, as_tensor=True)
        return cls.return_as(image, as_tensor)

    @classmethod
    def resize(cls, image, w=400, h=640, as_tensor=True):
        image = cls.as_batch(image, as_tensor=False)
        # img comes in bchw
        image = image.transpose(0, 2, 3, 1).astype(np.float)
        image = np.array([[cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)] for img in image])
        return cls.return_as(image, as_tensor)

    @classmethod
    def assert_range1(cls, img):
        min_val = torch.min(img)
        max_val = torch.max(img)
        assert min_val >= -1 - cls.eps, f"Invalid ranges for image. Min: {min_val}, max: {max_val}"
        assert max_val <= 1 + cls.eps, f"Invalid ranges for image. Min: {min_val}, max: {max_val}"

    @classmethod
    def get_error_map(cls, fake, target):
        assert fake.shape == target.shape
        cls.assert_range1(fake)
        cls.assert_range1(target)
        error_heatmap = torch.abs(fake - target)
        error_heatmap = (error_heatmap / torch.max(error_heatmap) * 2) - 1
        return error_heatmap

