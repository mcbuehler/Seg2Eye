import cv2
import torch
import numpy as np

from data.postprocessor import ImageProcessor


def get_text_image(text, dim=(100, 30), fontscale=1):
    img = np.zeros(dim, np.uint8)
    font                   = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10, dim[0] - 10)  # x, y
    fontScale              = fontscale
    fontColor              = (255,255,255)
    lineType               = 2

    cv2.putText(img, text,
        bottomLeftCornerOfText,
        font,
        fontScale,
        fontColor,
        lineType)
    # Convert to tensor in range [-1, 1]
    img = torch.from_numpy(img).double()
    img = ImageProcessor.normalize(img, as_tensor=True)
    return img
