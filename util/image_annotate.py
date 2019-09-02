import cv2
import torch
from PIL import Image, ImageFont, ImageDraw
import numpy as np

from data.postprocessor import ImagePostprocessor


def get_text_image(text, dim=(100, 30)):
    img = np.zeros(dim, np.uint8)
    font                   = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10, dim[0] - 10)  # x, y
    fontScale              = 1
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
    img = ImagePostprocessor.normalize(img, as_tensor=True)
    return img

    # img = Image.new('1', dim)
    #
    # fnt = ImageFont.truetype(font="datasets/font.ttf", size=36)
    # d = ImageDraw.Draw(img)
    # d.text((10, 10), text, font=fnt, fill=(255))
    # img = img.convert('1')
    # return np.copy(img).astype(np.float32)
