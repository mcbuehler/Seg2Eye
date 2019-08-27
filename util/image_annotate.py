from PIL import Image, ImageFont, ImageDraw
import numpy as np


def get_text_image(text, dim=(100, 30)):
    img = Image.new('1', dim)

    fnt = ImageFont.truetype(font="datasets/font.ttf", size=36)
    d = ImageDraw.Draw(img)
    d.text((10, 10), text, font=fnt, fill=(255))
    img = img.convert('1')
    return np.copy(img).astype(np.float32)
