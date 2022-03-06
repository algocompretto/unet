import numpy as np
import random

import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F

def pad_if_smaller(img, size: int, fill: int = 0):
    pass

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class RandomResize(object):
    def __init__(self, min_size: int, max_size: int = None):
        self.min_size = min_size
        if max_size is None:
            max_size = min_size
        self.max_size = max_size

    def __call__(self, image, target):
        size = random.randint(self.min_size, self.max_size)
        # Here size is passed in int type, so the minimum side length of the image is scaled to size X size
        image = F.resize(image, size)
        # Note for the interpolation here, InterpolationMode.NEAREST is only available after torchvision(0.9.0)
        # If it is a previous version, you need to use PIL.Image.NEAREST
        target = F.resize(target, size, interpolation=T.InterpolationMode.NEAREST)
        return image, target

