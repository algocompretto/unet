import numpy as np
import random

import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F

def pad_if_smaller(img, size, fill=0):
    pass

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

