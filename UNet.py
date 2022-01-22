import torch
import torch.nn as nn
from torch import autograd
from functools import partial
from torchvision import models
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
                )

    def forward(self, x):
        return self.conv(input)


class UNet(nn.Module):
    pass

class Decoder(nn.Module):
    pass

class ResNet34_UNet(nn.Module):
    pass
