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
    def __init__(self, in_ch, out_ch):
        super(UNet, self).__init__()

        # Double Conv Layers => Pool() and in the end, => Up-Conv
        self.conv1 = DoubleConv(in_ch, 32)
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = DoubleConv(32, 64)
        self.pool2 = nn.MaxPool2d(2)

        self.conv3 = DoubleConv(64, 128)
        self.pool3 = nn.MaxPool2d(2)

        self.conv4 = DoubleConv(128, 256)
        self.pool4 = nn.MaxPool2d(2)

        # Up-Conv stage
        self.conv5 = DoubleConv(256, 512)
        self.up1 = nn.ConvTranspose2d(512, 256, 2, stride=2)

        self.conv6 = DoubleConv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)

        self.conv7 = DoubleConv(256, 128)
        self.up3 = nn.ConvTranspose2d(128, 64, 2, stride=2)

        self.conv8 = DoubleConv(128, 64)
        self.up4 = nn.ConvTranspose2d(64, 32, 2, stride=2)

        self.conv9 = DoubleConv(64, 32)
        self.conv10 = nn.Conv2d(32, out_ch, 1)

    def forward(self, x):
        c1 = self.conv1(x)
        p1 = self.pool1(c1)

        c2 = self.conv2(p1)
        p2 = self.pool2(c2)

        c3 = self.conv3(p2)
        p3 = self.pool3(c3)

        c4 = self.conv4(p3)
        p4 = self.pool4(c4)

        c5 = self.conv5(p4)
        up1 = self.up1(c5)

        merge1 = torch.cat([up1, c4], dim=1)

        c6 = self.conv6(merge1)
        up2 = self.up2(c6)
        
        merge2 = torch.cat([up2, c3], dim=1)

        c7 = self.conv7(merge2)
        up3 = self.up3(c7)

        merge3 = torch.cat([up3, c2], dim=1)

        c8 = self.conv8(merge3)
        up4 = self.up4(c8)

        merge4 = torch.cat([up4, c1], dim=1)

        c9 = self.conv9(merge4)
        c10 = self.conv10(c9)
        out = nn.Sigmoid()(c10)
        
        return out



class Decoder(nn.Module):
    pass

class ResNet34_UNet(nn.Module):
    pass
