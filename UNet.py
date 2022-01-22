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

nonlinearity = partial(F.relu, inplace=True)

class Decoder(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(Decoder, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nonlinearity

        self.deconv = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3,
                stride=2, padding=1, output_padding=1)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nonlinearity

        self.conv2 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nonlinearity

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        x = self.deconv(x)
        x = self.norm2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)

        return x

class ResNet34_UNet(nn.Module):
    def __init__(self, num_classes=2, num_channels=3, pretrained=True):
        super(ResNet34_UNet, self).__init__()

        filters = [64, 128, 256, 512]
        
        resnet = models.resnet34(pretrained=pretrained)
        
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.decoder4 = DecoderBlock(512, filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.decoder4 = DecoderBlock(512, filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)


    def forward(self, x):
        # Encoding stage
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)

        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Decoding stage
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)

        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return nn.Sigmoid()(out)
