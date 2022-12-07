from unet_parts import *


class UNet(nn.Module):
    """U-net was originally invented and first used for biomedical image segmentation.
    Its architecture can be broadly thought of as an encoder network followed by a decoder network.
    Unlike classification where the end result of the the deep network is the only important thing,
    semantic segmentation not only requires discrimination at pixel level but also a mechanism to project
    the discriminative features learnt at different stages of the encoder onto the pixel space.

    Args:
        nn (nn.Module): Base class for all neural network modules.
    """

    def __init__(self, n_channels: int, n_classes: int, bilinear: bool =False):
        """The __init__ method is where we typically define the attributes of a class.
        In our case, all the "sub-components" of our model should be defined here.

        Args:
            n_channels (int): Number of channels for the data received. For example, RGB images are 3-channels.
            n_classes (int): Number of classes to segment, consider +1 for the background.
            bilinear (bool, optional): Bilinear interpolation for image resizing, used in upscaling step. Defaults to False.
        """
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        """The forward function computes output Tensors from input Tensors.
        The backward function receives the gradient of the output Tensors with respect to some scalar value,
        and computes the gradient of the input Tensors with respect to that same scalar value.

        Args:
            x (torch.Tensor): A tensor representing a node in a computational graph.

        Returns:
            torch.Tensor: The output of the segmentation network.
        """
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
