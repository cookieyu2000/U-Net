# unet.py
import torch    
import torch.nn as nn
from unet_part import DoubleConv, Downsampling, Upsampling

class UNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.down_conv1 = Downsampling(in_channels, 64)
        self.down_conv2 = Downsampling(64, 128)
        self.down_conv3 = Downsampling(128, 256)
        self.down_conv4 = Downsampling(256, 512)

        self.bottle_neck = DoubleConv(512, 1024)

        self.up_conv1 = Upsampling(1024, 512)
        self.up_conv2 = Upsampling(512, 256)
        self.up_conv3 = Upsampling(256, 128)
        self.up_conv4 = Upsampling(128, 64)

        self.out = nn.Conv2d(in_channels=64, out_channels=num_classes, kernel_size=1)

    def forward(self, x):
        down_1, p1 = self.down_conv1(x)
        down_2, p2 = self.down_conv2(p1)
        down_3, p3 = self.down_conv3(p2)
        down_4, p4 = self.down_conv4(p3)

        bottle_neck = self.bottle_neck(p4)

        up_1 = self.up_conv1(bottle_neck, down_4)
        up_2 = self.up_conv2(up_1, down_3)
        up_3 = self.up_conv3(up_2, down_2)
        up_4 = self.up_conv4(up_3, down_1)

        out = self.out(up_4)

        return out, [down_1, down_2, down_3, down_4, bottle_neck, up_1, up_2, up_3, up_4]
