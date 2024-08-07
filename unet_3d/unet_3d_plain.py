"""!@file: unet_3d_plain.py
@brief: Plain 3D U-Net architecture
@details: This script contains the plain 3D U-Net architecture described in the paper. Downsampling is done using strided convolutions and upsampling is done using transposed convolutions.
"""

import torch
import torch.nn as nn


class DownBlock(nn.Module):
    """!@brief: Down-sampling block in the plain unet
    @details: 3x3x3 strided convolution + IN + LeakyReLU + 3x3x3 convolution + IN + LeakyReLU
    @params: stride: stride of the convolution
                in_channels: number of input channels
                out_channels: number of output channels
    """

    def __init__(self, stride, in_channels, out_channels):
        super(DownBlock, self).__init__()
        assert (isinstance(stride, tuple) and len(stride) == 3) or isinstance(
            stride, int
        ), "stride should be a tuple of 3 integers or a single integer"
        self.conv1 = nn.Conv3d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1
        )
        self.norm1 = nn.InstanceNorm3d(out_channels)
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv3d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        self.norm2 = nn.InstanceNorm3d(out_channels)
        self.relu2 = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x = self.relu1(self.norm1(self.conv1(x)))
        x = self.relu2(self.norm2(self.conv2(x)))
        return x


class UpBlock(nn.Module):
    """!@brief: Up-sampling block in the plain unet
    @details: transposed convolution + concatenation + twice 3x3x3 convolution + IN + LeakyReLU
    @params: stride: stride of the transposed convolution
                in_channels: number of input channels
                out_channels: number of output channels
    """

    def __init__(self, stride, in_channels, out_channels):
        super(UpBlock, self).__init__()
        assert (isinstance(stride, tuple) and len(stride) == 3) or isinstance(
            stride, int
        ), "stride should be a tuple of 3 integers or a single integer"

        # transposed convolution
        if stride == (1, 2, 2):
            # special case for the last up-sampling block
            # keep the spatial dimensions in depth the same, but double the spatial dimensions in width and height
            self.upconv = nn.ConvTranspose3d(
                in_channels,
                out_channels,
                kernel_size=(3, 2, 2),
                stride=(1, 2, 2),
                padding=(1, 0, 0),
            )
        else:
            self.upconv = nn.ConvTranspose3d(
                in_channels, out_channels, kernel_size=2, stride=stride, padding=0
            )

        # the in channels is doubled the number of out channels because of concatenation
        self.conv1 = nn.Conv3d(
            2 * out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        self.norm1 = nn.InstanceNorm3d(out_channels)
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv3d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        self.norm2 = nn.InstanceNorm3d(out_channels)
        self.relu2 = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x, x_down):
        x = self.upconv(x)
        x = torch.cat((x, x_down), dim=1)  # concatenate along the channel dimension
        x = self.relu1(self.norm1(self.conv1(x)))
        x = self.relu2(self.norm2(self.conv2(x)))
        return x


class UNet3D(nn.Module):
    """!@brief: Plain 3D U-Net architecture
    @details: The architecture consists of 5 down-sampling blocks and 5 up-sampling blocks
    @params: in_channels: number of input channels
                out_channels: number of output channels
                base_channels: number of channels in the first layer
    """

    def __init__(self, in_channels, out_channels, base_channels=30):
        super(UNet3D, self).__init__()
        self.in_channels = in_channels  # number of input channels, 1
        self.out_channels = out_channels  # number of output channels, 1
        self.base_channels = base_channels  # number of channels in the first layer, 30

        # before downsampling
        self.conv1 = nn.Conv3d(
            in_channels, base_channels, kernel_size=3, stride=1, padding=1
        )
        self.norm1 = nn.InstanceNorm3d(base_channels)
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv3d(
            base_channels, base_channels, kernel_size=3, stride=1, padding=1
        )
        self.norm2 = nn.InstanceNorm3d(base_channels)
        self.relu2 = nn.LeakyReLU(0.2, inplace=True)

        # down-sampling
        self.down1 = DownBlock(
            stride=(1, 2, 2), in_channels=base_channels, out_channels=60
        )
        self.down2 = DownBlock(stride=(2, 2, 2), in_channels=60, out_channels=120)
        self.down3 = DownBlock(stride=(2, 2, 2), in_channels=120, out_channels=240)
        self.down4 = DownBlock(stride=(2, 2, 2), in_channels=240, out_channels=320)
        self.down5 = DownBlock(stride=(2, 2, 2), in_channels=320, out_channels=320)

        # up-sampling
        self.up1 = UpBlock(stride=(2, 2, 2), in_channels=320, out_channels=320)
        self.up2 = UpBlock(stride=(2, 2, 2), in_channels=320, out_channels=240)
        self.up3 = UpBlock(stride=(2, 2, 2), in_channels=240, out_channels=120)
        self.up4 = UpBlock(stride=(2, 2, 2), in_channels=120, out_channels=60)
        self.up5 = UpBlock(stride=(1, 2, 2), in_channels=60, out_channels=base_channels)

        # after upsampling
        # 1x1x1 convolution to get the desired number of output channels
        self.conv3 = nn.Conv3d(
            base_channels, out_channels, kernel_size=1, stride=1, padding=0
        )

    def forward(self, x):
        x = self.relu1(self.norm1(self.conv1(x)))
        x_down0 = self.relu2(self.norm2(self.conv2(x)))
        x_down1 = self.down1(x_down0)
        x_down2 = self.down2(x_down1)
        x_down3 = self.down3(x_down2)
        x_down4 = self.down4(x_down3)
        x_down5 = self.down5(x_down4)
        x_up1 = self.up1(x_down5, x_down4)
        # delete unnecessary variables to avoid memory leak
        del x_down5, x_down4
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        x_up2 = self.up2(x_up1, x_down3)
        del x_down3, x_up1
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        x_up3 = self.up3(x_up2, x_down2)
        del x_down2, x_up2
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        x_up4 = self.up4(x_up3, x_down1)
        del x_down1, x_up3
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        x_up5 = self.up5(x_up4, x_down0)
        del x_down0, x_up4
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        x = self.conv3(x_up5)
        del x_up5
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return x
