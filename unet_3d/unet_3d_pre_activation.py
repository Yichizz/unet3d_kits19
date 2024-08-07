"""!@file: unet_3d_pre_activation.py
@brief: 3D U-Net with pre-activation residual blocks
@details: This script contains the 3D U-Net architecture with pre-activation residual blocks described in the paper. 
            Downsampling is done using strided convolutions and upsampling is done using transposed convolutions.
            The number of residual blocks in the encoder increases with depth.
"""

import torch
import torch.nn as nn


class PreActivationResidualBlock(nn.Module):
    """!@brief: Pre-activation residual block
    @details: IN + ReLU + 3x3x3 convolution + IN + ReLU + 3x3x3 convolution + residual connection
    @params: in_channels: number of input channels
                out_channels: number of output channels
                repititions: number of residual blocks
    """

    def __init__(self, in_channels, out_channels, repititions=1):
        super(PreActivationResidualBlock, self).__init__()
        # repititions: number of residual blocks
        assert repititions >= 1 and isinstance(
            repititions, int
        ), "repititions should be a positive integer"

        self.res = nn.Sequential(
            nn.InstanceNorm3d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
        )
        self.repititions = repititions

    def forward(self, x):
        for _ in range(self.repititions):
            residual = x
            x = self.res(x)
            x = x + residual
        return x


class DownBlock(nn.Module):
    """!@brief: Down-sampling block in the unet with pre-activation
    @details: IN + ReLU + 3x3x3 strided convolution + IN + ReLU + 3x3x3 convolution + residual connection
    @params: stride: stride of the convolution
                in_channels: number of input channels
                out_channels: number of output channels
    """

    def __init__(self, stride, in_channels, out_channels):
        super(DownBlock, self).__init__()
        # stride should be a tuple of 3 integers, e.g. (1, 2, 2)
        assert (isinstance(stride, tuple) and len(stride) == 3) or isinstance(
            stride, int
        ), "stride should be a tuple of 3 integers or a single integer"

        self.norm1 = nn.InstanceNorm3d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv3d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1
        )
        self.norm2 = nn.InstanceNorm3d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )

        self.residual = nn.Conv3d(
            in_channels, out_channels, kernel_size=1, stride=stride, padding=0
        )

    def forward(self, x):
        residual = self.residual(x)
        x = self.relu1(self.norm1(x))
        x = self.relu2(self.norm2(self.conv1(x)))
        x = self.conv2(x)
        x = x + residual
        return x


class UpBlock(nn.Module):
    """!@brief: Up-sampling block in the unet with pre-activation
    @details: transposed convolution + concatenation + 3x3x3 convolution + IN + ReLU
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
        self.relu1 = nn.ReLU(inplace=True)

    def forward(self, x, x_down):
        x = self.upconv(x)
        x = torch.cat((x, x_down), dim=1)
        x = self.relu1(self.norm1(self.conv1(x)))
        return x


class UNet3DPreActivation(nn.Module):
    """
    @brief: 3D U-Net with pre-activation residual blocks
    @details: The architecture consists of 5 down-sampling
                blocks and 5 up-sampling blocks
    @params: in_channels: number of input channels
                out_channels: number of output channels
                base_channels: number of base channels
    """

    def __init__(self, in_channels, out_channels, base_channels=24):
        super(UNet3DPreActivation, self).__init__()
        # before the first down-sampling block
        self.block0 = nn.Sequential(
            nn.Conv3d(in_channels, base_channels, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(base_channels, base_channels, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(base_channels),
            nn.ReLU(inplace=True),
        )
        # down-sampling blocks
        self.down1 = DownBlock((1, 2, 2), base_channels, 48)
        self.res1 = PreActivationResidualBlock(48, 48, repititions=1)
        self.down2 = DownBlock((2, 2, 2), 48, 96)
        self.res2 = PreActivationResidualBlock(96, 96, repititions=2)
        self.down3 = DownBlock((2, 2, 2), 96, 192)
        self.res3 = PreActivationResidualBlock(192, 192, repititions=3)
        self.down4 = DownBlock((2, 2, 2), 192, 320)
        self.res4 = PreActivationResidualBlock(320, 320, repititions=4)
        self.down5 = DownBlock((2, 2, 2), 320, 320)
        self.res5 = PreActivationResidualBlock(320, 320, repititions=5)

        # up-sampling blocks
        self.up1 = UpBlock((2, 2, 2), 320, 320)
        self.up2 = UpBlock((2, 2, 2), 320, 192)
        self.up3 = UpBlock((2, 2, 2), 192, 96)
        self.up4 = UpBlock((2, 2, 2), 96, 48)
        self.up5 = UpBlock((1, 2, 2), 48, base_channels)

        # after the last up-sampling block
        self.out = nn.Conv3d(
            base_channels, out_channels, kernel_size=1, stride=1, padding=0
        )

    def forward(self, x):
        x0 = self.block0(x)
        x1 = self.down1(x0)
        x1 = self.res1(x1)
        x2 = self.down2(x1)
        x2 = self.res2(x2)
        x3 = self.down3(x2)
        x3 = self.res3(x3)
        x4 = self.down4(x3)
        x4 = self.res4(x4)
        x5 = self.down5(x4)
        x5 = self.res5(x5)

        x = self.up1(x5, x4)
        del x5, x4
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        x = self.up2(x, x3)
        del x3
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        x = self.up3(x, x2)
        del x2
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        x = self.up4(x, x1)
        del x1
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        x = self.up5(x, x0)
        del x0
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        x = self.out(x)
        return x
