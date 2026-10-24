import torch
import torch.nn as nn

class Block(nn.Module):
    """
    ResNet-50 V1.5 bottleneck block.

    Main branch:
        1x1 Conv: in_channels -> width
        3x3 Conv: width -> width, optional space reduction
        1x1 Conv: width -> 4 * width

    Each convolution is followed by BatchNorm.
    ReLU follows the first two BatchNorm layers.

    Shortcut:
        Identity when input/output dimensions match.
        1x1 projection + BatchNorm otherwise.

    The branches are added, followed by ReLU.

    Input:
        (B, in_channels, H, W)

    Output:
        (B, 4 * width, H/stride, W/stride)
    """
    expansion = 4 

    def __init__(self, in_channels, width, stride=1):
        super().__init__()

        out_channels = width * self.expansion

        # In: (B, in_channels, H, W)
        # 1x1 conv changes in_channels -> width
        # stride=1 keeps H and W the same
        # Out: (B, width, H, W) 
        self.conv1 = nn.Conv2d(
            in_channels, width, 
            kernel_size=1, stride=1, 
            bias=False
        )
        self.batch1 = nn.BatchNorm2d(width)

        # In: (B, width, H, W)
        # 3x3 conv preserves the width
        # padding=1 preserves H and W when stride=1
        # stride=2 downsizes H -> H/2 and W -> W/2
        # Out: (B, width, H/stride, W/stride)
        self.conv2 = nn.Conv2d(
            width, width, 
            kernel_size=3, stride=stride, padding=1, 
            bias=False
        )
        self.batch2 = nn.BatchNorm2d(width)

        # In: (B, width, H, W)
        # 1x1 conv expands width -> 4 * width
        # Out: (B, out_channels, H, W)
        self.conv3 = nn.Conv2d(
            width, out_channels, 
            kernel_size=1, stride=1, 
            bias=False
        )
        self.batch3 = nn.BatchNorm2d(out_channels)

        self.activation = nn.ReLU(inplace=True) # inplace reduces memory usage 

        # Presevrs (B, in_channels, H, W)
        self.shortcut = nn.Identity()

        if stride != 1 or in_channels != out_channels:
            # Projection convert in_channels -> out_channels
            # Uses same stride as conv2 to match H and W
            # Out: (B, out_channels, H/stride, W/stride)
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, 
                    kernel_size=1, stride=stride, 
                    bias=False
                ),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)

        features = self.conv1(x)
        features = self.batch1(features)
        self.activation(features)

        features = self.conv2(features)
        features = self.batch2(features)
        self.activation(features)

        features = self.conv3(features)
        features = self.batch3(features)

        features = features + identity
        self.activation(features)

        return features

class Stem(nn.Module):
    """
    Initial feature extraction and downsampling for ResNet-50.

    Input: (B, in_channels, H, W)

    Operations:
        7x7 Conv:   in_channels -> out_channels, H/2 x W/2
        BatchNorm
        ReLU
        3x3 MaxPool: Preserves channels, H/4 x W/4

    Output: (B, out_channels, H/4, W/4)

    For a 640x640 RGB input:
        (B, 3, 640, 640) -> (B, 64, 160, 160)
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels, out_channels, 
            kernel_size=7, stride=2, padding=3, 
            bias=False
        )
        self.batch = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)

        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        features = self.conv(x)
        features = self.batch(features)
        self.activation(features)

        return self.max_pool(features)

class ResNetBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stem = Stem(in_channels=3, out_channels=64)

        # 1 conv block, 2 identity
        self.stage1 = nn.Sequential(
            Block(64, 64), # conv_block as in != out
            Block(256, 64),
            Block(256, 64)
        )

        # 1 conv block, 3 identity
        self.stage2 = nn.Sequential(
            Block(256, 128, stride=2),
            Block(512, 128),
            Block(512, 128),
            Block(512, 128)
        )

        # 1 conv block, 5 identity
        self.stage3 = nn.Sequential(
            Block(512, 256, stride=2),
            Block(1024, 256),
            Block(1024, 256),
            Block(1024, 256),
            Block(1024, 256),
            Block(1024, 256)
        )

        # 1 conv block, 2 identity
        self.stage4 = nn.Sequential(
            Block(1024, 512, stride=2),
            Block(2048, 512),
            Block(2048, 512)
        )

    def forward(self, x):  # Input: 640 x 640 x 3
        # Output for each part : H x W x C
        x = self.stem(x) # 160 x 160 x 64
        x = self.stage1(x) # 160 x 160 x 256

        p3 = self.stage2(x) # 80 x 80 x 512
        p4 = self.stage3(p3) # 40 x 40 x 1024
        p5 = self.stage4(p4) # 20 x 20 x 2048

        return p3, p4, p5