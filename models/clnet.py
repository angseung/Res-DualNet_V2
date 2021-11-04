"""
CrossLink Network
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def swish(x):
    return x * x.sigmoid()


def mish(x):
    return x * torch.tanh(F.softplus(x))


class CrossLinkBlock(nn.Module):
    """Cross-Link Block"""

    def __init__(self, in_channels, out_channels, kernel_size, pool_enable):
        super(CrossLinkBlock, self).__init__()

        self.pool_enable = pool_enable

        # basic blocks
        self.dconv1_1 = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size[0],
            stride=1,
            padding="same",
            groups=1,
            bias=False,
        )

        self.dconv1_2 = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size[1],
            stride=1,
            padding="same",
            groups=1,
            bias=False,
        )

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(in_channels)

        self.pconv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding="same",
            groups=1,
            bias=False,
        )

        self.bn3 = nn.BatchNorm2d(out_channels)

        self.maxpool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        """add forward here"""

        out1 = self.dconv1_1(x)
        out2 = self.dconv1_2(x)

        out = self.bn1(out1 + mish(out2)) + self.bn2(out2 + mish(out1))
        out = self.bn3(self.pconv(out))

        if self.pool_enable:
            out = self.maxpool(out)

        return out


class CLNET(nn.Module):
    def __init__(self, cfg, num_classes=10):
        super(CLNET, self).__init__()
        self.cfg = cfg

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)

        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(
            32, 32, kernel_size=3, stride=1, padding=1, groups=1, bias=False
        )

        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 16, kernel_size=1, stride=1, padding=0, bias=False)

        self.layers = self._make_layers(in_channels=16)
        self.linear = nn.Linear(cfg["out_channels"][-1], num_classes)

    def _make_layers(self, in_channels):
        layers = []
        cfg = [self.cfg[k] for k in ["out_channels", "kernel_size", "pool_enable"]]

        for out_channels, kernel_size, pool_enable in zip(*cfg):
            layers.append(
                CrossLinkBlock(in_channels, out_channels, kernel_size, pool_enable)
            )
            in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = mish(self.bn1(self.pool1(self.conv1(x))))  # conv block
        out = self.conv3(swish(self.bn2(self.conv2(out))))  # sep block
        out = self.layers(out)
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        dropout_rate = self.cfg["dropout_rate"]
        if self.training and dropout_rate > 0:
            out = F.dropout(out, p=dropout_rate)
        out = self.linear(out)
        return out


def CLNet_V0(num_classes):
    cfg = {
        "out_channels": [24, 40, 80, 112, 160],
        "kernel_size": [(5, 3), (3, 5), (3, 3), (5, 5), (3, 3)],
        "pool_enable": [True, True, True, True, False],
        "dropout_rate": 0.2,
    }
    return CLNET(cfg, num_classes=num_classes)


import torchinfo


def test():
    net = CLNet_V0(10)
    torchinfo.summary(net, (1, 3, 224, 224))
    x = torch.randn(3, 3, 224, 224, device="cuda")
    y = net(x)
    print(y.shape)


if __name__ == "__main__":
    test()
