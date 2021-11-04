"""EfficientNet in PyTorch.
Paper: "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks".
Reference: https://github.com/keras-team/keras-applications/blob/master/keras_applications/efficientnet.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def swish(x):
    return x * x.sigmoid()


class Block(nn.Module):
    """expansion + depthwise + pointwise + squeeze-excitation"""

    def __init__(self, in_channels, out_channels, kernel_size):
        super(Block, self).__init__()

        # conv
        self.conv1 = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=1,
            # padding='same',
            bias=False,
        )

        self.conv2 = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=1,
            # padding='same',
            bias=False,
        )

        self.conv3 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=1, bias=False
        )

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(in_channels)
        self.bn3 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(x)
        out = self.bn1(conv1 + swish(conv2)) + self.bn2(conv2 + swish(conv1))
        out = self.bn3(swish(self.conv3(out)))

        return out


class L2NNet(nn.Module):
    def __init__(self, cfg, num_classes=10):
        super(L2NNet, self).__init__()
        self.cfg = cfg
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layers = self._make_layers(in_channels=32)
        self.linear = nn.Linear(cfg["out_channels"][-1], num_classes)

    def _make_layers(self, in_channels):
        layers = []
        cfg = [self.cfg[k] for k in ["out_channels", "kernel_size"]]

        for out_channels, kernel_size in zip(*cfg):
            layers.append(Block(in_channels, out_channels, kernel_size))
            in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = swish(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        dropout_rate = self.cfg["dropout_rate"]
        if self.training and dropout_rate > 0:
            out = F.dropout(out, p=dropout_rate)
        out = self.linear(out)
        return out


def L2NNetV0():
    cfg = {
        # 1,1,2,3,5,8,13,22,35,57,92,149,241,390,631
        "out_channels": [57, 92, 149, 241, 390],
        "kernel_size": [3, 3, 2, 1, 1],
        "dropout_rate": 0.2,
    }
    return L2NNet(cfg)


import torchinfo


def test():
    net = L2NNetV0()
    torchinfo.summary(net)
    x = torch.randn(2, 3, 32, 32)
    y = net(x)
    print(y.shape)


if __name__ == "__main__":
    test()
