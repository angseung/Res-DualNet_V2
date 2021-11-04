"""EfficientNet in PyTorch.
Paper: "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks".
Reference: https://github.com/keras-team/keras-applications/blob/master/keras_applications/efficientnet.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def swish(x):
    return x * x.sigmoid()


def mish(x):
    return x * torch.tanh(F.softplus(x))


class BasicLink2Block(nn.Module):
    """expansion + depthwise + pointwise + squeeze-excitation"""

    def __init__(self, in_channels, out_channels, kernel_size, strides, gdiv):
        super(BasicLink2Block, self).__init__()

        # basic blocks
        self.conv1 = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=in_channels // gdiv,
            bias=False,
        )

        self.conv2 = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=in_channels // gdiv,
            bias=True,
        )

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(in_channels)

        # link blocks
        self.conv3 = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=(1 if kernel_size == 3 else 2),
            groups=in_channels // 2,
            bias=False,
        )

        self.conv4 = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=(1 if kernel_size == 3 else 2),
            groups=in_channels // 4,
            bias=False,
        )

        self.conv5 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=strides,
            padding=1,
            groups=in_channels // 8,
            bias=False,
        )

        self.bn3 = nn.BatchNorm2d(in_channels)
        self.bn4 = nn.BatchNorm2d(in_channels)
        self.bn5 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        cout1 = self.conv3(self.bn1(mish(self.conv1(x))))
        cout2 = self.conv4(self.bn2(mish(self.conv2(x * 2))))

        out = self.bn3(cout1 + mish(cout2)) + self.bn4(cout2 + mish(cout1))
        out = self.bn5(mish(self.conv5(out)))

        return out


class LINKNet(nn.Module):
    def __init__(self, cfg, num_classes=10):
        super(LINKNet, self).__init__()
        self.cfg = cfg

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layers = self._make_layers(in_channels=32)
        self.linear = nn.Linear(cfg["out_channels"][-1], num_classes)

    def _make_layers(self, in_channels):
        layers = []
        cfg = [self.cfg[k] for k in ["out_channels", "kernel_size", "strides", "gdiv"]]

        for out_channels, kernel_size, strides, gdiv in zip(*cfg):
            layers.append(
                BasicLink2Block(in_channels, out_channels, kernel_size, strides, gdiv)
            )
            in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = mish(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        dropout_rate = self.cfg["dropout_rate"]
        if self.training and dropout_rate > 0:
            out = F.dropout(out, p=dropout_rate)
        out = self.linear(out)
        return out


def LINKNetV0(num_classes):
    cfg = {
        # 1,1,2,3,5,8,13,22,35,57,92,149,241,390,631
        "out_channels": [24, 96, 192, 384],
        "strides": [2, 2, 1, 1],
        "kernel_size": [3, 3, 5, 5],
        "gdiv": [32, 24, 96, 192],
        "dropout_rate": 0.2,
    }
    return LINKNet(cfg, num_classes=num_classes)


def LINKNetV1():
    cfg = {
        # 1,1,2,3,5,8,13,22,35,57,92,149,241,390,631
        "out_channels": [24, 96, 192, 384, 480],
        "strides": [2, 2, 1, 1, 2],
        "kernel_size": [3, 3, 5, 5, 3],
        "gdiv": [32, 24, 96, 192, 384],
        "dropout_rate": 0.2,
    }
    return LINKNet(cfg)


import torchinfo


def test():
    net = LINKNetV0(1000)
    torchinfo.summary(net, (1, 3, 224, 224))
    x = torch.randn(2, 3, 224, 224, device="cuda")
    y = net(x)
    print(y.shape)


if __name__ == "__main__":
    test()
