import warnings
from typing import Any, List, Optional, Union
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def swish(x: torch.Tensor = None) -> torch.Tensor:
    if x.__class__.__name__ != "Tensor":
        raise TypeError("Input must be torch Tensor")

    return x * x.sigmoid()


def channel_shuffle(x: torch.Tensor, groups: int) -> torch.Tensor:
    batchsize, num_channels, height, width = x.size()
    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups, channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x


class DWHT(nn.Module):
    def __init__(
        self,
        in_planes: int = 64,
        planes: int = 128,
        groups: int = 8,
        shuffle: bool = True,
    ) -> nn.Module:
        super(DWHT, self).__init__()
        self.n = int(math.log2(in_planes))
        self.N = in_planes
        self.M = planes
        self.groups = groups
        self.shuffle = shuffle

    def forward(self, x: torch.Tensor = None) -> torch.Tensor:
        if x.shape[1] != self.N:
            raise ValueError("input channel error")

        # pad zero along the channel axis
        if self.N < self.M:
            x = F.pad(x, (0, 0, 0, 0, 0, (self.M - self.N)), "constant", 0)

        for i in range(self.n):
            e = x[:, ::2, :, :]
            o = x[:, 1::2, :, :]
            x[:, : (self.M // 2), :, :] = e + o
            x[:, (self.M // 2) :, :, :] = e - o

        if self.N > self.M:
            x = x[:, : self.M, :, :]

        if self.shuffle:
            x = channel_shuffle(x, self.groups)

        return x


class CTPTBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int = 64, planes: int = 128, stride: int = 1) -> nn.Module:
        super(CTPTBlock, self).__init__()
        self.in_planes = in_planes
        self.planes = planes

        self.DWHT1 = DWHT(in_planes, planes)
        self.bn1 = nn.BatchNorm2d(planes)

        self.dconv1 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=planes,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.DWHT2 = DWHT(planes, planes)
        self.bn3 = nn.BatchNorm2d(planes)

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x):
        out = self.DWHT1(x)
        out = self.bn1(out)
        out = self.dconv1(out)
        out = self.bn2(out)
        out = self.DWHT2(out)
        out = self.bn3(out)

        shortcut = self.shortcut(x)

        out = torch.cat((out, shortcut), dim=1)

        return out


class DWHTBlock(nn.Module):
    expansion = 1

    def __init__(
        self, in_planes: int, planes: int, stride: int = 1, dropout_rate: float = 0.2
    ) -> nn.Module:
        super(DWHTBlock, self).__init__()
        self.in_planes = in_planes
        self.planes = planes
        self.dropout_rate = dropout_rate

        if dropout_rate > 0.5:
            warnings.warn(f"Dropout rate is too high, got {dropout_rate}", UserWarning)

        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv1_d1 = nn.Conv2d(
            in_planes,
            in_planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=in_planes,
            bias=False,
        )
        self.conv1_d2 = nn.Conv2d(
            in_planes,
            in_planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=in_planes,
            bias=False,
        )

        self.dwht = DWHT(in_planes, planes, groups=8, shuffle=False)

        self.bn1_dw1 = nn.BatchNorm2d(in_planes)
        self.bn1_dwht = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        if self.dropout_rate is not None:
            self.dropout = nn.Dropout(self.dropout_rate)

        self.conv2_d1 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=planes,
            bias=False,
        )

        self.conv2_d2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=planes,
            bias=False,
        )

        self.bn2_dw1 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = self.bn1_dw1(swish(self.conv1_d1(x) + self.conv1_d2(x)) * 0.5)
        if self.dropout_rate is not None:
            out = self.dropout(out)

        out = self.bn1_dwht(self.dwht(out))
        out = self.bn2_dw1(swish(self.conv2_d1(out) + self.conv2_d2(out)) * 0.5)

        out += self.shortcut(x)
        out = swish(out)

        return out


class ResDualNet(nn.Module):
    def __init__(
        self,
        block: nn.Module,
        num_blocks: List[int],
        num_classes: int = 10,
        dropout_rate: List[Union[float, None]] = [None, None, None, None],
    ) -> nn.Module:
        super(ResDualNet, self).__init__()
        self.in_planes = 64
        self.dropout_rate = dropout_rate

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(
            block, 64, num_blocks[0], stride=1, dropout_rate=self.dropout_rate[0]
        )
        self.layer2 = self._make_layer(
            block, 128, num_blocks[1], stride=2, dropout_rate=self.dropout_rate[1]
        )
        self.layer3 = self._make_layer(
            block, 256, num_blocks[2], stride=2, dropout_rate=self.dropout_rate[2]
        )
        self.layer4 = self._make_layer(
            block, 512, num_blocks[3], stride=2, dropout_rate=self.dropout_rate[3]
        )
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride, dropout_rate):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, dropout_rate))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResDaulNetV2():
    return ResDualNet(DWHTBlock, [2, 2, 2, 2])


def ResDaulNetV2Auto(
    block_config: List[int], dropout_rate: List[Union[float, None]]
):
    return ResDualNet(DWHTBlock, block_config, dropout_rate=dropout_rate)
