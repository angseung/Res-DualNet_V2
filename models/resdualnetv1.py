"""ResNet in PyTorch.
For Pre-activation ResNet, see 'preact_resnet.py'.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
"""
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def swish(x: Tensor) -> Tensor:
    return x * x.sigmoid()


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.in_planes = in_planes
        self.planes = planes

        # vanila resnet18 layer
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)

        # resdual18 layer - 1(DW stride 적용)
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
        self.conv1_p1 = nn.Conv2d(
            in_planes, planes, kernel_size=1, stride=1, padding=0, bias=False
        )

        self.bn1_dw1 = nn.BatchNorm2d(in_planes)
        self.bn1_dw2 = nn.BatchNorm2d(in_planes)
        self.bn1_pw = nn.BatchNorm2d(planes)

        # vanila resnet18 layer
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        # resdual18 layer - 2
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
        self.conv2_p1 = nn.Conv2d(
            planes, planes, kernel_size=1, stride=1, padding=0, bias=False
        )

        self.bn2_dw1 = nn.BatchNorm2d(planes)
        self.bn2_dw2 = nn.BatchNorm2d(planes)
        self.bn2_pw = nn.BatchNorm2d(planes)

        self.identify = nn.Identity()

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
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        out += self.shortcut(x)
        out = F.relu(out)
        return out


class DPBlock(BasicBlock):
    def forward(self, x):
        out = self.bn1_dw1(swish(self.conv1_d1(x) + self.conv1_d2(x)) * 0.5)
        out = self.bn1_pw(self.conv1_p1(out))
        out = self.bn2_dw1(swish(self.conv2_d1(out) + self.conv2_d2(out)) * 0.5)

        out += self.shortcut(x)
        out = swish(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(
            3, self.in_planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
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


class ResNetImageNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=1000):
        super(ResNetImageNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(
            3, self.in_planes, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.maxpool(F.relu(self.bn1(self.conv1(x))))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResDualNetV1(block_cfg: List[int] = [2, 2, 2, 2]):
    return ResNet(DPBlock, block_cfg)


def ResDualNetV1ImageNet(block_cfg: List[int] = [2, 2, 2, 2]):
    return ResNetImageNet(DPBlock, block_cfg)


def test():
    net = ResDualNetV1()
    # summary(net, (1, 3, 224, 224))
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())


# test()
