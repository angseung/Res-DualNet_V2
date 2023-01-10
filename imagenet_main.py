import torch
from models.resdualnetv1 import ResDualNetV1ImageNet
from models.resdualnetv2 import ResDualNetV2ImageNet

a = torch.randn((1, 3, 224, 224))

net = ResDualNetV1ImageNet().eval()
b1 = net(a)

net = ResDualNetV2ImageNet().eval()
b2 = net(a)
