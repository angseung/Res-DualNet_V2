import torch
from models.resdualnetv2 import ResDualNetV2

a = torch.randn((1, 3, 224, 224))
net = ResDualNetV2().eval()

b = net(a)
