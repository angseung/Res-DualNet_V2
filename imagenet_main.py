import torch
from models.resdualnetv1 import ResDualNetV1ImageNet, ResDualNetV1
from models.resdualnetv2 import ResDualNetV2ImageNet, ResDualNetV2
from models.shufflenetv1 import ShuffleNet_32

a = torch.randn((1, 3, 224, 224))
a1 = torch.randn((1, 3, 32, 32))

net1 = ResDualNetV1ImageNet().eval()
b1 = net1(a)

net2 = ResDualNetV2ImageNet().eval()
b2 = net2(a)

net1_1 = ResDualNetV1()
net1_1.layer1.load_state_dict(net1.layer1.state_dict())
net1_1.layer2.load_state_dict(net1.layer2.state_dict())
net1_1.layer3.load_state_dict(net1.layer3.state_dict())
net1_1.layer4.load_state_dict(net1.layer4.state_dict())

net2_1 = ResDualNetV2()
net2_1.layer1.load_state_dict(net2.layer1.state_dict())
net2_1.layer2.load_state_dict(net2.layer2.state_dict())
net2_1.layer3.load_state_dict(net2.layer3.state_dict())
net2_1.layer4.load_state_dict(net2.layer4.state_dict())

b1_1 = net1_1(a1)
b2_1 = net2_1(a1)
b3 = ShuffleNet_32()(a1)

mbv2 = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
