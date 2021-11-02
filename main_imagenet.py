import torch
import torchvision
import torchvision.transforms as transforms
from models import *
import matplotlib.pyplot as plt
net = RexNet18_T0()

activation = {}
ifmap = {}


def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
        ifmap[name] = input[0].detach()

    return hook


checkpoint = torch.load('outputs/rexnet18_0/ckpt.pth')
net.load_state_dict(checkpoint['net'])
net.to('cuda')

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                         std=[0.2023, 0.1994, 0.2010])
])
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
inputs, targets = testset[0]
inputs = inputs.reshape(1,3,32,32)
inputs = inputs.to('cuda')
print(inputs.shape)

net.layer1[1].identify.register_forward_hook(get_activation('identify'))
net.layer1[1].conv1_d1.register_forward_hook(get_activation('conv1_d1'))
net.layer1[1].conv1_d2.register_forward_hook(get_activation('conv1_d2'))
output = net(inputs)
print(output, targets)
x0 = activation['conv1_d1'].cpu().numpy().flatten()
x1 = activation['conv1_d2'].cpu().numpy().flatten()
y = ifmap['identify'].cpu().numpy().flatten()

ax = plt.axes(projection='3d')
ax.scatter3D(x0,x1,y)
plt.show()
