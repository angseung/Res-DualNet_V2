import random
import torch
import torch.nn as nn
import torchvision
from torchvision import models
from torchvision.utils import make_grid
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
import numpy as np
from torch.nn.functional import conv2d
from models.resnetCA import ResDaulNet18_TP5
from models.resnet import ResNet18
from utils import plot_filter_ch

random_seed = 1
g = torch.Generator()
g.manual_seed(random_seed)
torch.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)  # multi-GPU
np.random.seed(random_seed)

path = "outputs/resdual5_cifar-10_paper/ckpt.pth"
SAVEDAT = torch.load(path)

model = ResDaulNet18_TP5()
model = nn.DataParallel(model)
model.load_state_dict(SAVEDAT['net'])
model.eval()

resnet = ResNet18()
path = "outputs/resnet18/ckpt.pth"

normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                     std=[0.2023, 0.1994, 0.2010])

transform_train = transforms.Compose([
    transforms.ToTensor(),
    # normalize,
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    # normalize,
])

trainset = torchvision.datasets.CIFAR10(root='C:/cifar-10', train=False, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(root='C:/cifar-10', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=0)

for image, label in trainloader:
    break

input = (image[0, :, :, :][None, :, :, :]).float().cuda()

out = model.module.conv1(input)
out = model.module.bn1(out)

branch_1 = model.module.layer1[0].conv1_d1
branch_2 = model.module.layer1[1].conv1_d2

out_1 = branch_1(out)
out_2 = branch_2(out)

plot_filter_ch(out_1, title="out_dw1", fname="dw1", save_opt=True)
plot_filter_ch(out_2, title="out_dw2", fname="dw2", save_opt=True)

plot_filter_ch(out_1 + out_2, title="out", fname="summ", save_opt=True)