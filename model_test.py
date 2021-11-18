import torch
import torch.nn as nn
import torchvision
from torchvision import models
from torchvision.utils import make_grid
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from models.resnet import ResNet18
from models.shufflenetv2_32 import ShuffleNetV2
import numpy as np
from torch.nn.functional import conv2d
from utils import plot_filter_ch, plot_hist

# reproducible option
import random

random_seed = 123
torch.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)  # multi-GPU
np.random.seed(random_seed)


def make_grid_norm(kernels, nrows=12):
    kernels = kernels - kernels.min()
    kernels = kernels / kernels.max()

    make_grid(kernels, nrows=nrows)


model = ShuffleNetV2(net_size=0.5)
model = torch.nn.DataParallel(model)
optim = torch.optim.Adam(model.parameters())
INPUT_SIZE = 32

path = "outputs/dct_cifar-10/ckpt.pth"
SAVEDAT = torch.load(path)

model.load_state_dict(SAVEDAT["net"])
optim.load_state_dict(SAVEDAT["optimizer"])

params = {}

for name, p in model.named_parameters():
    a = p.cpu().clone().detach().numpy()
    params[name] = p.clone().detach()
    print(name)
    print(p.shape)

    # if 'conv' in name:
    #     p = p.cpu()
    #     filter_img = make_grid_norm(p)
    #     plt.imshow(filter_img.permute(1, 2, 0))

normalize = transforms.Normalize(
    mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]
)
if INPUT_SIZE == 32:
    transform_train = transforms.Compose([transforms.ToTensor(), normalize])

    transform_test = transforms.Compose([transforms.ToTensor(), normalize])

elif INPUT_SIZE == 224:
    transform_train = transforms.Compose(
        [
            transforms.RandomResizedCrop(112),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(112),
            transforms.ToTensor(),
            normalize,
        ]
    )

trainset = torchvision.datasets.CIFAR10(
    root="C:/cifar-10/", train=False, download=True, transform=transform_train
)
# trainset = torchvision.datasets.ImageNet(root='C:/imagenet/', split = 'train', transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=64, shuffle=True, num_workers=0
)

testset = torchvision.datasets.CIFAR10(
    root="C:/cifar-10/", train=False, download=True, transform=transform_test
)
# testset = torchvision.datasets.ImageNet(root='C:/imagenet/', split = 'val', transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=0
)
# classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

for image, label in trainloader:
    break

img = image[1, :, :, :]
plt.imshow(img.permute(1, 2, 0))
plt.show()

img = torch.reshape(img, (1, img.shape[0], img.shape[1], img.shape[2]))

conv_img = conv2d(img, params["module.conv1.weight"].cpu(), padding="same")

# plot_filter_ch(conv_img)

# model = ResNet18()
x = image.to("cuda")
# x = image
# model.to('cpu')
for net_name, model in enumerate(model.children()):
    pass

for name, layer in enumerate(model.children()):
    print(name, type(layer))

    x = layer(x)
    plot_filter_ch(x.detach().cpu().numpy(), title=layer.__class__.__name__)
    # plot_hist(
    #     x.detach().cpu().numpy(),
    #     title=layer.__class__.__name__)

    if "Sequential" in layer.__class__.__name__:
        for name_l, layer_l in enumerate(layer.children()):
            print("\t", name_l, type(layer_l))
            # x = layer_l(x)

            if "BasicBlock" in layer_l.__class__.__name__:
                for name_b, layer_b in enumerate(layer_l.children()):
                    print("\t\t", name_b, type(layer_b))
                    if "Split" in str(type(layer_b)):
                        x = layer_b(x)[0]
                    else:
                        x = layer_b(x)

                    plot_filter_ch(
                        x.detach().cpu().numpy(),
                        title=layer_b.__class__.__name__,
                        fname=layer_b.__class__.__name__,
                        save_opt=True,
                    )
                    plot_hist(
                        x.detach().cpu().numpy(),
                        title=layer_b.__class__.__name__,
                        fname=layer_b.__class__.__name__,
                        save_opt=True,
                    )
