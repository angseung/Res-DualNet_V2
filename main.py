import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import torch.onnx
from torchinfo import summary
from tqdm import tqdm
import os
import argparse
from utils import progress_bar, VisdomLinePlotter, VisdomImagePlotter, save_checkpoint
from models import *

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
# parser.add_argument('--lr', default=0.00001, type=float, help='learning rate')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args()

# reproducible option
import random
import numpy as np

random_seed = 1

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(random_seed)
torch.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)  # multi-GPU
np.random.seed(random_seed)
# torch.use_deterministic_algorithms(True)

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

config = {
    'max_epoch' : 100,
    'initial_lr' : 0.0025,
    'train_batch_size' : 256,
    'dataset' : 'CIFAR-10' # [ImageNet, CIFAR-10]
}

Dataset = config['dataset']
max_epoch = config['max_epoch']
batch_size = config['train_batch_size']

# Data Preparing  !!!
print('==> Preparing data..')

if Dataset == 'ImageNet':
    input_size = 224
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        # transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize]
    )

    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])
    trainset = torchvision.datasets.ImageNet(
        root='C:/imagenet/',
        split='train',
        transform=transform_train
    )
    testset = torchvision.datasets.ImageNet(
        root='C:/imagenet/',
        split='val',
        transform=transform_test
    )

elif Dataset == 'CIFAR-10':
    input_size = 32
    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010]
    )
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])
    trainset = torchvision.datasets.CIFAR10(
        root='C:/cifar-10/',
        train=True,
        download=True,
        transform=transform_train
    )
    testset = torchvision.datasets.CIFAR10(
        root='C:/cifar-10/',
        train=False,
        download=True,
        transform=transform_test
    )
    # testset = torchvision.datasets.CIFAR10(root='C:/cifar-10/', split='val', transform=transform_test)

trainloader = torch.utils.data.DataLoader(
    trainset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0
    # worker_init_fn=seed_worker,
    # generator=g,
)
testloader = torch.utils.data.DataLoader(
    testset,
    batch_size=100,
    shuffle=False,
    num_workers=0
)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Training
def train(epoch, dir_path=None, plotter=None):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    with tqdm(trainloader, unit="batch") as tepoch:
        for batch_idx, (inputs, targets) in enumerate(tepoch):
            tepoch.set_description(f"Train Epoch {epoch}")

            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            tepoch.set_postfix(
                loss=train_loss / (batch_idx + 1),
                accuracy=100. * correct / total
            )
            if plotter is not None:
                plotter[0].plot('batch_loss', 'train_epoch%d' % epoch, 'Batch Loss', batch_idx,
                                train_loss / (batch_idx + 1))
                plotter[0].plot('batch_acc', 'train_epoch%d' % epoch, 'Batch Acc', batch_idx, 100. * correct / total)

    if plotter is not None:
        plotter[0].plot('loss', 'train', 'Class Loss', epoch, train_loss / (batch_idx + 1))
        plotter[0].plot('acc', 'train', 'Class Accuracy', epoch, 100. * correct / total)

    with open('outputs/' + dir_path + '/log.txt', 'a') as f:
        f.write('Epoch [%d] |Train| Loss: %.3f, Acc: %.3f \t' % (
            epoch, train_loss / (batch_idx + 1), 100. * correct / total))

    return (epoch, train_loss / (batch_idx + 1), 100. * correct / total)


def test(epoch, dir_path=None, plotter=None):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0

    if dir_path is None:
        dir_path = 'outputs/checkpoint'
    else:
        dir_path = 'outputs/' + dir_path

    with torch.no_grad():
        with tqdm(testloader, unit="batch") as tepoch:
            for batch_idx, (inputs, targets) in enumerate(tepoch):
                tepoch.set_description(f"Test Epoch {epoch}")

                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                tepoch.set_postfix(loss=test_loss / (batch_idx + 1), accuracy=100. * correct / total)
    acc = 100. * correct / total

    # visualization
    if plotter is not None:
        plotter[0].plot('loss', 'val', 'Class Loss', epoch, test_loss / (batch_idx + 1))
        plotter[0].plot('acc', 'val', 'Class Accuracy', epoch, acc)

    # Save checkpoint.

    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'optimizer' : optimizer.state_dict(),
            'scheduler' : scheduler.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir(dir_path):
            os.mkdir(dir_path)
        torch.save(state, './' + dir_path + '/ckpt.pth')
        # torch.onnx.export(net,
        #                   torch.empty(1, 3, 224, 224, dtype=torch.float32, device=device),
        #                   dir_path + '/output.onnx')

        best_acc = acc

    with open(dir_path + '/log.txt', 'a') as f:
        f.write('|Test| Loss: %.3f, Acc: %.3f \n' % (test_loss / (batch_idx + 1), acc))

    return (epoch, test_loss, acc)


# Model
print('==> Building model..')

nets = {
    # 'resnet18_vanilla': ResDaulNet18_TP1(),
    # 'resnet18_mbstyle': ResDaulNet18_TP2(),
    # 'resdualnet18_pw': ResDaulNet18_TP3(),
    # 'resdualnet18': ResDaulNet18_TP4(),
    # 'resdualnet18_swish_1': ResDaulNet18_TP5(),
    # 'resdual5_imagenet': ResDaulNet18_TPI5(),
    'resdual5_cifar-10': ResDaulNet18_TP5(),
    # 'rexnet18_0_relu_relu': RexNet18_T0(),
    # 'rexnet18_1_crelu': RexNet18_T1(),
}

for netkey in nets.keys():
    # visualization
    # plotter = [VisdomLinePlotter(env_name='{} Training Plots'.format(netkey)),
    #            VisdomImagePlotter(env_name='{} Training Plots'.format(netkey))]
    plotter = None
    log_path = 'outputs/' + netkey
    net = nets[netkey]
    net = net.to(device)

    os.makedirs(log_path, exist_ok=True)
    with open(log_path + '/log.txt', 'w') as f:
        f.write('Networks : %s\n' % netkey)
        m_info = summary(
            net,
            (1, 3, input_size, input_size),
            verbose=0
        )
        f.write('%s\n' % str(m_info))

    if device == 'cuda':
        net = torch.nn.DataParallel(net)  # Not support ONNX converting
        # cudnn.benchmark = True
        pass

    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    # optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=5e-4)
    # optimizer = optim.Adam(net.parameters(), lr=0.0025)  ## Conf.2
    optimizer = optim.Adam(net.parameters(), lr=config['initial_lr'])  ## Conf.2
    # optimizer = optim.Adam(net.parameters(), lr=0.001)  ## Conf.2
    # optimizer = optim.RMSprop(net.parameters(), lr=0.256, alpha=0.99, eps=1e-08, weight_decay=0.9, momentum=0.9, centered=False) # Conf.1
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(max_epoch * 1.0))
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 2, gamma=0.97, last_epoch=-1, verbose=True)
    # from lr_scheduler import CosineAnnealingWarmUpRestarts
    # scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=50, T_mult=2, eta_max=0.1,  T_up=10, gamma=0.5)

    resume = True
    if resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        # assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(log_path + '/ckpt_bak.pth')
        net.load_state_dict(checkpoint['net'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch'] + 1

    for epoch in range(start_epoch, max_epoch - start_epoch):
        ep, train_loss, train_acc = train(epoch, netkey, plotter)
        ep, test_loss, test_acc = test(epoch, netkey, plotter)
        scheduler.step()
        # print('current lr : %.6f' % optimizer.defaults['lr'])
