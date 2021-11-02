import torch.optim as optim
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse
from models import *

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
# parser.add_argument('--lr', default=0.00001, type=float, help='learning rate')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
max_epoch = 1

print(device)

# Data Preparing  !!!

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
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

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    with open('l2n_point_depth_log.txt', 'a') as f:
        f.write('Epoch [%d] |Train| Loss: %.3f, Acc: %.3f \t' % (
            epoch, train_loss / (batch_idx + 1), 100. * correct / total))


def test(epoch, dir_path=None):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0

    if dir_path is None:
        dir_path = 'checkpoint'

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    # Save checkpoint.
    acc = 100. * correct / total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir(dir_path):
            os.mkdir(dir_path)
        torch.save(state, './' + dir_path + '/ckpt.pth')
        best_acc = acc

    with open('l2n_point_depth_log.txt', 'a') as f:
        f.write('|Test| Loss: %.3f, Acc: %.3f \n' % (test_loss / (batch_idx + 1), acc))


# Model
print('==> Building model..')
# net = VGG('VGG11')
# net = MobileNet()
nets = {'resnet_0': ResNet18(),
        'vgg': VGG('VGG16'),
        'mobilenet': MobileNet(),
        'effnet': EfficientNetB0()}
# from torchinfo import summary
# summary(net, (1, 3, 32, 32))


with open('l2n_point_depth_log.txt', 'w') as f:
    f.write('test log start\n')

for netkey in nets.keys():
    net = nets[netkey]
    net = net.to(device)

    from torchinfo import summary

    with open('l2n_point_depth_log.txt', 'a') as f:
        f.write('Networks : %s\n' % netkey)
        summary(net, (1, 3, 224, 224))

    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoint/ckpt.pth')
        net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr,
                          momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epoch)
    # from lr_scheduler import CosineAnnealingWarmUpRestarts
    # scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=50, T_mult=2, eta_max=0.1,  T_up=10, gamma=0.5)

    for epoch in range(start_epoch, start_epoch + max_epoch):
        train(epoch)
        test(epoch, netkey)
        scheduler.step()
