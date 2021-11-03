import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.nn as nn
from torchinfo import summary
from models.resnetCA import ResDaulNet18_TPI5, ResDaulNet18_TP5
from utils import data_loader, progress_bar
import math

config ={
    'mode' : 'train',
    'dataset' : 'ImageNet',
    'pth_path' : "./outputs/resdual5_imagenet/ckpt.pth",
    'input_size' : 224,
    'batch_size' : 256,
    'max_epoch' : 200
}

def train(ep):
    print('\nEpoch: %d' % ep)
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(dataloader):
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
        progress_bar(
            batch_idx,
            len(dataloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'% (
                train_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    train_acc = 100. * correct / total

    return train_acc

def test():
    # Model conversion to evaludation mode...
    net.eval()
    test_loss = 0
    correct = 0
    total = 0

    # Turn off back propagation...
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(
                batch_idx,
                len(dataloader),
                'Loss: %.3f | Acc: %.3f%% (%d/%d)' % (
                    test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    test_acc = 100. * correct / total

    return test_acc


# Check use GPU or not
use_gpu = torch.cuda.is_available()  # use GPU

if use_gpu:
    device = torch.device("cuda")
else:
    raise NotImplementedError('CUDA Device needed to run this code...')

# ImageNet
net = ResDaulNet18_TPI5()

# Optimizer
optimizer = optim.Adam(net.parameters(), lr=0.0025)

# CIFAR-10
# net = ResDaulNet18_TP5()

net.to(device)
net = torch.nn.DataParallel(net)
cudnn.benchmark = True
optimizer = optim.Adam(net.parameters(), lr=0.001)

# Load checkpoint data
# {net : net.state_dict,
#  acc : best test acc,
#  optimizer : optimizer.state_dict(),
#  epoch : best performed epoch}

pth_path = config['pth_path']
SAVEDAT = torch.load(pth_path)

net.load_state_dict(SAVEDAT['net'])
optimizer.load_state_dict(SAVEDAT['optimizer'])
acc = SAVEDAT['acc']
epoch = SAVEDAT['epoch']

model_name = net.module.__class__.__name__
print("%s model was loaded successfully... [best validation acc : %.3f at %03d epoch]"
      %(model_name, acc, epoch))

# input_size = 32
# dataset = "CIFAR-10"

dataloader = data_loader(
    mode=config['mode'],
    dataset=config['dataset'],
    input_size=config['input_size'],
    batch_size=config['batch_size'],
    shuffle_opt=True
)

print("Loading %s dataset completed..." % config['mode'])

# Get model params and macs...
modelinfo = summary(net, (1, 3, config['input_size'], config['input_size']), verbose=0)
total_params = modelinfo.total_params
total_macs = modelinfo.total_mult_adds

param_mil = total_params / (10 ** 6)
macs_bil = total_macs / (10 ** 9)

criterion = nn.CrossEntropyLoss()

if config['mode'] == 'train':
    max_epoch = config['max_epoch']
    print("Resume training from %03d epoch..." % (epoch + 1))
    for ep in range(epoch + 1, max_epoch):
        train_acc = train(ep)

    print("Training completed...\n\t[Training accuracy : %.3f at %03d epoch]"
          % (train_acc, ep))

elif config['mode'] == 'test':
    test_acc = test()

    netscore = 20 * math.log10((test_acc ** 2) / (math.sqrt(param_mil) * math.sqrt(macs_bil)))

    print("Test completed...")
    print("NetScore : %.3f, Params(M) : %.3f, Mac(G) : %.3f, Test Acc : %.3f"
          %(netscore, param_mil, macs_bil, test_acc))
