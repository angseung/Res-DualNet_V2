import math
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.nn as nn
from torchinfo import summary
from tqdm import tqdm
from models.resnetCA import ResDaulNet18_TPI5, ResDaulNet18_TP5
from utils import data_loader, progress_bar


# reproducible option
random_seed = 1
torch.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)  # multi-GPU
np.random.seed(random_seed)

# config ={
#     'mode' : 'test',
#     'dataset' : 'ImageNet',
#     'pth_path' : "./outputs/resdual5_imagenet/ckpt.pth",
#     'input_size' : 224,
#     'batch_size' : 100
# }

config = {
    "mode": "test",
    "dataset": "CIFAR-10",
    "pth_path": "./outputs/resdual5_cifar-10/ckpt.pth",
    "input_size": 32,
    "batch_size": 100,
}


def test():
    # Model conversion to evaludation mode...
    net.eval()
    test_loss = 0
    correct = 0
    total = 0

    # Turn off back propagation...
    with torch.no_grad():
        with tqdm(dataloader, unit="batch") as tepoch:
            for batch_idx, (inputs, targets) in enumerate(tepoch):
                tepoch.set_description(f"Test Epoch {epoch}")

                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                tepoch.set_postfix(
                    loss=test_loss / (batch_idx + 1), accuracy=100.0 * correct / total
                )
    test_acc = 100.0 * correct / total

    return test_acc


# Check use GPU or not
use_gpu = torch.cuda.is_available()  # use GPU

if use_gpu:
    device = torch.device("cuda")
else:
    raise NotImplementedError("CUDA Device needed to run this code...")

# ImageNet
# net = ResDaulNet18_TPI5()

# CIFAR-10
net = ResDaulNet18_TP5()

net.to(device)
net = torch.nn.DataParallel(net)
# cudnn.benchmark = True

# Load checkpoint data
# {net : net.state_dict,
#  acc : best test acc,
#  optimizer : optimizer.state_dict(),
#  epoch : best performed epoch}

pth_path = config["pth_path"]
SAVEDAT = torch.load(pth_path)

net.load_state_dict(SAVEDAT["net"])
acc = SAVEDAT["acc"]
epoch = SAVEDAT["epoch"]

model_name = net.module.__class__.__name__
print(
    "%s model was loaded successfully... [best validation acc : %.3f at %03d epoch]"
    % (model_name, acc, epoch)
)

dataloader = data_loader(
    mode=config["mode"],
    dataset=config["dataset"],
    input_size=config["input_size"],
    batch_size=config["batch_size"],
    shuffle_opt=True,
)

print("Loading %s dataset completed..." % config["mode"])

# Get model params and macs...
modelinfo = summary(net, (1, 3, config["input_size"], config["input_size"]), verbose=0)
total_params = modelinfo.total_params
total_macs = modelinfo.total_mult_adds

param_mil = total_params / (10 ** 6)
macs_bil = total_macs / (10 ** 9)

criterion = nn.CrossEntropyLoss()
test_acc = test()
netscore = 20 * math.log10(
    (test_acc ** 2) / (math.sqrt(param_mil) * math.sqrt(macs_bil))
)

print("Test completed...")
print(
    "NetScore : %.3f, Params(M) : %.3f, Mac(G) : %.3f, Test Acc : %.3f"
    % (netscore, param_mil, macs_bil, test_acc)
)
