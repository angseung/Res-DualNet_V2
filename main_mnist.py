import os
import argparse
import random
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import torch.onnx
from torchinfo import summary
from tqdm import tqdm
import numpy as np
from tensorboardX import SummaryWriter
from utils import progress_bar, VisdomLinePlotter, VisdomImagePlotter, save_checkpoint
from models.resdualnet import ResDaulNetMnist

## for tensorboard run this command
"""
>>> tensorboard --logdir ./tensorboard
"""

parser = argparse.ArgumentParser(description="PyTorch CIFAR10 Training")
# parser.add_argument('--lr', default=0.00001, type=float, help='learning rate')
parser.add_argument("--lr", default=0.1, type=float, help="learning rate")
parser.add_argument(
    "--resume", "-r", action="store_true", help="resume from checkpoint"
)
args = parser.parse_args()


def seed_worker(worker_id: None) -> None:
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


device = "cuda" if torch.cuda.is_available() else "cpu"
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

config = {
    "max_epoch": 100,
    "initial_lr": 0.0025,
    "train_batch_size": 256,
    "dataset": "MNIST",  # [ImageNet, CIFAR-10, MNIST]
    "train_resume": False,
    "set_random_seed": True,
}

if config["set_random_seed"]:
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
    # torch.use_deterministic_algorithms(True)

Dataset = config["dataset"]
max_epoch = config["max_epoch"]
batch_size = config["train_batch_size"]

# Data Preparing  !!!
print("==> Preparing data..")

if Dataset == "ImageNet":
    input_size = 224
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    transform_train = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]
    )
    trainset = torchvision.datasets.ImageNet(
        root="C:/imagenet/", split="train", transform=transform_train
    )
    testset = torchvision.datasets.ImageNet(
        root="C:/imagenet/", split="val", transform=transform_test
    )

elif Dataset == "CIFAR-10":
    input_size = 32
    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]
    )
    transform_train = transforms.Compose([transforms.ToTensor(), normalize])

    transform_test = transforms.Compose([transforms.ToTensor(), normalize])
    trainset = torchvision.datasets.CIFAR10(
        root="C:/cifar-10/", train=True, download=True, transform=transform_train
    )
    testset = torchvision.datasets.CIFAR10(
        root="C:/cifar-10/", train=False, download=True, transform=transform_test
    )

elif Dataset == "MNIST":
    input_size = 28
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    trainset = torchvision.datasets.MNIST(
        "./data", train=True, download=True, transform=transform
    )
    testset = torchvision.datasets.MNIST("./data", train=False, transform=transform)

trainloader = torch.utils.data.DataLoader(
    trainset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,
    worker_init_fn=seed_worker,
    generator=g,
)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=0
)

classes = (
    "plane",
    "car",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)

# Training
def train(epoch, dir_path=None, plotter=None) -> None:
    print("\nEpoch: %d" % epoch)
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
                loss=train_loss / (batch_idx + 1), accuracy=100.0 * correct / total
            )
            if plotter is not None:
                plotter[0].plot(
                    "batch_loss",
                    "train_epoch%d" % epoch,
                    "Batch Loss",
                    batch_idx,
                    train_loss / (batch_idx + 1),
                )
                plotter[0].plot(
                    "batch_acc",
                    "train_epoch%d" % epoch,
                    "Batch Acc",
                    batch_idx,
                    100.0 * correct / total,
                )

    if plotter is not None:
        plotter[0].plot(
            "loss", "train", "Class Loss", epoch, train_loss / (batch_idx + 1)
        )
        plotter[0].plot(
            "acc", "train", "Class Accuracy", epoch, 100.0 * correct / total
        )

    with open("outputs/" + dir_path + "/log.txt", "a") as f:
        f.write(
            "Epoch [%d] |Train| Loss: %.3f, Acc: %.3f \t"
            % (epoch, train_loss / (batch_idx + 1), 100.0 * correct / total)
        )

    return (epoch, train_loss / (batch_idx + 1), 100.0 * correct / total)


def test(epoch, dir_path=None, plotter=None) -> None:
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0

    if dir_path is None:
        dir_path = "outputs/checkpoint"
    else:
        dir_path = "outputs/" + dir_path

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

                tepoch.set_postfix(
                    loss=test_loss / (batch_idx + 1), accuracy=100.0 * correct / total
                )
    acc = 100.0 * correct / total

    # visualization
    if plotter is not None:
        plotter[0].plot("loss", "val", "Class Loss", epoch, test_loss / (batch_idx + 1))
        plotter[0].plot("acc", "val", "Class Accuracy", epoch, acc)

    # Save checkpoint.

    if acc > best_acc:
        print("\nSaving..")
        state = {
            "net": net.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "acc": acc,
            "epoch": epoch,
        }
        if not os.path.isdir(dir_path):
            os.mkdir(dir_path)
        torch.save(state, "./" + dir_path + "/ckpt.pth")

        best_acc = acc

    with open(dir_path + "/log.txt", "a") as f:
        f.write("|Test| Loss: %.3f, Acc: %.3f \n" % (test_loss / (batch_idx + 1), acc))

    return (epoch, test_loss, acc)


# Model
print("==> Building model..")

nets = {
    # 'resdual5_imagenet': ResDaulNet18_TPI5(),
    # "resdual5_cifar-10": ResDaulNet18_TP5(),
    "resdualnet_mnist": ResDaulNetMnist(),
}

for netkey in nets.keys():
    plotter = None
    log_path = "outputs/" + netkey
    net = nets[netkey]
    net = net.to(device)

    os.makedirs(log_path, exist_ok=True)

    if not config["train_resume"]:
        with open(log_path + "/log.txt", "w") as f:
            f.write("Networks : %s\n" % netkey)
            m_info = summary(net, (1, 1, input_size, input_size), verbose=0)
            f.write("%s\n" % str(m_info))
    elif config["train_resume"]:
        with open(log_path + "/log.txt", "a") as f:
            f.write("Train resumed from this point...\n")

    if device == "cuda":
        net = torch.nn.DataParallel(net)  # Not support ONNX converting
        # cudnn.benchmark = True

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=config["initial_lr"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(max_epoch * 1.0)
    )

    if config["train_resume"]:
        # Load checkpoint.
        print("==> Resuming from checkpoint..")
        # assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(log_path + "/ckpt.pth")
        net.load_state_dict(checkpoint["net"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        best_acc = checkpoint["acc"]
        start_epoch = checkpoint["epoch"] + 1

    for epoch in range(start_epoch, max_epoch):
        train(epoch, netkey, plotter)
        test(epoch, netkey, plotter)
        scheduler.step()
