import os
import platform
import argparse
import random
import datetime
import json
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import torch.onnx
from torchinfo import summary
from tqdm import tqdm
import numpy as np
import timm

parser = argparse.ArgumentParser(description="PyTorch CIFAR10 Training")
parser.add_argument("--lr", default=0.001, type=float, help="learning rate")
parser.add_argument(
    "--resume", "-r", action="store_true", help="resume from checkpoint"
)
args = parser.parse_args()


def seed_worker(worker_id: None) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


curr_os = platform.system()
print("Current OS : %s" % curr_os)

if "Windows" in curr_os:
    device = "cuda" if torch.cuda.is_available() else "cpu"
elif "Darwin" in curr_os:
    device = "mps" if torch.backends.mps.is_available() else "cpu"
elif "Linux" in curr_os:
    device = "cuda" if torch.cuda.is_available() else "cpu"

best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

config = {
    "max_epoch": 20,
    "initial_lr": 0.001,
    "train_batch_size": 64,
    "dataset": "CIFAR-10",  # [ImageNet, CIFAR-10]
    "train_resume": False,
    "set_random_seed": True,
    "l2_reg": 0.0005,
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
        root="./imagenet/", split="train", transform=transform_train
    )
    testset = torchvision.datasets.ImageNet(
        root="./imagenet/", split="val", transform=transform_test
    )

elif Dataset == "CIFAR-10":
    input_size = 224
    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]
    )
    transform_train = transforms.Compose(
        [
            transforms.Resize(256),
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
    trainset = torchvision.datasets.CIFAR10(
        root="./cifar-10/", train=True, download=True, transform=transform_train
    )
    testset = torchvision.datasets.CIFAR10(
        root="./cifar-10/", train=False, download=True, transform=transform_test
    )

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
def train(epoch, dir_path=None) -> None:
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

    with open("outputs/" + dir_path + "/log.txt", "a") as f:
        f.write(
            "Epoch [%d] |Train| Loss: %.3f, Acc: %.3f \t"
            % (epoch, train_loss / (batch_idx + 1), 100.0 * correct / total)
        )


def test(epoch, dir_path=None) -> None:
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

    # Save checkpoint.
    if acc > best_acc:
        print("Saving..")
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


# Model
print("==> Building model..")

nets = {
    "vit_model": timm.create_model(
        "vit_base_patch16_224", pretrained=True, num_classes=10
    )
}

for netkey in nets.keys():
    now = datetime.datetime.now().strftime("%y%m%d_%H%M%S")

    log_path = "outputs/" + netkey
    net = nets[netkey]

    os.makedirs(log_path, exist_ok=True)

    if not config["train_resume"]:
        with open(log_path + "/log.txt", "w") as f:
            f.write("Networks : %s\n" % netkey)
            f.write("Net Train Configs: \n %s\n" % json.dumps(config))
            m_info = summary(net, (1, 3, input_size, input_size), verbose=0)
            f.write("%s\n" % str(m_info))

    elif config["train_resume"]:
        with open(log_path + "/log.txt", "a") as f:
            f.write("Train resumed from this point...\n")

    if device == "cuda":
        net = torch.nn.DataParallel(net)  # Not support ONNX converting
        # cudnn.benchmark = True

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        net.parameters(), lr=config["initial_lr"], weight_decay=config["l2_reg"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(max_epoch * 1.0)
    )

    if config["train_resume"]:
        # Load checkpoint.
        print("==> Resuming from checkpoint..")
        checkpoint = torch.load(log_path + "/ckpt.pth")
        net.load_state_dict(checkpoint["net"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        best_acc = checkpoint["acc"]
        start_epoch = checkpoint["epoch"] + 1

    for epoch in range(start_epoch, max_epoch):
        net = net.to(device)
        train(epoch, netkey)
        test(epoch, netkey)
        scheduler.step()
