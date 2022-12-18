import time
import torch
from torchinfo import summary
from models.resdualnetv2 import ResDualNetV2
from models.resdualnetv1 import ResDualNetV1
from models.resnet import ResNet18

resdualnetv2 = ResDualNetV2().to("cpu").eval()
resdualnetv1 = ResDualNetV1().to("cpu").eval()
resnet = ResNet18().to("cpu").eval()

resnet = torch.jit.script(resnet)
resdualnetv1 = torch.jit.script(resdualnetv1)
# resdualnetv2 = torch.jit.script(resdualnetv2)

input_size = (1, 3, 32, 32)
input_tensor = torch.randn(input_size).to("cpu")

iters = 100
time_resdualnetv2: float = 0
time_resdualnetv1: float = 0
time_resnet: float = 0

for _ in range(iters):
    start = time.time()
    dummy_output = resdualnetv2(input_tensor)
    time_resdualnetv2 += time.time() - start

for _ in range(iters):
    start = time.time()
    dummy_output = resdualnetv1(input_tensor)
    time_resdualnetv1 += time.time() - start

for _ in range(iters):
    start = time.time()
    dummy_output = resnet(input_tensor)
    time_resnet += time.time() - start

avg_time_resdualnetv2 = time_resdualnetv2 / iters
avg_time_resdualnetv1 = time_resdualnetv1 / iters
avg_time_resnet = time_resnet / iters

print(
    f"RDV2 : {avg_time_resdualnetv2} RDV1: {avg_time_resdualnetv1} ResNet: {avg_time_resnet}"
)

# print(summary(resdualnetv2, input_size=input_size))
# print(summary(resdualnetv1, input_size=input_size))
# print(summary(resnet, input_size=input_size))
