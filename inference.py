import time
import torch
from thop import profile
from models.resdualnetv2 import ResDualNetV2
from models.resdualnetv1 import ResDualNetV1
from models.shufflenetv1 import ShuffleNet_32
from models.resnet import ResNet18

torch.set_num_threads(1)

resdualnetv2 = ResDualNetV2().to("cpu").eval()
resdualnetv1 = ResDualNetV1().to("cpu").eval()
resnet = ResNet18().to("cpu").eval()
shufflenet = ShuffleNet_32().to("cpu").eval()

# resnet = torch.jit.script(resnet)
# resdualnetv1 = torch.jit.script(resdualnetv1)
# resdualnetv2 = torch.jit.script(resdualnetv2)

input_size = (1, 3, 32, 32)
input_tensor = torch.randn(input_size).to("cpu")

iters = 10
time_resdualnetv2: float = 0
time_resdualnetv1: float = 0
time_resnet: float = 0
time_shufflenet: float = 0

for _ in range(iters):
    start = time.time()
    dummy_output = resdualnetv2(input_tensor)
    time_resdualnetv2 += time.time() - start

with torch.jit.optimized_execution(False):
    for _ in range(iters):
        start = time.time()
        dummy_output = resdualnetv1(input_tensor)
        time_resdualnetv1 += time.time() - start

with torch.jit.optimized_execution(False):
    for _ in range(iters):
        start = time.time()
        dummy_output = shufflenet(input_tensor)
        time_shufflenet += time.time() - start

with torch.jit.optimized_execution(False):
    for _ in range(iters):
        start = time.time()
        dummy_output = resnet(input_tensor)
        time_resnet += time.time() - start

avg_time_resdualnetv2 = time_resdualnetv2 / iters
avg_time_resdualnetv1 = time_resdualnetv1 / iters
avg_time_resnet = time_resnet / iters
avg_time_shufflenet = time_shufflenet / iters

print(
    f"RDV2 : {avg_time_resdualnetv2} RDV1: {avg_time_resdualnetv1} ResNet: {avg_time_resnet} SFNet: {avg_time_shufflenet}"
)

macs, params = profile(resdualnetv1, inputs=(input_tensor,))
print(f"resdualnetv1 : MACS {macs}, Params {params}")
macs, params = profile(resdualnetv2, inputs=(input_tensor,))
print(f"resdualnetv2 : MACS {macs}, Params {params}")
macs, params = profile(resnet, inputs=(input_tensor,))
print(f"resnet : MACS {macs}, Params {params}")
macs, params = profile(shufflenet, inputs=(input_tensor,))
print(f"shufflenet : MACS {macs}, Params {params}")
