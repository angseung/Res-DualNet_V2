# LiMENet: Improving the Inference Time of Neural Networks on CPU Through Discrete Walsh-Hadamard Transform

## Descriptions
This is a PyTorch implementation of ***LiMENet: Improving the Inference Time of Neural Networks on CPU Through Discrete Walsh-Hadamard Transform*** under review on the IJCNN 2023.

Artificial intelligence is increasingly being used as a core technology in various Internet of Things services.
However, most edge devices do not have a built-in GPU, so they have difficulty accelerating neural network processing.
Therefore, lightweight technology that can quickly process neural network models using only a CPU is becoming increasingly important.
In previous works, the depthwise separable convolution has been proposed to efficiently process convolutions with two steps, the depthwise and the pointwise convolution.
In addition, a groupwise convolution has been introduced to reduce the complexity of the pointwise convolution.
However, conventional methods still rely on the multiply-accumulate operation, which is unfavorable for CPUs.
To solve this problem, we propose LiMENet, a neural network model for faster processing on edge devices using the CPU with a novel lightweight architecture.
The proposed model uses the discrete Walsh-Hadamard transform instead of the pointwise convolution because the element-wise operation is more efficient than the multiply-accumulate operation in CPUs.
Experimental results show that the proposed method has lower computational complexity and memory consumption.
Furthermore, we demonstrate the efficiency of LiMENet by measuring the actual inference time on Raspberry Pi 4B and show that the proposed model has a faster inference time of 0.807 seconds, which is a 37.6% improvement compared to ResNet18.

Our main contributions are:
* We present LiMENet which has low complexity, small model size, and fast inference time on CPU environment.
* One of two pointwise convolution layer is replaced with two depthwise convolution layers and an add layer to decrease complexity and memory usage. 
* DWHT is employed as an alternative to the other pointwise convolution layer to reduce processing time on CPU, specifically through element-wise operation.  
* We demonstrate that our model has faster inference time compared to other lightweight models on Raspberry Pi 4B and other CPUs.


## Prerequisites
- Python 3.9 or higher
- PyTorch 1.12.0 or higher

<!-- ## Usage -->
## Installation
* Clone this repo: https://anonymous.4open.science/r/limenet

* install dependencies with pip

```bash
pip install -r requirements.txt
```


* CIFAR-10 dataset will be downloaded automatically.

Then you get the directory & file structure like this:

```
|---models
|  └───limenetv1.py
|  └───limenetv2.py
|  └───resnet.py
|---warmup_scheduler
|  └───run.py
|  └───scheduler.py
|---main.py
|---requirements.txt

```
## Quick Testing
### Train LiMENet models with CIFAR-10
You can configure hyper parameters by modifying values of "config" dictionary in main.py, line 40, then run:
```bash
$ python main.py
```
