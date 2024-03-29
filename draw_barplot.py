import numpy as np
from matplotlib import pyplot as plt

num_params = np.array([11.174, 1.446, 0.777])
mac_operations = np.array([0.5554, 0.0738, 0.0437])
acc_test = np.array([93.91, 92.67, 92.18])
netscore = np.array([70.98, 88.39, 93.27])

x_axis = [1, 2, 3]
labels = ["ResNet18", "ResNet18 with DSConv", "Res-DualNet (ours)"]
labels = ["", "", ""]

fig = plt.figure(figsize=(20, 8))
plt.subplot(141)
plt.bar(x_axis[0], num_params[0], align="center", color="r", label="ResNet18")
plt.bar(
    x_axis[1], num_params[1], align="center", color="g", label="ResNet18 with DSConv"
)
plt.bar(x_axis[2], num_params[2], align="center", color="b", label="Res-DualNet (ours)")
plt.axhline(num_params[1], linestyle="--", color="k")
plt.axhline(num_params[2], linestyle="--", color="k")
plt.xticks(x_axis, labels, rotation=0)
plt.ylim([0, 2.0])
plt.ylabel("MAC (G)", fontsize="xx-large")
plt.yticks(fontsize="xx-large")
plt.ylabel("The number of parameters (M)", fontsize="xx-large")
plt.title("Total Network Parameters (M)", fontsize="xx-large")

plt.subplot(142)
plt.bar(x_axis[0], mac_operations[0], align="center", color="r", label="ResNet18")
plt.bar(
    x_axis[1],
    mac_operations[1],
    align="center",
    color="g",
    label="ResNet18 with DSConv",
)
plt.bar(
    x_axis[2], mac_operations[2], align="center", color="b", label="Res-DualNet (ours)"
)
plt.axhline(mac_operations[1], linestyle="--", color="k")
plt.axhline(mac_operations[2], linestyle="--", color="k")
plt.xticks(x_axis, labels, rotation=0)
plt.yticks(fontsize="xx-large")
plt.ylim([0, 0.1])
plt.ylabel("MAC (G)", fontsize="xx-large")
plt.title("Total MAC (G)", fontsize="xx-large")

plt.subplot(143)
plt.bar(x_axis[0], acc_test[0], align="center", color="r", label="ResNet18")
plt.bar(x_axis[1], acc_test[1], align="center", color="g", label="ResNet18 with DSConv")
plt.bar(x_axis[2], acc_test[2], align="center", color="b", label="Res-DualNet (ours)")
plt.xticks(x_axis, labels, rotation=0)
plt.ylabel("MAC (G)", fontsize="xx-large")
plt.axhline(acc_test[1], linestyle="--", color="k")
plt.axhline(acc_test[2], linestyle="--", color="k")
plt.ylabel("Test Accuracy (%)", fontsize="xx-large")
plt.ylim([90, 96])
plt.yticks(fontsize="xx-large")
plt.title("Test Accuracy (%)", fontsize="xx-large")
plt.legend(loc="lower center", fontsize="xx-large")

plt.subplot(144)
plt.bar(x_axis[0], netscore[0], align="center", color="r", label="ResNet18")
plt.bar(x_axis[1], netscore[1], align="center", color="g", label="ResNet18 with DSConv")
plt.bar(x_axis[2], netscore[2], align="center", color="b", label="Res-DualNet (ours)")
plt.axhline(netscore[1], linestyle="--", color="k")
plt.axhline(netscore[2], linestyle="--", color="k")
plt.xticks(x_axis, labels, rotation=0)
plt.ylim([60, 100])
plt.yticks(fontsize="xx-large")
plt.ylabel("NetScore", fontsize="xx-large")
plt.title("NetScore", fontsize="xx-large")

plt.tight_layout()
plt.show()

fig_cos = plt.figure()
epoch = np.array(range(1, 201))
lr = 0.5 * (1 + np.cos(np.pi * epoch / 200))
plt.plot(epoch, lr)
plt.xlabel("Epoch", fontsize="large")
plt.ylabel("Learning Rate Scaling Factor", fontsize="large")
plt.grid(True)
plt.title("CosAnnealing Learning Rate Scheduler", fontsize="x-large")
plt.show()

fig2 = plt.figure()
x = np.linspace(-10, 10, 10000)
y = x * (1 / (1 + np.exp(-x)))
plt.plot(x, y)
plt.grid(True)
plt.xlim([-10, 10])
plt.ylim([-0.5, 10])
plt.title("Swish Activation function")
plt.show()
fig2.savefig("swish.png", dpi=300)
