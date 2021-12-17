import numpy as np
from matplotlib import pyplot as plt

num_params = np.array([11.174, 1.446, 0.777, 0.777])
mac_operations = np.array([0.5554, 0.0738, 0.0437, 0.0437])
acc_test = np.array([93.91, 92.67, 91.34, 92.18])
netscore = np.array([70.98, 88.39, 93.12, 93.27])

x_axis = [1, 2, 3, 4]
labels = ["ResNet18", "ResNet18 with DSConv", "Res-DualNet(ReLU, ours)", "Res-DualNet(Swish, ours)"]
labels = ["A", "B", "C", "D"]

fig = plt.figure(figsize=(16, 6))
plt.subplot(141)
plt.bar(x_axis, num_params, align="center")
plt.axhline(num_params[0], linestyle="--", color="r")
plt.axhline(num_params[3], linestyle="--", color="r")
plt.xticks(x_axis, labels, rotation=0)
plt.ylabel("The number of parameters (M)")
plt.title("Total Network Parameters (M)")

plt.subplot(142)
plt.bar(x_axis, mac_operations, align="center")
plt.axhline(mac_operations[0], linestyle="--", color="r")
plt.axhline(mac_operations[3], linestyle="--", color="r")
plt.xticks(x_axis, labels, rotation=0)
plt.ylabel("MAC (G)")
plt.title("Total MAC (G)")

plt.subplot(143)
plt.bar(x_axis, acc_test, align="center")
plt.xticks(x_axis, labels, rotation=0)
plt.ylabel("Test Accuracy (%)")
plt.ylim([0, 100])
plt.title("Test Accuracy (%)")

plt.subplot(144)
plt.bar(x_axis, netscore, align="center")
plt.axhline(netscore[0], linestyle="--", color="r")
plt.axhline(netscore[3], linestyle="--", color="r")
plt.xticks(x_axis, labels, rotation=0)
plt.ylabel("NetScore")
plt.title("NetScore")

plt.tight_layout()
plt.show()
