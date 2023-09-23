import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("ram_resnet18.csv")

querys = [
    ['mode == "default" & backward == False', "torchvision.models.resnet18", "solid"],
    ['mode == "ramsaving" & backward == False', "RSResNet", "dotted"],
]

fig = plt.figure(dpi=200)
ax = fig.add_subplot(111)

for query, label, line in querys:
    res = df.query(query)
    xs = res["size"]
    ys = res["result"]
    ax.plot(xs, ys, label=label, linestyle=line)
    ax.set_xlabel("size (px)")
    ax.set_ylabel("RAM consumption (MB)")
    ax.legend()
    ax.grid(color='b', linestyle=':', linewidth=0.2)
fig.tight_layout()
fig.savefig("ram_resnet18_forward.svg")

querys = [
    ['mode == "default" & backward == True', "torchvision.models.resnet18", "solid"],
    ['mode == "ramsaving" & backward == True', "RSResNet", "dotted"],
]

fig = plt.figure(dpi=200)
ax = fig.add_subplot(111)

for query, label, line in querys:
    res = df.query(query)
    xs = res["size"]
    ys = res["result"]
    ax.plot(xs, ys, label=label, linestyle=line)
    ax.set_xlabel("size (px)")
    ax.set_ylabel("RAM consumption (MB)")
    ax.legend()
    ax.grid(color='b', linestyle=':', linewidth=0.2)
fig.tight_layout()
fig.savefig("ram_resnet18_backward.svg")
