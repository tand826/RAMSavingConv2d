import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("ram.csv")

querys = [
    ['mode == "default" & backward == False', "torch.nn.Conv2d", "solid"],
    ['mode == "ramsaving" & backward == False', "RSConv2d", "dotted"],
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
fig.savefig("ram_forward.svg")

querys = [
    ['mode == "default" & backward == True', "torch.nn.Conv2d", "solid"],
    ['mode == "ramsaving" & skip_input_grad == False & backward == True', "RSConv2d (w/ input_grad)", "dotted"],
    ['mode == "ramsaving" & skip_input_grad == True & backward == True', "RSConv2d (w/o input_grad)", "dashed"],
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
fig.savefig("ram_backward.svg")
