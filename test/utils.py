import torch
import torch.nn as nn
import torch.optim as optim

from rsconv.model import SmallNet


def prepare(dtype, mode, skip_input_grad, channels_per_calc=1):
    image = torch.ones(2, 3, 256, 256, dtype=dtype).to("cuda")/255
    label = torch.tensor([1., 1.], dtype=dtype).to("cuda")

    assert mode in ["default", "ramsaving", "normal"]

    net = SmallNet(
        mode=mode, dtype=dtype, skip_input_grad=skip_input_grad, channels_per_calc=channels_per_calc).to("cuda")
    optimizer = optim.SGD(net.parameters(), lr=1., momentum=0.9)
    criterion = nn.MSELoss()

    return image, label, net, optimizer, criterion
