from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18

from rsconv.ramsaving_conv2d import RAMSavingConv2d, conv_to_rsconv
from rsconv.normal_conv2d import NormalConv2d


@dataclass
class MODE:
    DEFAULT = "default"
    NORMAL = "normal"
    RAMSaving = "ramsaving"


@dataclass
class CONV_LAYERS:
    FIRST = ["conv1"]
    SECOND = ["layer1.0.conv1", "layer1.0.conv2", "layer1.1.conv1", "layer1.1.conv2"]
    THIRD = ["layer2.0.conv1", "layer2.0.conv2", "layer2.0.downsample.0", "layer2.1.conv1", "layer2.1.conv2"]
    FOURTH = ["layer3.0.conv1", "layer3.0.conv2", "layer3.0.downsample.0", "layer3.1.conv1", "layer3.1.conv2"]
    LAST = ["layer4.0.conv1", "layer4.0.conv2", "layer4.0.downsample.0", "layer4.1.conv1", "layer4.1.conv2"]
    ALL = FIRST + SECOND + THIRD + FOURTH + LAST


class SingleLayer(nn.Module):

    def __init__(self, mode, dtype, skip_input_grad=False, channels_per_calc=1):
        super().__init__()
        self.mode = mode

        if self.mode == MODE.RAMSaving:
            self.conv = RAMSavingConv2d(
                3, 16, 7, 3, 2, dtype=dtype, skip_input_grad=skip_input_grad, channels_per_calc=channels_per_calc)
        elif self.mode == MODE.NORMAL:
            self.conv = NormalConv2d(3, 16, 7, 2, 3, dtype=dtype)
        elif self.mode == MODE.DEFAULT:
            self.conv = torch.nn.Conv2d(3, 16, 7, 2, 3, dtype=dtype)
        else:
            raise ValueError("invalid mode")

    def forward(self, x):
        return self.conv(x)


class SmallNet(nn.Module):
    def __init__(self, mode, dtype, skip_input_grad=False, channels_per_calc=1):
        super().__init__()
        self.mode = mode
        if mode == MODE.RAMSaving:
            self.conv = RAMSavingConv2d(
                3, 16, 7, 3, 2, dtype=dtype, skip_input_grad=skip_input_grad, channels_per_calc=channels_per_calc)
        elif mode == MODE.NORMAL:
            self.conv = NormalConv2d(3, 16, 7, 3, 2, dtype=dtype)
        elif mode == MODE.DEFAULT:
            self.conv = torch.nn.Conv2d(3, 16, 7, 3, 2, dtype=dtype)
        else:
            raise ValueError("invalid mode")

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten(start_dim=1)
        self.fc = nn.Linear(16, 1, dtype=dtype)

        for layer in [self.conv, self.fc]:
            nn.init.constant_(layer.weight, 0.5)
            nn.init.constant_(layer.bias, 0.1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv(x)))
        x = self.flatten(x)
        x = self.fc(x)
        return x


class RSResNet18(nn.Module):

    def __init__(self, rsconv_layers=["conv1"], device="cuda:0", dtype=torch.float32):
        super().__init__()
        self.model = resnet18()
        self.model.fc = nn.Linear(self.model.fc.in_features, 1)
        self.model = resnet18_to_rsconvs(self.model, rsconv_layers, device, dtype)

    def forward(self, x):
        return self.model(x)


def resnet18_to_rsconvs(model, layers, device, dtype):
    convs = [n for n, layer in model.named_modules() if isinstance(layer, nn.Conv2d)]
    for layer in layers:
        if layer not in convs:
            raise ValueError(f"layer {layer} not found in model")

    if "conv1" in layers:
        model.conv1 = conv_to_rsconv(model.conv1, device=device, dtype=dtype, skip_input_grad=True)
    if "layer1.0.conv1" in layers:
        model.layer1[0].conv1 = conv_to_rsconv(model.layer1[0].conv1, device=device, dtype=dtype)
    if "layer1.0.conv2" in layers:
        model.layer1[0].conv2 = conv_to_rsconv(model.layer1[0].conv2, device=device, dtype=dtype)
    if "layer1.1.conv1" in layers:
        model.layer1[1].conv1 = conv_to_rsconv(model.layer1[0].conv1, device=device, dtype=dtype)
    if "layer1.1.conv2" in layers:
        model.layer1[1].conv2 = conv_to_rsconv(model.layer1[0].conv2, device=device, dtype=dtype)
    if "layer2.0.conv1" in layers:
        model.layer2[0].conv1 = conv_to_rsconv(model.layer2[0].conv1, device=device, dtype=dtype)
    if "layer2.0.conv2" in layers:
        model.layer2[0].conv2 = conv_to_rsconv(model.layer2[0].conv2, device=device, dtype=dtype)
    if "layer2.0.downsample.0" in layers:
        model.layer2[0].downsample[0] = conv_to_rsconv(model.layer2[0].downsample[0], device=device, dtype=dtype)
    if "layer2.1.conv1" in layers:
        model.layer2[1].conv1 = conv_to_rsconv(model.layer2[1].conv1, device=device, dtype=dtype)
    if "layer2.1.conv2" in layers:
        model.layer2[1].conv2 = conv_to_rsconv(model.layer2[1].conv2, device=device, dtype=dtype)
    if "layer3.0.conv1" in layers:
        model.layer3[0].conv1 = conv_to_rsconv(model.layer3[0].conv1, device=device, dtype=dtype)
    if "layer3.0.conv2" in layers:
        model.layer3[0].conv2 = conv_to_rsconv(model.layer3[0].conv2, device=device, dtype=dtype)
    if "layer3.0.downsample.0" in layers:
        model.layer3[0].downsample[0] = conv_to_rsconv(model.layer3[0].downsample[0], device=device, dtype=dtype)
    if "layer3.1.conv1" in layers:
        model.layer3[1].conv1 = conv_to_rsconv(model.layer3[1].conv1, device=device, dtype=dtype)
    if "layer3.1.conv2" in layers:
        model.layer3[1].conv2 = conv_to_rsconv(model.layer3[1].conv2, device=device, dtype=dtype)
    if "layer4.0.conv1" in layers:
        model.layer4[0].conv1 = conv_to_rsconv(model.layer4[0].conv1, device=device, dtype=dtype)
    if "layer4.0.conv2" in layers:
        model.layer4[0].conv2 = conv_to_rsconv(model.layer4[0].conv2, device=device, dtype=dtype)
    if "layer4.0.downsample.0" in layers:
        model.layer4[0].downsample[0] = conv_to_rsconv(model.layer4[0].downsample[0], device=device, dtype=dtype)
    if "layer4.1.conv1" in layers:
        model.layer4[1].conv1 = conv_to_rsconv(model.layer4[1].conv1, device=device, dtype=dtype)
    if "layer4.1.conv2" in layers:
        model.layer4[1].conv2 = conv_to_rsconv(model.layer4[1].conv2, device=device, dtype=dtype)
    return model
