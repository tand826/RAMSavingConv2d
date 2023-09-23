from pathlib import Path
import argparse
import logging

import torch
import pandas as pd
from torchvision.models import resnet18

from rsconv.model import RSResNet18, CONV_LAYERS
from rsconv.utils import show_peak


logging.basicConfig(level=logging.DEBUG)


def check(args):
    device = torch.device("cuda:0")

    if args.mode == "default":
        model = resnet18()
    elif args.mode == "ramsaving":
        model = RSResNet18(
            rsconv_layers=CONV_LAYERS.ALL,
            device=device, dtype=torch.float16)
    else:
        raise ValueError("invalid mode")
    model = model.to(torch.float16)
    image = torch.randn(1, 3, args.size, args.size, dtype=torch.float16).to(device)
    model = model.to(device)

    if args.backward:
        pred = model(image).mean()
        pred.backward()
    else:
        model(image)

    peak = show_peak()
    save_result(args.mode, args.backward, args.size, peak)


def save_result(mode, backward, size, result):
    """
    Args:
        mode (str): "default" or "ramsaving"
        backward (bool): True or False
        size (int): size of input image for test
        result (int): result in MB
    """
    if not Path("ram_resnet18.csv").exists():
        df = pd.DataFrame({}, columns=["mode", "backward", "size", "result"])
    else:
        df = pd.read_csv("ram_resnet18.csv")

    df.loc[len(df)] = [mode, backward, size, result]
    df.to_csv("ram_resnet18.csv", index=False)


def main(args):
    print(args)
    check(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--size", type=int, default=24000)
    parser.add_argument("-m", "--mode", choices=["default", "ramsaving"])
    parser.add_argument("-b", "--backward", action="store_true")
    args = parser.parse_args()

    main(args)
