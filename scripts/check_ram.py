from pathlib import Path
import argparse
import logging

import torch
import pandas as pd

from rsconv.model import SingleLayer
from rsconv.utils import show_peak


logging.basicConfig(level=logging.DEBUG)


def check(args):
    device = torch.device("cuda:0")

    model = SingleLayer(mode=args.mode, dtype=torch.float16, skip_input_grad=args.skip_input_grad)
    model = model.to(torch.float16)
    image = torch.randn(1, 3, args.size, args.size, dtype=torch.float16).to(device)
    model = model.to(device)

    if args.backward:
        pred = model(image).mean()
        pred.backward()
    else:
        model(image)

    peak = show_peak()
    save_result(args.mode, args.skip_input_grad, args.backward, args.size, peak)


def save_result(mode, skip_input_grad, backward, size, result):
    """
    Args:
        mode (str): "default" or "ramsaving"
        skip_input_grad (bool): True or False
        backward (bool): True or False
        size (int): size of input image for test
        result (int): result in MB
    """
    if not Path("ram.csv").exists():
        df = pd.DataFrame({}, columns=["mode", "skip_input_grad", "backward", "size", "result"])
    else:
        df = pd.read_csv("ram.csv")

    df.loc[len(df)] = [mode, skip_input_grad, backward, size, result]
    df.to_csv("ram.csv", index=False)


def main(args):
    print(args)
    check(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--size", type=int, default=24000)
    parser.add_argument("-m", "--mode", choices=["default", "ramsaving"])
    parser.add_argument("-f", "--skip_input_grad", action="store_true")
    parser.add_argument("-b", "--backward", action="store_true")
    args = parser.parse_args()

    main(args)
