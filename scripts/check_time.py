import logging
import argparse
from pathlib import Path
from time import time

import torch
import pandas as pd

from rsconv.model import SingleLayer


logging.basicConfig(level=logging.DEBUG)


def check(args):
    device = torch.device("cuda:0")
    model = SingleLayer(args.mode, torch.float16, args.skip_input_grad)
    model = model.to(torch.float16)

    image = torch.randn(1, 3, args.size, args.size, dtype=torch.float16).to(device)
    model = model.to(device)
    model(image)

    try:
        start = time()

        if args.backward:
            pred = model(image).mean()
            pred.backward()
        else:
            model(image)

        end = time()
        result = (end-start) * 1000
        print(f"time {result:.4f}ms")
        save_result(args.mode, args.skip_input_grad, args.backward, args.size, result)

    except Exception as e:
        print(e)
        with open(Path.home()/"error.log", "a") as f:
            f.write(f"mode={args.mode}, size={args.size}\n")
            f.write(str(e) + "\n")


def save_result(mode, skip_input_grad, backward, size, result):
    """
    Args:
        mode (str): "default" or "ramsaving"
        skip_input_grad (bool): True or False
        backward (bool): True or False
        size (int): size of input image for test
        result (float): result in miliseconds.
    """
    if not Path("time.csv").exists():
        df = pd.DataFrame({}, columns=["mode", "skip_input_grad", "backward", "size", "result"])
    else:
        df = pd.read_csv("time.csv")

    df.loc[len(df)] = [mode, skip_input_grad, backward, size, result]
    df.to_csv("time.csv", index=False)


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
