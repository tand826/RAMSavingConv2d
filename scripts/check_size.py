import logging
import argparse
from pathlib import Path
from multiprocessing import Process, Value

import torch
import pandas as pd

from rsconv.model import SingleLayer
from rsconv.utils import show_peak


logging.basicConfig(level=logging.DEBUG)


def check(size, args, result):
    device = torch.device("cuda:0")

    model = SingleLayer(args.mode, torch.float16, skip_input_grad=args.skip_input_grad)
    model = model.to(device)
    image = torch.randn(1, 3, size, size, dtype=torch.float16).to(device)

    try:
        if args.backward:
            pred = model(image).mean()
            pred.backward()
        else:
            model(image)

        peak = show_peak()
        result.value = 1
        save_result(args.mode, args.skip_input_grad, args.backward, size, peak, True)
    except Exception as e:
        print(e)
        with open(Path.home()/"error.log", "a") as f:
            f.write(f"mode={args.mode}, size={size}\n")
            f.write(str(e) + "\n")
        result.value = 0
        peak = show_peak()


def save_result(mode, skip_input_grad, backward, size, peak, result):
    """
    Args:
        mode (str): "default" or "ramsaving"
        skip_input_grad (bool): True or False
        backward (bool): True or False
        size (int): size of input image for test
        peak (float): peak memory usage in MB.
        result (int): result in pixels.
    """
    if not Path("size.csv").exists():
        df = pd.DataFrame({}, columns=["mode", "skip_input_grad", "backward", "size", "ram", "result"])
    else:
        df = pd.read_csv("size.csv")

    df.loc[len(df)] = [mode, skip_input_grad, backward, size, peak, result]
    df.to_csv("size.csv", index=False)


def main(args):
    print(args)
    size = args.size
    result = Value("i", 0)
    passed = 0
    history = {}

    while True:
        print(f"trying size={size}")
        if history.get(size) == 0:
            result.value = 0
        else:
            test = Process(target=check, args=(size, args, result))
            test.start()
            test.join()

        if result.value == 0:
            print("failed")
            history[size] = 0
            if args.interval < 501:
                print(f"max size: {passed}")
                break
            size -= args.interval
            args.interval //= 2
        else:
            print("passed")
            history[size] = 1
            passed = size

        if args.single:
            exit()

        result.value = 0
        size += args.interval


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--size", type=int, default=24000)
    parser.add_argument("-m", "--mode", choices=["ramsaving", "default"])
    parser.add_argument("-i", "--interval", type=int, default=8000)
    parser.add_argument("-f", "--skip_input_grad", action="store_true")
    parser.add_argument("-b", "--backward", action="store_true")
    parser.add_argument("--single", action="store_true", help="run only 1 process.")
    args = parser.parse_args()

    main(args)
