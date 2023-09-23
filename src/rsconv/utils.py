import logging

import torch


def show_ram(memo, shape=False, dtype=False):
    if shape and dtype:
        if dtype == torch.float32:
            d = 4
        elif dtype == torch.float16:
            d = 2
        else:
            raise ValueError("unknown dtype")

        size = torch.prod(torch.tensor(shape)) * d // 1000**2  # MB
        msg = f"{memo} data={size}MB: "
    else:
        msg = f"{memo}: "
    peak = torch.cuda.memory_stats('cuda').get('reserved_bytes.all.peak')
    if peak is None:
        peak = 0
    msg += f"peak {peak//1000**2} MB, "
    current = torch.cuda.memory_stats('cuda').get('reserved_bytes.all.current')
    if current is None:
        current = 0
    msg += f"current {current//1000**2} MB."
    logging.debug(msg)


def show_peak():
    peak = torch.cuda.memory_stats('cuda').get('reserved_bytes.all.peak')
    if peak is None:
        peak = 0
    msg = f"peak {peak//1000**2} MB"
    logging.debug(msg)
    return peak//1000**2
