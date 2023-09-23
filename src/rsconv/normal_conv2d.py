from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from rsconv.utils import show_ram


class _NormalConv2d(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, id=False):
        show_ram("forward start")
        ctx.input = input
        ctx.weight = weight
        ctx.bias = bias

        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.groups = groups
        ctx.id = id

        return F.conv2d(input, weight, bias, stride, padding, dilation, groups)

    @staticmethod
    def backward(ctx, grad_output):
        show_ram("ctx.input + grad_output")
        grad_input = torch.nn.grad.conv2d_input(
            ctx.input.shape, ctx.weight, grad_output, ctx.stride, ctx.padding,
            ctx.dilation, ctx.groups)

        show_ram("ctx.input + grad_output + grad_input")
        grad_weight = torch.nn.grad.conv2d_weight(
            ctx.input, ctx.weight.shape, grad_output, ctx.stride, ctx.padding,
            ctx.dilation, ctx.groups)

        show_ram("ctx.input + grad_output + grad_input + grad_weight")
        bias = ctx.bias
        if bias is not None:
            grad_bias = grad_output.sum((0, 2, 3)).squeeze(0)

        show_ram("ctx.input + grad_output + grad_input + grad_weight + grad_bias")
        return grad_input, grad_weight, grad_bias, None, None, None, None, None, None


class NormalConv2d(nn.Module):
    def __init__(
            self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
            padding_mode='zeros', device=None, dtype=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias

        # self.padding_mode = padding_mode  # not used
        self.device = device  # not used
        self.dtype = dtype

        self.weight = nn.Parameter(
            torch.randn(
                self.out_channels, self.in_channels, self.kernel_size, self.kernel_size, dtype=self.dtype,
                device=self.device))
        self.bias = nn.Parameter(torch.randn(self.out_channels, dtype=self.dtype, device=self.device))

        self.id = str(id(self))

    def forward(self, x):
        return _NormalConv2d.apply(
            x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups, self.id)

    def __del__(self):
        for path in Path().glob(f"*{self.id}.pth"):
            path.unlink(missing_ok=True)
