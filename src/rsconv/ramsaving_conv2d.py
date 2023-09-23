from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.modules.conv import _ConvNd

from rsconv.utils import show_ram


class _RAMSavingConv2d(torch.autograd.Function):

    @staticmethod
    def forward(
            ctx, input, weight, bias=None, stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1, padding_mode='zeros',
            id=False, skip_input_grad=False, channels_per_calc=1, save_to="cpu"):
        show_ram("forward start", input.shape, input.dtype)
        # CPU:0
        # GPU: input
        ctx.device = input.device
        ctx.dtype = input.dtype
        ctx.input_shape = input.shape
        ctx.batch = input.shape[0]
        ctx.in_channels = input.shape[1]
        ctx.padding_mode = padding_mode
        in_height = input.shape[2]
        in_width = input.shape[3]
        kernel_height = weight.shape[2]
        kernel_width = weight.shape[3]

        show_ram("forward input to cpu", input.shape, input.dtype)
        # CPU:input
        # GPU: 0
        if save_to == "disk":
            for b in range(ctx.batch):
                for c in range(ctx.in_channels):
                    torch.save(input[b:b+1, c:c+1, :, :], f"input_b{b}_inc{c}_{id}.pth")
            del input
        elif save_to == "cpu":
            ctx.input = input.cpu()
            del input
        else:
            raise ValueError("invalid save_to")

        ctx.weight = weight
        ctx.bias = bias
        ctx.out_channels = weight.shape[0]

        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.groups = groups
        ctx.id = id
        ctx.skip_input_grad = skip_input_grad
        ctx.channels_per_calc = channels_per_calc
        ctx.save_to = save_to

        bias_ = torch.zeros(ctx.out_channels, device=ctx.device, dtype=ctx.dtype)  # dummy bias
        out_height = (in_height + 2 * ctx.padding[0] - ctx.dilation[0] * (kernel_height - 1) - 1) // ctx.stride[0] + 1
        out_width = (in_width + 2 * ctx.padding[1] - ctx.dilation[1] * (kernel_width - 1) - 1) // ctx.stride[1] + 1

        # CPU: input
        # GPU: output
        output = torch.zeros(ctx.batch, ctx.out_channels, out_width, out_height, device="cpu", dtype=ctx.dtype)
        show_ram("forward output created", output.shape, output.dtype)

        # to limit RAM usage, this block cannot be run in parallel.
        for b in range(ctx.batch):
            for c in range(ctx.in_channels):
                # CPU: input
                # GPU: input(1/batch*channel) + output
                show_ram(f"forward b={b} c={c}")
                if save_to == "cpu":
                    output[b:b+1, :, :, :] += F.conv2d(
                        ctx.input[b:b+1, c:c+1, :, :].to(ctx.device),
                        weight[:, c:c+1, :, :], bias_,
                        stride, padding, dilation, groups).to("cpu")
                elif save_to == "disk":
                    output[b:b+1, :, :, :] += F.conv2d(
                        torch.load(f"input_b{b}_inc{c}_{id}.pth", map_location=ctx.device),
                        weight[:, c:c+1, :, :], bias_,
                        stride, padding, dilation, groups).to("cpu")
                else:
                    raise ValueError("unknown save_to")
                torch.cuda.empty_cache()
            if isinstance(bias, torch.Tensor):
                output[b:b+1] += bias.reshape(1, ctx.out_channels, 1, 1).to("cpu")
            show_ram("forward output done", output.shape, output.dtype)

        torch.cuda.empty_cache()
        return output.to(ctx.device)

    @staticmethod
    def backward(ctx, grad_output):
        # Originally required: grad_input + grad_output + input
        show_ram("backward grad_output", grad_output.shape, grad_output.dtype)

        # CPU: input
        # GPU: grad_output
        bias = ctx.bias
        if isinstance(bias, torch.Tensor):
            grad_bias = grad_output.sum((0, 2, 3)).squeeze(0)
            show_ram("backward grad_bias", grad_bias.shape, grad_bias.dtype)
        else:
            grad_bias = None

        # CPU: input
        # GPU: grad_output
        # grad_output = grad_output.cpu()  # this is omitted because GPU consumption is larger with this line
        torch.cuda.empty_cache()

        grad_weight = torch.zeros(ctx.weight.shape, device=ctx.device, dtype=ctx.dtype)
        for b in range(ctx.batch):
            for c in range(ctx.in_channels):
                # CPU: input
                # GPU: grad_output + input(1/batch*channel)
                if ctx.save_to == "cpu":
                    grad_weight[:, c:c+1, :, :] += torch.nn.grad.conv2d_weight(
                        ctx.input[b:b+1, c:c+1, :, :].to(ctx.device),
                        ctx.weight[:, c:c+1, :, :].shape, grad_output[b:b+1, :, :, :].to(ctx.device),
                        ctx.stride, ctx.padding, ctx.dilation, ctx.groups)
                elif ctx.save_to == "disk":
                    grad_weight[:, c:c+1, :, :] += torch.nn.grad.conv2d_weight(
                        torch.load(f"input_b{b}_inc{c}_{ctx.id}.pth", map_location=ctx.device),
                        ctx.weight[:, c:c+1, :, :].shape, grad_output[b:b+1, :, :, :].to(ctx.device),
                        ctx.stride, ctx.padding, ctx.dilation, ctx.groups)
                else:
                    raise ValueError("unknown save_to")
                torch.cuda.empty_cache()
        show_ram("backward grad_weight", grad_weight.shape, grad_weight.dtype)
        del ctx.input

        if ctx.skip_input_grad:
            return None, grad_weight, grad_bias, None, None, None, None, None, None, None, None, None

        # CPU: input
        # GPU: grad_output
        if ctx.save_to == "disk":
            for b in range(ctx.batch):
                for out_c in range(ctx.out_channels):
                    torch.save(
                        grad_output[b:b+1, out_c:out_c+1, :, :], f"grad_output_b{b}_outc{out_c}_{ctx.id}.pth")
        elif ctx.save_to == "cpu":
            # grad_output = grad_output.cpu()  # this is omitted because GPU consumption is larger with this line
            torch.cuda.empty_cache()
        else:
            raise ValueError("unknown save_to")

        # if this is not the first layer, compute grad_input
        # CPU: grad_input
        # GPU: grad_output
        grad_input = torch.zeros(ctx.input_shape, device="cpu", dtype=ctx.dtype)
        show_ram("backward grad_input created", grad_input.shape, grad_input.dtype)
        for b in range(ctx.batch):
            for out_c in range(0, ctx.out_channels, ctx.channels_per_calc):
                # CPU: grad_input
                # GPU: grad_output + grad_input
                channels_per_calc = min(ctx.channels_per_calc, ctx.out_channels - out_c)
                input_shape = torch.Size([1] + list(ctx.input_shape)[1:])
                if ctx.save_to == "cpu":
                    grad_input[b:b+1, :, :, :] += torch.nn.grad.conv2d_input(
                        input_shape, ctx.weight[out_c:out_c+channels_per_calc, :, :, :],
                        grad_output[b:b+1, out_c:out_c+channels_per_calc, :, :].to(ctx.device),
                        ctx.stride, ctx.padding, ctx.dilation, ctx.groups).to("cpu")
                elif ctx.save_to == "disk":
                    grad_input[b:b+1, :, :, :] += torch.nn.grad.conv2d_input(
                        input_shape, ctx.weight[out_c:out_c+channels_per_calc, :, :, :],
                        torch.load(f"grad_output_b{b}_outc{out_c}_{ctx.id}.pth", map_location=ctx.weight.device),
                        ctx.stride, ctx.padding, ctx.dilation, ctx.groups).to("cpu")
                else:
                    raise ValueError("unknown save_to")
                torch.cuda.empty_cache()
                show_ram(f"backward grad_input, b={b}, out_c={out_c}", grad_input.shape, grad_input.dtype)

        # CPU: 0
        # GPU: grad_output + grad_input
        return grad_input.to(ctx.device), grad_weight, grad_bias, None, None, None, None, None, None, None, None, None


class RAMSavingConv2d(_ConvNd):
    """RAM Saving Convolutional Layer for 2D inputs"""

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            groups: int = 1,
            bias: bool = True,
            padding_mode: str = 'zeros',
            device=torch.device("cpu"),
            dtype=torch.float32,
            skip_input_grad=False,
            channels_per_calc: int = 1,
            save_to: str = "cpu"):
        """
        Args:
            in_channels (int): Number of channels in the input image.
            out_channels (int): Number of channels produced by the convolution.
            kernel_size (int or tuple): Size of the convolving kernel.
            stride (int or tuple, optional): Stride of the convolution. Default: 1
            padding (int or tuple, optional): Zero-padding added to both sides of the input. Default: 0
            dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
            groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
            bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``
            padding_mode (str, optional): 'zeros', 'reflect', 'replicate' or 'circular'. Default: 'zeros'
            device (torch.device, optional): Device to use. Default: ``torch.device('cpu')``
            dtype (torch.dtype, optional): Data type to use. Default: ``torch.float32``
            skip_input_grad (bool, optional): Set ``True``, if this layer is the first layer of the network and the
                gradient is not required. Default: ``False``
            channels_per_calc (int, optional): Number of channels to compute in backward at once. Default: 1
            save_to (str, optional): Where to save the input. ``cpu`` or ``disk``. Default: ``cpu``
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = int_to_tuple(kernel_size)
        self.stride = int_to_tuple(stride)
        self.padding = int_to_tuple(padding)
        self.dilation = int_to_tuple(dilation)
        self.groups = groups
        # self.bias = bias
        self.padding_mode = padding_mode
        self.device = device
        self.dtype = dtype
        super().__init__(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            transposed=False,
            output_padding=int_to_tuple(0),
            groups=self.groups,
            bias=bias,
            padding_mode=self.padding_mode,
            device=self.device,
            dtype=self.dtype)

        self.skip_input_grad = skip_input_grad
        self.channels_per_calc = channels_per_calc
        self.save_to = save_to

        self.id = str(id(self))

    def forward(self, x):
        show_ram("start forward", x.shape, x.dtype)
        if self.padding_mode != 'zeros':
            return _RAMSavingConv2d.apply(
                F.pad(x, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups, self.padding_mode,
                self.id, self.skip_input_grad, self.channels_per_calc, self.save_to)
        return _RAMSavingConv2d.apply(
            x,
            self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups, self.padding_mode,
            self.id, self.skip_input_grad, self.channels_per_calc, self.save_to)

    def __del__(self):
        for path in Path().glob(f"*{self.id}.pth"):
            path.unlink(missing_ok=True)

    def __str__(self):
        msg = "RAMSavingConv2d("
        msg += f"{self.in_channels}, "
        msg += f"{self.out_channels}, "
        msg += f"kernel_size={self.kernel_size}, "
        msg += f"stride={self.stride}, "
        msg += f"padding={self.padding}, "
        msg += f"bias={self.bias is not None}, "
        msg += f"skip_input_grad={self.skip_input_grad}, "
        msg += f"channels_per_calc={self.channels_per_calc}, "
        msg += f"save_to={self.save_to}"
        msg += ")"
        return msg

    def __repr__(self):
        return str(self)


def conv_to_rsconv(conv, skip_input_grad=False, device=torch.device("cpu"), dtype=torch.float32):
    return RAMSavingConv2d(
        in_channels=conv.in_channels,
        out_channels=conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=conv.bias is not None,
        padding_mode=conv.padding_mode,
        device=device,
        dtype=dtype,
        skip_input_grad=skip_input_grad)


def int_to_tuple(x):
    if isinstance(x, int):
        return (x, x)
    return x
