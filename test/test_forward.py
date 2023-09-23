import torch

from utils import prepare


def test_fp16():
    forward(dtype=torch.float16)


def test_fp32():
    forward(dtype=torch.float32)


def forward(dtype):
    image, _, net, _, _ = prepare(mode="default", dtype=dtype, skip_input_grad=False)
    normal_conv = {}
    outputs = net(image)
    normal_conv["init_weight"] = net.conv.weight.mean()
    normal_conv["init_grad"] = net.conv.weight.grad
    normal_conv["output"] = outputs.mean()

    image, _, net, _, _ = prepare(mode="ramsaving", dtype=dtype, skip_input_grad=False)
    cs_conv = {}
    outputs = net(image)
    cs_conv["init_weight"] = net.conv.weight.mean()
    cs_conv["init_grad"] = net.conv.weight.grad
    cs_conv["output"] = outputs.mean()

    assert normal_conv["init_weight"] == cs_conv["init_weight"]
    assert normal_conv["init_grad"] == cs_conv["init_grad"]
    assert normal_conv["output"] == cs_conv["output"]
