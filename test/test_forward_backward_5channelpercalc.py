import torch

from utils import prepare


def test_fp16():
    forward_backward(dtype=torch.float16)


def test_fp32():
    forward_backward(dtype=torch.float32)


def forward_backward(dtype):
    image, label, net, optimizer, criterion = prepare(
        mode="default", dtype=dtype, skip_input_grad=False, channels_per_calc=5)
    normal_conv = {}
    optimizer.zero_grad()
    outputs = net(image)
    normal_conv["init_weight"] = net.conv.weight.mean()
    normal_conv["init_grad"] = net.conv.weight.grad
    loss = criterion(outputs.flatten(), label)
    normal_conv["output"] = outputs.mean()
    normal_conv["loss"] = loss.mean()
    loss.backward()
    optimizer.step()
    normal_conv["after_weight"] = net.conv.weight.mean()
    normal_conv["after_grad"] = net.conv.weight.grad.mean()

    image, label, net, optimizer, criterion = prepare(
        mode="ramsaving", dtype=dtype, skip_input_grad=False, channels_per_calc=5)
    cs_conv = {}
    optimizer.zero_grad()
    outputs = net(image)
    cs_conv["init_weight"] = net.conv.weight.mean()
    cs_conv["init_grad"] = net.conv.weight.grad
    loss = criterion(outputs.flatten(), label)
    cs_conv["output"] = outputs.mean()
    cs_conv["loss"] = loss.mean()
    loss.backward()
    optimizer.step()
    cs_conv["after_weight"] = net.conv.weight.mean()
    cs_conv["after_grad"] = net.conv.weight.grad.mean()

    assert normal_conv["init_weight"] == cs_conv["init_weight"]
    assert normal_conv["init_grad"] == cs_conv["init_grad"]
    assert normal_conv["output"] == cs_conv["output"]
    assert normal_conv["loss"] == cs_conv["loss"]
    assert torch.isclose(normal_conv["after_weight"], cs_conv["after_weight"])
    assert torch.isclose(normal_conv["after_grad"], cs_conv["after_grad"], atol=1e-06)
