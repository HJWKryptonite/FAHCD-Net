#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/7/26 21:26
# @Author  : HuJiwei
# @File    : nn.py
# @Software: PyCharm
# @Project: AlignDiff

import math
from typing import Any, Tuple

import torch as th
import torch.nn as nn
import torch.nn.functional as F

# PyTorch 1.7 has SiLU, but we support PyTorch 1.5.
class SiLU(nn.Module):
    def forward(self, x):
        return x * th.sigmoid(x)


class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)


class ResBlockH(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        """
        (bn-relu-conv)-(bn-relu-conv)-(bn-relu-conv)+skip_layer(x)
        (64, 128) (128, 128) (128, 128)
        :param in_channels:
        :param out_channels:
        :param mid_channels: 64
        """
        super(ResBlockH, self).__init__()
        if mid_channels is None:
            mid_channels = out_channels // 2
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = ConvBlock(in_channels, mid_channels, kernel_size=(1, 1), relu=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.conv2 = ConvBlock(mid_channels, mid_channels, kernel_size=(3, 3), relu=False)
        self.bn3 = nn.BatchNorm2d(mid_channels)
        self.conv3 = ConvBlock(mid_channels, out_channels, kernel_size=(1, 1), relu=False)
        self.skip_layer = ConvBlock(in_channels, out_channels, (1, 1), relu=False)
        if in_channels == out_channels:
            self.need_skip = False
        else:
            self.need_skip = True

    def forward(self, x):
        if self.need_skip:
            residual = self.skip_layer(x)
        else:
            residual = x
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)
        out += residual
        return out


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), bn=False, relu=True, groups=1):
        """
        Conv2d + ReLU(optimal) + BatchNorm2d(optimal)
        """
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,  # 3
            out_channels,  # 64
            kernel_size,  # (7,7)
            stride,  # (2,2)
            padding=(kernel_size[0] - 1) // 2,  # 3
            groups=groups,  # 1
            bias=True
        )

        self.relu = nn.ReLU() if relu else None
        self.bn = nn.BatchNorm2d(out_channels) if bn else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class InputBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(InputBlock, self).__init__()
        planes = int(out_planes / 2)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=(1, 1), bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            in_channels=planes, out_channels=planes,
            kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
        )
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, out_planes, kernel_size=(1, 1), bias=False)

        if in_planes != out_planes:
            self.downsample = nn.Sequential(
                nn.BatchNorm2d(in_planes),
                nn.ReLU(True),
                nn.Conv2d(
                    in_planes, out_planes,
                    kernel_size=(1, 1), stride=(1, 1),
                    bias=False
                ),
            )
        else:
            self.downsample = None

    def forward(self, x):
        residual = x
        out1 = self.bn1(x)
        out1 = F.relu(out1, True)
        out1 = self.conv1(out1)
        out2 = self.bn2(out1)
        out2 = F.relu(out2, True)
        out2 = self.conv2(out2)
        out3 = self.bn3(out2)
        out3 = F.relu(out3, True)
        out3 = self.conv3(out3)
        if self.downsample is not None:
            residual = self.downsample(residual)
        out3 += residual
        return out3


def conv_nd(dims, *args, **kwargs):
    """
    Create a 1-D, 2-D or 3-D convolution module
    :param dims: determine the dimension of convolution
    :param args: args[]
    :param kwargs: kwargs{}
    :return: nn.Conv[dims]d
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}-D")


def linear(*args, **kwargs):
    """
    Create a linear module.
    """
    return nn.Linear(*args, **kwargs)


def avg_pool_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D average pooling module.
    """
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    create cos+sin timestep embeddings
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    :return:
    """
    half = dim // 2
    freqs = th.exp(
        -math.log(max_period) * th.arange(start=0, end=half, dtype=th.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = th.cat([th.cos(args), th.sin(args)], dim=-1)
    if dim % 2:
        embedding = th.cat([embedding, th.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def normalization(channels):
    """
    Make a standard normalization layer.

    :param channels: number of input channels.
    :return: an nn.Module for normalization.
    """
    return GroupNorm32(32, channels)


def update_ema(target_params, source_params, rate=0.99):
    """
    Update target parameters to be closer to those of source parameters using
    an exponential moving average.

    :param target_params: the target parameter sequence.
    :param source_params: the source parameter sequence.
    :param rate: the EMA rate (closer to 1 means slower).
    """
    for targ, src in zip(target_params, source_params):
        targ.detach().mul_(rate).add_(src, alpha=1 - rate)


def swap_ema(target_params, source_params):
    """
    Update target parameters to be closer to those of source parameters using
    an exponential moving average.

    :param target_params: the target parameter sequence.
    :param source_params: the source parameter sequence.
    """
    for targ, src in zip(target_params, source_params):
        temp = targ.data.clone()
        targ.data.copy_(src.data)
        src.data.copy_(temp)


def zero_module(module):
    """
    将模型的参数清零并返回
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def checkpoint(func, inputs, params, flag):
    """
    在不缓存中间激活的情况下评估函数，以牺牲向后传递中的额外计算为代价来减少内存。
    Evaluate a function without caching intermediate activations, allowing for
    reduced memory at the expense of extra compute in the backward pass.

    :param func: the function to evaluate.
    :param inputs: the argument sequence to pass to `func`.
    :param params: a sequence of parameters `func` depends on but does not
                   explicitly take as arguments.
    :param flag: if False, disable gradient checkpointing.
    """
    if flag:
        args = tuple(inputs) + tuple(params)
        return CheckpointFunction.apply(func, len(inputs), *args)
    else:
        return func(*inputs)


class CheckpointFunction(th.autograd.Function):
    @staticmethod
    def setup_context(ctx: Any, inputs: Tuple[Any], output: Any) -> Any:
        pass

    @staticmethod
    def jvp(ctx: Any, *grad_inputs: Any) -> Any:
        pass

    @staticmethod
    def vmap(info, in_dims, *args):
        pass

    @staticmethod
    def forward(ctx, run_function, length, *args):
        ctx.run_function = run_function
        ctx.input_tensors = list(args[:length])
        ctx.input_params = list(args[length:])
        with th.no_grad():
            output_tensors = ctx.run_function(*ctx.input_tensors)
        return output_tensors

    @staticmethod
    def backward(ctx, *output_grads):
        ctx.input_tensors = [x.detach().requires_grad_(True) for x in ctx.input_tensors]
        with th.enable_grad():
            # Fixes a bug where the first op in run_function modifies the
            # Tensor storage in place, which is not allowed for detach()'d
            # Tensors.
            shallow_copies = [x.view_as(x) for x in ctx.input_tensors]
            output_tensors = ctx.run_function(*shallow_copies)
        input_grads = th.autograd.grad(
            output_tensors,
            ctx.input_tensors + ctx.input_params,
            output_grads,
            allow_unused=True,
        )
        del ctx.input_tensors
        del ctx.input_params
        del output_tensors
        return (None, None) + input_grads


def main():
    pass


if __name__ == "__main__":
    main()
