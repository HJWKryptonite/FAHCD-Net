#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/7/30 13:20
# @Author  : HuJiwei
# @File    : rrdb.py
# @Software: PyCharm
# @Project: AlignDiff

import functools

import torch
import torch.nn as nn

from improved_diffusion.nn import ConvBlock, ResBlockH


def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)


class ResidualDenseBlock5C(nn.Module):
    def __init__(self, nf=128, gc=32, bias=True):
        super(ResidualDenseBlock5C, self).__init__()

        # gc: growth channel, i.e. intermediate channels
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # initialization
        # mutil.initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x):
        """

        :param x: (B, 128, H, W)
        :return:
        """
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    """
    Residual-in-Residual Dense Block

    :param nf:
    :param gc:
    """

    def __init__(self, nf=1, gc=32):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock5C(nf, gc)
        self.RDB2 = ResidualDenseBlock5C(nf, gc)
        self.RDB3 = ResidualDenseBlock5C(nf, gc)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + x


class RRDBNet(nn.Module):
    def __init__(self, in_nc=3, out_nc=128, nf=128, nb=5, gc=32):
        super(RRDBNet, self).__init__()
        RRDB_block_f = functools.partial(RRDB, nf=nf, gc=gc)

        self.conv_first = nn.Sequential(
            ConvBlock(3, 64, kernel_size=(7, 7), stride=(2, 2), bn=True, relu=True),
            ResBlockH(64, 128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ResBlockH(128, 128),
            ResBlockH(128, 128)
        )

        # self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.RRDB_trunk = make_layer(RRDB_block_f, nb)  # 2
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        """
        G有一个输入2D卷积层，
        一个RRDB，其周围有残差连接，
        然后是另一个2D卷积层、
        leaky RELU激活
        和最终的2D卷积输出层。
        :param x: test_x: ([9,3,256,256])
        :return:
        """

        fea = self.conv_first(x)  # 2x128x64x64 test_fea: ([9,128,64,64])
        trunk = self.trunk_conv(self.RRDB_trunk(fea))  # test_trunk: ([9,128,64,64])
        fea = fea + trunk  # test_fea: ([9,128,64,64])
        out = self.conv_last(self.lrelu(self.HRconv(fea)))
        return out  # test_out: ([9,128,64,64])


def main():
    pass


if __name__ == "__main__":
    main()
