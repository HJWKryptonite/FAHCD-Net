#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/7/26 20:49
# @Author  : HuJiwei
# @File    : unet.py
# @Software: PyCharm
# @Project: AlignDiff
from abc import abstractmethod

import numpy as np
import torch
from kornia.filters import gaussian_blur2d

from improved_diffusion.nn import *
from improved_diffusion.rrdb import RRDBNet
from utils.fp16_util import convert_module_to_f16, convert_module_to_f32
from utils.network_util import print_network


class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb):  # 对 layer 进行遍历
        """
        forward()
        :param x: x_t at each time
        :param emb: timestep_embeddings + condition_embeddings
        :return:
        """
        for layer in self:
            if isinstance(layer, TimestepBlock):
                # 如果当前层是 TimestepBlock 的对象（仅 Resblock）
                x = layer(x, emb)  # 传入 emb
            else:
                x = layer(x)
        return x


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    """

    def __init__(self, channels, use_conv, dims=2):
        super().__init__()
        self.channels = channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, channels, channels, 3, padding=1)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(
                x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest"
            )
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    """
    下采样层，带有选择卷积

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2):
        super().__init__()
        self.channels = channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)  # 如果是三维卷积，则 stride = (1, 2, 2)，空间维度不变，其他维度缩小1/2
        if use_conv:  # 如果使用卷积
            # conv(ks=3, stride=2, padding=1)
            self.op = conv_nd(dims, channels, channels, 3, stride=stride, padding=1)
        else:
            self.op = avg_pool_nd(stride)  # 平均池化

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels
    """

    def __init__(
            self,
            in_channels,  # 输入通道数
            emb_channels,  # timestep embedding 通道数
            out_channels,  # 输出通道数
            dropout,  # dropout 率
            use_conv=False,  # 是否使用 3x3(padding=1)卷积 代替 1x1 卷积在跳连接中改变 channel 数
            use_checkpoint=False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.emb_channels = emb_channels
        self.out_channels = out_channels or in_channels
        self.dropout = dropout
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint

        self.in_layers = nn.Sequential(
            normalization(self.in_channels),
            SiLU(),
            conv_nd(2, self.in_channels, self.out_channels, kernel_size=(3, 3), padding=(1, 1))
        )

        self.emb_layers = nn.Sequential(
            SiLU(),
            linear(self.emb_channels, self.out_channels)
        )

        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            SiLU(),
            nn.Dropout(p=self.dropout),
            zero_module(
                conv_nd(2, self.out_channels, self.out_channels, kernel_size=(3, 3), padding=(1, 1))
            )
        )

        if self.out_channels == self.in_channels:  # 如果输入输出通道数相同
            self.skip_connection = nn.Identity()  # 占位层，不做任何改变
        elif use_conv:  # 使用 3x3 跳连接卷积改变通道数
            self.skip_connection = conv_nd(
                2, self.in_channels, self.out_channels, kernel_size=(3, 3), padding=(1, 1)
            )
        else:  # 使用 1x1 卷积
            self.skip_connection = conv_nd(
                2, self.in_channels, self.out_channels, kernel_size=(1, 1)
            )

    def forward(self, x, emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.
        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return checkpoint(
            self._forward, (x, emb), self.parameters(), self.use_checkpoint
        )

    def _forward(self, x, emb):
        h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):  # 使 emb_ou 与 h 维度一致
            emb_out = emb_out[..., None]  # [..., None] 新增一个维度
        h = h + emb_out
        h = self.out_layers(h)
        out = self.skip_connection(x) + h
        return out


class QKVAttention(nn.Module):
    """
    A module which performs QKV attention.
    """

    def forward(self, qkv):
        """
        Apply QKV attention.
        :param qkv: an [N x (C * 3) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x C x T] tensor after attention.
        """
        ch = qkv.shape[1] // 3
        q, k, v = th.split(qkv, ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        return th.einsum("bts,bcs->bct", weight, v)

    @staticmethod
    def count_flops(model, _x, y):
        """
        A counter for the `thop` package to count the operations in an
        attention operation.

        Meant to be used like:

            macs, params = thop.profile(
                model,
                inputs=(inputs, timestamps),
                custom_ops={QKVAttention: QKVAttention.count_flops},
            )

        """
        b, c, *spatial = y[0].shape
        num_spatial = int(np.prod(spatial))
        # We perform two matmuls with the same number of ops.
        # The first computes the weight matrix, the second computes
        # the combination of the value vectors.
        matmul_ops = 2 * b * (num_spatial ** 2) * c
        model.total_ops += th.DoubleTensor([matmul_ops])


class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.

    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(self, channels, num_heads=1, use_checkpoint=False):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.use_checkpoint = use_checkpoint

        self.norm = normalization(channels)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        self.attention = QKVAttention()
        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x):
        return checkpoint(self._forward, (x,), self.parameters(), self.use_checkpoint)

    def _forward(self, x):
        b, c, *spatial = x.shape  # b, c, spatial=[H, W]
        x = x.reshape(b, c, -1)  # 将 x 变为 [B, C, HxW]
        norm_x = self.norm(x)
        qkv = self.qkv(norm_x)
        qkv = qkv.reshape(b * self.num_heads, -1, qkv.shape[2])
        h = self.attention(qkv)

        h = h.reshape(b, -1, h.shape[-1])  # 恢复形状
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)  # 带有残差的注意力


class MFAM(nn.Module):
    def __init__(self, kernel_size=3):  # channel=1024, kernel_size=3
        super().__init__()
        self.ks = kernel_size  # 3
        self.sigma_rate = 1

        params = torch.ones((4, 1), requires_grad=True)  # ([[1],[1],[1],[1]])
        self.params = nn.Parameter(params)  # 可训练的参数，初始化为 1

    def forward(self, x):
        # k=(3, 3), sigma=(1, 1)
        x1 = gaussian_blur2d(x, (self.ks, self.ks), (1 * self.sigma_rate, 1 * self.sigma_rate))
        R1 = x - x1
        # k=(5, 5), sigma=(2, 2)
        x2 = gaussian_blur2d(x, (self.ks * 2 - 1, self.ks * 2 - 1), (2 * self.sigma_rate, 2 * self.sigma_rate))
        # k=(7, 7), sigma=(3, 3)
        x3 = gaussian_blur2d(x, (self.ks * 2 + 1, self.ks * 2 + 1), (3 * self.sigma_rate, 3 * self.sigma_rate))
        R2 = x1 - x2
        R3 = x2 - x3

        R1 = R1.unsqueeze(dim=-1)  # ([1,1024,32,32,1])
        R2 = R2.unsqueeze(dim=-1)  # ([1,1024,32,32,1])
        R3 = R3.unsqueeze(dim=-1)  # ([1,1024,32,32,1])
        R_cat = torch.cat([R1, R2, R3, x.unsqueeze(dim=-1)], dim=-1)  # ([1,1024,32,32,4])

        sum_ = torch.matmul(R_cat, self.params).squeeze(dim=-1)

        return sum_  # ([1,1024,32,32])


class HighPassFilter(nn.Module):
    def __init__(self, num_channels, size=3, sigma=1.0):
        """
        self.high_pass1 = HighPassFilter(num_filters, 3, 1.0)
        :param size:
        :param sigma:
        :param num_channels:
        """
        super(HighPassFilter, self).__init__()
        x = torch.arange(-size // 2 + 1, size // 2 + 1, dtype=torch.float32)
        g = torch.exp(-x ** 2 / (2 * sigma ** 2))
        g = g / torch.sum(g)
        g = g.view(size, 1)
        self.filter = nn.Parameter(g.expand(num_channels, 1, -1, -1), requires_grad=False)

    def forward(self, x):
        return F.conv2d(
            x,
            self.filter,
            padding=(self.filter.size(2) // 2, self.filter.size(3) // 2),
            groups=x.size(1)
        )


class UNet(nn.Module):
    def __init__(self, classes_num, model_channels, channel_mult, num_res_blocks, time_embed_dim, dropout,
                 use_checkpoint, num_heads, conv_resample, attention_resolutions, first_unet):
        super().__init__()

        self.freq_adap = False
        self.edge_enhance = False

        # the Left side of the Unet
        if first_unet:
            self.input_blocks = nn.ModuleList([
                TimestepEmbedSequential(
                    conv_nd(
                        dims=2,
                        in_channels=classes_num[0] * 2,  # this[68*2] or 128
                        out_channels=model_channels,  # 128
                        kernel_size=(3, 3),
                        stride=(1, 1),
                        padding=(1, 1)
                    ),
                    ResBlockH(model_channels, model_channels),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    ResBlockH(model_channels, model_channels),
                    ResBlockH(model_channels, model_channels),
                )
            ])
        else:
            self.input_blocks = nn.ModuleList([
                TimestepEmbedSequential(
                    conv_nd(
                        dims=2,
                        in_channels=model_channels,  # 68*2 or this[128]
                        out_channels=model_channels,  # 128
                        kernel_size=(3, 3),
                        stride=(1, 1),
                        padding=(1, 1)
                    )
                )
            ])

        # 设立一个堆栈保存左侧每一层中的输入通道数，用于右侧使用
        input_block_channels = [model_channels]
        cur_channels = model_channels  # 128
        ds = 1  # 缩放尺寸

        for level, mult in enumerate(channel_mult):  # (1,1,2,2,4,4)
            for _ in range(num_res_blocks):  # 对于每一层，都有若干个 num_res_block
                layers = [ResBlock(
                    in_channels=cur_channels,  # 128
                    emb_channels=time_embed_dim,  # 512
                    out_channels=mult * model_channels,  # 1*128, 1*128, 2*128, 2*128, 4*128, 4*128
                    dropout=dropout,
                    use_checkpoint=use_checkpoint
                )]
                cur_channels = mult * model_channels  # 每一次循环通道数 1*128, 1*128, 2*128, 2*128, 4*128, 4*128

                if ds in attention_resolutions:  # 如果 ds 在 attention_resolutions 列表中
                    layers.append(AttentionBlock(
                        channels=cur_channels, use_checkpoint=use_checkpoint, num_heads=num_heads
                    ))
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                input_block_channels.append(cur_channels)  # 将此刻的 input_channel 存入堆栈

            if level != len(channel_mult) - 1:  # 每一层之后进行一次下采样，除了最下一层
                self.input_blocks.append(TimestepEmbedSequential(
                    Downsample(channels=cur_channels, use_conv=conv_resample, dims=2)
                ))
                input_block_channels.append(cur_channels)  # 将 input_channel 存入堆栈
                ds *= 2

        # the Middle side of the Unet，ResBlock + AttentionBlock + ResBlock
        self.middle_block = TimestepEmbedSequential(
            # out_channel = in_channel
            # cur_channels = 4*128
            ResBlock(
                in_channels=cur_channels, emb_channels=time_embed_dim, out_channels=None,
                dropout=dropout, use_checkpoint=use_checkpoint
            ),
            AttentionBlock(
                channels=cur_channels, use_checkpoint=use_checkpoint, num_heads=num_heads
            ),
            ResBlock(
                in_channels=cur_channels, emb_channels=time_embed_dim, out_channels=None,
                dropout=dropout, use_checkpoint=use_checkpoint
            ),
        )

        # the Right side of the Unet
        mfam_list = []
        hpf_list = []
        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:  # (4,4,2,2,1,1)
            for i in range(num_res_blocks + 1):
                layers = [ResBlock(
                    in_channels=cur_channels + input_block_channels.pop(),
                    emb_channels=time_embed_dim,
                    dropout=dropout,
                    out_channels=model_channels * mult,
                    use_checkpoint=use_checkpoint
                )]

                mfam = MFAM() if self.freq_adap else None
                mfam_list.append(mfam)

                hpf = HighPassFilter(num_channels=cur_channels) if self.edge_enhance else None
                hpf_list.append(hpf)

                cur_channels = model_channels * mult  # 128*8, 128*4, 128*2, 128*1
                if ds in attention_resolutions:
                    layers.append(AttentionBlock(
                        channels=cur_channels, use_checkpoint=use_checkpoint, num_heads=num_heads
                    ))
                if level and i == num_res_blocks:
                    layers.append(Upsample(cur_channels, conv_resample, dims=2))  # 上采样

                    mfam = MFAM() if self.freq_adap else None
                    mfam_list.append(mfam)

                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))

        self.mfam_list = nn.ModuleList(mfam_list)
        self.hpf_list = nn.ModuleList(hpf_list)

    def forward(self, x, features, emb):
        """

        :param x: ([N,128,64,64])
        :param features: ([N,128,64,64])
        :param emb: ([N,512])
        :return:
        """

        data_type = x.dtype
        hs = []
        # 1. Left
        for i, module in enumerate(self.input_blocks):
            x = module(x, emb)
            if i == 0:
                x = x + features  # 1: ([N,128,64,64])  2:
            hs.append(x)

        # 2. Middle  # x ([N,256,4,4])
        x = self.middle_block(x, emb)

        # 3. Right
        for module, mfam in zip(self.output_blocks, self.mfam_list):
            tmp = hs.pop()
            if self.freq_adap:
                tmp = mfam(tmp)
            cat_in = th.cat([x, tmp], dim=1)
            x = module(cat_in, emb)
        x = x.type(data_type)  # ([N,128,64,64])

        return x  # ([N,128,64,64])


class CDHN(nn.Module):
    """
    """

    def __init__(
            self, model_channels, channel_mult, # (1, 1, 1, 2, 2)
            num_res_blocks,  # 3
            dropout, attention_resolutions, # [16,32]
            use_checkpoint=False,
            classes_num=None, conv_resample=True,
            num_heads=4,  # 4
            rrdb_blocks=2,  # 1
            edge_info=None
    ):
        super().__init__()
        self.model_channels = model_channels  # 128
        time_embed_dim = model_channels * 4  # 512

        # classes_num: [68,13,68]
        if classes_num is None:
            classes_num = [68, 9, 68] * 4

        # time_embed: model_channel 变换到 time_embed_dim
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        # RRDB Block
        self.rrdb = RRDBNet(nb=rrdb_blocks, out_nc=model_channels)  # nb=2, out_nc=128

        self.unet_1 = UNet(
            classes_num=classes_num,
            model_channels=model_channels,
            channel_mult=channel_mult,
            num_res_blocks=num_res_blocks,
            time_embed_dim=time_embed_dim,
            dropout=dropout,
            use_checkpoint=use_checkpoint,
            num_heads=num_heads,
            conv_resample=conv_resample,
            attention_resolutions=attention_resolutions,
            first_unet=True
        )

        self.unet_2 = UNet(
            classes_num=classes_num,
            model_channels=model_channels,
            channel_mult=channel_mult,
            num_res_blocks=num_res_blocks,
            time_embed_dim=time_embed_dim,
            dropout=dropout,
            use_checkpoint=use_checkpoint,
            num_heads=num_heads,
            conv_resample=conv_resample,
            attention_resolutions=attention_resolutions,
            first_unet=False
        )

        self.unet_3 = UNet(
            classes_num=classes_num,
            model_channels=model_channels,
            channel_mult=channel_mult,
            num_res_blocks=num_res_blocks,
            time_embed_dim=time_embed_dim,
            dropout=dropout,
            use_checkpoint=use_checkpoint,
            num_heads=num_heads,
            conv_resample=conv_resample,
            attention_resolutions=attention_resolutions,
            first_unet=False
        )

        self.unet_4 = UNet(
            classes_num=classes_num,
            model_channels=model_channels,
            channel_mult=channel_mult,
            num_res_blocks=num_res_blocks,
            time_embed_dim=time_embed_dim,
            dropout=dropout,
            use_checkpoint=use_checkpoint,
            num_heads=num_heads,
            conv_resample=conv_resample,
            attention_resolutions=attention_resolutions,
            first_unet=False
        )

        # output part
        self.out = nn.Sequential(
            ResBlockH(in_channels=model_channels, out_channels=model_channels),
            ConvBlock(
                in_channels=model_channels, out_channels=model_channels,
                kernel_size=(1, 1), bn=True, relu=True
            ),
            ConvBlock(
                in_channels=model_channels, out_channels=model_channels,
                kernel_size=(1, 1), bn=False, relu=False
            ),
            Upsample(channels=128, use_conv=True, dims=2),
            ConvBlock(
                in_channels=model_channels, out_channels=classes_num[0],
                kernel_size=(1, 1), bn=False, relu=False
            )
        )

    def forward(self, x, timesteps, conditioned_image=None, heatmap=None):
        """

        :param x: ([N,68,128,128])
        :param timesteps: ([N])
        :param conditioned_image: ([N,3,256,256])
        :param heatmap: ([N,68,128,128])
        :return:
        """

        # 1. time embed
        emb = self.time_embed(
            timestep_embedding(timesteps, self.model_channels)
        )  # emb: ([N,512])

        # 2. RRDB
        # from conditioned_image extract features: ([N,128,64,64])
        features = self.rrdb(conditioned_image.type(self.inner_dtype))

        # 3. Concat h and heatmap
        h = x.type(self.inner_dtype)  # ([N,68,128,128])
        h = th.cat((h, heatmap), dim=1)  # ([N,136,128,128])

        h = self.unet_1(h, features, emb)  # ([N,128,64,64])
        h = self.unet_2(h, features, emb)  # ([N,128,64,64])
        h = self.unet_3(h, features, emb)  # ([N,128,64,64])
        h = self.unet_4(h, features, emb)  # ([N,128,64,64])

        out = self.out(h)  # ([N,68,128,128])
        return out

    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.time_embed.apply(convert_module_to_f16)
        self.rrdb.apply(convert_module_to_f16)
        self.unet_1.apply(convert_module_to_f16)
        self.unet_2.apply(convert_module_to_f16)
        self.unet_3.apply(convert_module_to_f16)
        self.unet_4.apply(convert_module_to_f16)
        self.out.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.time_embed.apply(convert_module_to_f32)
        self.rrdb.apply(convert_module_to_f32)
        self.unet_1.apply(convert_module_to_f32)
        self.unet_2.apply(convert_module_to_f32)
        self.unet_3.apply(convert_module_to_f32)
        self.unet_4.apply(convert_module_to_f32)
        self.out.apply(convert_module_to_f32)

    @property
    def inner_dtype(self):
        """
        Get the dtype used by the torso of this model
        :return: dtype
        """
        return next(self.time_embed.parameters()).dtype


def main():
    model_channels = 128
    channel_mult = (1, 1, 2, 2, 4, 4)
    channel_mult = (1, 1, 1, 2, 2)
    num_res_blocks = 3
    dropout = 0.0
    attention_resolutions = (16, 32)
    use_checkpoint = False
    classes_num = [68, 13, 68]
    conv_resample = True
    num_heads = 4
    rrdb_blocks = 1

    net = CDHN(
        model_channels, channel_mult, num_res_blocks, dropout, attention_resolutions, use_checkpoint,
        classes_num, conv_resample, num_heads, rrdb_blocks
    )
    net.cuda()

    print_network(net, "UnetModel")

    x = torch.rand([10, 68, 128, 128], device='cuda:0')
    t = torch.tensor([4, 102, 265, 375, 465, 588, 629, 735, 813, 962], device='cuda:0')

    cond = torch.rand([10, 3, 256, 256], device='cuda:0')

    heatmap = x

    result = net(x, t, cond, heatmap)
    print(result.shape)


if __name__ == "__main__":
    main()
