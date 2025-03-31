#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @FileName : hfns.py
# @Time     : 2024/10/10 2:47
# @Author   : HuJiwei

import torch
import torch.nn as nn
import torch.nn.functional as F


def gaussian_blur2d(input, kernel_size, sigma):
    """
    Apply a 2D Gaussian blur to the input tensor.
    """
    # Create Gaussian kernel
    kernel = create_gaussian_kernel(kernel_size, sigma, input.shape[1]).to(input.device)
    # Apply convolution
    return F.conv2d(input, kernel, padding=kernel_size[0] // 2, groups=input.shape[1])


def create_gaussian_kernel(kernel_size, sigma, channels):
    """
    Create a Gaussian kernel for 2D convolution, adjusted for multiple channels.
    """
    kx = torch.arange(kernel_size[0], dtype=torch.float32) - (kernel_size[0] - 1) / 2
    ky = torch.arange(kernel_size[1], dtype=torch.float32) - (kernel_size[1] - 1) / 2
    x, y = torch.meshgrid(kx, ky, indexing='ij')
    gaussian_kernel = torch.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    gaussian_kernel /= gaussian_kernel.sum()  # Normalize
    # Expand dimensions to create a multi-channel kernel
    return gaussian_kernel.unsqueeze(0).unsqueeze(0).repeat(channels, 1, 1, 1)  # [C_out, C_in, H, W]


class HighFreqSuppressor(nn.Module):
    def __init__(self, kernel_size=3):
        super(HighFreqSuppressor, self).__init__()
        self.kernel_size = kernel_size
        self.sigma_rate = 1.0  # 控制模糊程度的参数

    def forward(self, x):
        # 使用高斯模糊提取低频信息
        x1 = gaussian_blur2d(x, (self.kernel_size, self.kernel_size), (1 * self.sigma_rate))
        x2 = gaussian_blur2d(x, (self.kernel_size * 2 - 1, self.kernel_size * 2 - 1), (2 * self.sigma_rate))

        # 计算高频残差
        R1 = x - x1  # 高频信息
        R2 = x1 - x2  # 中频信息

        # 将残差和低频信息结合
        R_cat = torch.cat((R1, R2, x), dim=1)  # 按通道维度拼接

        return R_cat


# 使用示例
if __name__ == "__main__":
    # 假设输入特征图的尺寸为 (batch_size, channels, height, width)
    input_tensor = torch.randn(1, 1024, 32, 32)  # 示例输入
    suppressor = HighFreqSuppressor(kernel_size=3)
    output = suppressor(input_tensor)
    print("Output shape:", output.shape)  # 输出特征图的形状
