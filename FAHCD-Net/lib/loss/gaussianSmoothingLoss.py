#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @FileName : gaussianSmoothingLoss.py
# @Time     : 2024/10/8 19:10
# @Author   : HuJiwei

import torch
import torch.nn as nn
import torch.nn.functional as F


class GaussianSmoothingLoss(nn.Module):
    def __init__(self, sigma=1.0):
        super(GaussianSmoothingLoss, self).__init__()
        self.sigma = sigma

    def __repr__(self):
        return "GaussianSmoothingLoss()"

    def forward(self, output):
        """
        input:  b x n x h x w
        output: Gaussian smoothing loss
        """
        # Create Gaussian kernel based on the given sigma
        kernel_size = int(2 * (3 * self.sigma) + 1)  # Kernel size depends on sigma
        gaussian_kernel = self.create_gaussian_kernel(kernel_size, self.sigma).to(output.device)

        # Apply Gaussian filter to the output
        smoothed = F.conv2d(output, gaussian_kernel, padding=kernel_size // 2, groups=output.shape[1])

        # Calculate the Gaussian smoothing loss as the difference between original and smoothed images
        loss = torch.sum(torch.pow(output - smoothed, 2))
        return loss.mean()

    def create_gaussian_kernel(self, kernel_size, sigma):
        """
        Creates a 2D Gaussian kernel using the given kernel size and standard deviation (sigma).
        """
        ax = torch.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
        xx, yy = torch.meshgrid(ax, ax)
        kernel = torch.exp(-(xx ** 2 + yy ** 2) / (2. * sigma ** 2))
        kernel = kernel / torch.sum(kernel)
        kernel = kernel.unsqueeze(0).unsqueeze(0)  # Reshape to 4D tensor
        return kernel
