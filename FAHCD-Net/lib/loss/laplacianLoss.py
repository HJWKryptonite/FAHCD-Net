#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @FileName : laplacianLoss.py
# @Time     : 2024/10/8 19:09
# @Author   : HuJiwei

import torch
import torch.nn as nn
import torch.nn.functional as F


class LaplacianLoss(nn.Module):
    def __init__(self):
        super(LaplacianLoss, self).__init__()

    def __repr__(self):
        return "LaplacianLoss()"

    def forward(self, output):
        """
        input:  b x n x h x w
        output: Laplacian smoothing loss
        """
        # Define Laplacian kernel
        laplacian_kernel = torch.tensor(
            [[0, 1, 0], [1, -4, 1], [0, 1, 0]],
            dtype=torch.float32
        ).unsqueeze(0).unsqueeze(0)
        laplacian_kernel = laplacian_kernel.to(output.device)  # Move kernel to the same device as output

        # Apply Laplacian filter to the output
        laplacian = F.conv2d(output, laplacian_kernel, padding=1, groups=output.shape[1])

        # Compute the Laplacian smoothing loss
        loss = torch.sum(torch.pow(laplacian, 2))
        return loss.mean()
