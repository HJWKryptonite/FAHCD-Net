#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @FileName : hgLoss.py
# @Time     : 2024/10/8 19:13
# @Author   : HuJiwei


import torch
import torch.nn as nn

class HeatmapGradientLoss(nn.Module):
    def __init__(self):
        super(HeatmapGradientLoss, self).__init__()

    def __repr__(self):
        return "HeatmapGradientLoss()"

    def forward(self, heatmap):
        """
        input:  b x n x h x w
        output: Heatmap gradient-based loss
        """
        # Compute gradients in the x and y directions using finite differences
        gradient_x = heatmap[:, :, :, 1:] - heatmap[:, :, :, :-1]  # Difference in the x-direction
        gradient_y = heatmap[:, :, 1:, :] - heatmap[:, :, :-1, :]  # Difference in the y-direction

        # Compute the gradient loss as the sum of squared gradients
        loss = torch.sum(gradient_x**2) + torch.sum(gradient_y**2)
        return loss.mean()
