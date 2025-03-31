#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/03/28 10:30
# @Author  : HuJiwei
# @File    : awingLoss.py
# @Software: PyCharm
# @Project: AlignDiff_5_server

import torch
import torch.nn as nn
import torch.nn.functional as F


class AWingLoss(nn.Module):
    def __init__(self, omega=14, theta=0.5, epsilon=1, alpha=2.1, use_weight_map=True):
        super(AWingLoss, self).__init__()
        self.omega = omega
        self.theta = theta
        self.epsilon = epsilon
        self.alpha = alpha
        self.use_weight_map = use_weight_map

    def __repr__(self):
        return "AWingLoss()"

    def generate_weight_map(self, heatmap, k_size=3, w=10):
        dilate = F.max_pool2d(heatmap, kernel_size=k_size, stride=1, padding=1)
        weight_map = torch.where(dilate < 0.2, torch.zeros_like(heatmap), torch.ones_like(heatmap))
        return w * weight_map + 1

    def forward(self, output, groundtruth):
        """
        input:  b x n x h x w
        output: b x n x h x w => 1
        """
        delta = (output - groundtruth).abs()
        A = (
                self.omega
                * (1 / (1 + torch.pow(self.theta / self.epsilon, self.alpha - groundtruth)))
                * (self.alpha - groundtruth)
                * (torch.pow(self.theta / self.epsilon, self.alpha - groundtruth - 1))
                * (1 / self.epsilon)
        )  # A = omega * {1 / [1 + (theta/eps)^(alpha-y)]} * (alpha-y) * ((theta/eps)^(alpha-y-1)) * (1/eps)

        # C = thetaA - omega*ln[1 + (theta/eps)^(alpha-y)]
        C = self.theta * A - self.omega * torch.log(1 + torch.pow(self.theta / self.epsilon, self.alpha - groundtruth))

        # loss = delta < θ ? omega*ln[1 + (theta/eps)^(alpha-y)] : (A*delta - C)
        loss = torch.where(
            delta < self.theta,
            self.omega * torch.log(1 + torch.pow(delta / self.epsilon, self.alpha - groundtruth)),
            (A * delta - C)
        )
        if self.use_weight_map:
            weight = self.generate_weight_map(groundtruth)
            loss = loss * weight
        return loss.mean()


if __name__ == "__main__":
    loss_function = AWingLoss()
    y = torch.ones(2, 68, 64, 64)
    y_pred = torch.zeros(2, 68, 64, 64)
    y_pred.requires_grad_(True)
    loss = loss_function(y_pred, y)
    loss.backward()
    print(loss)
