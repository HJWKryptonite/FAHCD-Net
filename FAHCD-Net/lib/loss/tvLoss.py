#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @FileName : tvLoss.py
# @Time     : 2024/10/8 17:43
# @Author   : HuJiwei

import torch
import torch.nn as nn
import torch.nn.functional as F


class TVLoss(nn.Module):
    def __init__(self):
        super(TVLoss, self).__init__()

    def __repr__(self):
        return "TVLoss()"

    def forward(self, output):
        """
        :param output: batch x C x H x W
        :return: Total Variation Loss
        """

        # Calculate the total variation loss
        # by computing differences between neighboring pixels
        loss_h = torch.abs(output[:, :, 1:, :] - output[:, :, :-1, :])
        loss_w = torch.abs(output[:, :, :, 1:] - output[:, :, :, :-1])

        # Summing over height and width directions
        loss = torch.sum(loss_h) + torch.sum(loss_w)
        return loss.mean()


if __name__ == "__main__":
    loss_function = TVLoss()
    y_pred = torch.randn(2, 68, 64, 64)
    y_pred.requires_grad_(True)
    loss = loss_function(y_pred)
    loss.backward()
    print(loss)
