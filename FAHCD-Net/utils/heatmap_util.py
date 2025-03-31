#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/6/2 19:52
# @Author  : HuJiwei
# @File    : heatmap_util.py
# @Software: PyCharm
# @Project: AlignDiff_8_stack_cascade
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def draw_gaussian(img, pt, sigma, type='Gaussian'):
    # Draw a 2D gaussian
    # Adopted from https://github.com/anewell/pose-hg-train/blob/master/src/pypose/draw.py

    # Check that any part of the gaussian is in-bounds
    ul = [int(pt[0] - 3 * sigma), int(pt[1] - 3 * sigma)]
    br = [int(pt[0] + 3 * sigma + 1), int(pt[1] + 3 * sigma + 1)]
    if (ul[0] >= img.shape[1] or ul[1] >= img.shape[0] or
            br[0] < 0 or br[1] < 0):
        # If not, just return the image as is
        return torch.from_numpy(img)

    # Generate gaussian
    size = 6 * sigma + 1
    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    # The gaussian is not normalized, we want the center value to equal 1
    if type == 'Gaussian':
        g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
    elif type == 'Cauchy':
        g = sigma / (((x - x0) ** 2 + (y - y0) ** 2 + sigma ** 2) ** 1.5)

    # Usable gaussian range
    g_x = max(0, -ul[0]), min(br[0], img.shape[1]) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], img.shape[0]) - ul[1]
    # Image range
    img_x = max(0, ul[0]), min(br[0], img.shape[1])
    img_y = max(0, ul[1]), min(br[1], img.shape[0])

    img[img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
    return img


def heatmap2landmarks(heatmap):
    landmark_num = heatmap.size(0)
    max_n1, index_n1 = torch.max(heatmap, 2)
    max_n2, index_n2 = torch.max(max_n1, 1)
    landmarks = torch.FloatTensor(landmark_num * 2).to(heatmap)
    for i in range(int(landmark_num)):
        landmarks[2 * i + 1] = index_n2[i]
        landmarks[2 * i + 0] = index_n1[i, index_n2[i]]
    return landmarks


def main():
    sample = dict()

    tsv_file = "test/test.tsv"
    items = pd.read_csv(tsv_file, sep="\t")  # 读取tsv文件，含有标签信息

    pts_name = "test/test.pts"
    add_pts_name = "test/test_.pts"  # "test_.pts"




if __name__ == "__main__":
    main()
