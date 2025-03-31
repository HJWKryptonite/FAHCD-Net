#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/7/27 13:45
# @Author  : HuJiwei
# @File    : tensor_util.py
# @Software: PyCharm
# @Project: AlignDiff


def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    # ([N,81,128,128])
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def main():
    pass


if __name__ == "__main__":
    main()
