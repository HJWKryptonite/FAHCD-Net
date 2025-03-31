#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/7/27 13:46
# @Author  : HuJiwei
# @File    : losses.py
# @Software: PyCharm
# @Project: AlignDiff

import numpy as np
import torch as th


def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    计算两个高斯分布的 KL 散度
    """
    tensor = None
    for obj in (mean1, logvar1, mean2, logvar2):
        if isinstance(obj, th.Tensor):
            tensor = obj
            break
    assert tensor is not None, "at least one argument must be a Tensor"

    # Force variances to be Tensors. Broadcasting helps convert scalars to
    # Tensors, but it does not work for th.exp().
    logvar1, logvar2 = [
        x if isinstance(x, th.Tensor) else th.tensor(x).to(tensor)
        for x in (logvar1, logvar2)
    ]

    return 0.5 * (
            -1.0
            + logvar2
            - logvar1
            + th.exp(logvar1 - logvar2)
            + ((mean1 - mean2) ** 2) * th.exp(-logvar2)
    )


def approx_standard_normal_cdf(x):
    """
    计算标准分布的累积分布函数
    A fast approximation of the cumulative distribution function of the
    standard normal.
    """
    return 0.5 * (1.0 + th.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * th.pow(x, 3))))


def discretized_gaussian_log_likelihood(x, *, means, log_scales):
    """
    离散的高斯分布对数似然
    使用连续分布的累积分布的差分来模拟离散分布
    Compute the log-likelihood of a Gaussian distribution discretizing to a
    given image.

    :param x: the target images. It is assumed that this was uint8 values,
              rescaled to the range [-1, 1].
    :param means: the Gaussian mean Tensor.
    :param log_scales: the Gaussian log stddev(标准差) Tensor.
    :return: a tensor like x of log probabilities (in nats).
    """
    assert x.shape == means.shape == log_scales.shape

    # 减去均值
    centered_x = x - means
    inv_stdv = th.exp(-log_scales)

    # 将 [-1, 1] 分成 255 个 bins，最右边的 CDF 记为 1，最左边的 CDF 记为 0
    # 右侧
    plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
    cdf_plus = approx_standard_normal_cdf(plus_in)  # 右侧的累积分布函数
    # 左侧
    min_in = inv_stdv * (centered_x - 1.0 / 255.0)
    cdf_min = approx_standard_normal_cdf(min_in)  # 左侧的累积分布函数
    # 确保稳定性，不能让数据太小取不到对数
    log_cdf_plus = th.log(cdf_plus.clamp(min=1e-12))
    log_one_minus_cdf_min = th.log((1.0 - cdf_min).clamp(min=1e-12))

    # 用小范围的 CDF 之差来表示 PDF
    cdf_delta = cdf_plus - cdf_min

    # 考虑两个极限的地方
    # if x < -0.999, log_cdf_plus
    # if x > 0.999, log_one_minus_cdf_min
    # if x in [-0.999, 0.999], log(cdf_delta)
    log_probs = th.where(
        x < -0.999,
        log_cdf_plus,
        th.where(x > 0.999, log_one_minus_cdf_min, th.log(cdf_delta.clamp(min=1e-12))),
    )
    assert log_probs.shape == x.shape
    return log_probs


def main():
    pass


if __name__ == "__main__":
    main()
