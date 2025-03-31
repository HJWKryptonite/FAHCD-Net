#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/7/26 12:33
# @Author  : HuJiwei
# @File    : seed_generator.py
# @Software: PyCharm
# @Project: AlignDiff
import random

import numpy as np
import torch


def set_random_seed(seed, deterministic=False):
    """
    Set random seed.
    :param seed: int, seed to be used
    :param deterministic:
        bool, whether to set the deterministic option for CUDNN backend,
        i.e., set "torch.backends.cudnn.deterministic" to True
        and "torch.backends.cudnn.benchmark" to False.
        Default: False
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def set_random_seed_for_iterations(seed):
    """
    Set random seed.
    :param seed: int, seed to be used
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
