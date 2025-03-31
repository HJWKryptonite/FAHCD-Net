#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/10/9 19:24
# @Author  : HuJiwei
# @File    : network_util.py
# @Software: PyCharm
# @Project: AlignDiff


def print_network(model, name):
    """Print out the network information, grouped by layer."""
    print("Model name:", name)

    # 计算总参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Total params: {total_params / 1e6:.2f}M')

    # 分组统计每一层的参数
    print("\nLayer-wise parameter counts:")
    layer_params = {}

    for name, param in model.named_parameters():
        # 获取每一层的名字，去除参数名称中的 '.weight' 和 '.bias'
        layer_name = name.split('.')[0]

        # 如果该层还没有被加入字典，则初始化
        if layer_name not in layer_params:
            layer_params[layer_name] = 0

        # 累加该层的参数数量
        layer_params[layer_name] += param.numel()

    # 打印每一层的参数总量
    for layer, params in layer_params.items():
        print(f'{layer}: {params / 1e6:.2f}M')


def main():
    pass


if __name__ == "__main__":
    main()
