#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/7/27 13:50
# @Author  : HuJiwei
# @File    : respace.py
# @Software: PyCharm
# @Project: AlignDiff

import numpy as np
import torch as th

from improved_diffusion.gaussian_diffusion import GaussianDiffusion


class SpacedDiffusion(GaussianDiffusion):
    """
    A diffusion process which can skip steps in a base diffusion process，
    即基于 spaced-timesteps 的 diffusion
    """
    def __init__(self, use_timesteps, **kwargs):
        """
        :param use_timesteps: a collection (sequence or set) of timesteps from the
                            original diffusion process to retain.
        要使用（保留）的时间步，
        可能是步长为1，也可能是大于1（respaceing）
        :param kwargs: the kwargs to create the base diffusion process.
        """
        self.use_timesteps = set(use_timesteps)
        self.timestep_map = []  # 列表状态的 use_timesteps
        self.original_num_steps = len(kwargs["betas"])  # 原始 T

        base_diffusion = GaussianDiffusion(**kwargs)  # pylint: disable=missing-kwoa

        # 计算全新的采样时刻后的 betas
        last_alpha_cumprod = 1.0
        # 重新定义 betas 序列
        new_betas = []
        for i, alpha_cumprod in enumerate(base_diffusion.alphas_cumprod):
            if i in self.use_timesteps:
                new_betas.append(1 - alpha_cumprod / last_alpha_cumprod)
                last_alpha_cumprod = alpha_cumprod
                self.timestep_map.append(i)
        # 更新 self.betas 成员变量
        kwargs["betas"] = np.array(new_betas)
        super().__init__(**kwargs)

    def p_mean_variance(self, model, *args, **kwargs):  # pylint: disable=signature-differs
        return super().p_mean_variance(self._wrap_model(model), *args, **kwargs)

    def training_losses(self, model, *args, **kwargs):  # pylint: disable=signature-differs
        return super().training_losses(self._wrap_model(model), *args, **kwargs)

    def _wrap_model(self, model):
        if isinstance(model, _WrappedModel):
            return model
        return _WrappedModel(
            model, self.timestep_map, self.rescale_timesteps, self.original_num_steps
        )

    def _scale_timesteps(self, t):
        # Scaling is done by the wrapped model.
        return t


class _WrappedModel:
    """ WrappedModel 类 """

    def __init__(self, model, timestep_map, rescale_timesteps, original_num_steps):
        self.model = model
        self.timestep_map = timestep_map
        self.rescale_timesteps = rescale_timesteps
        self.original_num_steps = original_num_steps

    def __call__(self, x, ts, **kwargs):
        """
        :param ts: 连续的索引，map_tensor 中包含的是 spacing 后的索引
        """
        map_tensor = th.tensor(self.timestep_map, device=ts.device, dtype=ts.dtype)
        new_ts = map_tensor[ts]
        # 转换为真正 respacing 的时间序列，比如说从 [0,1,...,499] => [0,2,...,998]
        if self.rescale_timesteps:
            # 始终控制 new_ts 在 [0,1000] 以内的浮点数
            new_ts = new_ts.float() * (1000.0 / self.original_num_steps)
        return self.model(x, new_ts, **kwargs)


def space_timesteps(num_timesteps, section_counts):
    """
    创建一个要从原始扩散过程中使用的时间步长列表，
    给定我们希望从原始过程的同等大小部分中获取的时间步长的数量。

    例如，如果有300个时间步长，并且区间计数为[10，15，20]，
    则前100个时间步长被跨越为10个时间步长、
    第二个100被跨越为15个时间步长以及最后100被跨越至20个时间步长。

    如果步幅是以“ddim”开头的字符串，则固定步幅。

    :param num_timesteps: T
    :param section_counts: "ddimN" or a list of numbers
    :return: a set of diffusion steps from the original process to use.
        print(space_timesteps(1000, "ddim10"))
        {0, 800, 100, 900, 200, 300, 400, 500, 600, 700}
        print(space_timesteps(1000, [1,2,3]))
        {0, 833, 999, 334, 666, 667}
    """
    if isinstance(section_counts, str):
        if section_counts.startswith("ddim"):  # ddimN
            desired_count = int(section_counts[len("ddim"):])  # N
            for i in range(1, num_timesteps):
                if len(range(0, num_timesteps, i)) == desired_count:
                    return set(range(0, num_timesteps, i))
            raise ValueError(
                f"cannot create exactly {num_timesteps} steps with an integer stride"
            )
        section_counts = [int(x) for x in section_counts.split(",")]
    size_per = num_timesteps // len(section_counts)
    extra = num_timesteps % len(section_counts)
    start_idx = 0
    all_steps = []
    for i, section_count in enumerate(section_counts):
        size = size_per + (1 if i < extra else 0)
        if size < section_count:
            raise ValueError(
                f"cannot divide section of {size} steps into {section_count}"
            )
        if section_count <= 1:
            frac_stride = 1
        else:
            frac_stride = (size - 1) / (section_count - 1)
        cur_idx = 0.0
        taken_steps = []
        for _ in range(section_count):
            taken_steps.append(start_idx + round(cur_idx))
            cur_idx += frac_stride
        all_steps += taken_steps
        start_idx += size
    return set(all_steps)


def main():
    # steps = space_timesteps(100, [1])  # {0}
    # steps = space_timesteps(100, [2])  # {0, 99}
    # steps = space_timesteps(100, [1, 2])  # {0, 50, 99}
    # steps = space_timesteps(100, [1, 2, 3])  # {0, 34, 67, 66, 99, 83}
    # steps = space_timesteps(100, "ddim10")  # {0, 70, 40, 10, 80, 50, 20, 90, 60, 30}

    steps = space_timesteps(1000, "ddim50")
    """
    {
    0, 640, 260, 900, 520, 140, 780, 400, 20, 660, 
    280, 920, 540, 160, 800, 420, 40, 680, 300, 940, 
    560, 180, 820, 440, 60, 700, 320, 960, 580, 200, 
    840, 460, 80, 720, 340, 980, 600, 220, 860, 480, 
    100, 740, 360, 620, 240, 760, 880, 500, 120, 380
    }
    """
    print(steps)


if __name__ == "__main__":
    main()
