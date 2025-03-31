#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/7/27 13:43
# @Author  : HuJiwei
# @File    : gaussian_diffusion.py
# @Software: PyCharm
# @Project: AlignDiff


import enum
import math

import numpy as np
import torch as th

from lib.loss.awingLoss import AWingLoss
from lib.loss.tvLoss import TVLoss
from utils.tensor_util import mean_flat
from improved_diffusion.losses import normal_kl, discretized_gaussian_log_likelihood


def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    """
    根据 schedule_name 选择一个加噪方案
    :param schedule_name: 方案选择，"linear" or "cosine"
    :param num_diffusion_timesteps: T
    :return: betas
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of diffusion steps.
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == "cosine":
        # f(t) = cos(\frac{t/T+s}{1+s} \dot \frac{\pi}{2})^2
        # \overline{\alpha}_t = \frac{f(t)}{f(0)}
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    创建一个离散化的给定 alpha_T_Bar 函数的 β 方案，
    定义 从 t=[0,1] 开始（1-β）随时间的累积乘积
    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps  # t/T
        t2 = (i + 1) / num_diffusion_timesteps  # (t+1)/T

        # \beta_t = 1 - \frac{\overline{\alpha}_t}{\overline{\alpha}_{t-1}}
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))  # 控制上界 0.999
    return np.array(betas)


class ModelMeanType(enum.Enum):
    """
    模型预测的输出，可选项为 x_{t-1} or x_0 or ε
    """
    PREVIOUS_X = enum.auto()  # the model predicts x_{t-1}
    START_X = enum.auto()  # the model predicts x_0
    EPSILON = enum.auto()  # the model predicts epsilon


class ModelVarType(enum.Enum):
    """
    模型的输出方差，可选项为 LEARNED/FIXED_SMALL/FIXED_LARGE/LEARNED_RANGE
    预测方差、方差线性加权的权重、固定方差 betas、固定方差 betas_bar
    增加了LEARNED_RANGE选项，使模型能够预测 FIXED_SMALL 和 FIXED_LARGE 之间的值，使其工作更容易。
    """
    LEARNED = enum.auto()
    FIXED_SMALL = enum.auto()
    FIXED_LARGE = enum.auto()
    LEARNED_RANGE = enum.auto()


class LossType(enum.Enum):
    """
    损失函数类型，可选项为 MSE/RESCALED_MSE/KL/RESCALED_KL
    """
    MSE = enum.auto()  # use raw MSE loss (and KL when learning variances)
    RESCALED_MSE = (
        enum.auto()
    )  # use raw MSE loss (with RESCALED_KL when learning variances)
    KL = enum.auto()  # use the variational lower-bound
    RESCALED_KL = enum.auto()  # like KL, but rescale to estimate the full VLB

    AWING = enum.auto()  # Awing Loss

    def is_vb(self):
        return self == LossType.KL or self == LossType.RESCALED_KL


class GaussianDiffusion:
    """
    用于训练和采样扩散模型的类
    Ported directly from here, and then adapted over time to further experimentation.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py#L42

    :param betas: a 1-D numpy array of betas for each diffusion timestep,
                  starting at T and going to 1.
                  每个扩散时间步长的 β 的 1-D numpy 阵列，从 T 开始到 1。
    :param model_mean_type: 决定模型的输出
    :param model_var_type: 决定如何输出方差
    :param loss_type: 确定要使用的损失函数的类型。
    :param rescale_timesteps: if True, pass floating point timesteps into the
                              model so that they are always scaled like in the
                              original paper (0 to 1000).
    """

    def __init__(self, *, betas, model_mean_type, model_var_type, loss_type, rescale_timesteps=False):
        self.model_mean_type = model_mean_type  # 模型预测的输出
        self.model_var_type = model_var_type  # 模型预测的方差
        self.loss_type = loss_type  # 损失函数类型
        self.rescale_timesteps = rescale_timesteps  # 缩放在 (0, 1000) 内的 timeStep

        betas = np.array(betas, dtype=np.float64)  # 原始的 betas，Use float64 for accuracy.
        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1-D"  # 检查 betas 是否是一维的
        assert (betas > 0).all() and (betas <= 1).all()  # 检查 betas 中的 beta 在 (0, 1] 中

        self.num_timesteps = int(betas.shape[0])  # num_timesteps = betas 中 beta 的数目

        alphas = 1.0 - betas  # α_i = 1 - β_i
        self.alphas_cumprod = np.cumprod(alphas, axis=0)  # \bar{α}_t := A_t
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])  # \bar{α}_{t-1} := A_{t-1}
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)  # \bar{α}_{t+1} := A_{t+1}
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)  # 判断形状是否保持一致

        # 计算扩散过程 q(x_t|x_{t-1}) 的参数
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)  # \sqrt{A_t}
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)  # \sqrt{1-A_t}
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)  # log(1-A_t)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)  # 1 / \sqrt{A_t}
        self.sqrt_recip_alphas_cumprod_m1 = np.sqrt(1.0 / self.alphas_cumprod - 1)  # \sqrt{1/A_t - 1)

        # 计算后验分布 q(x_{t-1}|x_t,x_0)
        self.posterior_variance = (
                betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )  # 后验方差 \widetilde\beta_t := \beta × \frac{1-A_{t-1}}{1-A_t}

        # 对数计算被截断，因为在扩散链的开始处后验方差为 0，使用第一项代替
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )

        # 计算后验分布的均值的两个系数
        self.posterior_mean_coef1 = (
                betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )  # coef1 = \frac{\sqrt{A_{t-1} \beta_t}}{1-A_t}
        self.posterior_mean_coef2 = (
                (1.0 - self.alphas_cumprod_prev)
                * np.sqrt(alphas)
                / (1.0 - self.alphas_cumprod)
        )  # coef2 = \sqrt{\alpha_t} × \frac{1-A_{t-1}}{1-A_t}

    def q_mean_variance(self, x_start, t):
        """
        获得分布 q(x_t|x_0) = N(x_t; sqrt{A_t}x_0, (1-A_t)I) 的均值、方差和 log 方差
        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        # x_0 * sqrt{A_t}
        mean = (
                _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        )
        # (1 - A_t)
        variance = _extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        # log(1-A_t)
        log_variance = _extract_into_tensor(
            self.log_one_minus_alphas_cumprod, t, x_start.shape
        )
        return mean, variance, log_variance

    def q_sample(self, x_start, t, noise=None):
        """
        从 q(x_t|x_0) 中重参数取样，即给定 x_0 和 t 求 x_t
        :param x_start: the initial data batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :return: A noisy version of x_start, in other word, x_t
        """
        if noise is None:
            noise = th.randn_like(x_start)  # 随机(random)标准(normal)噪声
        assert noise.shape == x_start.shape
        return (
                _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
                + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
                * noise
        )  # x_t = x_0 * \sqrt{A_t} + (1-A_t) * noise

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        计算后验分布 q(x_{t-1}|x_t, x_0) 的均值和方差
        widetilde{beta_t} := beta × frac{1-A_{t-1}}{1-A_t}
        widetilde{mu_t}(x_t,x_0) := frac{sqrt{A_{t-1}beta_t}}{1-A_t}x_0 + sqrt{alpha_t} × frac{1-A_{t-1}}{1-A_t}x_t
        :param x_start: the initial data batch.
        :param x_t: the data batch at the timestep of t
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (posterior mean, posterior variance, posterior log_variance),
                 all of x_start's shape.
        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
                _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
                + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )  # tilde{mu_t}(x_t,x_0) := coef1 × x_0 + coef2 × x_t

        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )

        assert (
                posterior_mean.shape[0]
                == posterior_variance.shape[0]
                == posterior_log_variance_clipped.shape[0]
                == x_start.shape[0]
        )  # 判断形状
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(
            self, model, x, t, clip_denoised=True, denoised_fn=None, model_kwargs=None
    ):
        """
        网络预测的模型的分布 p(x_{t-1}|x_t} 的均值和方差，以及预测得到的 x_0
        即给定模型 model 和 t 时刻的数据 x，求出 t-1 时刻的均值和方差，以及 0 时刻的数据
        :param model: the model, which takes a signal and a batch of timesteps as input.
        :param x: the [N x C x ...] tensor at time t.
        :param t: a 1-D Tensor of timesteps.
        :param clip_denoised: if True, clip the denoised signal into [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample. Applies before
            clip_denoised.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict with the following keys:
                 - 'mean': the model mean output.
                 - 'variance': the model variance output.
                 - 'log_variance': the log of 'variance'.
                 - 'pred_xstart': the prediction for x_0.
        """
        if model_kwargs is None:
            model_kwargs = {}

        B, C = x.shape[:2]  # [B, C, H, W][0:2] = [B, C]
        assert t.shape == (B,)
        model_output = model(x, self._scale_timesteps(t), **model_kwargs)

        # 得到方差和对数方差
        if self.model_var_type in [ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE]:
            # 可学习的方差
            assert model_output.shape == (B, C * 2, *x.shape[2:])
            model_output, model_var_values = th.split(model_output, C, dim=1)  # 在 C 通道分割
            if self.model_var_type == ModelVarType.LEARNED:
                # 直接预测
                model_log_variance = model_var_values
                model_variance = th.exp(model_log_variance)
            else:
                # 预测方差插值系数
                # 范围 [-1, 1]
                # log(widetilde(β)_t)
                min_log = _extract_into_tensor(
                    self.posterior_log_variance_clipped, t, x.shape
                )
                # log(β_t)
                max_log = _extract_into_tensor(np.log(self.betas), t, x.shape)
                # The model_var_values is [-1, 1] for [min_var, max_var].
                frac = (model_var_values + 1) / 2  # [-1, 1] => [0, 1]

                # Sigma(theta)(x_t, t) = exp[vlog(beta_t) + (1-v)log(widetilde(beta)_t)]
                model_log_variance = frac * max_log + (1 - frac) * min_log
                model_variance = th.exp(model_log_variance)
        else:
            # 固定的方差
            model_variance, model_log_variance = {
                # for fixedlarge, we set the initial (log-)variance like so
                # to get a better decoder log likelihood.
                # 使用 beta_t，初始时刻使用 先验分布方差
                ModelVarType.FIXED_LARGE: (
                    np.append(self.posterior_variance[1], self.betas[1:]),
                    np.log(np.append(self.posterior_variance[1], self.betas[1:])),
                ),
                # 使用 先验分布方差 和 先验分布log方差
                ModelVarType.FIXED_SMALL: (
                    self.posterior_variance,
                    self.posterior_log_variance_clipped,
                ),
            }[self.model_var_type]
            model_variance = _extract_into_tensor(model_variance, t, x.shape)
            model_log_variance = _extract_into_tensor(model_log_variance, t, x.shape)

        def process_xstart(x):
            """
            对 x 进行后处理
            :param x:
            :return:
            """
            if denoised_fn is not None:
                x = denoised_fn(x)
            if clip_denoised:
                return x.clamp(-1, 1)
            return x

        if self.model_mean_type == ModelMeanType.PREVIOUS_X:
            # 预测 x_{t-1} 的期望值，即 widetilde(mu)
            pred_xstart = process_xstart(self._predict_xstart_from_xprev(x_t=x, t=t, xprev=model_output))  # x_0
            model_mean = model_output  # x{t-1}, or mu

        elif self.model_mean_type in [ModelMeanType.START_X, ModelMeanType.EPSILON]:
            # 预测 x_0 或者 epsilon
            if self.model_mean_type == ModelMeanType.START_X:
                pred_xstart = process_xstart(model_output)  # x_0
            else:
                pred_xstart = process_xstart(self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output))
            model_mean, _, _ = self.q_posterior_mean_variance(x_start=pred_xstart, x_t=x, t=t)  # posterior mean
        else:
            raise NotImplementedError(self.model_mean_type)

        assert (model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape)
        return {
            "output": model_output,
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }

    def _predict_xstart_from_eps(self, x_t, t, eps):
        """
        从 epsilon, x_t, t 预测 x_0
        x_t = sqrt(A_t)x_0 + sqrt{1-A_t}epsilon
        x_0 = (x_t -sqrt{1-A_t}epsilon) / sqrt{A_t}
        x_0 = 1/sqrt{A_t}*x_t - sqrt(1/A_t -1)*epsilon
        :param x_t: x_t
        :param t: t
        :param eps: epsilon
        :return: x_0
        """
        assert x_t.shape == eps.shape
        return (
                _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
                - _extract_into_tensor(self.sqrt_recip_alphas_cumprod_m1, t, x_t.shape) * eps
        )  # x_0 = 1/sqrt{A_t}*x_t - sqrt(1/A_t -1)*epsilon

    def _predict_xstart_from_xprev(self, x_t, t, xprev):
        """
        给出 x_t, x_{t-1} 预测 x_0
        :param x_t: x_t
        :param t: t
        :param xprev: x_{t-1}, 即 widetilde(mu)_t(x_t,x_0)
        :return: widetilde{mu}_t(x_t, x_0)
        """
        assert x_t.shape == xprev.shape
        return (
                _extract_into_tensor(1.0 / self.posterior_mean_coef1, t, x_t.shape) * xprev
                - _extract_into_tensor(
            self.posterior_mean_coef2 / self.posterior_mean_coef1, t, x_t.shape
        )
                * x_t
        )  # (xprev - coef2 * x_t) / coef1

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        """
        从 x_0, x_t, t 预测 epsilon
        x_t = sqrt{A_t}x_0 + sqrt{1-A_t}*epsilon
        (x_t - sqrt{A_t}x_0) / sqrt{1-A_t} = epsilon
        epsilon = (x_t / sqrt{A_t} - x_0) / (sqrt{1-A_t} / sqrt{A_t})
        epsilon = (x_t / sqrt{A_t} - x_0) / sqrt(1/A_t - 1)
        :param x_t: x_t
        :param t: t
        :param pred_xstart: x_0
        :return:
        """

        # epsilon = (x_t / sqrt{A_t} - x_0) / sqrt(1/A_t - 1)
        return (
                       _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
                       - pred_xstart
               ) / _extract_into_tensor(self.sqrt_recip_alphas_cumprod_m1, t, x_t.shape)

    def _scale_timesteps(self, t):
        """
        是否对 t 进行预处理
        :param t: t
        :return: 1000t/T
        """
        if self.rescale_timesteps:  # True
            return t.float() * (1000.0 / self.num_timesteps)
        return t

    def p_sample(self, model, x, t, clip_denoised=True, denoised_fn=None, model_kwargs=None):
        """
        从 x_t 采样 x_{t-1}
        :param model: the model to sample from.
        :param x: the current tensor at x_{t-1}.
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param clip_denoised: if True, clip the x_start prediction to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict containing the following keys:
                 - 'sample': a random sample from the model.
                 - 'pred_xstart': a prediction of x_0.
        """
        # 得到 t-1 时刻的均值和方差
        out = self.p_mean_variance(model, x, t, clip_denoised=clip_denoised, denoised_fn=denoised_fn,
                                   model_kwargs=model_kwargs, )

        # 根据均值和方差采样出 x_{t-1}
        noise = th.randn_like(x)
        nonzero_mask = ((t != 0).float().view(-1, *([1] * (len(x.shape) - 1))))  # no noise when t == 0
        # e^(0.5*log(var)) = e^log(sqrt(var))=sqrt(var)=标准差
        sample = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"], "output": out["output"]}

    def p_sample_loop(self, model, shape, noise=None, clip_denoised=True, denoised_fn=None, model_kwargs=None,
                      device=None, progress=False, ):
        """
        为模型生成采样
        :param model: the model module.
        :param shape: the shape of the samples, (N, C, H, W).
        :param noise: if specified, the noise from the encoder to sample.
                      Should be of the same shape as `shape`.
        :param clip_denoised: if True, clip x_start predictions to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param device: if specified, the device to create the samples on.
                       If not specified, use a model parameter's device.
        :param progress: if True, show a tqdm progress bar.
        :return: a non-differentiable batch of samples.
        """
        final = None
        for sample in self.p_sample_loop_progressive(model, shape, noise=noise, clip_denoised=clip_denoised,
                                                     denoised_fn=denoised_fn, model_kwargs=model_kwargs, device=device,
                                                     progress=progress, ):
            final = sample
        return final["output"]

    def p_sample_loop_progressive(self, model, shape, noise=None, clip_denoised=True, denoised_fn=None,
                                  model_kwargs=None, device=None, progress=False, ):
        """
        从模型中生成样本，并从扩散的每个时间步长中生成中间样本。
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.
        Arguments are the same as p_sample_loop().
        Returns a generator over dicts, where each dict is the return value of
        p_sample().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise  # x_T 时刻的标准噪音
        else:
            img = th.randn(*shape).to(device=device)

        # 对 t 进行倒序索引 [::-1] 表示逆序
        indices = list(range(self.num_timesteps))[::-1]

        # if progress:
        #     # Lazy import so that we don't depend on tqdm.
        #     from tqdm.auto import tqdm
        #     indices = tqdm(indices)

        # 循环执行 p_sample
        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            with th.no_grad():  # 不需要计算梯度
                out = self.p_sample(model, img, t, clip_denoised=clip_denoised, denoised_fn=denoised_fn,
                                    model_kwargs=model_kwargs, )
                yield out
                img = out["sample"]

    def ddim_sample(
            self, model, x, t, clip_denoised=True,
            denoised_fn=None, model_kwargs=None, eta=0.0,
    ):
        """
        使用 DDIM 从 x_t 预测 x_{t-1}

        :param model: the model to sample from.
        :param x: the current tensor at x_{t-1}.
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param clip_denoised: if True, clip the x_start prediction to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param eta: eta
        :return: a dict containing the following keys:
                 - 'sample': a random sample from the model.
                 - 'pred_xstart': a prediction of x_0.
        """
        # 获取 t-1 时刻的均值与方差
        out = self.p_mean_variance(
            model, x, t, clip_denoised=clip_denoised,
            denoised_fn=denoised_fn, model_kwargs=model_kwargs
        )

        # 根据重采样公式，计算此时预测的 eps
        eps = self._predict_eps_from_xstart(x, t, out["pred_xstart"])
        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
        alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod_prev, t, x.shape)
        sigma = (
                eta
                * th.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
                * th.sqrt(1 - alpha_bar / alpha_bar_prev)
        )  # 设置 eta=0.0 以采用 DDIM
        # σ = η x sqrt{(1-A_{τ_{i-1}})/(1-A_{τ_i})}sqrt{1-A_{τ_i}/A_{τ_{i-1}}}
        noise = th.randn_like(x)

        # x_{t-1} = sqrt{A_{t-1}}x_0 + sqrt{1-A_{t-1}-σ_t^2}ε_θ^{(t)}(x_t) + σ_tε_t
        mean_pred = (out["pred_xstart"] * th.sqrt(alpha_bar_prev) + th.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps)
        nonzero_mask = ((t != 0).float().view(-1, *([1] * (len(x.shape) - 1))))  # no noise when t == 0
        sample = mean_pred + nonzero_mask * sigma * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def ddim_reverse_sample(
            self, model, x, t, clip_denoised=True, denoised_fn=None,
            model_kwargs=None, eta=0.0,
    ):
        """
        Sample x_{t+1} from the model using DDIM reverse ODE.
        """
        assert eta == 0.0, "Reverse ODE only for deterministic path"
        out = self.p_mean_variance(
            model, x, t, clip_denoised=clip_denoised,
            denoised_fn=denoised_fn, model_kwargs=model_kwargs,
        )
        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        eps = (
                      _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x.shape) * x
                      - out["pred_xstart"]
              ) / _extract_into_tensor(self.sqrt_recip_alphas_cumprod_m1, t, x.shape)
        alpha_bar_next = _extract_into_tensor(self.alphas_cumprod_next, t, x.shape)

        # Equation 12. reversed
        mean_pred = (
                out["pred_xstart"] * th.sqrt(alpha_bar_next)
                + th.sqrt(1 - alpha_bar_next) * eps
        )

        return {"sample": mean_pred, "pred_xstart": out["pred_xstart"]}

    def ddim_sample_loop_remains_process(
            self, model, shape, noise=None, clip_denoised=True, denoised_fn=None,
            model_kwargs=None, device=None, progress=False, eta=0.0,
    ):
        """
        使用 DDIM 为模型生成采样
        :param model: the model module.
        :param shape: the shape of the samples, (N, C, H, W).
        :param noise: if specified, the noise from the encoder to sample.
                      Should be of the same shape as `shape`.
        :param clip_denoised: if True, clip x_start predictions to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param device: if specified, the device to create the samples on.
                       If not specified, use a model parameter's device.
        :param progress: if True, show a tqdm progress bar.
        :param eta: eta
        :return: a non-differentiable batch of samples.
        """
        process = []
        for sample in self.ddim_sample_loop_progressive(
                model, shape, noise=noise, clip_denoised=clip_denoised,
                denoised_fn=denoised_fn, model_kwargs=model_kwargs,
                device=device, progress=progress, eta=eta,
        ):
            process.append(sample)
        return process

    def ddim_sample_loop(
            self, model, shape, noise=None, clip_denoised=True, denoised_fn=None,
            model_kwargs=None, device=None, progress=False, eta=0.0,
    ):
        """
        使用 DDIM 为模型生成采样
        :param model: the model module.
        :param shape: the shape of the samples, (N, C, H, W).
        :param noise: if specified, the noise from the encoder to sample.
                      Should be of the same shape as `shape`.
        :param clip_denoised: if True, clip x_start predictions to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param device: if specified, the device to create the samples on.
                       If not specified, use a model parameter's device.
        :param progress: if True, show a tqdm progress bar.
        :param eta: eta
        :return: a non-differentiable batch of samples.
        """
        final = None
        for sample in self.ddim_sample_loop_progressive(
                model, shape, noise=noise, clip_denoised=clip_denoised,
                denoised_fn=denoised_fn, model_kwargs=model_kwargs,
                device=device, progress=progress, eta=eta,
        ):
            final = sample
        return final["sample"]

    def ddim_sample_loop_progressive(
            self, model, shape, noise=None, clip_denoised=True, denoised_fn=None,
            model_kwargs=None, device=None, progress=False, eta=0.0,
    ):
        """
        使用 DDIM 从模型中进行采样，并从DDIM的每个时间步长中 yield 中间采样。
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape).to(device=device)

        # 从 numsteps 中 range 出来，因此需要一个 [0-1000] 的存在 dilation 的时间序列
        indices = list(range(self.num_timesteps))[::-1]

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm
            indices = tqdm(indices)

        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            with th.no_grad():
                out = self.ddim_sample(
                    model, img, t, clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn, model_kwargs=model_kwargs, eta=eta,
                )
                yield out
                img = out["sample"]

    def _vb_terms_bpd(
            self, model, x_start, x_t, t, clip_denoised=True, model_kwargs=None
    ):
        # 需要优化的 KL 散度
        """
        得到变分下界的一个项 bit per demension
        Get a term for the variational lower-bound.

        The resulting units are bits (rather than nats, as one might expect).
        This allows for comparison to other papers.
        产生的单位是位（而不是nat，正如人们所期望的那样）。
        这允许与其他论文进行比较。

        :return: a dict with the following keys:
                 - 'output': a shape [N] tensor of NLLs or KLs.
                 - 'pred_xstart': the x_0 predictions.
        """

        # 传入 x_0, x_t, t 计算 x_{t-1} 的真实的均值和方差
        true_mean, _, true_log_variance_clipped = self.q_posterior_mean_variance(
            x_start=x_start, x_t=x_t, t=t
        )

        # 传入 x_t, t 预测的 x_{t-1} 的均值和方差
        out = self.p_mean_variance(
            model, x_t, t, clip_denoised=clip_denoised, model_kwargs=model_kwargs
        )

        # 计算 p_theta 和 q 分布的 KL 散度
        # L_{t-1} = D_{KL}(q(x_{t-1}|x_t,x_0)||p_\theta(x_{t-1}|x_t))
        kl = normal_kl(
            true_mean, true_log_variance_clipped, out["mean"], out["log_variance"]
        )  # KL
        kl = mean_flat(kl) / np.log(2.0)  # mean(KL)/log2 = bpd

        # L_0 = -log[p_\theta(x_0|x_1)]
        decoder_nll = -discretized_gaussian_log_likelihood(
            x_start, means=out["mean"], log_scales=0.5 * out["log_variance"]
        )
        assert decoder_nll.shape == x_start.shape
        decoder_nll = mean_flat(decoder_nll) / np.log(2.0)

        # At the first timestep return the decoder NLL,
        # otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
        # t=0 时刻，用离散的高斯分布计算似然
        # t>0 时刻，直接用 KL 散度
        output = th.where((t == 0), decoder_nll, kl)
        return {"output": output, "pred_xstart": out["pred_xstart"]}

    def training_losses(self, model, x_start, t, model_kwargs=None, noise=None):
        """
        Compute training losses for a single timestep.

        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs. ([N,81,128,128])
        :param t: a batch of timestep indices. ([20])
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.  # ([N,3,256,256])
        :param noise: if specified, the specific Gaussian noise to try to remove.
        :return: a dict with the key "loss" containing a tensor of shape [N].
                 Some mean or variance settings may also have other keys.
        """
        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = th.randn_like(x_start)

        # x_0, t, epsilon 采样出 x_t([N,68,128,128])
        x_t = self.q_sample(x_start, t, noise=noise)

        terms = {}

        if self.loss_type == LossType.KL or self.loss_type == LossType.RESCALED_KL:
            terms["loss"] = self._vb_terms_bpd(
                model=model,
                x_start=x_start,
                x_t=x_t,
                t=t,
                clip_denoised=False,
                model_kwargs=model_kwargs,
            )["output"]
            if self.loss_type == LossType.RESCALED_KL:
                terms["loss"] *= self.num_timesteps
            terms["output"] = None
        elif self.loss_type == LossType.MSE or self.loss_type == LossType.RESCALED_MSE:  # RESCALED_MSE(old)

            model_output = model(x_t, self._scale_timesteps(t), **model_kwargs)

            terms["output"] = model_output

            if self.model_var_type in [  # FIXED_LARGE
                ModelVarType.LEARNED,
                ModelVarType.LEARNED_RANGE,
            ]:
                B, C = x_t.shape[:2]
                assert model_output.shape == (B, C * 2, *x_t.shape[2:])
                model_output, model_var_values = th.split(model_output, C, dim=1)
                # Learn the variance using the variational bound, but don't let
                # it affect our mean prediction.
                frozen_out = th.cat([model_output.detach(), model_var_values], dim=1)
                terms["vb"] = self._vb_terms_bpd(
                    model=lambda *args, r=frozen_out: r, x_start=x_start,
                    x_t=x_t, t=t, clip_denoised=False,
                )["output"]
                if self.loss_type == LossType.RESCALED_MSE:
                    # Divide by 1000 for equivalence with initial implementation.
                    # Without a factor of 1/1000, the VB term hurts the MSE term.
                    terms["vb"] *= self.num_timesteps / 1000.0

            target = {  # target = noise
                ModelMeanType.PREVIOUS_X: self.q_posterior_mean_variance(
                    x_start=x_start, x_t=x_t, t=t
                )[0],
                ModelMeanType.START_X: x_start,
                ModelMeanType.EPSILON: noise,
            }[self.model_mean_type]  # epsilon
            assert model_output.shape == target.shape == x_start.shape
            terms["mse"] = mean_flat((target - model_output) ** 2)  # x_start 和模型输出求损失
            # terms["sum"] = (target - model_output).pow(2).sum(dim=(1, 2, 3))
            if "vb" in terms:
                terms["loss"] = terms["mse"] + terms["vb"]
            else:
                terms["loss"] = terms["mse"]

        elif self.loss_type == LossType.AWING:  # this
            model_output = model(x_t, self._scale_timesteps(t), **model_kwargs)
            terms["output"] = model_output

            target = {  # target = START_X
                ModelMeanType.PREVIOUS_X: self.q_posterior_mean_variance(
                    x_start=x_start, x_t=x_t, t=t
                )[0],
                ModelMeanType.START_X: x_start,
                ModelMeanType.EPSILON: noise,
            }[self.model_mean_type]  # START_X
            assert model_output.shape == target.shape == x_start.shape
            awl_function = AWingLoss()
            terms["awl"] = awl_function(model_output, target)
            smooth_function = TVLoss()
            terms["smooth"] = smooth_function(model_output)

            terms["kl"] = self._vb_terms_bpd(
                model=model,
                x_start=x_start,
                x_t=x_t,
                t=t,
                clip_denoised=False,
                model_kwargs=model_kwargs,
            )["output"]

            terms["kl"] *= self.num_timesteps

            l = 3e-7
            terms["loss"] = terms["awl"] + l * terms["smooth"]

        else:
            raise NotImplementedError(self.loss_type)

        return terms

    def _prior_bpd(self, x_start):
        """
        先验分布 D(KL)(q(x_T|x_0)||p(x_T))
        Get the prior KL term for the variational lower-bound, measured in
        bits-per-dim.

        This term can't be optimized, as it only depends on the encoder.

        :param x_start: the [N x C x ...] tensor of inputs.
        :return: a batch of [N] KL values (in bits), one per batch element.
        """
        batch_size = x_start.shape[0]
        t = th.tensor([self.num_timesteps - 1] * batch_size, device=x_start.device)
        qt_mean, _, qt_log_variance = self.q_mean_variance(x_start, t)
        kl_prior = normal_kl(
            mean1=qt_mean, logvar1=qt_log_variance, mean2=0.0, logvar2=0.0
        )
        return mean_flat(kl_prior) / np.log(2.0)

    def calc_bpd_loop(self, model, x_start, clip_denoised=True, model_kwargs=None):
        """
        计算整个变分下界（以 bit/dimension 为单位测量），以及其他相关数据。
        Compute the entire variational lower-bound, measured in bits-per-dim,
        as well as other related quantities.

        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param clip_denoised: if True, clip denoised samples.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.

        :return: a dict containing the following keys:
                 - total_bpd: the total variational lower-bound, per batch element.
                 - prior_bpd: the prior term in the lower-bound.
                 - vb: an [N x T] tensor of terms in the lower-bound.
                 - xstart_mse: an [N x T] tensor of x_0 MSEs for each timestep.
                 - mse: an [N x T] tensor of epsilon MSEs for each timestep.
        """
        device = x_start.device
        batch_size = x_start.shape[0]

        vb = []
        xstart_mse = []
        mse = []
        for t in list(range(self.num_timesteps))[::-1]:
            t_batch = th.tensor([t] * batch_size, device=device)
            noise = th.randn_like(x_start)
            x_t = self.q_sample(x_start=x_start, t=t_batch, noise=noise)
            # Calculate VLB term at the current timestep
            with th.no_grad():
                out = self._vb_terms_bpd(
                    model,
                    x_start=x_start,
                    x_t=x_t,
                    t=t_batch,
                    clip_denoised=clip_denoised,
                    model_kwargs=model_kwargs,
                )
            vb.append(out["output"])
            xstart_mse.append(mean_flat((out["pred_xstart"] - x_start) ** 2))
            eps = self._predict_eps_from_xstart(x_t, t_batch, out["pred_xstart"])
            mse.append(mean_flat((eps - noise) ** 2))

        vb = th.stack(vb, dim=1)
        xstart_mse = th.stack(xstart_mse, dim=1)
        mse = th.stack(mse, dim=1)

        prior_bpd = self._prior_bpd(x_start)
        total_bpd = vb.sum(dim=1) + prior_bpd
        return {
            "total_bpd": total_bpd,
            "prior_bpd": prior_bpd,
            "vb": vb,
            "xstart_mse": xstart_mse,
            "mse": mse,
        }


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    按照索引 timesteps 从 arr 中提取形状符合 broadcast_shape 形状的张量
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)
