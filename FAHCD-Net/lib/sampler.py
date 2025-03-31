#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @FileName : sampler.py
# @Time     : 2024/9/11 0:06
# @Author   : HuJiwei
import cv2
import numpy as np
import torch
import torch as th
from matplotlib import pyplot as plt

from lib.metric.NME import NME
from utils import dist_util


class Sampler(object):
    def __init__(
            self, config, model, diffusion, loader, logger
    ):
        self.config = config
        self.model = model
        self.diffusion = diffusion
        self.loader = loader
        self.logger = logger

        self.save_path = "D:\\Workspace\\Python\\Diffusion\\AlignDiff_8_stack_cascade\\output"

        self.metrics = NME(self.config.nme_left_index, self.config.nme_right_index)

    def default(self):

        with torch.no_grad():
            dataset_size = len(self.loader.dataset)  # dataset_size =
            batch_size = self.loader.batch_size
            batch_num = max(dataset_size / max(batch_size, 1), 1)  # batch_num =
            self.logger.log("Default")

            for i, sample in enumerate(self.loader):
                pass

    def sample(self, mode="predict"):
        self.model = self.model.float().to(dist_util.dev())
        self.model.eval()
        with torch.no_grad():
            dataset_size = len(self.loader.dataset)  # dataset_size =
            batch_size = self.loader.batch_size
            batch_num = max(dataset_size / max(batch_size, 1), 1)  # batch_num =
            self.logger.log(
                "The " + mode + " process, Dataset size: %d, Batch size: %d" % (dataset_size, batch_size)
            )

            for i, sample in enumerate(self.loader):

                # ================ Get Data =============== #
                # ([1,3,256,256}) Image
                cond = sample["data"].float().to(dist_util.dev(), non_blocking=True)
                # ([1,68,128,128]) 标签heatmap
                batch = sample["label"][1].float().to(dist_util.dev(), non_blocking=True)
                # ([1,68,128,128]) 条件heatmap
                heatmap = sample["heatmap"].float().to(dist_util.dev(), non_blocking=True)
                image_name = sample["image_name"][0]
                img_rgb = sample["rgb"]
                labels = sample["label"]

                # ================ Test ================ #
                model_kwargs = {
                    "conditioned_image": cond,
                    "heatmap": heatmap
                }

                sample_fn = self.diffusion.ddim_sample_loop
                if mode == "process":
                    sample_fn = self.diffusion.ddim_sample_loop_remains_process
                output = sample_fn(
                    self.model,
                    (
                        batch.shape[0], batch.shape[1],
                        self.config.heatmap_size, self.config.heatmap_size
                    ),
                    progress=True,
                    clip_denoised=self.config.clip_denoised,
                    model_kwargs=model_kwargs
                )
                print(output.shape)

                metrics = self.train_metrics(
                    output[:, :self.config.classes_num[0], :, :],
                    labels
                )

                if mode == "default":
                    pass
                if mode == "lmk":

                    pass
                if mode == "predict":
                    # 预测，打印热图

                    self.save_image(
                        img_rgb[0],
                        save_path=self.save_path + "\\predict\\" + image_name + ".png"
                    )

                    self.save_heatmap(
                        output[0],
                        save_path=self.save_path + "\\predict\\" + image_name + "_heatmap.png"
                    )

                    if self.logger is not None:
                        self.logger.log(
                            "Predict Process Start: %d/%d" % (
                                i, batch_num
                            )
                        )
                if mode == "exp":
                    combined_heatmap = self.save_heatmap(
                        batch[0],
                        save_path=self.save_path + "\\exp\\" + image_name + ".png"
                    )
                    self.exp(
                        combined_heatmap,
                        save_dir=self.save_path + "\\exp\\" + image_name,
                        metrics=None
                    )

                    combined_heatmap = self.save_heatmap(
                        output[0],
                        save_path=self.save_path + "\\exp\\" + image_name + "_output.png"
                    )
                    self.exp(
                        combined_heatmap,
                        save_dir=self.save_path + "\\exp\\" + image_name + "_output",
                        metrics=metrics
                    )
                    if self.logger is not None:
                        self.logger.log(
                            "Exp Process Start: %d/%d" % (
                                i, batch_num
                            )
                        )

                if mode == "process":
                    # 扩散过程
                    self.save_heatmap(
                        output[0]["sample"][0],
                        save_path=self.save_path + "\\process\\" + image_name + "_" + str(0) + ".png"
                    )

                    for i in range(45, 50):
                        self.save_heatmap(
                            output[i]["sample"][0],
                            save_path=self.save_path + "\\process\\" + image_name + "_" + str(i) + ".png"
                        )
                    if self.logger is not None:
                        self.logger.log(
                            "Process Process Start: %d/%d" % (
                                i, batch_num
                            )
                        )

                if mode == "generate_psd_all":
                    combined_heatmap = self.save_heatmap(
                        batch[0], save_path=self.save_path + "\\psd\\" + image_name + ".png"
                    )
                    self.gen_psd_from_heatmap(
                        combined_heatmap, save_path=self.save_path + "\\psd\\" + image_name + "_psd.png"
                    )

                    comb_heatmap_output = self.save_heatmap(
                        output[0], save_path=self.save_path + "\\psd\\" + image_name + "_output.png"
                    )

                    self.gen_psd_from_heatmap(
                        comb_heatmap_output, save_path=self.save_path + "\\psd\\" + image_name + "_output_psd.png"
                    )

                    if self.logger is not None:
                        self.logger.log(
                            "PSD Process Start: %d/%d" % (
                                i, batch_num
                            )
                        )

                if mode == "generate_psd_single":
                    pass
                if mode == "generate_psd_multi":
                    pass

    def generate_psd_single(self):
        """
        关注遮挡点的 psd
        :return:
        """
        self.model = self.model.float().to(dist_util.dev())
        self.model.eval()
        with torch.no_grad():
            dataset_size = len(self.loader.dataset)  # dataset_size =
            batch_size = self.loader.batch_size  # batch_size =
            batch_num = max(dataset_size / max(batch_size, 1), 1)  # batch_num =
            self.logger.log(
                "Generate process, Dataset size: %d, Batch size: %d" % (dataset_size, batch_size)
            )

            for i, sample in enumerate(self.loader):
                heatmap = sample["heatmap"].float().to(dist_util.dev(), non_blocking=True)
                # ([1,3,256,256})
                cond = sample["data"].float().to(dist_util.dev(), non_blocking=True)
                # ([])
                batch = sample["label"][1].float().to(dist_util.dev(), non_blocking=True)
                image_name = sample["image_name"][0]

                img_rgb = sample["rgb"]
                # self.show_heatmap_on_image(heatmap[0], img_rgb[0])
                combined_heatmap = self.show_heatmap_range(
                    heatmap[0], range_l=0, range_r=1,
                    save_path=self.save_path + "\\occlusion\\" + image_name + ".png"
                )

                self.gen_psd_from_heatmap(
                    combined_heatmap, save_path=self.save_path + "\\occlusion\\" + image_name + "_psd.png"
                )

                # Test
                model_kwargs = {
                    "conditioned_image": cond,
                    "heatmap": heatmap
                }
                sample_fn = self.diffusion.ddim_sample_loop
                output = sample_fn(
                    self.model,
                    (
                        batch.shape[0], batch.shape[1],
                        self.config.heatmap_size, self.config.heatmap_size
                    ),
                    progress=True,
                    clip_denoised=self.config.clip_denoised,
                    model_kwargs=model_kwargs
                )

                print(output.shape)

                comb_heatmap_output = self.show_heatmap_range(
                    output[0], range_l=0, range_r=1,
                    save_path=self.save_path + "\\occlusion\\" + image_name + "_output.png"
                )

                self.gen_psd_from_heatmap(
                    comb_heatmap_output, save_path=self.save_path + "\\occlusion\\" + image_name + "_output_psd.png"
                )

    def save_image(self, image, save_path=None):
        image = image.detach().cpu()
        plt.imshow(image)
        plt.axis('off')
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            print(f"Saved Image to {save_path}")
        # plt.show()

    def show_heatmap_range(self, heatmaps, range_l=0, range_r=68, save_path=None):
        heatmaps = heatmaps.detach().cpu().numpy()

        # 创建一个用于叠加的空白图像
        combined_heatmap = np.zeros_like(heatmaps[0])

        # 叠加每一个关键点的热图
        for i in range(range_l, range_r):
            temp = heatmaps[i]
            # 归一化热图，避免数值过大
            if np.max(temp) > 0:
                temp = temp / np.max(temp)
            combined_heatmap += temp

        # 最终归一化，防止图像亮度过高
        if np.max(combined_heatmap) > 0:
            combined_heatmap = combined_heatmap / np.max(combined_heatmap)

        # 显示叠加后的热图
        plt.imshow(combined_heatmap, cmap='viridis')
        # plt.title('Combined Heatmap of Facial Keypoints')
        # plt.colorbar()
        plt.axis('off')  # 隐藏坐标轴

        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            print(f"Saved Image to {save_path}")
        else:
            plt.show()
        return combined_heatmap

    def save_heatmap(self, heatmaps, save_path=None):

        heatmaps = heatmaps.detach().cpu().numpy()
        combined_heatmap = np.zeros_like(heatmaps[0])

        # 叠加每一个关键点的热图
        for i in range(heatmaps.shape[0]):
            temp = heatmaps[i]
            combined_heatmap += temp

        # 归一化
        combined_heatmap = np.clip(combined_heatmap, 0, 1)  # 确保不超过 1
        # 显示叠加后的热图
        plt.figure(dpi=300)
        plt.imshow(combined_heatmap, cmap='viridis')
        # plt.title('Combined Heatmap of Facial Landmarks')
        # plt.colorbar()
        plt.axis('off')  # 隐藏坐标轴

        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            print(f"Saved Image to {save_path}")
        else:
            # plt.show()
            pass
        return combined_heatmap

    def gen_psd_from_image(self, image):
        # 1. 获取灰度图
        image_np = image.detach().cpu().numpy()
        img_gray = 0.2989 * image_np[:, :, 0] + 0.5870 * image_np[:, :, 1] + 0.1140 * image_np[:, :, 2]

        # 2. 获取图像的尺寸
        rows, cols = img_gray.shape

        # 3. 对图像进行傅里叶变换
        f_transform = np.fft.fft2(img_gray)
        f_shifted = np.fft.fftshift(f_transform)

        # 4. 计算功率谱密度（PSD）
        psd = np.abs(f_shifted) ** 2

        # 5. 将功率谱密度转换为对数尺度以便可视化
        log_psd = np.log(psd + 1)

        # 6. 计算频率范围（Cycles/Image）
        frequencies_x = np.fft.fftfreq(cols)
        frequencies_y = np.fft.fftfreq(rows)
        frequencies_x_shifted = np.fft.fftshift(frequencies_x)
        frequencies_y_shifted = np.fft.fftshift(frequencies_y)

        # 7. 绘制功率谱密度（PSD）图
        plt.figure(figsize=(8, 6))
        plt.imshow(log_psd, cmap='gray')
        plt.colorbar(label='Log Power Spectral Density')
        plt.title('Power Spectral Density (PSD) in Frequency Domain')
        plt.xlabel('Cycles/Image (Horizontal)')
        plt.ylabel('Cycles/Image (Vertical)')
        plt.show()

    def gen_psd_from_heatmap(self, combined_heatmap, save_path=None):
        # 获取热图的行列数
        rows, cols = combined_heatmap.shape

        # 对热图进行二维傅里叶变换
        f_transform = np.fft.fft2(combined_heatmap)  # 将空间域的热图转换到频率域
        f_shifted = np.fft.fftshift(f_transform)  # 将变换结果的零频分量移动到频谱中心，便于可视化分析

        # 计算功率谱密度（PSD）
        psd = np.abs(f_shifted) ** 2  # 取傅里叶变换结果的幅值的平方（np.abs(f_shifted) ** 2），得到功率谱密度。

        # 5. 将功率谱密度转换为对数尺度以便可视化
        log_psd = np.log(psd + 1)  # 对PSD值取对数（np.log(psd + 1)），增强可视化效果，特别是较小的功率值。

        # 6. 计算频率范围（Cycles/Image）
        frequencies_x = np.fft.fftfreq(cols)  # 计算频率分量，分别在横向和纵向上计算周期/图像的频率。
        frequencies_y = np.fft.fftfreq(rows)
        frequencies_x_shifted = np.fft.fftshift(frequencies_x)  # 将频率分量对齐，确保频率与PSD图对应。
        frequencies_y_shifted = np.fft.fftshift(frequencies_y)

        # 7. 绘制功率谱密度（PSD）图
        plt.figure(figsize=(8, 6))  # 使用 Matplotlib 绘制对数尺度的PSD图。
        plt.imshow(log_psd, cmap='gray')  # 使用 cmap='jet' 设置伪彩色图表（红-蓝渐变）。
        # plt.imshow(log_psd, cmap='jet')  # 使用 cmap='jet' 设置伪彩色图表（红-蓝渐变）。

        # 默认隐藏坐标轴，生成简洁的图像。
        # plt.colorbar(label='Log Power Spectral Density')
        # plt.title('Power Spectral Density (PSD) of Heatmap')
        # plt.xlabel('Cycles/Image (Horizontal)')
        # plt.ylabel('Cycles/Image (Vertical)')
        plt.axis('off')  # 隐藏坐标轴
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            print(f"Saved Image to {save_path}")

        # plt.show()

    def get_coords_from_heatmap(self, heatmap):
        """
        :param heatmap: tensor([N,68,128,128])
        :return:
        """
        map_size = self.config.heatmap_size  # 128
        predict_landmarks = get_predicts(heatmap)  # return([N,68,2])
        predict_landmarks = norm_points(predict_landmarks, map_size, map_size)  # return([N,68,2])

        return predict_landmarks

    def exp(self, heatmap, save_dir=None, metrics=None):
        # 进行二维傅里叶变换
        f = np.fft.fft2(heatmap)

        # 将频谱中心化，将零频率移到频谱的中心
        fshift = np.fft.fftshift(f)

        # 计算幅度（能量）
        magnitude_spectrum = np.abs(fshift) ** 2  # 计算频率成分的能量（幅度的平方）
        magnitude_spectrum = np.log(magnitude_spectrum + 1)  # 对数尺度显示，便于可视化

        # 显示频谱图像
        plt.imshow(magnitude_spectrum, cmap='gray')
        # plt.title('Magnitude Spectrum (Energy)')
        # plt.colorbar()
        plt.axis('off')  # 隐藏坐标轴
        if save_dir:
            path = save_dir + "_heatmap_gray.jpg"
            plt.savefig(path, bbox_inches='tight', dpi=300)
            print(f"Saved Image to {path}")
        else:
            plt.show()

        plt.imshow(magnitude_spectrum, cmap='jet')
        # plt.title('Magnitude Spectrum (Energy)')
        # plt.colorbar()
        plt.axis('off')  # 隐藏坐标轴
        if save_dir:
            path = save_dir + "_heatmap_jet.jpg"
            plt.savefig(path, bbox_inches='tight', dpi=300)
            print(f"Saved Image to {path}")
        else:
            plt.show()

        # 获取图像尺寸
        rows, cols = heatmap.shape
        crow, ccol = rows // 2, cols // 2  # 频谱中心

        # 设计低通滤波器
        radius = 10  # 低通滤波器的半径，可以根据需要调整
        mask_lp = np.zeros((rows, cols), dtype=np.uint8)
        cv2.circle(mask_lp, (ccol, crow), radius, 1, thickness=-1)  # 在频谱中心添加圆形区域（低频保留）

        # 应用低通滤波器
        fshift_lp = fshift * mask_lp
        magnitude_spectrum_lp = np.abs(fshift_lp) ** 2  # 低频区域的能量
        magnitude_spectrum_lp = np.log(magnitude_spectrum_lp + 1)

        # 显示低通滤波后的频谱
        plt.imshow(magnitude_spectrum_lp, cmap='jet')
        # plt.title('Low-pass Filtered Magnitude Spectrum')
        # plt.colorbar()
        plt.axis('off')  # 隐藏坐标轴
        if save_dir:
            path = save_dir + "_lpf_magnitude_spectrum.jpg"
            plt.savefig(path, bbox_inches='tight', dpi=300)
            print(f"Saved Image to {path}")
        else:
            plt.show()

        # 设计高通滤波器
        mask_hp = np.ones((rows, cols), dtype=np.uint8)
        cv2.circle(mask_hp, (ccol, crow), radius, 0, thickness=-1)  # 将中心区域设为0（去除低频）
        fshift_hp = fshift * mask_hp
        magnitude_spectrum_hp = np.abs(fshift_hp) ** 2  # 高频区域的能量
        magnitude_spectrum_hp = np.log(magnitude_spectrum_hp + 1)

        # 显示高通滤波后的频谱
        plt.imshow(magnitude_spectrum_hp, cmap='jet')
        # plt.title('High-pass Filtered Magnitude Spectrum')
        # plt.colorbar()
        plt.axis('off')  # 隐藏坐标轴
        if save_dir:
            path = save_dir + "_hpf_magnitude_spectrum.jpg"
            plt.savefig(path, bbox_inches='tight', dpi=300)
            print(f"Saved Image to {path}")
        else:
            plt.show()

        # 逆傅里叶变换
        f_ishift_lp = np.fft.ifftshift(fshift_lp)
        img_back_lp = np.fft.ifft2(f_ishift_lp)  # 对低通滤波后的频域进行逆傅里叶变换
        img_back_lp = np.abs(img_back_lp)  # 取复数的幅度值

        f_ishift_hp = np.fft.ifftshift(fshift_hp)
        img_back_hp = np.fft.ifft2(f_ishift_hp)  # 对高通滤波后的频域进行逆傅里叶变换
        img_back_hp = np.abs(img_back_hp)  # 取复数的幅度值

        # 显示低通和高通滤波后的图像
        plt.imshow(img_back_lp, cmap='viridis')
        # plt.title('Low-pass Filtered Image')
        # plt.colorbar()
        plt.axis('off')  # 隐藏坐标轴
        if save_dir:
            path = save_dir + "_lpf_image.jpg"
            plt.savefig(path, bbox_inches='tight', dpi=300)
            print(f"Saved Image to {path}")
        else:
            plt.show()

        plt.imshow(img_back_hp, cmap='viridis')
        # plt.title('High-pass Filtered Image')
        # plt.colorbar()
        plt.axis('off')  # 隐藏坐标轴
        if save_dir:
            path = save_dir + "_hpf_image.jpg"
            plt.savefig(path, bbox_inches='tight', dpi=300)
            print(f"Saved Image to {path}")
        else:
            plt.show()

        # 计算高低频能量
        low_freq_energy = np.sum(np.abs(fshift_lp) ** 2)
        high_freq_energy = np.sum(np.abs(fshift_hp) ** 2)

        if save_dir:
            path = save_dir + "_energy.txt"
            with open(path, 'a') as f:
                f.write("low freq energy: " + str(low_freq_energy) + "\n")
                f.write("high freq energy: " + str(high_freq_energy) + "\n")
                if metrics is not None:
                    f.write("NME_IP: " + str(metrics[0]) + "\n")
                    f.write("NME_IO: " + str(metrics[1]) + "\n")
        else:
            print("Low Frequency Energy:", low_freq_energy)
            print("High Frequency Energy:", high_freq_energy)
            if metrics is not None:
                print("NME_IP: " + str(metrics[0]))
                print("NME_IO: " + str(metrics[1]))
    def train_metrics(self, label_predict, label_groundtruth):
        """
        :param label_predict: tensor([N,68,128,128])
        :param label_groundtruth: [tensor([N,68,2]), tensor([N,81,128,128])]
        :return:
        """
        map_size = self.config.heatmap_size  # 128
        predict_landmarks = get_predicts(label_predict)  # return([N,68,2])
        predict_landmarks = norm_points(predict_landmarks, map_size, map_size)  # return([N,68,2])
        metrics_value = self.metrics.test(predict_landmarks, label_groundtruth[0])  # return (sum_nme, total_cnt)

        return metrics_value  # (value:float64, N)

def get_predicts(scores):
    """
    get predictions from score maps in torch Tensor
    :param scores: tensor([4N,68,128,128])
    :return preds: torch.LongTensor ([N,68,2])
    """
    assert scores.dim() == 4, 'Score maps should be 4-dim'
    # scores.view(N, 68, -1) => [N, 68, 128x128]
    # maxval: tensor([N, 68]): max value
    # idx: tensor([N, 68]): index of max value
    maxval, idx = th.max(scores.view(scores.size(0), scores.size(1), -1), 2)

    # ([N,68]) => ([N,68,1])
    maxval = maxval.view(scores.size(0), scores.size(1), 1)
    # ([N,68]) => ([N,68,1])
    idx = idx.view(scores.size(0), scores.size(1), 1) + 1
    # ([N,68,1]) => ([N,68,2])
    preds = idx.repeat(1, 1, 2).float()

    preds[:, :, 0] = (preds[:, :, 0] - 1) % scores.size(3)  # 每个元素取余128
    preds[:, :, 1] = th.floor((preds[:, :, 1] - 1) / scores.size(3))  # 每个元素除以128向下取整

    return preds  # ([N,68,2])


def norm_points(points, h, w, align_corners=False):
    """
    normalize points
    :param points: ([N,68,2])
    :param h: 128
    :param w: 128
    :param align_corners:
    :return:
    """
    if align_corners:
        # [0, SIZE-1] -> [-1, +1]
        des_points = points / th.tensor([w - 1, h - 1]).to(points).view(1, 2) * 2 - 1
    else:
        # [-0.5, SIZE-0.5] -> [-1, +1]
        # torch.tensor([w,h]).to(points).view = ([[128.,128.]])
        des_points = (points * 2 + 1) / th.tensor([w, h]).to(points).view(1, 2) - 1
    des_points = th.clamp(des_points, -1, 1)
    return des_points
