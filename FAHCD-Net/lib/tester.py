#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/9/7 16:46
# @Author  : HuJiwei
# @File    : tester.py
# @Software: PyCharm
# @Project: AlignDiff
import matplotlib.pyplot as plt
import numpy as np
import torch

from lib.metric.NME import NME
from utils import dist_util


class Tester:
    def __init__(
            self, config, model, diffusion, loader, logger,
    ):
        self.config = config
        self.model = model
        self.diffusion = diffusion
        self.loader = loader
        self.logger = logger

        self.clip_denoised = self.config.clip_denoised

        self.major_vote_number = self.config.major_vote_number
        self.metrics = NME(self.config.nme_left_index, self.config.nme_right_index)

    def test(self):

        avg_metrics = [0, 0, 0]
        self.model = self.model.float().to(dist_util.dev())
        self.model.eval()

        with torch.no_grad():
            dataset_size = len(self.loader.dataset)  # dataset_size = 134
            batch_size = self.loader.batch_size  # batch_size = 20
            batch_num = max(dataset_size / max(batch_size, 1), 1)  # batch_num = 6.7
            self.logger.log(
                "Validate process, Dataset size: %d, Batch size: %d" % (dataset_size, batch_size)
            )

            for i, sample in enumerate(self.loader):
                # ([n,2,68,128,128)]
                heatmap = sample["heatmap"].float().to(dist_util.dev(), non_blocking=True)
                # ([2n,68,128,128)]
                heatmap = torch.reshape(
                    heatmap, (heatmap.size(0) * 2, heatmap.size(2), heatmap.size(3), heatmap.size(4))
                )
                # Vote ([2nv,68,128,128)]
                heatmap = torch.cat([heatmap] * self.major_vote_number)  # ([2nv,68,128,128])

                # ([n,2,3,256,256])
                cond = sample["data"].float().to(dist_util.dev(), non_blocking=True)
                # ([2n,3,256,256])
                cond = torch.reshape(cond, (cond.size(0) * 2, 3, cond.size(3), cond.size(4)))
                # Vote ([2nv,3,256,256])
                cond = torch.cat([cond] * self.major_vote_number)  # ([2nv,3,256,256])

                # ([2n,68,128,128])
                batch = sample["label"][1].float().to(dist_util.dev(), non_blocking=True)  # ([2n,68,128,128])

                model_kwargs = {
                    "conditioned_image": cond,  # ([2nv,3,256,256])
                    "heatmap": heatmap  # ([2nv,68,128,128)]
                }

                labels = sample["label"]  # ([n,68,2]), ([n,68,128,128])

                sample_fn = self.diffusion.ddim_sample_loop
                sample = sample_fn(
                    self.model,
                    (
                        batch.shape[0] * 2 * self.major_vote_number,  # 2nv
                        batch.shape[1],  # 68
                        self.config.heatmap_size,  # 128
                        self.config.heatmap_size  # 128
                    ),  # ([2nv,68,128,128])
                    progress=True,
                    clip_denoised=self.config.clip_denoised,
                    model_kwargs=model_kwargs
                )  # sample=([2nv,68,128,128])

                print(sample.shape)

                metrics = self.test_metrics(
                    sample[:, :self.config.classes_num[0], :, :],  # sample([4n,0~68,128,128])
                    labels,  # landmarks: ([n,68,2]) target: ([n,81,128,128])
                    labels[0].size(0) * self.config.major_vote_number
                )

                for x in range(len(metrics)):
                    avg_metrics[x] += metrics[x]

                if self.logger is not None:
                    self.logger.log(
                        "Val Process Start: %d/%d" % (
                            i, batch_num
                        )
                    )

        avg_nme_ip = round((
                avg_metrics[0] / max(avg_metrics[2], 1)
        ), 6)

        avg_nme_io = round((
                avg_metrics[1] / max(avg_metrics[2], 1)
        ), 6)
        print("NME_IO: " + str(avg_nme_io) + "\nNME_IP: " + str(avg_nme_ip))
        print("NME_IP: " + str(avg_nme_ip) + "")

    def test_metrics(self, label_predict, label_groundtruth, batch_size):

        """
        :param label_predict: tensor([2N,68,128,128])
        :param label_groundtruth: [tensor([N,68,2]), tensor([N,68,128,128])]
        :return:
        """

        map_size = self.config.heatmap_size  # 128
        # 对位于基数索引的热力图进行翻转  img_1, img_flip_1, img_2, img_flip_2, ...img_batch, img_flip_batch,
        # img_1, img_flip_1, img_2, img_flip_2, ...img_batch, img_flip_batch,
        for ii in range(1, batch_size * 2, 2):
            temp = label_predict[ii]
            temp = flip_channels(temp.cpu())
            temp = shuffle_channels_for_horizontal_flipping(temp)
            label_predict[ii] = temp

        predict_heatmaps = []  # 原图和翻转对应的热力图求均值
        for ii in range(0, 2 * batch_size, 2):
            temp = (label_predict[ii] + label_predict[ii + 1]) / 2
            predict_heatmaps.append(temp)
        predict_heatmaps = torch.stack(predict_heatmaps)  # return([N,68,128,128])

        tmp_landmarks = get_predicts(predict_heatmaps)  # return([N,68,2])
        predict_landmarks = []
        batch_tmp = int(batch_size / self.config.major_vote_number)
        for ii in range(0, batch_tmp):
            temp = tmp_landmarks[ii]
            for jj in range(1, self.config.major_vote_number):  # 多个投票相加的结果的均值作为最终的结果
                temp = temp + tmp_landmarks[ii + jj * batch_tmp]
            temp = temp / self.config.major_vote_number
            predict_landmarks.append(temp)
        predict_landmarks = torch.stack(predict_landmarks)  # ([N,68,2])
        predict_landmarks = norm_points(predict_landmarks, map_size, map_size)  # return([2N,68,2])
        metrics_value = self.metrics.test(predict_landmarks, label_groundtruth[0])

        return metrics_value


def test(tensor: torch.Tensor):
    """
    show tensor image
    :param tensor: ([B, C, H, W])
    :return:
    """
    assert len(tensor.shape) == 4 and tensor.shape[0] == 1, "error format"

    chw_numpy = tensor[0].cpu().numpy()
    img_array = np.transpose(chw_numpy, (1, 2, 0))  # HWC
    plt.imshow(img_array)
    plt.show()


def compute_metrics(label_predict, label_groundtruth):
    """
    :param label_predict: tensor([N,68,128,128])
    :param label_groundtruth: [tensor([N,68,2]), tensor([N,81,128,128])]
    :return:
    """
    map_size = 128

    predict_landmarks = get_predicts(label_predict)  # return([N,68,2])
    predict_landmarks = norm_points(predict_landmarks, map_size, map_size)  # return([N,68,2])

    nme = NME(60, 72)
    metrics_value = nme.test(predict_landmarks, label_groundtruth[0])  # return (sum_nme, total_cnt)

    return metrics_value  # (value:float64, N)


def get_predicts(scores):
    """
    get predictions from score maps in torch Tensor
    :param scores: Tensor([N,68,128,128])
    :return preds: torch.LongTensor ([24,68,2])
    """
    assert scores.dim() == 4, 'Score maps should be 4-dim'
    # scores.view(N, 68, -1) => [N, 68, 128x128]
    # maxval: tensor([N, 68]): max value
    # idx: tensor([N, 68]): index of max value
    maxval, idx = torch.max(scores.view(scores.size(0), scores.size(1), -1), 2)

    # ([N,68]) => ([N,68,1])
    maxval = maxval.view(scores.size(0), scores.size(1), 1)
    # ([N,68]) => ([N,68,1])
    idx = idx.view(scores.size(0), scores.size(1), 1) + 1
    # ([N,68,1]) => ([N,68,2])
    preds = idx.repeat(1, 1, 2).float()

    preds[:, :, 0] = (preds[:, :, 0] - 1) % scores.size(3)  # 每个元素取余128
    preds[:, :, 1] = torch.floor((preds[:, :, 1] - 1) / scores.size(3))  # 每个元素除以128向下取整

    return preds  # ([N,68,2])


def norm_points(points, h, w, align_corners=False):
    """
    normalize points
    :param points: ([24,68,2])
    :param h: 128
    :param w: 128
    :param align_corners:
    :return:
    """
    if align_corners:
        # [0, SIZE-1] -> [-1, +1]
        des_points = points / torch.tensor([w - 1, h - 1]).to(points).view(1, 2) * 2 - 1
    else:
        # [-0.5, SIZE-0.5] -> [-1, +1]
        # torch.tensor([w,h]).to(points).view = ([[128.,128.]])
        des_points = (points * 2 + 1) / torch.tensor([w, h]).to(points).view(1, 2) - 1
    des_points = torch.clamp(des_points, -1, 1)
    return des_points


def flip_channels(maps):
    # horizontally flip the channels
    # maps is a tensor of dimension n x c x h x w or c x h x w
    if maps.ndimension() == 4:
        maps = maps.numpy()
        maps = maps[:, :, :, ::-1].copy()
    elif maps.ndimension() == 3:
        maps = maps.numpy()
        maps = maps[:, :, ::-1].copy()
    else:
        exit('tensor dimension is not right')

    return torch.from_numpy(maps).float()


def shuffle_channels_for_horizontal_flipping(maps):
    match_parts_68 = np.array([
        [0, 16], [1, 15], [2, 14], [3, 13], [4, 12], [5, 11], [6, 10], [7, 9],  # outline
        [17, 26], [18, 25], [19, 24], [20, 23], [21, 22],  # eyebrow
        [36, 45], [37, 44], [38, 43], [39, 42], [40, 47], [41, 46],  # eye
        [31, 35], [32, 34],  # nose
        [48, 54], [49, 53], [50, 52], [59, 55], [58, 56],  # outer mouth
        [60, 64], [61, 63], [67, 65]
    ])
    match_parts_98 = np.array([
        [0, 32], [1, 31], [2, 30], [3, 29], [4, 28], [5, 27], [6, 26], [7, 25], [8, 24], [9, 23], [10, 22], [11, 21],
        [12, 20], [13, 19], [14, 18], [15, 17],  # outline
        [33, 46], [34, 45], [35, 44], [36, 43], [37, 42], [41, 47], [40, 48], [39, 49], [38, 50],  # eyebrow
        [60, 72], [61, 71], [62, 70], [63, 69], [64, 68], [67, 73], [66, 74], [65, 75], [96, 97],  # eye
        [55, 59], [56, 58],  # nose
        [76, 82], [77, 81], [78, 80], [87, 83], [86, 84],  # outer mouth
        [88, 92], [89, 91], [95, 93]
    ])

    # when the image is horizontally flipped, its corresponding groundtruth maps should be shuffled.
    # maps is a tensor of dimension n x c x h x w or c x h x w
    if maps.ndimension() == 4:
        dim = 1
        nPoints = maps.size(1)
    elif maps.ndimension() == 3:
        dim = 0
        nPoints = maps.size(0)
    else:
        exit('tensor dimension is not right')
    if nPoints == 98:
        match_parts = match_parts_98
    else:
        match_parts = match_parts_68
    for i in range(0, match_parts.shape[0]):
        idx1, idx2 = match_parts[i]
        idx1 = int(idx1)
        idx2 = int(idx2)
        tmp = maps.narrow(dim, idx1, 1).clone()  # narrow(dimension, start, length) dimension是要压缩的维度
        maps.narrow(dim, idx1, 1).copy_(maps.narrow(dim, idx2, 1))
        maps.narrow(dim, idx2, 1).copy_(tmp)
    return maps


if __name__ == "__main__":
    pass
