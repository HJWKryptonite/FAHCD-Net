#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/7/27 15:23
# @Author  : HuJiwei
# @File    : trainer.py
# @Software: PyCharm
# @Project: AlignDiff
import functools

import numpy as np
import torch
import torch as th
from torch.optim import AdamW, lr_scheduler

from improved_diffusion.resample import UniformSampler
from lib.metric.NME import NME
from utils import dist_util

INITIAL_LOG_LOSS_SCALE = 20.0


class Trainer:
    def __init__(
            self, config, model, train_diffusion, val_diffusion,
            train_loader, val_loader, schedule_sampler,
            logger, resume_checkpoint, ema
    ):

        self.config = config
        self.model = model
        # train
        self.train_diffusion = train_diffusion
        self.train_loader = train_loader
        self.log_interval = self.config.log_interval
        self.schedule_sampler = schedule_sampler

        # val
        self.val_diffusion = val_diffusion
        self.val_loader = val_loader
        self.val_interval = self.config.val_interval

        self.logger = logger

        self.resume_checkpoint = resume_checkpoint

        self.ema = ema

        # EMA
        self.model_params = list(self.model.parameters())

        # Constant
        self.lg_loss_scale = INITIAL_LOG_LOSS_SCALE  # 20.0

        self.optimizer = AdamW(
            self.model_params, lr=self.config.learn_rate, weight_decay=self.config.weight_decay
        )
        self.schedule = lr_scheduler.MultiStepLR(
            optimizer=self.optimizer,
            milestones=self.config.milestones,
            gamma=self.config.gamma
        )
        self.metrics = NME(self.config.nme_left_index, self.config.nme_right_index)

    def run_loop(self):

        best_metric = None
        best_net = None

        # Create model path
        self.config.model_dir.mkdir(exist_ok=True)
        if not self.config.model_dir.is_dir():
            raise RuntimeError("Failed to create model dir.")

        for epoch in range(self.config.max_epoch + 1):
            # memory occupation
            # os.system("nvidia-smi")

            # ========================================================================
            #                         1. Training
            # ========================================================================
            self.forward_backward(epoch)

            # ========================================================================
            #                         2. Validating
            # ========================================================================

            # if (epoch > 99 and ((epoch + 1) % self.val_interval == 0)) or epoch == 0:
            # if (epoch > 99 and ((epoch + 1) % self.val_interval == 0)) or (epoch > 159 and (epoch + 1) % 10 == 0) or epoch == 0:
            if (epoch + 1) % 10 == 0 or epoch == 0:
                epoch_nets = {"model": self.model}
                for name, model in epoch_nets.items():
                    if model is None:  # 如果模型不存在，进行下一次循环
                        continue

                    nme_ip, nme_io = self.forward(model)
                    self.logger.log("Validate/Epoch: %d/%d, Test NME_IP: %.6f, Test NME_IO: %.6f" % (
                        epoch, self.config.max_epoch, nme_ip, nme_io
                    ))

                    if best_metric is None or best_metric > nme_io:
                        best_metric = nme_io
                        best_net = model

                        # if best_metric is None or best_metric > nme:
                        #     best_metric = nme
                        #     best_net = model

                        if best_metric < 0.10:
                            # {best_metric}.pth.tar
                            file_name = str(best_metric) + ".pth.tar"

                            # ./model/aligndiff/{time}/{best_metric}.pth.tar
                            model_path = str(self.config.model_dir) + "/" + file_name
                            self.logger.log("saving model %s" % model_path)
                            self.save_model(
                                epoch=epoch,
                                net=best_net,
                                optimizer=self.optimizer,
                                scheduler=self.schedule,
                                pytorch_model_path=model_path
                            )

                    if self.ema:
                        self.ema.restore()

        if best_metric is not None:
            self.logger.log("Val/Best_Metric in this epoch: %.6f" % best_metric)

    def forward_backward(self, epoch):

        avg_metrics = [0, 0, 0]

        self.model = self.model.float().to(dist_util.dev())
        self.model.train()

        dataset_size = len(self.train_loader.dataset)  # 3147
        batch_size = self.train_loader.batch_size  # B
        batch_num = max(dataset_size / max(batch_size, 1), 1)  # 3147/B

        self.logger.log(
            "Train process, Dataset size: %d, Batch size: %d" % (dataset_size, batch_size)
        )

        for i, sample in enumerate(self.train_loader):
            # ([N,68,128,128])
            heatmap = sample["heatmap"].float().to(dist_util.dev(), non_blocking=True)

            # cond: origin image, ([N,3,256,256])
            cond = sample["data"].float().to(dist_util.dev(), non_blocking=True)
            cond = {
                "conditioned_image": cond,
                "heatmap": heatmap
            }

            # landmarks: ([N,68,2])
            # target: ([N,81,128,128])
            # target{point_info}: ([N, 0~67,128,128])
            # target{edge_info}: ([N, 68~80,128,128])
            # sample["label"] = [landmarks, target]
            labels = sample["label"]

            # labels[1] = target
            # batch: target, ([N,81,128,128])
            batch = labels[1].float().to(dist_util.dev(), non_blocking=True)

            """
            :param batch: ([N,81,128,128])
            :param cond: {
                    "conditioned_image": ([N,3,256,256]),
                    "heatmap": ([N,68,128,128])
                }
            :param labels: [landmarks, target]
            """

            # diffusion get time_embedding
            # t: [N], in which n belongs (0, diffusion_step)
            # weights: [N], n = 1
            t, weights = self.schedule_sampler.sample(batch.shape[0], dist_util.dev())

            compute_losses = functools.partial(
                self.train_diffusion.training_losses,
                self.model, batch, t,
                model_kwargs=cond,
            )

            # losses: tensor([N])
            losses = compute_losses()
            # loss: ([1])
            # loss = (losses["mse"] * weights).mean()
            loss = losses["loss"]

            output = losses["output"]
            if output is None:
                raise ValueError("Error Loss Type")

            metrics = self.train_metrics(
                output[:, :self.config.classes_num[0], :, :],
                labels
            )

            for x in range(len(metrics)):
                avg_metrics[x] += metrics[x]

            if (i + 1) % self.log_interval == 0:
                if self.logger is not None:
                    avg_nme_ip = round((
                            avg_metrics[0] / max(avg_metrics[2], 1)
                    ), 6)
                    avg_nme_io = round((
                            avg_metrics[1] / max(avg_metrics[2], 1)
                    ), 6)
                    self.logger.log(
                        "Train/Epoch: %d/%d, Iter: %d/%d, Loss: %.6f, AVG NME_IP is: %.6f, AVG NME_IO is: %.6f" % (
                            epoch, self.config.max_epoch, i, batch_num, loss, avg_nme_ip, avg_nme_io
                        )
                    )

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # ema
            if self.ema:
                self.ema.update()

    def forward(self, model):

        if self.ema:
            self.ema.apply_shadow()

        avg_metrics = [0, 0, 0]
        model = model.float().to(dist_util.dev())
        model.eval()

        dataset_size = len(self.val_loader.dataset)  # dataset_size = 134
        batch_size = self.val_loader.batch_size  # batch_size = 20
        batch_num = max(dataset_size / max(batch_size, 1), 1)  # batch_num = 6.7

        self.logger.log(
            "Validate process, Dataset size: %d, Batch size: %d" % (dataset_size, batch_size)
        )

        for i, sample in enumerate(self.val_loader):

            heatmap = sample["heatmap"].float().to(dist_util.dev(), non_blocking=True)  # ([n,2,68,128,128)]
            # ([2n,68,128,128)]
            heatmap = th.reshape(heatmap, (heatmap.size(0) * 2, heatmap.size(2), heatmap.size(3), heatmap.size(4)))
            # Vote
            heatmap = th.cat([heatmap] * self.config.major_vote_number)  # ([2nv,68,128,128])

            cond = sample["data"].float().to(dist_util.dev(), non_blocking=True)  # ([n,2,3,256,256])
            cond = th.reshape(cond, (cond.size(0) * 2, 3, cond.size(3), cond.size(4)))  # ([2n,3,256,256])
            # Vote
            cond = th.cat([cond] * self.config.major_vote_number)  # ([2nv,3,256,256])

            batch = sample["label"][1].float().to(dist_util.dev(), non_blocking=True)  # ([2n,68,128,128])

            model_kwargs = {
                "conditioned_image": cond,  # ([2nv,3,256,256])
                "heatmap": heatmap
            }

            labels = sample["label"]  # ([n,68,2]), ([n,68,128,128])

            sample_fn = self.val_diffusion.ddim_sample_loop
            sample = sample_fn(
                model,
                (
                    batch.shape[0] * 2 * self.config.major_vote_number,  # 2nv
                    batch.shape[1],  # 68
                    self.config.heatmap_size,  # 128
                    self.config.heatmap_size  # 128
                ),  # ([2nv,68,128,128])
                progress=True,
                clip_denoised=self.config.clip_denoised,
                model_kwargs=model_kwargs
            )  # sample=([2n,68,128,128])

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

        return avg_nme_ip, avg_nme_io

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
        predict_heatmaps = th.stack(predict_heatmaps)  # return([N,68,128,128])

        tmp_landmarks = get_predicts(predict_heatmaps)  # return([N,68,2])
        predict_landmarks = []
        batch_tmp = int(batch_size / self.config.major_vote_number)
        for ii in range(0, batch_tmp):
            temp = tmp_landmarks[ii]
            for jj in range(1, self.config.major_vote_number):  # 多个投票相加的结果的均值作为最终的结果
                temp = temp + tmp_landmarks[ii + jj * batch_tmp]
            temp = temp / self.config.major_vote_number
            predict_landmarks.append(temp)
        predict_landmarks = th.stack(predict_landmarks)  # ([N,68,2])
        predict_landmarks = norm_points(predict_landmarks, map_size, map_size)  # return([2N,68,2])
        metrics_value = self.metrics.test(predict_landmarks, label_groundtruth[0])

        return metrics_value

    def save_model(self, epoch, net, optimizer, scheduler, pytorch_model_path):
        # save pytorch model
        state = {
            "net": net.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch
        }

        th.save(state, pytorch_model_path)
        self.logger.log("Epoch: %d/%d, model saved in this epoch" % (epoch, self.config.max_epoch))


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


def main():
    pass


if __name__ == "__main__":
    main()
