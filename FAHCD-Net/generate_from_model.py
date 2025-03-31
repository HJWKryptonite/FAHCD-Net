#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/5/12 19:08
# @Author  : HuJiwei
# @File    : generate_from_model.py
# @Software: PyCharm
# @Project: AlignDiff_8_stack_cascade
import functools
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from config.align_diff import AlignDiff
from improved_diffusion.resample import create_named_schedule_sampler
from lib import logger
from lib.dataset.aligndiff_dataset import show_preds
from lib.script import create_model, create_diffusion, get_dataloader
from utils import dist_util
from utils.file_util import write_to_pts


def generate_from_model():
    # 1. Get config info
    config = AlignDiff("./")
    config.init_instance()
    point_dir = config.point_dir

    # 2. Get trained model and diffusion
    logger.log("Getting trained model and diffusion...")
    model = create_model(config)
    train_diffusion = create_diffusion(config, "train")
    schedule_sampler = create_named_schedule_sampler(config.schedule_sampler, train_diffusion)

    checkpoint = dist_util.load_state_dict(config.val_model_path, map_location="cpu")
    model.load_state_dict(checkpoint["net"])

    model.to(dist_util.dev())
    model.eval()

    # 3. Creating data loader
    logger.log("Creating test data loader...")
    train_loader = get_dataloader(config, "generate")

    with torch.no_grad():
        for i, sample in enumerate(train_loader):

            image_paths = sample["image_path"]

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
            # sample["label"] = [landmarks, target]
            labels = sample["label"]

            batch = labels[1].float().to(dist_util.dev(), non_blocking=True)

            # diffusion get time_embedding
            # t: [N], in which n belongs (0, diffusion_step)
            # weights: [N], n = 1
            t, weights = schedule_sampler.sample(batch.shape[0], dist_util.dev())

            compute_losses = functools.partial(
                train_diffusion.training_losses,
                model, batch, t,
                model_kwargs=cond,
            )

            # losses: tensor([N])
            losses = compute_losses()

            output = losses["output"]
            if output is None:
                raise ValueError("Error Loss Type")

            print(output)  # ([N,68,128,128])

            landmarks = get_predicts(output)  # ([N,68,2])
            pts_names = sample["pts_name"]  # [N]

            inverse = sample["inverse"]
            inverse = inverse.view(3, 3)
            inverse = inverse.cpu().detach().numpy()

            for i, landmark in enumerate(landmarks):
                landmark = landmark.cpu().detach().numpy()
                image_path = image_paths[i]
                img = cv2.imread(image_path, cv2.IMREAD_COLOR)
                reverse_landmark = transform(landmark, inverse)
                show_preds(img, landmark)
                show_preds(img, reverse_landmark * 2)

                file_path = point_dir / Path(pts_names[i])
                file_path = str(file_path)
                write_to_pts(file_path, landmark)
                print(1)


def get_predicts(scores):
    """
    get predictions from score maps in torch Tensor
    :param scores: ([N,68,128,128])
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


def transform(points: np.ndarray, matrix: np.ndarray):
    """
        points (nx2), matrix (3x3) -> points (nx2)
    """
    dtype = points.dtype

    # nx3
    points = np.concatenate([points, np.ones_like(points[:, [0]])], axis=1)
    points = points @ np.transpose(matrix)
    points = points[:, :2] / points[:, [2, 2]]
    return points.astype(dtype)


def transformImage(image, matrix):
    return cv2.warpPerspective(
        image, matrix,
        dsize=(256, 256),
        flags=cv2.INTER_LINEAR, borderValue=0
    )


if __name__ == "__main__":
    generate_from_model()
