#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time     : 2024/12/20 14:58
# @Author   : HuJiwei
# @FileName : Face300WDataset.py
# @Software : PyCharm
# @Project  : AlignDiff_8_stack_cascade
import math
import os

import cv2
import imageio
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from lib.dataset.augmentation import Augmentation


class Face300WDataset(Dataset):
    def __init__(self, config, is_train=True, transform=None):
        super(Face300WDataset, self).__init__()

        self.database = config.database_300W

        self.cond_path = config.train_cond_dir if is_train else config.test_cond_dir

        tsv_file = config.train_tsv_file
        self.datas = pd.read_csv(tsv_file, sep="\t")  # 读取tsv文件，含有标签信息

        std_lmk_5pts = np.array([
            196.0, 226.0, 316.0, 226.0, 256.0, 286.0, 220.0, 360.4, 292.0, 360.4], np.float32
        ) / 256.0 - 1.0
        std_lmk_5pts = np.reshape(std_lmk_5pts, (5, 2))  # 排列成5行2列 [-1 1]
        target_face_scale = 1.0 if config.crop_op else 1.25  # 是否做了图片裁剪，决定目标人脸的尺寸为1倍还是1.25倍
        self.augmentation = Augmentation(
            is_train=is_train,  # 是否为训练
            aug_prob=config.aug_prob,  # 做 augmentation 的概率/比例
            image_size=config.image_width,  # 图片尺寸，设置为图片宽度
            crop_op=config.crop_op,  # 是否做图片裁剪
            std_lmk_5pts=std_lmk_5pts,  # std_lmk_5pts
            target_face_scale=target_face_scale,  # 目标尺寸
            flip_rate=0.5,  # 翻转比例
            flip_mapping=config.flip_mapping,  # 翻转映射
            random_shift_sigma=0.05,  # 随机shift
            random_rot_sigma=math.pi / 180 * 18,  # 随机旋转（18°）
            random_scale_sigma=0.1,  # 随机缩放（±10%）
            random_gray_rate=0.2,  # 随机灰度（±20%）
            random_occ_rate=0.4,  # 随机遮挡（±40%）
            random_blur_rate=0.3,  # 随机模糊（±30%）
            random_gamma_rate=0.2,
            random_nose_fusion_rate=0.2
        )

    def _norm_points(self, points, h, w, align_corners=False):
        if align_corners:
            # [0, SIZE-1] -> [-1, +1]
            des_points = points / torch.tensor([w - 1, h - 1]).to(points).view(1, 2) * 2 - 1
        else:
            # [-0.5, SIZE-0.5] -> [-1, +1]
            des_points = (points * 2 + 1) / torch.tensor([w, h]).to(points).view(1, 2) - 1
        des_points = torch.clamp(des_points, -1, 1)
        return des_points

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index):
        res = dict()

        # 0: image_path
        image_path = self.database + self.datas.iloc[index, 0]
        res["image_path"] = image_path
        image_name = image_path.split("/")[-1][:-4]
        res["image_name"] = image_name
        pts_name = image_name + "_.pts"
        res["pts_name"] = pts_name
        cond_lmk_path = self.cond_path + "/" + pts_name
        cond = self._load_from_pts(cond_lmk_path)

        # 1: lmk_5pts
        lmk_5pts = self.datas.iloc[index, 1]
        lmk_5pts = np.array(list(
            # 语法：map(function，iterable,…)
            # 将函数 function 依次作用到列表 iterable 的每个元素上，并返回以函数结果作为元素的列表
            map(float, lmk_5pts.split(","))
        ), dtype=np.float32).reshape(5, 2)

        # 2：ldk label
        lmk = self.datas.iloc[index, 2]
        lmk = np.array(list(
            map(float, lmk.split(","))
        ), dtype=np.float32).reshape(68, 2)

        # 3,4,5: scale, center_w, center_h
        scale = float(self.datas.iloc[index, 3])
        center_w = float(self.datas.iloc[index, 4])
        center_h = float(self.datas.iloc[index, 5])

        img = self._load_image(image_path)
        assert img is not None

        # Aug Process(I, lmk) => (I, lmk)
        img, lmk, _ = self.augmentation.process(
            img=img, lmk=lmk, lmk_5pts=lmk_5pts, scale=scale, center_w=center_w, center_h=center_h
        )  # ndarray(H,W,C), (68,2)

        # Aug Process(I, cond) => (_, cond)
        _, cond, _ = self.augmentation.process(
            img=img, lmk=cond, lmk_5pts=lmk_5pts, scale=scale, center_w=center_w, center_h=center_h
        )

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 标准化关键点
        landmarks = self._norm_points(
            points=torch.from_numpy(lmk),
            h=256,  # image_height
            w=256  # image_width
        )  # ([68,2]) 0-255



        return res

    def _load_image(self, image_path):
        if not os.path.exists(image_path):
            assert image_path is not None
        try:
            img = cv2.imread(image_path, cv2.IMREAD_COLOR)  # HWC, BGR, [0-255]
            assert img is not None and len(img.shape) == 3 and img.shape[2] == 3
        except:
            try:
                img = imageio.imread(image_path)  # HWC, RGB, [0-255]
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # HWC, BGR, [0-255]
                assert img is not None and len(img.shape) == 3 and img.shape[2] == 3
            except:
                try:
                    gifImg = imageio.mimread(image_path)  # BHWC, RGB, [0-255]
                    img = gifImg[0]  # HWC, RGB, [0-255]
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # HWC, BGR, [0-255]
                    assert img is not None and len(img.shape) == 3 and img.shape[2] == 3
                except:
                    img = None
        return img

    def _load_from_pts(self, filename: str):
        landmarks = np.genfromtxt(filename, skip_header=3, skip_footer=1)
        landmarks = landmarks - 1
        return landmarks
