#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/7/27 14:57
# @Author  : HuJiwei
# @File    : aligndiff_dataset.py
# @Software: PyCharm
# @Project: AlignDiff

import copy
import hashlib
import math
import os

import cv2
import imageio
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy import interpolate

from torch.utils.data import Dataset
from scipy.interpolate import splprep, splev

from lib.dataset.augmentation import Augmentation
from utils.file_util import load_from_pts

sigma2 = 3
boundary_keys = ['chin', 'leb', 'reb', 'bon', 'breath', 'lue', 'lle', 'rue', 'rle', 'usul', 'lsul', 'usll', 'lsll']

interp_points_num = {
    'chin': 120,
    'leb': 32,
    'reb': 32,
    'bon': 32,
    'breath': 25,
    'lue': 25,
    'lle': 25,
    'rue': 25,
    'rle': 25,
    'usul': 32,
    'lsul': 32,
    'usll': 32,
    'lsll': 32
}

dataset_pdb_numbins = {
    '300W': 9,
    'AFLW': 17,
    'COFW': 7,
    'WFLW': 13
}
dataset_size = {
    '300W': {
        'train': 3148,
        'common_subset': 554,
        'challenge_subset': 135,
        'fullset': 689,
        '300W_testset': 600,
        'COFW68': 507  # 该数据集用于300W数据集上训练模型的测试
    },
    'AFLW': {
        'train': 20000,
        'test': 24386,
        'frontal': 1314
    },
    'COFW': {
        'train': 1345,
        'test': 507
    },
    'WFLW': {
        'train': 7500,
        'test': 2500,
        'pose': 326,
        'expression': 314,
        'illumination': 698,
        'makeup': 206,
        'occlusion': 736,
        'blur': 773
    }
}

kp_num = {
    '300W': 68,
    'AFLW': 19,
    'COFW': 29,
    'WFLW': 98
}

point_num_per_boundary = {
    '300W': [17., 5., 5., 4., 5., 4., 4., 4., 4., 7., 5., 5., 7.],
    'AFLW': [1., 3., 3., 1., 2., 3., 3., 3., 3., 3., 3., 3., 3.],
    'COFW': [1., 3., 3., 1., 3., 3., 3., 3., 3., 3., 1., 1., 3.],
    'WFLW': [33., 9., 9., 4., 5., 5., 5., 5., 5., 7., 5., 5., 7.]
}

boundary_special = {  # 有些边界线条使用的关键点和其他边界形成不连续交集，特殊处理
    'lle': ['300W', 'COFW', 'WFLW'],
    'rle': ['300W', 'COFW', 'WFLW'],
    'usll': ['300W', 'WFLW'],
    'lsll': ['300W', 'COFW', 'WFLW']
}

duplicate_point = {  # 需要重复使用的关键点的序号，从0开始计数
    '300W': {
        'lle': 36,
        'rle': 42,
        'usll': 60,
        'lsll': 48
    },
    'COFW': {
        'lle': 13,
        'rle': 17,
        'lsll': 21
    },
    'WFLW': {
        'lle': 60,
        'rle': 68,
        'usll': 88,
        'lsll': 76
    }
}

point_range = {  # notice: this is 'range', the later number pluses 1; the order is boundary order; index starts from 0
    '300W': [
        [0, 17], [17, 22], [22, 27], [27, 31], [31, 36],
        [36, 40], [39, 42], [42, 46], [45, 48], [48, 55],
        [60, 65], [64, 68], [54, 60]
    ],
    'AFLW': [
        [0, 1], [1, 4], [4, 7], [7, 8], [8, 10],
        [10, 13], [10, 13], [13, 16], [13, 16], [16, 19],
        [16, 19], [16, 19], [16, 19]
    ],
    'COFW': [
        [0, 1], [1, 4], [5, 8], [9, 10], [10, 13],
        [13, 16], [15, 17], [17, 20], [19, 21], [21, 24],
        [25, 26], [26, 27], [23, 25]
    ],
    'WFLW': [
        [0, 33], [33, 38], [42, 47], [51, 55], [55, 60],
        [60, 65], [64, 68], [68, 73], [72, 76], [76, 83],
        [88, 93], [92, 96], [82, 88]
    ]
}

flip_relation = {
    '300W': [
        [0, 16], [1, 15], [2, 14], [3, 13], [4, 12], [5, 11],
        [6, 10], [7, 9], [8, 8], [9, 7], [10, 6], [11, 5],
        [12, 4], [13, 3], [14, 2], [15, 1], [16, 0], [17, 26],
        [18, 25], [19, 24], [20, 23], [21, 22], [22, 21], [23, 20],
        [24, 19], [25, 18], [26, 17], [27, 27], [28, 28], [29, 29],
        [30, 30], [31, 35], [32, 34], [33, 33], [34, 32], [35, 31],
        [36, 45], [37, 44], [38, 43], [39, 42], [40, 47], [41, 46],
        [42, 39], [43, 38], [44, 37], [45, 36], [46, 41], [47, 40],
        [48, 54], [49, 53], [50, 52], [51, 51], [52, 50], [53, 49],
        [54, 48], [55, 59], [56, 58], [57, 57], [58, 56], [59, 55],
        [60, 64], [61, 63], [62, 62], [63, 61], [64, 60], [65, 67],
        [66, 66], [67, 65]
    ],
    'AFLW': [
        [0, 0], [1, 6], [2, 5], [3, 4], [4, 3], [5, 2],
        [6, 1], [7, 7], [8, 9], [9, 8], [10, 15], [11, 14],
        [12, 13], [13, 12], [14, 11], [15, 10], [16, 18], [17, 17],
        [18, 16]
    ],
    'COFW': [
        [0, 0], [1, 7], [2, 6], [3, 5], [4, 8], [5, 3],
        [6, 2], [7, 1], [8, 4], [9, 9], [10, 12], [11, 11],
        [12, 10], [13, 19], [14, 18], [15, 17], [16, 20], [17, 15],
        [18, 14], [19, 13], [20, 16], [21, 23], [22, 22], [23, 21],
        [24, 24], [25, 25], [26, 26], [27, 28], [28, 27]
    ],
    'WFLW': [
        [0, 32], [1, 31], [2, 30], [3, 29], [4, 28], [5, 27],
        [6, 26], [7, 25], [8, 24], [9, 23], [10, 22], [11, 21],
        [12, 20], [13, 19], [14, 18], [15, 17], [16, 16], [17, 15],
        [18, 14], [19, 13], [20, 12], [21, 11], [22, 10], [23, 9],
        [24, 8], [25, 7], [26, 6], [27, 5], [28, 4], [29, 3],
        [30, 2], [31, 1], [32, 0], [33, 46], [34, 45], [35, 44],
        [36, 43], [37, 42], [38, 50], [39, 49], [40, 48], [41, 47],
        [42, 37], [43, 36], [44, 35], [45, 34], [46, 33], [47, 41],
        [48, 40], [49, 39], [50, 38], [51, 51], [52, 52], [53, 53],
        [54, 54], [55, 59], [56, 58], [57, 57], [58, 56], [59, 55],
        [60, 72], [61, 71], [62, 70], [63, 69], [64, 68], [65, 75],
        [66, 74], [67, 73], [68, 64], [69, 63], [70, 62], [71, 61],
        [72, 60], [73, 67], [74, 66], [75, 65], [76, 82], [77, 81],
        [78, 80], [79, 79], [80, 78], [81, 77], [82, 76], [83, 87],
        [84, 86], [85, 85], [86, 84], [87, 83], [88, 92], [89, 91],
        [90, 90], [91, 89], [92, 88], [93, 95], [94, 94], [95, 93],
        [96, 97], [97, 96]
    ]
}

lo_eye_corner_index_x = {'300W': 72, 'AFLW': 20, 'COFW': 26, 'WFLW': 120}
lo_eye_corner_index_y = {'300W': 73, 'AFLW': 21, 'COFW': 27, 'WFLW': 121}
ro_eye_corner_index_x = {'300W': 90, 'AFLW': 30, 'COFW': 38, 'WFLW': 144}
ro_eye_corner_index_y = {'300W': 91, 'AFLW': 31, 'COFW': 39, 'WFLW': 145}
l_eye_center_index_x = {'300W': [72, 74, 76, 78, 80, 82], 'AFLW': 22, 'COFW': 54, 'WFLW': 192}
l_eye_center_index_y = {'300W': [73, 75, 77, 79, 81, 83], 'AFLW': 23, 'COFW': 55, 'WFLW': 193}
r_eye_center_index_x = {'300W': [84, 86, 88, 90, 92, 94], 'AFLW': 28, 'COFW': 56, 'WFLW': 194}
r_eye_center_index_y = {'300W': [85, 87, 89, 91, 93, 95], 'AFLW': 29, 'COFW': 57, 'WFLW': 195}
nparts = {  # [chin, brow, nose, eyes, mouth], totally 5 parts
    '300W': [
        [0, 17], [17, 27], [27, 36], [36, 48], [48, 68]
    ],
    'WFLW': [
        [0, 33], [33, 51], [51, 60], [60, 76], [76, 96]
    ]
}


def generate_Bounday_heatmap(dataset, gt_coords, boundary_num):
    coord_x, coord_y, gt_heatmap = [], [], []
    for index in range(boundary_num):
        gt_heatmap.append(np.ones((128, 128)))
        gt_heatmap[index].tolist()
    boundary_x = {'chin': [], 'leb': [], 'reb': [], 'bon': [], 'breath': [], 'lue': [], 'lle': [],
                  'rue': [], 'rle': [], 'usul': [], 'lsul': [], 'usll': [], 'lsll': []}
    boundary_y = {'chin': [], 'leb': [], 'reb': [], 'bon': [], 'breath': [], 'lue': [], 'lle': [],
                  'rue': [], 'rle': [], 'usul': [], 'lsul': [], 'usll': [], 'lsll': []}
    points = {'chin': [], 'leb': [], 'reb': [], 'bon': [], 'breath': [], 'lue': [], 'lle': [],
              'rue': [], 'rle': [], 'usul': [], 'lsul': [], 'usll': [], 'lsll': []}

    for boundary_index in range(boundary_num):  # boundary_index：0-12
        for kp_index in range(point_range[dataset][boundary_index][0], point_range[dataset][boundary_index][1]):
            boundary_x[boundary_keys[boundary_index]].append(gt_coords[kp_index, 0])
            boundary_y[boundary_keys[boundary_index]].append(gt_coords[kp_index, 1])
        if boundary_keys[boundary_index] in boundary_special.keys() and dataset in boundary_special[
            boundary_keys[boundary_index]]:
            # tmp = gt_coords[duplicate_point[dataset][boundary_keys[boundary_index]],0]
            boundary_x[boundary_keys[boundary_index]].append(
                gt_coords[duplicate_point[dataset][boundary_keys[boundary_index]], 0])
            boundary_y[boundary_keys[boundary_index]].append(
                gt_coords[duplicate_point[dataset][boundary_keys[boundary_index]], 1])

    for k_index, k in enumerate(boundary_keys):
        if point_num_per_boundary[dataset][k_index] >= 2.:
            if len(boundary_x[k]) == len(set(boundary_x[k])) or len(boundary_y[k]) == len(set(boundary_y[k])):
                # print(k_index)
                points[k].append(boundary_x[k])
                points[k].append(boundary_y[k])
                res = splprep(points[k], s=0.0, k=1)
                u_new = np.linspace(res[1].min(), res[1].max(), interp_points_num[k])
                boundary_x[k], boundary_y[k] = splev(u_new, res[0], der=0)  # 利用B样条和它的导数进行插值，

    for index, k in enumerate(boundary_keys):
        if point_num_per_boundary[dataset][index] >= 2.:  # 边界包含的点的数量大于等于2
            for i in range(len(boundary_x[k]) - 1):  # i 从0 到边界包含点的数量-1
                # 起点到终点划线，元素值设为0
                cv2.line(gt_heatmap[index], (int(boundary_x[k][i]), int(boundary_y[k][i])),
                         (int(boundary_x[k][i + 1]), int(boundary_y[k][i + 1])), 0)
        else:
            cv2.circle(gt_heatmap[index], (int(boundary_x[k][0]), int(boundary_y[k][0])), 2, 0, -1)
        gt_heatmap[index] = np.uint8(gt_heatmap[index])
        # 利用distanceTransform计算像素距离矩阵，离边界越近值越接近于0，相差一个像素距离为1
        gt_heatmap[index] = cv2.distanceTransform(gt_heatmap[index], cv2.DIST_L2, 5)
        gt_heatmap[index] = np.float32(np.array(gt_heatmap[index]))
        gt_heatmap[index] = gt_heatmap[index].reshape(128 * 128)  # 拉成一列，像素距离小于6的，使用指数进行概率转换
        # 将与边界线距离小于3* sigma的点，使用指数处理/2 * sigma * sigma计算概率值
        (gt_heatmap[index])[(gt_heatmap[index]) < 6] = \
            np.exp(-(gt_heatmap[index])[(gt_heatmap[index]) < 6] *
                   (gt_heatmap[index])[(gt_heatmap[index]) < 6] / (2. * sigma2 * sigma2))
        (gt_heatmap[index])[(gt_heatmap[index]) >= 6] = 0.001
        gt_heatmap[index] = gt_heatmap[index].reshape([128, 128])
        # gt_heatmap_tmp = gt_heatmap[0]
    return np.array(gt_heatmap)


def draw_gaussian(img, pt, sigma, type='Gaussian'):
    # Draw a 2D gaussian
    # Adopted from https://github.com/anewell/pose-hg-train/blob/master/src/pypose/draw.py

    # Check that any part of the gaussian is in-bounds
    ul = [int(pt[0] - 3 * sigma), int(pt[1] - 3 * sigma)]
    br = [int(pt[0] + 3 * sigma + 1), int(pt[1] + 3 * sigma + 1)]
    if (ul[0] >= img.shape[1] or ul[1] >= img.shape[0] or
            br[0] < 0 or br[1] < 0):
        # If not, just return the image as is
        return torch.from_numpy(img)

    # Generate gaussian
    size = 6 * sigma + 1
    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    # The gaussian is not normalized, we want the center value to equal 1
    if type == 'Gaussian':
        g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
    elif type == 'Cauchy':
        g = sigma / (((x - x0) ** 2 + (y - y0) ** 2 + sigma ** 2) ** 1.5)

    # Usable gaussian range
    g_x = max(0, -ul[0]), min(br[0], img.shape[1]) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], img.shape[0]) - ul[1]
    # Image range
    img_x = max(0, ul[0]), min(br[0], img.shape[1])
    img_y = max(0, ul[1]), min(br[1], img.shape[0])

    img[img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
    return img


def show_preds(image, preds, image_name):

    image = cv2.UMat(image).get()
    for idx, pred in enumerate(preds):
        # plt.scatter(pred[:, 0], pred[:, 1], s=10, marker='.', c='r')
        image = cv2.circle(image, (int(pred[0]), int(pred[1])), 1, (0, 255, 0), 2)
        image = cv2.putText(image, str(idx), (int(pred[0] + 3), int(pred[1]) - 3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

    plt.imshow(image)  # pause a bit so that plots are updated
    plt.savefig("D:\\Download\\" + image_name + ".png", bbox_inches='tight', dpi=500)



def show_img(image):
    plt.figure()
    plt.imshow(image)
    plt.show()


class AlignDiffDataset(Dataset):
    def __init__(
            self, tsv_file, pic_dir="", condition_dir="", label_num=1,
            transform=None,
            width=256, height=256, channels=3,
            means=(127.5, 127.5, 127.5), scale=1 / 127.5,
            classes_num=None, crop_op=True, aug_prob=0.0,
            edge_info=None, flip_mapping=None, is_train=True, is_generate=False,
            debug=False
    ):
        super(AlignDiffDataset, self).__init__()

        self.items = pd.read_csv(tsv_file, sep="\t")  # 读取tsv文件，含有标签信息
        self.pic_dir = pic_dir  # 图片路径
        self.condition_dir = condition_dir  # 条件图路径

        # 检查 classes_num 列表的长度和给定的标签信息数目是否一致
        assert label_num == len(classes_num)

        self.landmark_num = classes_num[0]  # COFW: 29, 300W: 68, WFLW: 98
        self.transform = transform  # transform 预设为 transforms.Compose([transforms.ToTensor()])

        self.image_width = width  # W
        self.image_height = height  # H
        self.channels = channels  # C
        assert self.image_width == self.image_height  # 检查图片长宽是否相同
        self.means = means  # WHC三个参数的平均值，means=[127, 127, 127]
        self.scale = scale  # 缩放比例，scale = 1/127.5

        self.aug_prob = aug_prob  # 做 augmentation 的概率/比率
        self.edge_info = edge_info  # 边信息，在 aligndiff.py 中给出
        self.is_train = is_train  # 是否是训练
        self.is_generate = is_generate  # 是否是生成模式
        self.debug = debug  # 是否是debug

        std_lmk_5pts = np.array([
            196.0, 226.0, 316.0, 226.0, 256.0, 286.0, 220.0, 360.4, 292.0, 360.4], np.float32
        ) / 256.0 - 1.0
        std_lmk_5pts = np.reshape(std_lmk_5pts, (5, 2))  # 排列成5行2列 [-1 1]
        target_face_scale = 1.0 if crop_op else 1.25  # 是否做了图片裁剪，决定目标人脸的尺寸为1倍还是1.25倍

        self.augmentation = Augmentation(
            is_train=self.is_train,  # 是否为训练
            aug_prob=self.aug_prob,  # 做 augmentation 的概率/比例
            image_size=self.image_width,  # 图片尺寸，设置为图片宽度
            crop_op=crop_op,  # 是否做图片裁剪
            std_lmk_5pts=std_lmk_5pts,  # std_lmk_5pts
            target_face_scale=target_face_scale,  # 目标尺寸
            flip_rate=0.5,  # 翻转比例
            flip_mapping=flip_mapping,  # 翻转映射
            random_shift_sigma=0.05,  # 随机shift
            random_rot_sigma=math.pi / 180 * 18,  # 随机旋转（18°）
            random_scale_sigma=0.1,  # 随机缩放（±10%）
            random_gray_rate=0.2,  # 随机灰度（±20%）
            random_occ_rate=0.4,  # 随机遮挡（±40%）
            random_blur_rate=0.3,  # 随机模糊（±30%）
            random_gamma_rate=0.2,
            random_nose_fusion_rate=0.2
        )

    def _circle(self, img, pt, sigma=1.0, label_type='Gaussian'):
        # 检查是否在边界内
        # Check that any part of the gaussian is in-bounds
        tmp_size = sigma * 3
        ul = [int(pt[0] - tmp_size), int(pt[1] - tmp_size)]
        br = [int(pt[0] + tmp_size + 1), int(pt[1] + tmp_size + 1)]
        if (ul[0] > img.shape[1] - 1 or ul[1] > img.shape[0] - 1 or
                br[0] - 1 < 0 or br[1] - 1 < 0):
            # 如果图片不在边界内，则按原图片返回
            # If not, just return the image as is
            return img

        # Generate gaussian
        size = 2 * tmp_size + 1
        x = np.arange(0, size, 1, np.float32)
        y = x[:, np.newaxis]
        x0 = y0 = size // 2
        # The gaussian is not normalized, we want the center value to equal 1
        if label_type == 'Gaussian':
            g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
        else:
            g = sigma / (((x - x0) ** 2 + (y - y0) ** 2 + sigma ** 2) ** 1.5)

        # Usable gaussian range
        g_x = max(0, -ul[0]), min(br[0], img.shape[1]) - ul[0]
        g_y = max(0, -ul[1]), min(br[1], img.shape[0]) - ul[1]
        # Image range
        img_x = max(0, ul[0]), min(br[0], img.shape[1])
        img_y = max(0, ul[1]), min(br[1], img.shape[0])

        img[img_y[0]:img_y[1], img_x[0]:img_x[1]] = 255 * g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
        return img

    def _polylines(self, img, lmks, is_closed, color=255, thickness=1, draw_mode=cv2.LINE_AA,
                   interpolate_mode=cv2.INTER_AREA, scale=4):
        h, w = img.shape
        img_scale = cv2.resize(img, (w * scale, h * scale), interpolation=interpolate_mode)
        lmks_scale = (lmks * scale + 0.5).astype(np.int32)
        cv2.polylines(img_scale, [lmks_scale], is_closed, color, thickness * scale, draw_mode)
        img = cv2.resize(img_scale, (w, h), interpolation=interpolate_mode)
        return img

    def _generate_pointmap(self, points, scale=0.25, sigma=1.5):
        """
        generate point heatmap
        :param points: ndarray (68,2) for each landmark
        :param scale:
        :param sigma:
        :return: pointmaps:([68,64,64])
        """
        h, w = self.image_height, self.image_width  # 256, 256
        pointmaps = []
        for i in range(len(points)):
            pointmap = np.zeros([h, w], dtype=np.float32)
            # align_corners: False.
            point = copy.deepcopy(points[i])
            point[0] = max(0, min(w - 1, point[0]))
            point[1] = max(0, min(h - 1, point[1]))
            pointmap = self._circle(pointmap, point, sigma=sigma)

            pointmaps.append(pointmap)
        pointmaps = np.stack(pointmaps, axis=0) / 255.0  # [68*(256,256)] => (68,256,256)
        pointmaps = torch.from_numpy(pointmaps).float().unsqueeze(0)  # (68,256,256) => ([1,68,256,256)]
        pointmaps = F.interpolate(
            pointmaps,
            size=(int(w * scale), int(h * scale)),
            mode='bilinear',
            align_corners=False
        ).squeeze()

        # print("pointmaps: ", pointmaps.sum())
        return pointmaps  # ([68,64,64])

    def _generate_edgemap(self, points, scale=0.25, thickness=1):
        """
        generate edge heatmap
        :param points: ndarray (68,2) for each landmark
        :param scale:
        :param thickness:
        :return: edgemaps:([9,64,64])
        """
        h, w = self.image_height, self.image_width  # 256, 256
        edgemaps = []
        for is_closed, indices in self.edge_info:
            edgemap = np.zeros([h, w], dtype=np.float32)

            # indices: (17,2), (5,2), (5,2), (4,2), (5,2), (6,2), (6,2), (12,2), (8,2)
            part = copy.deepcopy(points[np.array(indices)])

            part = self._fit_curve(part, is_closed)
            part[:, 0] = np.clip(part[:, 0], 0, w - 1)
            part[:, 1] = np.clip(part[:, 1], 0, h - 1)
            edgemap = self._polylines(edgemap, part, is_closed, 255, thickness)

            # offset = 0.5
            # part = (part + offset).astype(np.int32)
            # part[:, 0] = np.clip(part[:, 0], 0, w-1)
            # part[:, 1] = np.clip(part[:, 1], 0, h-1)
            # cv2.polylines(edgemap, [part], is_closed, 255, thickness, cv2.LINE_AA)

            edgemaps.append(edgemap)
        edgemaps = np.stack(edgemaps, axis=0) / 255.0  # [9*(256,256)] => (9,256,256)
        edgemaps = torch.from_numpy(edgemaps).float().unsqueeze(0)  # ([1,9,256,256])
        edgemaps = F.interpolate(
            edgemaps,
            size=(int(w * scale), int(h * scale)),
            mode='bilinear',
            align_corners=False
        ).squeeze()

        # print("edgemaps: ", edgemaps.sum())
        return edgemaps  # ([9,64,64])

    def _fit_curve(self, lmks, is_closed=False, density=5):  # (17,2), False
        try:
            x = lmks[:, 0].copy()  # col1
            y = lmks[:, 1].copy()  # col2
            if is_closed:
                x = np.append(x, x[0])
                y = np.append(y, y[0])
            tck, u = interpolate.splprep([x, y], s=0, per=is_closed, k=3)
            # bins = (x.shape[0] - 1) * density + 1
            # lmk_x, lmk_y = interpolate.splev(np.linspace(0, 1, bins), f)
            intervals = np.array([])
            for i in range(len(u) - 1):
                intervals = np.concatenate((intervals, np.linspace(u[i], u[i + 1], density, endpoint=False)))
            if not is_closed:
                intervals = np.concatenate((intervals, [u[-1]]))
            lmk_x, lmk_y = interpolate.splev(intervals, tck, der=0)
            # der_x, der_y = interpolate.splev(intervals, tck, der=1)
            curve_lmks = np.stack([lmk_x, lmk_y], axis=-1)
            # curve_ders = np.stack([der_x, der_y], axis=-1)
            # origin_indices = np.arange(0, curve_lmks.shape[0], density)

            return curve_lmks
        except:
            return lmks

    def _image_id(self, image_path):
        if not os.path.exists(image_path):
            image_path = os.path.join(self.pic_dir, image_path)
        return hashlib.md5(open(image_path, "rb").read()).hexdigest()

    def _load_image(self, image_path):
        if not os.path.exists(image_path):
            image_path = os.path.join(self.pic_dir, image_path)

        try:
            # img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)#HWC, BGR, [0-255]
            img = cv2.imread(image_path, cv2.IMREAD_COLOR)  # HWC, BGR, [0-255]
            # img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # RGB
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

    def _compose_rotate_and_scale(self, angle, scale, shift_xy, from_center, to_center):
        cosv = math.cos(angle)
        sinv = math.sin(angle)

        fx, fy = from_center
        tx, ty = to_center

        acos = scale * cosv
        asin = scale * sinv

        a0 = acos
        a1 = -asin
        a2 = tx - acos * fx + asin * fy + shift_xy[0]

        b0 = asin
        b1 = acos
        b2 = ty - asin * fx - acos * fy + shift_xy[1]

        rot_scale_m = np.array([
            [a0, a1, a2],
            [b0, b1, b2],
            [0.0, 0.0, 1.0]
        ], np.float32)
        return rot_scale_m

    def _transformPoints2D(self, points, matrix):
        """
        points (nx2), matrix (3x3) -> points (nx2)
        """
        dtype = points.dtype

        # nx3
        points = np.concatenate([points, np.ones_like(points[:, [0]])], axis=1)
        points = points @ np.transpose(matrix)  # nx3
        points = points[:, :2] / points[:, [2, 2]]
        return points.astype(dtype)

    def _transformPerspective(self, image, matrix, target_shape):
        """
        image, matrix3x3 -> transformed_image
        """
        return cv2.warpPerspective(
            image, matrix,
            dsize=(target_shape[1], target_shape[0]),
            flags=cv2.INTER_LINEAR, borderValue=0
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

    def _denorm_points(self, points, h, w, align_corners=False):
        if align_corners:
            # [-1, +1] -> [0, SIZE-1]
            des_points = (points + 1) / 2 * torch.tensor([w - 1, h - 1]).to(points).view(1, 1, 2)
        else:
            # [-1, +1] -> [-0.5, SIZE-0.5]
            des_points = ((points + 1) * torch.tensor([w, h]).to(points).view(1, 1, 2) - 1) / 2
        return des_points

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):

        path_prefix = "D:/Datasets/300W/"
        # path_prefix = "/home/info/hjw/300W/"

        sample = dict()

        # 0: image_path
        image_path = self.items.iloc[index, 0]
        image_path = path_prefix + image_path

        image_name = image_path.split("/")[-1][:-4]
        sample["image_name"] = image_name

        pts_name = image_name + "_.pts"
        sample["pts_name"] = pts_name  # 新参考点的文件名

        con_kps_path = self.condition_dir + "/" + pts_name
        con_kps = load_from_pts(con_kps_path)

        # 1: landmarks_5pts
        landmarks_5pts = self.items.iloc[index, 1]
        # 将 landmarks_5pts 的元素转化为 float，然后返回迭代器，再转化为 list，变成5行2列
        landmarks_5pts = np.array(list(
            # 语法：map(function，iterable,…)
            # 将函数 function 依次作用到列表 iterable 的每个元素上，并返回以函数结果作为元素的列表
            map(float, landmarks_5pts.split(","))
        ), dtype=np.float32).reshape(5, 2)

        # 2: landmarks_target
        landmarks_target = self.items.iloc[index, 2]
        landmarks_target = np.array(list(
            map(float, landmarks_target.split(","))
        ), dtype=np.float32).reshape(self.landmark_num, 2)  # in 300W landmark_num = 68

        # 3: scale
        scale = float(self.items.iloc[index, 3])

        # 4, 5: center_w, center_h
        center_w, center_h = float(self.items.iloc[index, 4]), float(self.items.iloc[index, 5])

        # if it has 6, 6: tags
        if len(self.items.iloc[index]) > 6:
            tags = np.array(list(map(lambda x: int(float(x)), self.items.iloc[index, 6].split(","))))
        else:
            tags = np.array([])

        # image path
        sample["image_path"] = image_path

        # HWC, BGR, [0, 255] (256,256,3)
        img = self._load_image(image_path)
        assert img is not None

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        show_preds(img, con_kps, image_name)

        # augmentation
        _, con_kps, matrix = self.augmentation.process(
            img=img,
            lmk=con_kps,
            lmk_5pts=landmarks_5pts,
            scale=scale,
            center_w=center_w,
            center_h=center_h
        )  # ndarray(H,W,C), (68,2)

        inverse = np.linalg.inv(matrix)  # 获取逆变换矩阵
        sample["inverse"] = inverse

        # show_img(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        img, landmarks_target, _ = self.augmentation.process(
            img=img,
            lmk=landmarks_target,
            lmk_5pts=landmarks_5pts,
            scale=scale,
            center_w=center_w,
            center_h=center_h
        )  # ndarray(H,W,C), (68,2)

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # show_img(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        # 标准化关键点
        landmarks = self._norm_points(
            points=torch.from_numpy(landmarks_target),  # (68,2)
            h=self.image_height,  # 256
            w=self.image_width  # 256
        )  # ([68,2]) 0-255

        landmarks_tmp = landmarks_target / 2
        cond_lmk_tmp = con_kps / 2

        target_points = np.zeros([
            landmarks.shape[0],
            self.image_height // 2,
            self.image_height // 2
        ])
        for n in range(landmarks.shape[0]):
            target_points[n] = draw_gaussian(
                target_points[n], landmarks_tmp[n], sigma=3
            )  # 构造训练的 heatmap 的标签
        # target_points: (68,128,128)

        condition_heatmap = np.zeros([
            landmarks.shape[0],
            self.image_height // 2,
            self.image_width // 2
        ])
        for n in range(landmarks.shape[0]):
            condition_heatmap[n] = draw_gaussian(
                condition_heatmap[n], cond_lmk_tmp[n], sigma=3
            )  # 构造训练的 heatmap 的标签

        # for i in range(68):
        #     temp = condition_heatmap[i]
        #     plt.imshow(temp)
        #     plt.show()

        # test = np.zeros_like(condition_heatmap[0])
        # for i in range(condition_heatmap.shape[0]):
        #     temp = condition_heatmap[i]
        #     test += temp
        # # 归一化
        # test = np.clip(test, 0, 1)  # 确保不超过 1
        # plt.figure(dpi=300)
        # plt.imshow(test)
        # plt.show()

        target = target_points
        target = torch.from_numpy(target).float()

        # landmarks = tensor([N, 68, 2])
        # target = tensor([N,68,128,128])
        sample["label"] = [landmarks, target]

        # image normalization
        img = img.transpose(2, 0, 1).astype(np.float32)  # CHW, BGR, [0, 255]
        img[0, :, :] = (img[0, :, :] - self.means[0]) * self.scale
        img[1, :, :] = (img[1, :, :] - self.means[1]) * self.scale
        img[2, :, :] = (img[2, :, :] - self.means[2]) * self.scale

        if not self.is_train and not self.is_generate:  # only test
            images_flip = img[:, :, ::-1].copy()
            images_flip = torch.from_numpy(images_flip)  # ([3,256,256])
            imgs = torch.stack([torch.from_numpy(img), images_flip])  # ([2,3,256,256])
            condition_heatmap_flip = condition_heatmap[:, :, ::-1].copy()  # (81,128,128)
            heatmap = torch.stack([
                torch.from_numpy(condition_heatmap),
                torch.from_numpy(condition_heatmap_flip)
            ])  # ([2,81,128,128])
            sample["heatmap"] = heatmap
            sample["data"] = imgs

        else:  # is train or generate
            heatmap = condition_heatmap
            sample["heatmap"] = heatmap
            sample["data"] = torch.from_numpy(img)

        # landmarks = tensor([N, 68, 2])
        # target = tensor([N,68,128,128])
        sample["label"] = [landmarks, target]
        sample["tags"] = tags
        sample["rgb"] = img_rgb
        return sample


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


if __name__ == "__main__":
    run_code = 0
