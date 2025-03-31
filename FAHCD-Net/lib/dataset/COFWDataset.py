#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time     : 2024/12/20 14:53
# @Author   : HuJiwei
# @FileName : COFWDataset.py
# @Software : PyCharm
# @Project  : AlignDiff_8_stack_cascade
from torch.utils.data import Dataset
import h5py
import numpy as np
import cv2
import matplotlib.pyplot as plt


class COFWDataset(Dataset):
    def __init__(self, config, is_train=True, transform=None):
        if is_train:
            self.mat_file = "D:/Datasets/COFW/COFW_train_color.mat"
        else:
            self.mat_file = "D:/Datasets/COFW/COFW_test_color.mat"

        self.mat = h5py.File(self.mat_file, "r")
        if is_train:
            images = self.mat['IsTr']
            pts = self.mat['phisTr']
            data_size = len(images)
            repeat_num = int(1000 / data_size) + 1
            dataset_dicts_new = []
            dataset_pts_new = []
            for ii in range(repeat_num):
                dataset_dicts_new = dataset_dicts_new + list(images)
                dataset_pts_new = dataset_pts_new + list(pts)
            self.images = dataset_dicts_new[:1000]
            self.pts = dataset_pts_new[:1000]
        else:
            self.images = self.mat['IsT']
            self.pts = self.mat['phisT']

        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):

        img = self.images[index][0]

        if len(img.shape) == 2:
            img = img.reshape(img.shape[0], img.shape[1], 1)
            img = np.repeat(img, 3, axis=2)

        pts = self.pts[index][0:58].reshape(2, -1).transpose()

        show_preds(img, pts, str(index) + ".png")
        sample = {'img': img, 'pts': pts}
        return sample


def show_preds(image, preds, image_name):

    image = cv2.UMat(image).get()
    for idx, pred in enumerate(preds):
        # plt.scatter(pred[:, 0], pred[:, 1], s=10, marker='.', c='r')
        image = cv2.circle(image, (int(pred[0]), int(pred[1])), 1, (0, 255, 0), 2)
        image = cv2.putText(image, str(idx), (int(pred[0] + 3), int(pred[1]) - 3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

    plt.imshow(image)  # pause a bit so that plots are updated
    plt.savefig("D:\\Download\\" + image_name + ".png", bbox_inches='tight', dpi=500)
