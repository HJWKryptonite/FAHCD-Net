#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/7/30 13:06
# @Author  : HuJiwei
# @File    : NME.py
# @Software: PyCharm
# @Project: AlignDiff

import numpy as np


class NME:
    def __init__(self, nme_left_index, nme_right_index):
        self.nme_left_index = nme_left_index
        self.nme_right_index = nme_right_index

    def __repr__(self):
        return "NME()"

    def test(self, label_pd, label_gt):
        """

        :param label_pd: label of predict data, ([N,68,2])
        :param label_gt: label of ground truth, ([N,68,2])
        :return: ip, io, total
        """
        sum_nme_ip = 0
        sum_nme_io = 0
        total_cnt = 0
        label_pd = label_pd.data.cpu().numpy()
        label_gt = label_gt.data.cpu().numpy()
        for i in range(label_gt.shape[0]):
            landmarks_gt = label_gt[i]
            landmarks_pv = label_pd[i]
            lcenter = (
                              landmarks_gt[36, :] + landmarks_gt[37, :] +
                              landmarks_gt[38, :] + landmarks_gt[39, :] +
                              landmarks_gt[40, :] + landmarks_gt[41, :]
                      ) / 6
            rcenter = (
                              landmarks_gt[42, :] + landmarks_gt[43, :] +
                              landmarks_gt[44, :] + landmarks_gt[45, :] +
                              landmarks_gt[46, :] + landmarks_gt[47, :]
                      ) / 6
            interpupil = np.linalg.norm(lcenter - rcenter)

            interocular = np.linalg.norm(landmarks_gt[self.nme_left_index] - landmarks_gt[self.nme_right_index])

            # landmarks_gt = {ndarray: (68, 2)}
            # landmarks_pv = {ndarray: (68, 2)}
            landmarks_delta = landmarks_pv - landmarks_gt
            nme_ip = (np.linalg.norm(landmarks_delta, axis=1) / interpupil).mean()
            nme_io = (np.linalg.norm(landmarks_delta, axis=1) / interocular).mean()

            sum_nme_ip += nme_ip
            sum_nme_io += nme_io
            total_cnt += 1
        return sum_nme_ip, sum_nme_io, total_cnt


def main():
    pass


if __name__ == "__main__":
    main()
