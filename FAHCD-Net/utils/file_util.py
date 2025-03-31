#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/11/16 15:37
# @Author  : HuJiwei
# @File    : file_util.py
# @Software: PyCharm
# @Project: AlignDiff_4_fixed_cond_heatmap
import os.path

import numpy as np


def load_from_pts(filename: str):
    landmarks = np.genfromtxt(filename, skip_header=3, skip_footer=1)
    landmarks = landmarks - 1
    return landmarks


def write_to_pts(filename: str, landmarks: np.ndarray):

    directory = os.path.dirname(filename)

    if not os.path.exists(directory):
        os.mkdir(directory)
        print(f"Directory {directory} created.")

    if not os.path.exists(filename):
        with open(filename, 'w') as f:
            # 可以根据需要写入文件的头部信息
            f.write("version: 1\n")
            f.write("n_points: {}\n".format(len(landmarks)))
            f.write("{\n")
            # 写入默认值或初始值
            for landmark in landmarks:
                f.write(f"{landmark[0]} {landmark[1]}\n")
            f.write("}\n")
        print(f"File {filename} created.")
    else:
        # 使用 np.savetxt 写入 landmarks 数据
        with open(filename, 'w') as f:
            # 写入文件的头部信息
            f.write("version: 1\n")
            f.write("n_points: {}\n".format(len(landmarks)))
            f.write("{\n")
            np.savetxt(f, landmarks, fmt='%.13f')
            f.write("}\n")
        print(f"Landmarks saved to {filename}.")


def main():
    # 示例使用
    landmarks = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    filename = "example.pts"
    write_to_pts(filename, landmarks)


if __name__ == "__main__":
    main()
