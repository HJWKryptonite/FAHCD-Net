#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/7/21 22:15
# @Author  : HuJiwei
# @File    : align_diff.py
# @Software: PyCharm
# @Project: AlignDiff
import json
import os
from pathlib import Path

from config.base import Base
from lib import logger
from utils.json_encoder import JSONEncoder


class AlignDiff(Base):
    def __init__(self, work_dir="../", data_definition="300W"):
        super(AlignDiff, self).__init__("aligndiff", work_dir)

        self.description = "AlignDiff Configuration"

        # backbone
        self.net = "diffusionUNet"
        self.data_definition = data_definition
        self.loader_type = "aligndiff"

        # data

        self.data_dir = "./data/aligndiff"
        self.batch_size = 4  # origin=24
        self.val_batch_size = 20  # origin=20
        self.test_batch_size = 20
        self.gen_batch_size = 1
        self.height = 256
        self.width = 256
        self.channels = 3
        self.image_size = 256
        self.heatmap_size = 128
        self.means = (127.5, 127.5, 127.5)
        self.scale = 1 / 127.5
        self.aug_prob = 1.0

        # train_settings
        self.class_name = "train"
        self.log_interval = 60  # 打印训练信息/iteration

        # val_settings
        self.val_interval = 20  # val/epoch
        self.major_vote_number = 4
        self.val_model_path = "model/aligndiff/0.046374.pth.tar"
        self.is_flipped_val = False

        # parameters
        self.optimizer = "adam"
        self.learn_rate = 1e-4  # seg-diff lr  # 1e-4
        self.resume_checkpoint = ""
        self.display_iteration = 10  # 展示/iteration
        self.model_save_epoch = 10  # 保存/epoch
        self.milestones = [80, 150, 200]  # [200, 350, 450]
        self.gamma = 0.9  # MultistepLR Rate
        self.max_epoch = 300  # 500
        self.val_epoch = 10

        # model
        self.num_stack = 4
        self.num_channels = 128
        self.num_res_blocks = 3
        self.dropout = 0.0
        self.attention_resolutions = "16,8"
        self.use_checkpoint = False
        self.num_heads = 4
        self.num_heads_upsample = -1
        self.rrdb_blocks = 1  # origin = 10
        self.deeper_net = False

        # EMA
        self.ema = True
        self.ema_rate = 0.999

        # diffusion
        self.schedule_sampler = "uniform"  # 采样方式
        # self.schedule_sampler = "loss-second-moment"  # 采样方式
        self.class_cond = False

        self.noise_schedule = "cosine"
        self.diffusion_steps = 1000  # IDDPM 是 1000
        self.use_kl = False  # 是否使用 rescaled KL 散度
        self.rescale_learned_sigmas = True  # 是否使用 rescaled MSE，仅当 use_kl=False 时生效
        self.timestep_respacing = ""
        self.timestep_respacing_val = "ddim50"  # 50 step ddim respace
        self.timestep_respacing_test = "ddim50"  # 50 step ddim respace
        self.predict_xstart = True  # 预测 x_0
        self.sigma_small = False
        self.learn_sigma = False
        self.rescale_timesteps = True

        self.weight_decay = 0.0  # 权重衰减

        self.use_scale_shift_norm = False
        self.seed = None
        self.clip_denoised = True  # True or False

        # COFW
        if self.data_definition == "COFW":
            self.edge_info = (
                (True, (0, 4, 2, 5)),  # RightEyebrow
                (True, (1, 6, 3, 7)),  # LeftEyebrow
                (True, (8, 12, 10, 13)),  # RightEye
                (False, (9, 14, 11, 15)),  # LeftEye
                (True, (18, 20, 19, 21)),  # Nose
                (True, (22, 26, 23, 27)),  # LowerLip
                (True, (22, 24, 23, 25)),  # UpperLip
            )
            # self.nme_left_index = 8 # ocular
            # self.nme_right_index = 9 # ocular
            self.nme_left_index = 16  # pupils
            self.nme_right_index = 17  # pupils

            # 4 * [29, 7, 29] = [29, 7, 29, 29, 7, 29, 29, 7, 29, 29, 7, 29]
            self.classes_num = self.num_stack * [29, 7, 29]
            self.crop_op = True
            self.flip_mapping = (
                [0, 1], [4, 6], [2, 3], [5, 7], [8, 9], [10, 11], [12, 14], [16, 17], [13, 15], [18, 19], [22, 23],
            )

        # 300W
        elif self.data_definition == "300W":
            self.edge_info = (
                (False, (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)),  # FaceContour
                (False, (17, 18, 19, 20, 21)),  # RightEyebrow
                (False, (22, 23, 24, 25, 26)),  # LeftEyebrow
                (False, (27, 28, 29, 30)),  # NoseLine
                (False, (31, 32, 33, 34, 35)),  # Nose
                (True, (36, 37, 38, 39, 40, 41)),  # RightEye
                (True, (42, 43, 44, 45, 46, 47)),  # LeftEye
                (True, (48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59)),  # OuterLip
                (True, (60, 61, 62, 63, 64, 65, 66, 67)),  # InnerLip
            )
            self.nme_left_index = 36  # ocular
            self.nme_right_index = 45  # ocular

            # 4 * [68, 9, 68] = [68, 9, 68, 68, 9, 68, 68, 9, 68, 68, 9, 68]
            self.classes_num = self.num_stack * [68, 13, 68]

            self.crop_op = True
            self.flip_mapping = (
                [0, 16], [1, 15], [2, 14], [3, 13], [4, 12], [5, 11], [6, 10], [7, 9],
                [17, 26], [18, 25], [19, 24], [20, 23], [21, 22],
                [31, 35], [32, 34],
                [36, 45], [37, 44], [38, 43], [39, 42], [40, 47], [41, 46],
                [48, 54], [49, 53], [50, 52], [61, 63], [60, 64], [67, 65], [58, 56], [59, 55],
            )  # 翻转映射点关系
        # WFLW
        elif self.data_definition == "WFLW":
            self.edge_info = (
                (False, (
                    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
                    27,
                    28, 29, 30, 31, 32)),  # FaceContour
                (True, (33, 34, 35, 36, 37, 38, 39, 40, 41)),  # RightEyebrow
                (True, (42, 43, 44, 45, 46, 47, 48, 49, 50)),  # LeftEyebrow
                (False, (51, 52, 53, 54)),  # NoseLine
                (False, (55, 56, 57, 58, 59)),  # Nose
                (True, (60, 61, 62, 63, 64, 65, 66, 67)),  # RightEye
                (True, (68, 69, 70, 71, 72, 73, 74, 75)),  # LeftEye
                (True, (76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87)),  # OuterLip
                (True, (88, 89, 90, 91, 92, 93, 94, 95)),  # InnerLip
            )
            # self.nme_left_index = 96 # pupils
            # self.nme_right_index = 97 # pupils
            self.nme_left_index = 60  # ocular
            self.nme_right_index = 72  # ocular

            # 4 * [98, 9, 98] = [98, 9, 98, 98, 9, 98, 98, 9, 98, 98, 9, 98]
            self.classes_num = self.num_stack * [98, 9, 98]
            self.crop_op = True
            self.flip_mapping = (
                [0, 32], [1, 31], [2, 30], [3, 29], [4, 28], [5, 27], [6, 26], [7, 25], [8, 24], [9, 23], [10, 22],
                [11, 21], [12, 20], [13, 19], [14, 18], [15, 17],  # cheek
                [33, 46], [34, 45], [35, 44], [36, 43], [37, 42], [38, 50], [39, 49], [40, 48], [41, 47],  # elbrow
                [60, 72], [61, 71], [62, 70], [63, 69], [64, 68], [65, 75], [66, 74], [67, 73],
                [55, 59], [56, 58],
                [76, 82], [77, 81], [78, 80], [87, 83], [86, 84],
                [88, 92], [89, 91], [95, 93], [96, 97]
            )

        self.label_num = len(self.classes_num)  # label_num = 12

        # loss, criterion, metrics
        self.loss_lambda = 2.0
        self.loss_weights = []
        self.criterions = []
        self.metrics = []
        for i in range(self.num_stack):
            # 2^i/2^3=2^(i-3), i in range(4) => factor = 1/8, 1/4, 1/2, 1
            factor = (2 ** i) / (2 ** (self.num_stack - 1))
            # loss_weights = [1/8, 10/8, 10/8, 1/4, 10/4, 10/4, 1/2, 5, 5, 1, 10, 10]
            self.loss_weights += [factor * weight for weight in [1.0, 10.0, 10.0]]

            # 修改后的损失函数
            self.criterions += [
                "AnisotropicDirectionLoss",
                "AWingLoss",
                "AWingLoss"
            ]

            self.metrics += ["NME", None, None]
        self.key_metric_index = (self.num_stack - 1) * 3  # 3*3=9
        self.use_tags = False

        # self.platform = "linux"
        self.platform = "windows"

        if self.platform == "windows":
            self.data_prefix = "D:/Datasets/300W/"
        elif self.platform == "linux":
            self.data_prefix = "/home/info/hjw/300W/"

        # database
        self.database_300W = "D:/Datasets/300W"
        self.database_COFW = "D:/Datasets/COFW"
        self.database_WFLW = "D:/Datasets/WFLW"
        self.database_AFLW = "D:/Datasets/AFLW"

        # data
        self.train_tsv_file = os.path.join(self.data_dir, self.data_definition, "train.tsv")
        self.val_tsv_file = os.path.join(self.data_dir, self.data_definition, "test1.tsv")
        # self.test_tsv_file = os.path.join(self.data_dir, self.data_definition, "test1.tsv")
        self.test_tsv_file = os.path.join(self.data_dir, self.data_definition, "test1.tsv")
        self.occ_tsv_file = os.path.join(self.data_dir, self.data_definition, "occlusion.tsv")
        self.train_pic_dir = self.data_prefix
        self.val_pic_dir = self.data_prefix
        self.test_pic_dir = self.data_prefix
        self.occ_pic_dir = self.data_prefix
        # condition data
        self.train_cond_dir = self.data_prefix + "predict_300w/trainset"
        self.val_cond_dir = self.data_prefix + "predict_300w/ibug"
        self.test_cond_dir = self.data_prefix + "predict_300w/ibug"
        # self.test_cond_dir = "D:/datasets/300W/predict_300w/testset"


if __name__ == "__main__":
    aligndiff = AlignDiff("D:\\Workspace\\Python\\Diffusion\\AlignDiff")
    aligndiff.init_instance()
    aligndiff.logger.info(aligndiff.__dict__)
    (Path(aligndiff.log_dir) / 'args.json').write_text(
        json.dumps(aligndiff.__dict__, indent=4, cls=JSONEncoder)
    )
    logger.info(f"log folder path: {Path(aligndiff.log_dir).resolve()}")
