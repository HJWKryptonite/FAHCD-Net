#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/7/21 22:15
# @Author  : HuJiwei
# @File    : base.py
# @Software: PyCharm
# @Project: AlignDiff
import datetime
import json
from pathlib import Path

from tensorboardX import SummaryWriter

from lib import logger
from utils.json_encoder import JSONEncoder


class Base:
    def __init__(self, config_name, work_dir):
        # Base
        self.description = "Base Configuration"
        self.config_name = config_name
        self.time = str(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f"))

        # Path of data, model & log
        self.work_dir = work_dir  # default="../"
        self.data_dir = Path(self.work_dir) / "data" / Path(self.config_name)  # ./data/aligndiff
        self.model_dir = Path(self.work_dir) / "model" / Path(self.config_name, self.time)  # ./model/aligndiff/{time}
        self.log_dir = Path(self.work_dir) / "log" / Path(self.config_name, self.time)  # ./log/aligndiff/{time}
        self.point_dir = Path(self.work_dir) / "data/predict" / Path(self.time)  # ./data/predict/{time}

        # Path of data file
        self.train_tsv_file = Path(self.data_dir, "train.tsv")
        self.train_pic_dir = Path(self.data_dir, "images")
        self.val_tsv_file = Path(self.data_dir, "val.tsv")
        self.val_pic_dir = Path(self.data_dir, "images")
        self.test_tsv_file = Path(self.data_dir, "test1.tsv")
        self.test_pic_dir = Path(self.data_dir, "images")

        # net
        self.net = ""

        # num of cpu
        self.train_num_workers = 0
        self.val_num_workers = 0
        self.test_num_workers = 0

        # visualization
        self.writer = None

        # wandb
        self.wandb = None

        # log file
        self.logger = None

    def init_instance(self):
        # visualization
        self.writer = SummaryWriter(log_dir=str(self.log_dir), comment=self.config_name)

        # wandb
        """
        wandb_key = "3462de1f0c2817d194002922c8ffd438ff4c5b6c"  # to be changed to yours.
        if wandb_key is not None:
            wandb.login(key=wandb_key)
            wandb.init(project=self.type, dir=self.log_dir,
                       name=self.time, tensorboard=True, sync_tensorboard=True)
            self.wandb = wandb
        """

        # log file
        logger.configure(dir=str(self.log_dir))
        self.logger = logger

    def __del__(self):
        # tensorboard --logdir self.log_dir
        if self.writer is not None:
            # self.writer.export_scalars_to_json(self.log_dir + "visual.json")
            self.writer.close()


if __name__ == "__main__":
    base = Base("aligndiff", "D:\\Workspace\\Python\\Diffusion\\AlignDiff")
    # base = Base("aligndiff", "../")
    base.init_instance()
    base.logger.info(base.__dict__)
    (Path(base.log_dir) / 'args.json').write_text(
        json.dumps(base.__dict__, indent=4, cls=JSONEncoder)
    )
    logger.info(f"log folder path: {Path(base.log_dir).resolve()}")
