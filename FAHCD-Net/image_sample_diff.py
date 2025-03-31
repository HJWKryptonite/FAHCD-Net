#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/9/6 11:38
# @Author  : HuJiwei
# @File    : image_sample_diff_300W.py
# @Software: PyCharm
# @Project: AlignDiff
from pathlib import Path

from lib.sampler import Sampler
from config.align_diff import AlignDiff
from lib import logger
from lib.script import create_model, get_dataloader, create_diffusion
from utils import dist_util


def main():
    # 1. Get config info
    config = AlignDiff("./", data_definition="300W")
    config.init_instance()
    log_dir = config.log_dir

    # 2. Get trained model and diffusion
    # logger.log("Getting trained model and diffusion...")
    # model = create_model(config)
    # diffusion = create_diffusion(config, "test")

    # checkpoint = dist_util.load_state_dict(config.val_model_path, map_location="cpu")
    # model.load_state_dict(checkpoint["net"])
    # model.to(dist_util.dev())
    # model.eval()

    # 3. Creating data loader
    logger.log("Creating test data loader...")
    loader = get_dataloader(config, "generate")

    # 4. Set random seed
    seed = 1234 if config.seed is None else config.seed

    # 5. Create sample path
    logger.log("Creating sample log dir")
    (Path(log_dir) / "major_vote").mkdir(exist_ok=True)

    # 6. Sampling
    logger.log("Sampling...")
    sample = Sampler(
        config=config,
        model=None,
        diffusion=None,
        loader=loader,
        logger=logger
    )
    sample.default()


if __name__ == "__main__":
    main()
