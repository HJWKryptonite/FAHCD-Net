#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/7/21 22:13
# @Author  : HuJiwei
# @File    : image_train_diff_300W.py
# @Software: PyCharm
# @Project: AlignDiff
import json
import os
from pathlib import Path

from config.align_diff import AlignDiff
from improved_diffusion.resample import create_named_schedule_sampler
from lib import logger
from lib.ema import EMA
from lib.script import create_model, get_dataloader, create_diffusion
from lib.trainer import Trainer
from utils import dist_util
from utils.json_encoder import JSONEncoder
from utils.network_util import print_network
from utils.seed_generator import set_random_seed, set_random_seed_for_iterations


def main():
    # 1. Get config info
    config = AlignDiff("./")
    config.init_instance()
    log_dir = config.log_dir
    os.environ["OPENAI_LOGDIR"] = str(log_dir)
    # set_random_seed(MPI.COMM_WORLD.Get_rank(), deterministic=True)
    set_random_seed(0, deterministic=True)
    # set_random_seed_for_iterations(MPI.COMM_WORLD.Get_rank())
    set_random_seed_for_iterations(0)

    # 2. Load checkpoint
    if config.resume_checkpoint:
        resumed_checkpoint_arg = config.resume_checkpoint
        config.__dict__.update(json.loads((Path(config.resume_checkpoint) / 'args.json').read_text()))
        config.resume_checkpoint = resumed_checkpoint_arg

    # 3. Save config info into json file
    logger.info(config.__dict__)
    (Path(log_dir) / 'args.json').write_text(
        json.dumps(config.__dict__, indent=4, cls=JSONEncoder)
    )
    logger.info(f"log folder path: {Path(log_dir).resolve()}")

    # 4. Creating model and diffusion
    logger.log("Creating model and diffusion")
    model = create_model(config)
    model.to(dist_util.dev())
    # print_network(model, "UNetModel")
    if config.ema:
        ema = EMA(model, config.ema_rate)
        ema.register()
    else:
        ema = None

    train_diffusion = create_diffusion(config, "train")
    val_diffusion = create_diffusion(config, "val")
    schedule_sampler = create_named_schedule_sampler(config.schedule_sampler, train_diffusion)

    # 5. Creating data loader
    logger.log("creating data loader...")
    train_loader = get_dataloader(config, "train", world_rank=0, world_size=1)
    val_loader = get_dataloader(config, "val")

    # 6. Training
    logger.log("training...")
    Trainer(
        config=config,
        model=model,
        train_diffusion=train_diffusion,
        val_diffusion=val_diffusion,
        train_loader=train_loader,
        val_loader=val_loader,
        schedule_sampler=schedule_sampler,
        logger=logger,
        resume_checkpoint=None,
        ema=ema,
    ).run_loop()


if __name__ == "__main__":
    main()
