#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/7/26 13:00
# @Author  : HuJiwei
# @File    : script.py
# @Software: PyCharm
# @Project: AlignDiff
from torch.utils.data import DistributedSampler, DataLoader
from torchvision import transforms

from improved_diffusion.respace import SpacedDiffusion, space_timesteps
from improved_diffusion.unet import CDHN
from improved_diffusion import gaussian_diffusion as gd
from lib.dataset.aligndiff_dataset import AlignDiffDataset


def create_model(config):
    model = create_unet_model(
        config.image_size, config.deeper_net, config.attention_resolutions,
        config.num_channels, config.num_res_blocks, config.dropout,
        config.use_checkpoint, config.classes_num, config.num_heads,
        config.rrdb_blocks, config.edge_info
    )
    return model


def create_diffusion(config, phase):
    diffusion_steps = config.diffusion_steps  # 1000
    if phase == "train":
        timestep_respacing = config.timestep_respacing  # ""
    elif phase == "val":
        timestep_respacing = config.timestep_respacing_val  # "ddim50"
    elif phase == "test":
        timestep_respacing = config.timestep_respacing_test  # "ddim50"
    else:
        assert False
    diffusion = create_gaussian_diffusion(
        noise_schedule=config.noise_schedule,  # "cosine"
        steps=diffusion_steps,  # 1000
        use_kl=config.use_kl,  # False
        rescale_learned_sigmas=config.rescale_learned_sigmas,  # True
        timestep_respacing=timestep_respacing,  # ""/"ddim50"
        predict_xstart=config.predict_xstart,  # True
        sigma_small=config.sigma_small,  # False
        learn_sigma=config.learn_sigma,  # False
        rescale_timesteps=config.rescale_timesteps  # True
    )
    return diffusion


def create_unet_model(
        image_size, deeper_net, attention_resolution, num_channels,
        num_res_block, dropout, use_checkpoint, classes_num, num_heads,
        rrdb_blocks, edge_info
):
    if image_size == 256:  # This
        if deeper_net:
            channel_mult = (1, 1, 2, 2, 4, 4)
        else:  # True
            channel_mult = (1, 1, 1, 2, 2)
    elif image_size == 128:
        channel_mult = (1, 1, 2, 2, 4, 4)
    elif image_size == 64:
        channel_mult = (1, 2, 3, 4)
    elif image_size == 32:
        channel_mult = (1, 2, 2, 2)
    else:
        raise ValueError(f"unsupported image size: {image_size}px")

    attention_ds = []
    for res in attention_resolution.split(","):  # "16,8"  [256//16, 256//8]=[16,32]
        attention_ds.append(image_size // int(res))

    return CDHN(
        model_channels=num_channels,  # 128
        channel_mult=channel_mult,  # (1, 1, 1, 2, 2)
        num_res_blocks=num_res_block,  # 3
        dropout=dropout,  # 0.0
        attention_resolutions=tuple(attention_ds),  # [16,32]
        use_checkpoint=use_checkpoint,  # False
        classes_num=classes_num,  # [68, 13, 68]
        conv_resample=True,
        num_heads=num_heads,  # 4
        rrdb_blocks=rrdb_blocks,  # 1
        edge_info=edge_info
    )


def create_gaussian_diffusion(
        noise_schedule,  # cosine
        steps,  # 1000
        use_kl,  # False
        rescale_learned_sigmas,  # True
        timestep_respacing,  # "ddim50" / ""
        predict_xstart,  # True
        sigma_small,  # False
        learn_sigma,  # False
        rescale_timesteps  # True
):
    # 确定加噪方案
    betas = gd.get_named_beta_schedule(noise_schedule, steps)

    if use_kl:  # False
        loss_type = gd.LossType.RESCALED_KL
    elif rescale_learned_sigmas:  # True
        loss_type = gd.LossType.RESCALED_MSE
    else:
        loss_type = gd.LossType.MSE

    loss_type = gd.LossType.AWING

    if not timestep_respacing:  # "ddim50"
        timestep_respacing = [steps]

    return SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),  # set{}
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.EPSILON
            if not predict_xstart  # predict_xstart = True
            else gd.ModelMeanType.START_X  # this
        ),  # START_X
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE  # this
                if not sigma_small  # False
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma  # False
            else gd.ModelVarType.LEARNED_RANGE
        ),  # FIXED_LARGE
        loss_type=loss_type,  # AWING
        rescale_timesteps=rescale_timesteps  # True
    )


def get_dataset(config, tsv_file, pic_dir, condition_dir, is_train, is_generate):
    if config.loader_type == "aligndiff":
        dataset = AlignDiffDataset(
            tsv_file=tsv_file,
            pic_dir=pic_dir,
            condition_dir=condition_dir,
            label_num=config.label_num,
            transform=transforms.Compose([transforms.ToTensor()]),
            width=config.width,
            height=config.height,
            channels=config.channels,
            means=config.means,
            scale=config.scale,
            classes_num=config.classes_num,
            crop_op=config.crop_op,
            aug_prob=config.aug_prob,
            edge_info=config.edge_info,
            flip_mapping=config.flip_mapping,
            is_train=is_train,
            is_generate=is_generate
        )
    else:
        assert False
    return dataset


def get_dataloader(config, data_type, world_rank=0, world_size=1):
    if data_type == "train":
        dataset = get_dataset(
            config=config,
            tsv_file=config.train_tsv_file,
            pic_dir=config.train_pic_dir,
            condition_dir=config.train_cond_dir,
            is_train=True,
            is_generate=False
        )
        if world_size > 1:
            sampler = DistributedSampler(
                dataset, rank=world_rank, num_replicas=world_size, shuffle=True
            )
            loader = DataLoader(
                dataset, sampler=sampler, batch_size=config.batch_size // world_size,
                num_workers=config.train_num_workers, pin_memory=True, drop_last=True
            )
        else:
            loader = DataLoader(
                dataset, batch_size=config.batch_size, shuffle=True,
                num_workers=config.train_num_workers
            )
    elif data_type == "val":
        dataset = get_dataset(
            config=config,
            tsv_file=config.val_tsv_file,
            pic_dir=config.val_pic_dir,
            condition_dir=config.val_cond_dir,
            is_train=False,
            is_generate=False
        )
        loader = DataLoader(
            dataset, shuffle=False, batch_size=config.val_batch_size,
            num_workers=config.val_num_workers
        )

    elif data_type == "test":
        dataset = get_dataset(
            config=config,
            tsv_file=config.test_tsv_file,
            pic_dir=config.test_pic_dir,
            condition_dir=config.test_cond_dir,
            is_train=False,
            is_generate=False
        )
        loader = DataLoader(
            dataset, shuffle=False, batch_size=config.test_batch_size,
            num_workers=config.test_num_workers
        )
    elif data_type == "generate":
        dataset = get_dataset(
            config=config,
            tsv_file=config.test_tsv_file,
            pic_dir=config.test_pic_dir,
            condition_dir=config.test_cond_dir,
            is_train=False,
            is_generate=True
        )
        loader = DataLoader(
            dataset, shuffle=False, batch_size=config.gen_batch_size,
            num_workers=config.test_num_workers
        )
    elif data_type == "occlusion":
        dataset = get_dataset(
            config=config,
            tsv_file=config.occ_tsv_file,
            pic_dir=config.occ_pic_dir,
            condition_dir=config.test_cond_dir,
            is_train=False,
            is_generate=True
        )
        loader = DataLoader(
            dataset, shuffle=False, batch_size=config.gen_batch_size,
            num_workers=config.test_num_workers
        )
    else:
        assert False
    return loader


def main():
    train_diffusion = create_gaussian_diffusion(
            noise_schedule="cosine",
            steps=1000,
            use_kl=False,
            rescale_learned_sigmas=True,
            timestep_respacing="",
            predict_xstart=True, 
            sigma_small=False,
            learn_sigma=False,
            rescale_timesteps=True 
    )
    
    test_diffusion = create_gaussian_diffusion(
            noise_schedule="cosine",
            steps=1000,
            use_kl=False,
            rescale_learned_sigmas=True,
            timestep_respacing="ddim50",
            predict_xstart=True,
            sigma_small=False,
            learn_sigma=False,
            rescale_timesteps=True 
    )

    print(train_diffusion)
    print(test_diffusion)


if __name__ == "__main__":
    main()
