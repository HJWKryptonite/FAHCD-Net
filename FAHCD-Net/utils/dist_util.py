#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/7/26 12:46
# @Author  : HuJiwei
# @File    : dist_util.py
# @Software: PyCharm
# @Project: AlignDiff

"""
Helpers for distributed training.
"""

import io
import socket

import blobfile as bf
import torch as th
import torch.distributed as dist


def dev():
    """
    Get the device to use for torch.distributed.
    """
    if th.cuda.is_available():
        # return th.device(f"cuda:{MPI.COMM_WORLD.Get_rank() % GPUS_PER_NODE}")
        return th.device(f"cuda:0")
    return th.device("cpu")


def load_state_dict(path, **kwargs):
    """
    Load a PyTorch file without redundant fetches across MPI ranks.
    """
    with bf.BlobFile(path, "rb") as f:
        data = f.read()
    return th.load(io.BytesIO(data), **kwargs)


def sync_params(params):
    """
    Synchronize a sequence of Tensors across ranks from rank 0.
    """
    for p in params:
        with th.no_grad():
            dist.broadcast(p, 0)


def _find_free_port():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]
    finally:
        s.close()


def main():
    pass


if __name__ == "__main__":
    main()
