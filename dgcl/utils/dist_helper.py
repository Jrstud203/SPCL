import copy
import math
import os
import random
import subprocess

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data.sampler import Sampler


def setup_distributed(backend="nccl", port=None):
    """AdaHessian Optimizer
    Lifted from https://github.com/BIGBALLON/distribuuuu/blob/master/distribuuuu/utils.py
    Originally licensed MIT, Copyright (c) 2020 Wei Li
    """
    num_gpus = torch.cuda.device_count()

    if "SLURM_JOB_ID" in os.environ:
        # 可用作全局rank(进程的唯一标识符)  | # (用于多节点时)local_rank: os.environ['SLURM_LOCALID']
        rank = int(os.environ["SLURM_PROCID"])
        # World可以认为是一个集合，由一组能够互相发消息的进程组成
        # world_size: 参这组能够互相通信的进程的总数，也就是与job的进程数, 实际就是（一个GPU一个进程）GPU的个数；
        world_size = int(os.environ["SLURM_NTASKS"])
        # #从中取得一个ip作为通讯ip
        node_list = os.environ["SLURM_NODELIST"]
        addr = subprocess.getoutput(f"scontrol show hostname {node_list} | head -n1")

        # specify master port
        # MASTER_ADDR和MASTER_PORT是通信模块初始化需要的两个环境变量。
        # 由于是在单机上，所以用localhost的ip就可以了。
        if port is not None:
            os.environ["MASTER_PORT"] = str(port)
        elif "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = "10685"
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = addr
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["LOCAL_RANK"] = str(rank % num_gpus)
        os.environ["RANK"] = str(rank)
    else:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(rank % num_gpus)

    # 多进程初始化通信环境
    # init_method: 机器之间交换数据需要指定一个主节点, 这个参数用来指定主节点的。
    dist.init_process_group(
        # 后端, 实际上是多个机器之间交换数据的协议，官方和很多用户都强烈推荐'nccl'作为backend。
        # 但是nccl的接口只有5个，如果有其他诉求nccl比较受限，mpi也可考虑。
        backend=backend,
        world_size=world_size,
        rank=rank,
    )
    return rank, world_size


def gather_together(data):
    world_size = dist.get_world_size()
    gather_data = [torch.zeros_like(data).cuda() for _ in range(world_size)]
    dist.all_gather(gather_data, data)
    return gather_data


class DistributedGivenIterationSampler(Sampler):
    def __init__(
        self, dataset, total_iter, batch_size, world_size=None, rank=None, last_iter=-1
    ):
        if world_size is None:
            world_size = dist.get_world_size()
        if rank is None:
            rank = dist.get_rank()
        assert rank < world_size
        self.dataset = dataset
        self.total_iter = total_iter
        self.batch_size = batch_size
        self.world_size = world_size
        self.rank = rank
        self.last_iter = last_iter

        self.total_size = self.total_iter * self.batch_size

        self.indices = self.gen_new_list()
        self.call = 0

    def __iter__(self):
        if self.call == 0:
            self.call = 1
            return iter(self.indices[(self.last_iter + 1) * self.batch_size :])
        else:
            raise RuntimeError(
                "this sampler is not designed to be called more than once!!"
            )

    def gen_new_list(self):
        # each process shuffle all list with same seed, and pick one piece according to rank
        np.random.seed(0)

        all_size = self.total_size * self.world_size
        indices = np.arange(len(self.dataset))
        indices = indices[:all_size]
        num_repeat = (all_size - 1) // indices.shape[0] + 1
        indices = np.tile(indices, num_repeat)
        indices = indices[:all_size]

        np.random.shuffle(indices)
        beg = self.total_size * self.rank
        indices = indices[beg : beg + self.total_size]

        assert len(indices) == self.total_size

        return indices

    def __len__(self):
        # note here we do not take last iter into consideration, since __len__
        # should only be used for displaying, the correct remaining size is
        # handled by dataloader
        # return self.total_size - (self.last_iter+1)*self.batch_size
        return self.total_size
