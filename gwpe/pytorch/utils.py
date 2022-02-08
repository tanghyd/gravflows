import os

from typing import Optional
from datetime import timedelta

import random
import numpy as np

import torch

import torch.distributed as distributed


def set_seed(seed: Optional[int]=None):
    # set random seed
    random.seed(seed)  # not required
    np.random.seed(seed)

    # cudnn reproducibility
    if seed is None:
        # non-deterministic
        seed = np.random.randint((2**32-1))
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
    else:
        # deterministic
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # torch seeds cannot be None
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # ddp

# distributed training
def setup_nccl(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(12355)
    distributed.init_process_group("nccl", rank=rank, world_size=world_size, timeout=timedelta(0, 180))

def cleanup_nccl():
    distributed.destroy_process_group()

# used to check effectiveness of ZeroRedundancyOptimizer
def print_peak_memory(prefix, device):
    if device == 0:
        print(f"{prefix}: {torch.cuda.max_memory_allocated(device) // 1e6}MB ")