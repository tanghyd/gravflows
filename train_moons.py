import os
import random

import numpy as np

from tqdm import tqdm
from argparse import ArgumentParser
from pathlib import Path

import torch
import torch.nn as nn

from torch.utils.data import DataLoader
# from torch.cuda.amp import autocast

import torch.distributed as distributed
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

import torchvision

# local imports
from models import flows
from moons import TwoMoonsDataset, map_colours, cascade_log_probs

def train(
    batch_size: int=1024,
    epochs: int=5,
    interval: int=1,
    video_title: str='flow',
    cmap: str='viridis',
    fps: float=50,
    cascade: bool=True,
):
    # Reproducible State (random seed)
    seed = 0 
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    device = torch.device('cuda')

    n = batch_size*1000
    dataset = TwoMoonsDataset(n)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
    )

    flow = flows.create_NDE_model(
        input_dim=2,
        context_dim=1,
        num_flow_steps=5,
        base_transform_kwargs={'base_transform_type': 'rq-coupling'}
    )
    
    optimizer = torch.optim.Adam(flow.parameters())
    flow.to(device)
    flow.train()

    # video
    video_dir = Path('moons/figures')
    video_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    nrow = len(flow._transform._transforms) if cascade else 1

    # input tensors for evaluating likelihood with flow model
    if interval != 0:
        xline = torch.linspace(-1.5, 2.5, 100)
        yline = torch.linspace(-.75, 1.25, 100)
        xgrid, ygrid = torch.meshgrid(xline, yline)  # (x,y) co-ordinate grid
        xyinput = torch.cat([xgrid.reshape(-1, 1), ygrid.reshape(-1, 1)], dim=1).to(device)
        zeroes = torch.zeros(10000, 1, device=device)
        ones = torch.ones(10000, 1, device=device)

    # train
    with tqdm(total=epochs*len(dataloader), desc='Training on Two Moons Dataset') as progress:
        for epoch in range(epochs):
            progress.set_postfix(epoch=epoch)
            for i, (x, y) in enumerate(dataloader):
                optimizer.zero_grad()               
                
                x = x.to(device)
                y = y.to(device)
                
                # estimate negative log likelihood conditional on class
                # each class corresponds to a cluster of data points - a "moon"
                loss = -flow.log_prob(x, context=y).mean()
                    
                loss.backward()
                optimizer.step()

                if (interval != 0) and ((i + 1) % interval == 0):
                    with torch.inference_mode():
                        # compute likelihood of observing 10,000 (x,y) co-ordinate pairs conditioned on label
                        if cascade:
                            # compute intermediate log probabilities and stack
                            zgrid0 = cascade_log_probs(flow, xyinput, zeroes).exp()
                            zgrid1 = cascade_log_probs(flow, xyinput, ones).exp()
                        else:
                            zgrid0 = flow.log_prob(xyinput, zeroes).exp()
                            zgrid1 = flow.log_prob(xyinput, ones).exp()
                    
                        # reshape all samples to 2D images (100x100); rotate 90 degrees counter-clockerwise
                        zgrids = torchvision.transforms.functional.rotate(
                            torch.cat([
                                zgrid.reshape(nrow, 100, 100)
                                for zgrid in (zgrid0, zgrid1)
                            ]),
                            angle=90.,
                        )
                        
                        # append grid of visualised posterior densities for each of 0 and 1
                        frames.append(torchvision.utils.make_grid(
                            torch.stack([
                                torch.tensor(map_colours(zgrid, cmap=cmap)[:,:,:3])  # RGB from RGB-A
                                for zgrid in zgrids
                            ]).permute(0,3,1,2),  # make_grid requires (B, C, H, W)
                            nrow=nrow  # number of flow transforms (1 if cascade is False)
                        ))
                        
                progress.update()
    
    if interval != 0:
        torchvision.io.write_video(
            filename=str(video_dir / f'{video_title}.mp4'),
            video_array=torch.stack(frames).permute(0,2,3,1),  # write_video requires (B, H, W, C)
            fps=fps
        )

# utility functions for distributed training
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(12355)
    distributed.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    distributed.destroy_process_group()

def train_distributed(
    rank: int,
    world_size: int,
    batch_size: int=1024,
    epochs: int=5,
    interval: int=1,
    video_title: str='flow',
    cmap: str='viridis',
    fps: float=50,
    cascade: bool=True,
):
    # Reproducible State (random seed)
    seed = 0 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # train function is run on each gpu with a local rank
    setup(rank, world_size)  # world size is total gpus
    torch.cuda.set_device(rank)

    n = batch_size*1000
    dataset = TwoMoonsDataset(n)

    # recommended for pytorch native DDP
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=rank,
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        sampler=sampler,
        pin_memory=True,
        persistent_workers=True,
    )

    flow = flows.create_NDE_model(
        input_dim=2,
        context_dim=1,
        num_flow_steps=5,
        base_transform_kwargs={'base_transform_type': 'rq-coupling'}
    )

    # Initialise data distributed parallel wrapper for pytorch model
    # flow = nn.SyncBatchNorm.convert_sync_batchnorm(flow)
    flow = DDP(flow.to(rank), device_ids=[rank], output_device=rank)
    optimizer = torch.optim.Adam(flow.parameters())

    if rank==0:    
        # video
        video_dir = Path('moons/figures')
        video_dir.mkdir(parents=True, exist_ok=True)
        frames = []
        nrow = len(flow.module._transform._transforms) if cascade else 1
        
        if interval != 0:
            # input tensors for evaluating likelihood with flow model
            xline = torch.linspace(-1.5, 2.5, 100)
            yline = torch.linspace(-.75, 1.25, 100)
            xgrid, ygrid = torch.meshgrid(xline, yline)  # (x,y) co-ordinate grid
            xyinput = torch.cat([xgrid.reshape(-1, 1), ygrid.reshape(-1, 1)], dim=1).to(rank)
            zeroes = torch.zeros(10000, 1, device=rank)
            ones = torch.ones(10000, 1, device=rank)

    # train
    disable_pbar = False if (rank == 0) else True  # tqdm progress bar
    with tqdm(total=epochs*len(dataloader), disable=disable_pbar, desc='Training on Two Moons Dataset') as progress:
        for epoch in range(epochs):
            if rank == 0: progress.set_postfix(epoch=epoch)
            
            # let all processes sync up before starting with a new epoch of training
            distributed.barrier()

            for i, (x, y) in enumerate(dataloader):
                if not flow.training: flow.train()
                optimizer.zero_grad()

                x = x.to(rank)
                y = y.to(rank)

                # estimate negative log likelihood conditional on class
                # each class corresponds to a cluster of data points - a "moon"
                loss = -flow.module.log_prob(x, context=y).mean()

                loss.backward()
                    
                if rank==0:
                    progress.update(1)

                    # validation and visualisation on gpu 0
                    if (interval != 0) and ((i + 1) % interval == 0):
                        with torch.inference_mode():
                            
                            # compute likelihood of observing data for 10,000 (x,y) co-ordinate pairs conditioned on class
                            if cascade:
                                # compute intermediate log probabilities and stack for video
                                zgrid0 = cascade_log_probs(flow.module, xyinput, zeroes).exp()  # class = 0
                                zgrid1 = cascade_log_probs(flow.module, xyinput, ones).exp()  # class = 1
                            else:
                                zgrid0 = flow.module.log_prob(xyinput, zeroes).exp()
                                zgrid1 = flow.module.log_prob(xyinput, ones).exp()
                        
                            zgrid0 = zgrid0.detach().cpu()
                            zgrid1 = zgrid1.detach().cpu()

                            # reshape all samples to 2D images (100x100)
                            # rotate 90 degrees counter-clockerwise to fix indexing on image
                            zgrids = torchvision.transforms.functional.rotate(
                                torch.cat([
                                    zgrid.reshape(nrow, 100, 100)
                                    for zgrid in (zgrid0, zgrid1)
                                ]),
                                angle=90.,
                            )
                            
                            # append grid of visualised posterior densities for each of 0 and 1
                            frames.append(torchvision.utils.make_grid(
                                torch.stack([
                                    torch.tensor(map_colours(zgrid, cmap=cmap)[:,:,:3])  # RGB from RGB-A
                                    for zgrid in zgrids
                                ]).permute(0,3,1,2),  # make_grid requires (B, C, H, W)
                                nrow=nrow  # number of flow transforms (1 if cascade is False)
                            ))

    if interval != 0:
        # write video to disk
        if rank==0:
            torchvision.io.write_video(
                filename=str(video_dir / f'{video_title}.mp4'),
                video_array=torch.stack(frames).permute(0,2,3,1),  # write_video requires (B, H, W, C)
                fps=fps
            )

    # destroy processes from distributed training
    cleanup()

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--num_gpus', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--interval', type=int, default=1)
    parser.add_argument('--video_title', type=str, default='flow')
    parser.add_argument('--cmap', type=str, default='viridis')
    parser.add_argument('--fps', type=float, default=50)
    parser.add_argument('--cascade', default=False, action="store_true", help="Whether to show intermediate layers in visualiation.")
    args = parser.parse_args()

    assert isinstance(args.num_gpus, int)
    assert args.num_gpus > 0 and args.num_gpus <= torch.cuda.device_count()

    if args.num_gpus == 1:
        train(**args.__dict__),
    else:
        # torch multiprocessing
        mp.spawn(
            train_distributed,  # train function specifically for DDP
            args=(args.batch_size, args.num_gpus, args.epochs, args.interval, args.video_title, args.cmap, args.fps, args.cascade),
            nprocs=args.num_gpus,
            join=True
        )

