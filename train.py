import os
import traceback
import json
import datetime

from tqdm import tqdm
from typing import Union, Optional
from argparse import ArgumentParser
from pathlib import Path

import numpy as np

import corner
import matplotlib.pyplot as plt

import torch

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# data distributed parallel
import torch.distributed as distributed
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# local imports
from gravflows import nde_flows
from gravflows.utils.parameters import (
    ParameterGenerator, compute_parameter_statistics,
    get_parameter_latex_labels
)
from gravflows.utils.strain import get_standardization_factor
from gravflows.utils.pytorch import StrainDataset, BasisDataset


def setup_nccl(rank, world_size):
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(12355)
    distributed.init_process_group("nccl", rank=rank, world_size=world_size, timeout=datetime.timedelta(0, 180))

def cleanup_nccl():
    distributed.destroy_process_group()

def sample_flow(
    flow,
    nsamples: int=50000,
    context: Optional[Union[np.ndarray, torch.Tensor]]=None,
    batch_size: int=512,
    output_device: Union[str, torch.device]='cpu',
    inference_mode: bool=True,
):
    """Draw samples from the posterior.
    
    The nsf package concatenates on the wrong dimension (dim=0 instead of dim=1).
        
    Arguments:
        flow {Flow} -- NSF model
        y {array} -- strain data
        nsamples {int} -- number of samples desired

    Keyword Arguments:
        device {torch.device} -- model device (CPU or GPU) (default: {None})
        batch_size {int} -- batch size for sampling (default: {512})

    Returns:
        Tensor -- samples
    """
    single_batch = len(context.shape) == 1
    with torch.inference_mode(inference_mode):
        if context is not None:
            if not isinstance(context, torch.Tensor):
                context = torch.from_numpy(context)
            if single_batch:
                # if 1 context tensor provided, unsqueeze batch dim
                context = context.unsqueeze(0)

        num_batches = nsamples // batch_size
        num_leftover = nsamples % batch_size

        samples = [flow.sample(batch_size, context).to(output_device) for _ in range(num_batches)]
        if num_leftover > 0:
            samples.append(flow.sample(num_leftover, context).to(output_device))


        samples = torch.cat(samples, dim=1)
        
        if single_batch:
            return samples[0]
        return samples

def generate_figures(queue: mp.Queue, tb: SummaryWriter):
    # to do - better handle latex for figure formatting
    param_labels = get_parameter_latex_labels()
    labels = [
        param_labels[param] for param in [
        'mass_1', 'mass_2', 'phase', 'a_1', 'a_2',
        'tilt_1', 'tilt_2', 'phi_12', 'phi_jl', 'theta_jn',
        'psi', 'ra', 'dec', 'time', 'distance']
    ]
    
    while True:
        try:
            (samples, ground_truths, epoch) = queue.get()
            
            fig = corner.corner(
                samples,
                levels=[0.5, 0.9],
                scale_hist=True,
                plot_datapoints=False,
                labels=labels,
                show_titles=True,
                fontsize=18,
                truths=ground_truths[0].numpy()
            #     range=corner_range,
            )

            fig.suptitle('Neural Spline Flow Posteriors', fontsize=22)
            fig.tight_layout()
            tb.add_figure('corner', fig, epoch)
        except Exception as e:
            traceback.print_exc()


# Training with PyTorch native DataDistributedParallel (DDP)
def train_distributed(
    rank: int,
    world_size: int,
    lr: float=0.0002,
    epochs: int=500,
    interval: int=100,
    verbose: bool=False,
):
    assert epochs > interval > 0, "Interval must be a positive integer between 0 and epochs."

    # setup data distributed parallel training
    setup_nccl(rank, world_size)  # world size is total gpus
    torch.cuda.set_device(rank)  # rank is gpu index
    device = torch.device('cuda')

    # visualisation setup
    if rank == 0:  
        tb = SummaryWriter()  # tensorboard

        # setup asynchronous figure generation worker process
        queue = mp.SimpleQueue()
        figure_writer = mp.Process(target=generate_figures, args=(queue, tb))
        figure_writer.start()
    
    # repository for pre-generated waveform data
    waveform_dir = Path('/mnt/datahole/daniel/gravflows/complex128_4s_centered')
    with (waveform_dir / 'static_args.json').open('r') as arg_file:
        static_args = json.load(arg_file)
        
    save_dir = Path('model_weights/')
    experiment_dir = save_dir / datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    # experiment_dir = save_dir / '2021-09-09_01:49:21'
    experiment_dir.mkdir(exist_ok=True)

    n = 200  # number of reduced basis elements
    batch_size = 2000

    dataset = BasisDataset(
        waveform_dir,
        static_args,
        file_name='bandpassed_standardized_basis.npy'
    )

    sampler = DistributedSampler(
        dataset,
        shuffle=True,
        num_replicas=world_size,
        rank=rank,
        seed=rank,
    )


    dataloader = DataLoader(
        dataset,
        shuffle=False,
        num_workers=6,
        batch_size=batch_size,
        sampler=sampler,
        pin_memory=True,
        persistent_workers=True,
        worker_init_fn=dataset._worker_init_fn,
    )


    # parameter standardization factor
    intrinsics_ini = 'gravflows/config_files/intrinsic_params.ini'
    extrinsics_ini = 'gravflows/config_files/extrinsic_params.ini'
    IntrinsicsGenerator = ParameterGenerator(config_files=intrinsics_ini)
    ExtrinsicsGenerator = ParameterGenerator(config_files=extrinsics_ini)
    prior_bounds = {
        **IntrinsicsGenerator.distribution.bounds,
        **ExtrinsicsGenerator.distribution.bounds,
    }

    # [2, 15] tensor (0: mean; 1: std)
    parameter_stats = (
        torch.from_numpy(compute_parameter_statistics(prior_bounds).values)
        .to(device, torch.float32)
    )

    flow = nde_flows.create_NDE_model(
        input_dim=15, 
        context_dim=4*n,
        num_flow_steps=15,
        base_transform_kwargs={
            'base_transform_type': 'rq-coupling',
            'batch_norm': True,
            'num_transform_blocks': 10,
        }
    )

    # sync_bn_flow = nn.SyncBatchNorm.convert_sync_batchnorm(flow, process_group=None)

    flow = DDP(flow.to(rank), device_ids=[rank], output_device=rank)
    optimizer = torch.optim.Adam(flow.parameters(), lr=lr)

    # tqdm progress bar
    disable = True
    if verbose:
        if rank == 0:
            disable = False

    # train
    flow.train()
    train_loss = torch.zeros((1,), device=device, requires_grad=False)
    with tqdm(total=len(dataloader)*epochs, desc='Training Neural Spline Flow', disable=disable) as loop:
        for epoch in range(1, 1+epochs):
            if rank == 0: loop.set_postfix({'epoch': epoch})
            iterator = iter(dataloader)
            basis, parameters = next(iterator) 
            basis = basis[:, :, :n].to(device, non_blocking=True)
            parameters = parameters.to(device, torch.float32, non_blocking=True)
            
            complete = False
            while not complete:
                optimizer.zero_grad()

                # standardize and flatten complex values
                parameters = (parameters - parameter_stats[None, :, 0]) / parameter_stats[None, :, 1]
                # basis *= standardization[None]  # scale real and imag for unit variance  [already done]
                basis = torch.cat([basis.real, basis.imag], dim=1).to(torch.float32)

                # flatten for 1-d residual network input
                basis = basis.reshape(basis.shape[0], basis.shape[1]*basis.shape[2])

                # negative log-likelihood conditional on strain over mini-batch
                loss = -flow.module.log_prob(parameters, context=basis).mean()

                try:
                    # async get data from CPU and move to GPU during model forward
                    basis, parameters = next(iterator) 
                    basis = basis[:, :, :n].to(device, non_blocking=True)
                    parameters = parameters.to(device, dtype=torch.float32, non_blocking=True)

                except StopIteration:
                    # exit while loop if iterator is complete
                    complete = True

                loss.backward()
                optimizer.step()

                # total loss summed over each sample in batch
                train_loss += loss.detach() * basis.shape[0]  # * 15 ??
                if rank == 0: loop.update(1)

            # gather total loss during epoch between each GPU worker as list of tensors
            world_loss = [torch.ones_like(train_loss) for _ in range(world_size)]
            distributed.all_gather(world_loss, train_loss)
            train_loss *= 0.0  # reset loss for next epoch

            if rank == 0:
                # log average epoch loss across DDP to tensorboard
                loss = torch.cat(world_loss).mean() / len(dataloader.dataset)
                tb.add_scalar(f'loss', loss.item(), epoch)
                tb.flush()

            if (interval != 0) and (epoch % interval == 0):
                # validation
                n_samples = 50000
                world_samples = [
                    torch.empty(
                        (n_samples // world_size, 15),
                        dtype=torch.float32,
                        device=device
                    ) for _ in range(world_size)
                ]

                # generate samples for corner plots etc.
                samples = sample_flow(flow, n_samples, device=rank)
                distributed.all_gather(world_samples, samples)
                world_samples = world_samples.cpu()

                if rank == 0:
                    # send samples to async process to generate matplotlib figures
                    samples = (samples - parameter_stats[None, :, 0]) / samples[None, :, 1]
                    ground_truths = (parameters * parameter_stats[None, :, 1]) + parameter_stats[None, :, 0]
                    queue.put((tb, world_samples, ground_truths, epoch))
                
                    # save checkpoint and write computationally expensive data to tb
                    torch.save(flow.module.state_dict(), experiment_dir / f'flow_{epoch}.pt')
                    torch.save(optimizer.state_dict(), experiment_dir / f'optimizer_{epoch}.pt')
                

            distributed.barrier()

        cleanup_nccl()  # destroy processes from distributed training

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--num_gpus', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--interval', type=int, default=2)
    parser.add_argument('--verbose', default=False, action="store_true")
    args = parser.parse_args()

    assert isinstance(args.num_gpus, int), "num_gpus argument must be an integer."
    assert args.num_gpus > 0 and args.num_gpus <= torch.cuda.device_count(), f"{args.num_gpus} not a valid number of GPU devices."

    # run pytorch training
    if args.num_gpus == 1:
        raise NotImplementedError()
        train()

    else:
        # data distributed parallel
        mp.spawn(
            train_distributed,
            args=(args.num_gpus, args.epochs, args.verbose),
            nprocs=args.num_gpus,
            join=True
        )