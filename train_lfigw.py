import os
import argparse

from tqdm import tqdm
from datetime import datetime, timedelta

from pathlib import Path

import torch

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# data distributed parallel
import torch.distributed as distributed
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from torch.distributed.optim import ZeroRedundancyOptimizer

# local imports
from models import flows
from gwpe.pytorch.lfigw_datasets import lfigwWaveformDataset

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def train(
    lr: float=2e-4,
    batch_size: int=1000,
    epochs: int=500,
    interval: int=10,
    save: int=0,
    num_workers: int=4,
    num_basis: int=100,
    dataset: str='datasets',
    coefficient_noise: bool=False,
    verbose: bool=False,
    profile: bool=False,
):
    assert 0 < batch_size, "batch_size must be a positive integer."
    assert 0 < epochs, "epochs must be a positive integer."

    device = torch.device('cuda')

    # directories
    data_dir = Path(f'/mnt/datahole/daniel/gravflows/{dataset}/train/')
    log_dir = f"{datetime.now().strftime('%b%d_%H-%M-%S')}_{os.uname().nodename}"
    tb = SummaryWriter(f'lfigw/runs/{log_dir}')

    save_dir = Path('lfigw/model_weights/')
    experiment_dir = save_dir / log_dir
    experiment_dir.mkdir(parents=True, exist_ok=True)

    # config files
    waveform_params_ini = str(data_dir / 'config_files/parameters.ini')
    extrinsics_ini = 'gwpe/config_files/extrinsics.ini'
    static_args_ini = str(data_dir / 'config_files/static_args.ini')

    # training data
    dataset = lfigwWaveformDataset(
        n=num_basis,
        data_dir='lfigw/data/train',
        basis_dir='lfigw/data/basis/',
        psd_dir='data/events/GW150914',
        data_file='coefficients.npy',
        static_args_ini=static_args_ini,
        intrinsics_ini=waveform_params_ini,
        extrinsics_ini=extrinsics_ini,
        ifos=['H1','L1'],
        ref_ifo='H1',
        downcast=True,
        add_noise=True,
        coefficient_noise=coefficient_noise,
        distance_scale=True,
        time_shift=False,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        persistent_workers=True,
        num_workers=num_workers,
        worker_init_fn=dataset._worker_init_fn,
        collate_fn=dataset._collate_fn,
    )


    # instantiate neural spline coupling flow
    flow = flows.create_NDE_model(
        input_dim=14,  # we do not predict coalescence time 
        context_dim=4*100,
        num_flow_steps=15,
        base_transform_kwargs={
            'base_transform_type': 'rq-coupling',
            'batch_norm': True,
            'num_transform_blocks': 10,
            'activation': 'elu',
        }
    )

    flow = flow.to(device)
    optimizer = torch.optim.Adam(flow.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # run training loop
    flow.train()
    train_loss = torch.zeros((1,), device=device, requires_grad=False)

    with tqdm(
        total=len(dataloader)*epochs,
        disable=not verbose,
        ncols=150,
        postfix={'epoch': 0},
        desc=f'[{log_dir}] Training'
    ) as progress:

        # set torch profiling runs
        # to do: optional pytorch profiler context manager https://www.py4u.net/discuss/192558
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=4, repeat=2),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(f'gwpe/runs/{log_dir}'),
            profile_memory=True,
            record_shapes=True,
            with_stack=True
        ) as profiler:

            for epoch in range(1, 1+epochs):
                for coefficients, parameters in dataloader:
                    progress.set_postfix({'epoch': epoch})
                    progress.set_description(f'[{log_dir}] Training', refresh=True)

                    optimizer.zero_grad()
                    
                    coefficients = coefficients.to(device, non_blocking=True)
                    parameters = parameters.to(device, non_blocking=True)

                    # negative log-likelihood conditional on strain over mini-batch
                    loss = -flow.log_prob(parameters, context=coefficients).mean()
                        
                    loss.backward()
                    optimizer.step()
                    if profile: profiler.step()

                    # total loss summed over each sample in batch (scale to match lfigw)
                    train_loss += loss.detach() * coefficients.shape[0]
                    train_loss *= (15/14)
                    progress.update(1)

                tb.add_scalar('loss/train', train_loss.item() / len(dataloader.dataset), epoch)
                train_loss *= 0  # reset epoch loss tracking

                scheduler.step()

                if (save != 0) and (epoch % save == 0):
                    # save checkpoint and write computationally expensive data to tb
                    torch.save(flow.state_dict(), experiment_dir / f'flow_{epoch}.pt')
                    torch.save(optimizer.state_dict(), experiment_dir / f'optimizer_{epoch}.pt')
                    torch.save(scheduler.state_dict(), experiment_dir / f'scheduler_{epoch}.pt')

def setup_nccl(rank, world_size):
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(12355)
    distributed.init_process_group("nccl", rank=rank, world_size=world_size, timeout=timedelta(0, 180))

def cleanup_nccl():
    distributed.destroy_process_group()

def print_peak_memory(prefix, device):
    if device == 0:
        print(f"{prefix}: {torch.cuda.max_memory_allocated(device) // 1e6}MB ")

# Training with PyTorch native DataDistributedParallel (DDP)
def train_distributed(
    rank: int,
    world_size: int,
    lr: float=2e-4,
    batch_size: int=1000,
    epochs: int=500,
    interval: int=10,
    save: int=0,
    num_workers: int=4,
    num_basis: int=100,
    dataset: str='datasets',
    coefficient_noise: bool=False,
    use_zero: bool=False,
    verbose: bool=False,
):
    assert 0 < batch_size, "batch_size must be a positive integer."
    assert 0 < epochs, "epochs must be a positive integer."
    assert (0 <= interval) and (interval <= epochs), "Interval must be a non-negative integer between 0 and epochs."
    assert (0 <= save) and (save <= epochs), "Save must be a non-negative integer between 0 and epochs."


    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    # setup data distributed parallel training
    setup_nccl(rank, world_size)  # world size is total gpus
    torch.cuda.set_device(rank)  # rank is gpu index

    # directories
    data_dir = Path(f'/mnt/datahole/daniel/gravflows/{dataset}/train/')
    log_dir = f"{datetime.now().strftime('%b%d_%H-%M-%S')}_{os.uname().nodename}"
    save_dir = Path('lfigw/model_weights/')
    experiment_dir = save_dir / log_dir
    experiment_dir.mkdir(parents=True, exist_ok=True)

    # config files
    waveform_params_ini = str(data_dir / 'config_files/parameters.ini')
    extrinsics_ini = 'gwpe/config_files/extrinsics.ini'
    static_args_ini = 'gwpe/config_files/static_args.ini'

    # tensorboard
    if rank == 0:
        tb = SummaryWriter(f'lfigw/runs/{log_dir}')

    # training data
    dataset = lfigwWaveformDataset(
        n=num_basis,
        data_dir='lfigw/data/train',
        basis_dir='lfigw/data/basis/',
        psd_dir='data/events/GW150914',
        data_file='coefficients.npy',
        static_args_ini=static_args_ini,
        intrinsics_ini=waveform_params_ini,
        extrinsics_ini=extrinsics_ini,
        ifos=['H1','L1'],
        ref_ifo='H1',
        downcast=True,
        add_noise=True,
        coefficient_noise=coefficient_noise,
        distance_scale=True,
        time_shift=False,
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
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        persistent_workers=True,
        sampler=sampler,
        num_workers=num_workers,
        prefetch_factor=4,
        worker_init_fn=dataset._worker_init_fn,
        collate_fn=dataset._collate_fn,
    )

    # instantiate neural spline coupling flow
    flow = flows.create_NDE_model(
        input_dim=14,  # we do not predict coalescence time 
        context_dim=4*100,
        num_flow_steps=15,
        base_transform_kwargs={
            'base_transform_type': 'rq-coupling',
            'batch_norm': True,
            'num_transform_blocks': 10,
            'activation': 'elu',
            'hidden_dim': 512,  # default
            'dropout_probability': 0.0,  # default
            'num_bins': 8,  # default
            'tail_bound': 1.0,  # default
            'apply_unconditional_transform': False,  # default
        }
    )

    flow = flow.to(rank)
    # print_peak_memory("Max memory allocated after creating local model", rank)

    # sync_bn_flow = nn.SyncBatchNorm.convert_sync_batchnorm(flow)
    flow = DDP(flow, device_ids=[rank], output_device=rank)

    # print_peak_memory("Max memory allocated after creating DDP", rank)

    if use_zero:
        #https://pytorch.org/tutorials/recipes/zero_redundancy_optimizer.html
        from torch.distributed.optim import ZeroRedundancyOptimizer
        optimizer = ZeroRedundancyOptimizer(
            flow.parameters(),
            optimizer_class=torch.optim.Adam,
            lr=lr,
            parameters_as_bucket_view=True,
        )
    else:
        optimizer = torch.optim.Adam(flow.parameters(), lr=lr)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)


    disable_pbar = False if verbose and (rank == 0) else True  # tqdm progress bar
    with tqdm(
        total=len(dataloader)*epochs,
        disable=disable_pbar,
        ncols=150,
        postfix={'epoch': 0},
        desc=f'[{log_dir}] Training'
    ) as progress:

        # run training loop
        train_loss = torch.zeros((1,), device=rank, requires_grad=False)

        for epoch in range(1, 1+epochs):
            flow.train()
            distributed.barrier()

            for coefficients, parameters in dataloader:
                if rank == 0:
                    progress.set_postfix({'epoch': epoch})
                    progress.set_description(f'[{log_dir}] Training', refresh=True)

                optimizer.zero_grad()
                
                coefficients = coefficients.to(rank, non_blocking=True)
                parameters = parameters.to(rank, non_blocking=True)

                # negative log-likelihood conditional on strain over mini-batch
                loss = -flow.module.log_prob(parameters, context=coefficients).mean()
                
                loss.backward()

                # print_peak_memory("Max memory allocated before optimizer step()", rank)
                optimizer.step()
                # print_peak_memory("Max memory allocated after optimizer step()", rank)

                # total loss summed over each sample in batch (scaled to lfigw)
                train_loss += loss.detach() * coefficients.shape[0] * (15/14)
                progress.update(1)

            scheduler.step()

            # gather total loss during epoch between each GPU worker as list of tensors
            world_loss = [torch.ones_like(train_loss) for _ in range(world_size)]
            distributed.all_gather(world_loss, train_loss)
            train_loss *= 0.0  # reset loss for next epoch

            if rank == 0:
                epoch_loss = torch.cat(world_loss).sum().item() / len(dataloader.dataset)
                tb.add_scalar('loss/train', epoch_loss, epoch)
                # if (interval != 0) and (epoch % interval == 0):

                if (save != 0) and (epoch % save == 0):
                    # save checkpoint and write computationally expensive data to tb
                    torch.save(flow.module.state_dict(), experiment_dir / f'flow_{epoch}.pt')
                    torch.save(optimizer.state_dict(), experiment_dir / f'optimizer_{epoch}.pt')
                    torch.save(scheduler.state_dict(), experiment_dir / f'scheduler_{epoch}.pt')


    cleanup_nccl()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_gpus', type=int, default=1)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--interval', type=int, default=1)
    parser.add_argument('--save', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=4, help="Number of workers for dataloader.")
    parser.add_argument('--num_basis', type=int, default=100, help="Number of reduced basis elements.")
    parser.add_argument('--dataset', type=str, default='datasets', help="Dataset directory name.")
    parser.add_argument('--coefficient_noise', default=False, action="store_true")
    parser.add_argument('--use_zero', default=False, action="store_true", help="Whether to use ZeroRedudancyOptimizer.")
    parser.add_argument('--verbose', default=False, action="store_true")

    args = parser.parse_args()

    # for k, v in args.__dict__.items():
    #     print(k, v)


    assert isinstance(args.num_gpus, int), "num_gpus argument must be an integer."
    assert args.num_gpus > 0 and args.num_gpus <= torch.cuda.device_count(), f"{args.num_gpus} not a valid number of GPU devices."
    
    if args.num_gpus == 1:
        # single gpu training (and profiler)
        ddp_keys = ('num_gpus, use_zero')
        train(**{key: val for key, val in args.__dict__.items() if key not in ddp_keys})
    else:
        # data distributed parallel
        mp.spawn(
            train_distributed,
            args=tuple(args.__dict__.values()),  # assumes parser loaded in correct order
            nprocs=args.num_gpus,
            join=True
        )